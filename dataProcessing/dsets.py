# 导入必要的库
import copy
import csv
import functools
import glob
import os
from typing import List, Tuple, Dict, Optional
from collections import namedtuple
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

# 1. 基础配置

# 配置日志记录器
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# 初始化缓存系统
# raw_cache用于缓存处理后的数据块
raw_cache = getCache('part2ch10_raw')
# 用于跟踪已加载的CT数据
_loaded_ct_cache = {}

# 定义数据结构：用于存储候选结节的信息
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)

# 定义处理相关的常量
CT_HU_MIN = -1000  # CT值的最小值（空气的HU值）
CT_HU_MAX = 1000   # CT值的最大值（骨骼的典型HU值范围内）
DEFAULT_CHUNK_SIZE = (32, 48, 48)  # 默认数据块大小，用于提取结节区域

# 2. 数据读取

@functools.lru_cache(1)  # 使用LRU缓存，避免重复读取
def getCandidateInfoList(requireOnDisk_bool: bool = True) -> List[CandidateInfoTuple]:
    """
    获取所有候选结节信息
    
    Args:
        requireOnDisk_bool: 是否要求数据文件必须在磁盘上存在
        
    Returns:
        包含所有候选结节信息的列表
    """
    # 查找所有的.mhd文件
    mhd_list = glob.glob('data/subset*/*.mhd')
    # 获取所有可用的序列ID（不含扩展名）
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    # 读取直径信息，用字典存储以优化查找效率
    diameter_dict = {}
    with open('data/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:  # 跳过表头
            series_uid = row[0]
            # 将坐标字符串转换为浮点数元组
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            # 使用setdefault确保列表存在后再添加
            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    # 处理候选结节信息
    candidateInfo_list = []
    with open('data/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            
            # 如果要求数据在磁盘上存在且当前序列不存在，则跳过
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            
            # 转换结节标记（0/1）为布尔值
            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            # 确定候选点的直径
            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                # 检查候选点是否在标注结节的范围内
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:  # 使用1/4直径作为阈值
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            # 创建并存储候选结节信息
            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))

    # 按指定顺序排序（通常是按重要性）
    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

# 3. CT数据处理

class Ct:
    """CT扫描数据处理类"""
    # 类级别的缓存，用于存储CT实例
    _cache = {}
    _max_cache_size = 3  # 限制最大缓存数量，避免内存溢出
    
    def __init__(self, series_uid: str):
        """
        初始化CT实例
        
        Args:
            series_uid: CT序列的唯一标识符
        """
        # 查找并读取.mhd文件
        mhd_path = glob.glob('data/subset*/{}.mhd'.format(series_uid))[0]
        ct_mhd = sitk.ReadImage(mhd_path)  # 自动关联加载对应的.raw文件
        
        # 转换为numpy数组并进行HU值范围裁剪
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        ct_a.clip(CT_HU_MIN, CT_HU_MAX, ct_a)  # 裁剪到[-1000, 1000]范围
        
        self.series_uid = series_uid
        self.hu_a = ct_a  # 存储HU值数组
        
        # 保存用于坐标转换的空间信息
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())      # 物理空间原点（mm）
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())     # 体素大小（mm/voxel）
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)  # 方向矩阵

    def getRawCandidate(self, center_xyz: tuple, width_irc: tuple) -> Tuple[np.ndarray, tuple]:
        """
        获取指定位置和大小的数据块
        
        Args:
            center_xyz: 物理空间中的中心坐标（mm）
            width_irc: 需要提取的数据块大小（体素数）
            
        Returns:
            (数据块数组, 中心点的图像坐标)
        """
        # 将物理坐标转换为图像坐标
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        # 计算每个维度的切片范围
        slice_list = []
        for axis, (center_val, width) in enumerate(zip(center_irc, width_irc)):
            start_ndx = int(round(center_val - width/2))
            end_ndx = int(start_ndx + width)
            
            # 验证坐标有效性
            assert 0 <= center_val < self.hu_a.shape[axis], \
                f"Invalid center coordinate for {self.series_uid}"
            
            # 处理边界情况，确保不会超出数组范围
            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width)
            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(end_ndx - width)
                
            slice_list.append(slice(start_ndx, end_ndx))

        # 提取数据块
        ct_chunk = self.hu_a[tuple(slice_list)]
        return ct_chunk, center_irc

    @classmethod
    def clear_cache(cls):
        """清理类级别的缓存"""
        cls._cache.clear()

# 4. 缓存机制

@functools.lru_cache(1, typed=True)
def getCt(series_uid: str) -> Ct:
    """
    获取CT实例的缓存版本
    
    Args:
        series_uid: CT序列的唯一标识符
        
    Returns:
        缓存的CT实例
    """
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid: str, center_xyz: tuple, width_irc: tuple) -> Tuple[np.ndarray, tuple]:
    """
    获取处理后的CT数据块（带缓存）
    
    Args:
        series_uid: CT序列的唯一标识符
        center_xyz: 目标位置的物理坐标
        width_irc: 数据块的大小
        
    Returns:
        (处理后的数据块, 中心点的图像坐标)
    """
    ct = getCt(series_uid)
    return ct.getRawCandidate(center_xyz, width_irc)

# 5. 数据集实现

class LunaDataset(Dataset):
    """Luna数据集类，用于管理和访问CT数据"""
    
    def __init__(self,
                 val_stride: int = 0,
                 isValSet_bool: Optional[bool] = None,
                 series_uid: Optional[str] = None,
                 transform = None):
        """
        初始化数据集
        
        Args:
            val_stride: 验证集划分的步长
            isValSet_bool: 是否作为验证集
            series_uid: 指定的CT序列ID
            transform: 数据转换函数
        """
        self.transform = transform
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        # 如果指定了特定序列，只保留该序列的数据
        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]

        # 验证集划分逻辑
        # 使用 Python 的切片语法 [::val_stride] 来实现数据划分
        # 当 isValSet_bool=True 时，使用 self.candidateInfo_list[::val_stride] 提取验证集
        # 当 isValSet_bool=False 或未指定时，使用 del self.candidateInfo_list[::val_stride]
        # 这会删除验证集使用的样本，剩余的样本作为训练集。
        if isValSet_bool:
            assert val_stride > 0, "val_stride must be positive for validation set"
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list, "Empty validation set"
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list, "Empty training set"

        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
        ))

    def __len__(self) -> int:
        """返回数据集中样本的数量"""
        return len(self.candidateInfo_list)

    # 数据访问接口
    def __getitem__(self, ndx: int) -> tuple:
        """
        获取指定索引的数据项
        
        Args:
            ndx: 数据索引
            
        Returns:
            (数据张量, 标签, 序列ID, 中心坐标)
        """
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = DEFAULT_CHUNK_SIZE

        # 获取数据块，数据预处理：裁剪、归一化
        # 这里使用缓存机制，避免重复读取CT数据
        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

        # 转换为PyTorch张量并添加通道维度
        candidate_t = torch.from_numpy(candidate_a).float()
        candidate_t = candidate_t.unsqueeze(0)

        # 应用数据转换（如果有）
        if self.transform is not None:
            candidate_t = self.transform(candidate_t)

        # 准备标签：[不是结节的概率, 是结节的概率]
        pos_t = torch.tensor([
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool
        ], dtype=torch.long)

        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc),
        )

    def clear_cache(self):
        """清理所有相关的缓存"""
        if hasattr(self, 'preloaded_data'):
            del self.preloaded_data
        Ct.clear_cache()
        getCt.cache_clear()
        log.info(f"Cleared cache for dataset {self!r}")

    def __del__(self):
        """析构函数，确保在对象销毁时清理缓存"""
        self.clear_cache()