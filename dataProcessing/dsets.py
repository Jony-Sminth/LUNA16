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
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch10_raw')

# 定义数据结构
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)

# 处理相关的常量
CT_HU_MIN = -1000
CT_HU_MAX = 1000
DEFAULT_CHUNK_SIZE = (32, 48, 48)

@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool: bool = True) -> List[CandidateInfoTuple]:
    """获取所有候选结节信息"""
    # 查找所有的.mhd文件
    mhd_list = glob.glob('data/subset*/*.mhd')
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    # 读取直径信息
    diameter_dict = {}
    with open('data/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )

    # 处理候选结节信息
    candidateInfo_list = []
    with open('data/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            
            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list

class Ct:
    def __init__(self, series_uid: str):
        """初始化CT实例"""
        mhd_path = glob.glob('data/subset*/{}.mhd'.format(series_uid))[0]
        
        # 读取CT图像
        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        
        # 记录图像尺寸
        self.dims = ct_a.shape
        
        # 裁剪HU值范围
        ct_a.clip(CT_HU_MIN, CT_HU_MAX, ct_a)
        
        self.series_uid = series_uid
        self.hu_a = ct_a
        
        # 保存用于坐标转换的空间信息
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)
        
        # Debug信息
        log.debug(f"CT {series_uid} loaded:")
        log.debug(f"Shape: {self.hu_a.shape}")
        log.debug(f"Origin: {self.origin_xyz}")
        log.debug(f"Spacing: {self.vxSize_xyz}")
        log.debug(f"Direction: {self.direction_a}")

    def getRawCandidate(self, center_xyz: tuple, width_irc: tuple) -> Tuple[np.ndarray, tuple]:
        """获取指定位置和大小的数据块"""
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        log.debug(f"Coordinates - XYZ: {center_xyz} -> IRC: {center_irc}")
        
        # 安全的坐标边界处理
        slice_list = []
        for axis, (center_val, width) in enumerate(zip(center_irc, width_irc)):
            start_ndx = int(round(center_val - width/2))
            end_ndx = int(round(start_ndx + width))
            
            # 确保不会超出图像边界
            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width)
            
            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(end_ndx - width)
                
            # 再次确保边界有效
            start_ndx = max(0, start_ndx)
            end_ndx = min(self.hu_a.shape[axis], end_ndx)
            
            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]
        
        # 如果提取的数据块大小不正确，进行填充
        final_shape = width_irc
        if ct_chunk.shape != width_irc:
            pad_list = []
            for i in range(3):
                pad_needed = final_shape[i] - ct_chunk.shape[i]
                if pad_needed > 0:
                    pad_before = pad_needed // 2
                    pad_after = pad_needed - pad_before
                    pad_list.append((pad_before, pad_after))
                else:
                    pad_list.append((0, 0))
            ct_chunk = np.pad(ct_chunk, pad_list, mode='constant', constant_values=CT_HU_MIN)

        return ct_chunk, center_irc

    @classmethod
    def clear_cache(cls):
        """清理类级别的缓存"""
        pass

@functools.lru_cache(1, typed=True)
def getCt(series_uid: str) -> Ct:
    """获取CT实例的缓存版本"""
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid: str, center_xyz: tuple, width_irc: tuple) -> Tuple[np.ndarray, tuple]:
    """获取处理后的CT数据块（带缓存）"""
    ct = getCt(series_uid)
    return ct.getRawCandidate(center_xyz, width_irc)

class LunaDataset(Dataset):
    def __init__(self,
                 val_stride: int = 0,
                 isValSet_bool: Optional[bool] = None,
                 series_uid: Optional[str] = None,
                 transform = None):
        """初始化数据集"""
        self.transform = transform
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]

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

    def __getitem__(self, ndx: int) -> tuple:
        """获取指定索引的数据项"""
        candidateInfo_tup = self.candidateInfo_list[ndx]
        width_irc = DEFAULT_CHUNK_SIZE

        try:
            candidate_a, center_irc = getCtRawCandidate(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
            )

            candidate_t = torch.from_numpy(candidate_a).float()
            candidate_t = candidate_t.unsqueeze(0)

            if self.transform is not None:
                candidate_t = self.transform(candidate_t)

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
            
        except Exception as e:
            log.error(f"Error processing candidate {candidateInfo_tup}: {str(e)}")
            raise

    def clear_cache(self):
        """清理所有相关的缓存"""
        getCt.cache_clear()  # 清理 getCt 的 lru_cache
        if hasattr(self, 'candidateInfo_list'):
            self.candidateInfo_list = []
        getCandidateInfoList.cache_clear()  # 清理 getCandidateInfoList 的 lru_cache
        log.info(f"Cleared cache for dataset {self!r}")

    def __del__(self):
        """析构函数，确保在对象销毁时清理缓存"""
        self.clear_cache()