# LUNA16
A repository only for learning LUNA16.
not for any commercial purpose just for learning purpose.

```
- project_root/
    - README.md            # 项目说明文档
    - requirements.txt        # 项目所需库
    - model/                # 模型代码
        - model.py            # 模型定义
    - train/                # 训练代码
        - train.py            # 训练脚本
        - metrics.py      # 新增：评估指标相关代码
        - trainer.py      # 新增：训练器类
        - checkpoint_manager.py  # 检查点管理
    - checkpoints/           # 新建文件夹，存放模型文件
        - luna_models/       # 具体模型的检查点
            - version_1/     # 版本控制
                - best.pth   # 最佳模型
                - last.pth   # 最新模型
                - config.json # 训练配置
    - util/                # 工具代码
        - disk.py            # 内存缓存系统，缓存函数的返回结果，避免重复计算
        - logconf.py         # 统一的日志配置，帮助记录和调试程序运行过程
        - util.py           # 坐标转换相关
    - data/
        - annotations.csv        # 标注文件
        - candidates.csv         # 候选文件
    - dataProcessing/
        - transforms.ipynb      # 构建器模式的数据转换流程
        - transforms.py       # 数据转换代码
        - dsets.py            # 数据集定义
        - dsets_test.py       # 测试数据集定义
    - requirements.txt          # 项目所需库
    - .gitignore            # Git 忽略文件
```

## 代码结构
```python
# 基础库导入
import SimpleITK as sitk  # 医学影像处理库
import numpy as np
import torch
from torch.utils.data import Dataset
```

1. `CandidateInfoTuple` 的定义：
```python
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)
# 定义了一个命名元组，用于存储结节候选区信息：
# - isNodule_bool: 是否为结节
# - diameter_mm: 直径(毫米)
# - series_uid: 序列ID
# - center_xyz: 中心坐标
```

1. `getCandidateInfoList` 函数：
```python
@functools.lru_cache(1)  # 使用LRU缓存优化性能
def getCandidateInfoList(requireOnDisk_bool=True):
    # 读取磁盘上的CT数据文件
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    # 从annotations.csv读取标注信息
    # 从candidates.csv读取候选区信息
    # 返回处理后的候选区信息列表
```

1. `Ct` 类：
```python
class Ct:
    def __init__(self, series_uid):
        # 初始化CT图像对象
        # 读取.mhd文件
        # 进行Hounsfield单位的裁剪(-1000到1000)
        
    def getRawCandidate(self, center_xyz, width_irc):
        # 获取指定中心位置的CT图像块
        # 处理边界情况
        # 返回裁剪后的CT数据块
```

1. `LunaDataset` 类：
```python
class LunaDataset(Dataset):
    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None):
        # 初始化数据集
        # 可以选择训练集或验证集
        
    def __getitem__(self, ndx):
        # 获取单个数据样本
        # 返回:
        # - candidate_t: CT图像张量
        # - pos_t: 标签张量(是否为结节)
        # - series_uid: 序列ID
        # - center_irc: 中心坐标
```

主要功能点：
1. 数据加载和预处理：读取CT扫描数据(.mhd文件)和相关标注信息
2. 数据转换：将医学图像数据转换为PyTorch可用的张量格式
3. 结节检测：提供了结节候选区的提取和处理功能
4. 性能优化：使用了缓存机制(@functools.lru_cache)来提高性能
5. 数据集划分：支持训练集和验证集的划分

是医学图像处理流水线的预处理部分。