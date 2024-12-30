"""
此模块实现了数据变换相关的功能。
主要与 dsets.py 中的 LunaDataset 类配合使用，完成 CT 图像数据的预处理工作。
采用构建器模式和责任链模式实现灵活的数据处理流水线。
"""

from dataclasses import dataclass
import numpy as np
import torch
from typing import Callable, Optional, List, Union
from abc import ABC, abstractmethod

class Transform(ABC):
    """
    Transform的基类
    定义了所有变换类必须实现的接口
    配合 dsets.py 中的 LunaDataset.__getitem__ 方法使用
    """
    @abstractmethod
    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        执行数据变换
        Args:
            x: 输入数据，可以是 torch.Tensor 或 numpy.ndarray
               来自 dsets.py 中的 CT 数据
        Returns:
            变换后的数据
        """
        pass

class TransformBuilder:
    """
    Transform构建器，通过参数配置构建变换方法
    使用构建器模式，实现灵活的变换流水线配置
    
    用于处理 dsets.py 中 LunaDataset 加载的 CT 数据
    可以配置多个变换步骤，如归一化、窗位窗宽调整等
    """
    def __init__(self):
        """初始化变换列表"""
        self.transforms = []
        
    def add_normalize(self, min_val: float = -1000.0, max_val: float = 1000.0) -> 'TransformBuilder':
        """
        添加归一化处理
        默认值与 dsets.py 中的 CT_HU_MIN 和 CT_HU_MAX 对应
        
        Args:
            min_val: 数据最小值，默认为 CT 图像的最小 HU 值
            max_val: 数据最大值，默认为 CT 图像的最大 HU 值
            
        Returns:
            self: 返回构建器实例，支持链式调用
        """
        def normalize_func(x):
            return (x - min_val) / (max_val - min_val) * 2 - 1
        self.transforms.append(('normalize', normalize_func))
        return self
        
    def add_window(self, center: float = 40, width: float = 400) -> 'TransformBuilder':
        """
        添加窗位窗宽处理
        用于调整 CT 图像的显示范围，突出感兴趣的组织结构
        
        Args:
            center: 窗位值，表示感兴趣区域的中心 HU 值
            width: 窗宽值，表示显示范围的宽度
            
        Returns:
            self: 返回构建器实例，支持链式调用
        """
        def window_func(x):
            min_value = center - (width / 2)
            max_value = center + (width / 2)
            return torch.clamp(x, min_value, max_value)
        self.transforms.append(('window', window_func))
        return self
        
    def add_custom(self, func: Callable, name: str) -> 'TransformBuilder':
        """
        添加自定义变换
        允许用户添加特定的数据处理步骤
        
        Args:
            func: 自定义变换函数
            name: 变换名称，用于调试和日志
            
        Returns:
            self: 返回构建器实例，支持链式调用
        """
        self.transforms.append((name, func))
        return self
        
    def build(self) -> Callable:
        """
        构建最终的变换函数
        使用责任链模式串联所有变换步骤
        
        Returns:
            transform_func: 组合后的变换函数，可直接传递给 LunaDataset 的 transform 参数
        """
        def transform(x):
            # 按添加顺序依次应用所有变换
            for name, func in self.transforms:
                x = func(x)
            return x
        return transform

# 使用示例注释：
# def create_transform(normalize: bool = True, 
#                     window: bool = False,
#                     custom_func: Optional[Callable] = None) -> Callable:
#     """
#     创建数据变换函数
#     根据参数动态构建数据处理流水线
#     
#     Args:
#         normalize: 是否进行归一化，默认为True
#         window: 是否进行窗位窗宽处理，默认为False
#         custom_func: 可选的自定义变换函数
#         
#     Returns:
#         Callable: 组合后的变换函数，可直接用于 LunaDataset 初始化
#     """
#     builder = TransformBuilder()
    
#     if normalize:
#         builder.add_normalize()  # 使用默认的 HU 值范围
#     if window:
#         builder.add_window()     # 使用默认的窗位窗宽值
#     if custom_func:
#         builder.add_custom(custom_func, "custom")
        
#     return builder.build()