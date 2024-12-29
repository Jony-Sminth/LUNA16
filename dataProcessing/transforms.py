from dataclasses import dataclass
import numpy as np
import torch
from typing import Callable, Optional, List, Union
from abc import ABC, abstractmethod

# 首先定义Transform基类
class Transform(ABC):
    """Transform的基类"""
    @abstractmethod
    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        pass

# 构建器模式
class TransformBuilder:
    """Transform构建器,通过参数配置构建变换方法"""
    def __init__(self):
        self.transforms = []
        
    def add_normalize(self, min_val: float = -1000.0, max_val: float = 1000.0) -> 'TransformBuilder':
        """添加归一化处理"""
        def normalize_func(x):
            return (x - min_val) / (max_val - min_val) * 2 - 1
        self.transforms.append(('normalize', normalize_func))
        return self
        
    def add_window(self, center: float = 40, width: float = 400) -> 'TransformBuilder':
        """添加窗位窗宽处理"""
        def window_func(x):
            min_value = center - (width / 2)
            max_value = center + (width / 2)
            return torch.clamp(x, min_value, max_value)
        self.transforms.append(('window', window_func))
        return self
        
    def add_custom(self, func: Callable, name: str) -> 'TransformBuilder':
        """添加自定义变换"""
        self.transforms.append((name, func))
        return self
        
    # 责任链模式
    def build(self) -> Callable:
        """构建最终的变换函数"""
        def transform(x):
            for name, func in self.transforms:
                x = func(x)
            return x
        return transform

# 使用:
# def create_transform(normalize: bool = True, 
#                     window: bool = False,
#                     custom_func: Optional[Callable] = None) -> Callable:
#     """
#     创建数据变换函数
#     动态构建处理链：根据参数决定使用哪些处理步骤
#     Args:
#         normalize: 是否进行归一化
#         window: 是否进行窗位窗宽处理
#         custom_func: 可选的自定义变换函数
#     """
#     builder = TransformBuilder()
    
#     if normalize:
#         builder.add_normalize()
#     if window:
#         builder.add_window()
#     if custom_func:
#         builder.add_custom(custom_func, "custom")
        
#     return builder.build()