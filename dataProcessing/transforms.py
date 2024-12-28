# transforms.py
import numpy as np
import torch
from torch.nn.functional import interpolate

class Compose:
    """组合多个transform"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class NormalizeHU:
    """将HU值归一化到[-1,1]范围"""
    def __init__(self, min_hu=-1000.0, max_hu=1000.0):
        self.min_hu = min_hu
        self.max_hu = max_hu
        
    def __call__(self, x):
        x = (x - self.min_hu) / (self.max_hu - self.min_hu) * 2 - 1
        return x

class ToTensor:
    """将numpy数组转换为torch tensor"""
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x

class AddChannel:
    """添加channel维度"""
    def __call__(self, x):
        return x.unsqueeze(0) if isinstance(x, torch.Tensor) else x[None]

class Resize:
    """调整数据块大小"""
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            # 确保输入是5D的 (B,C,D,H,W)
            if x.dim() == 3:
                x = x.unsqueeze(0).unsqueeze(0)
            elif x.dim() == 4:
                x = x.unsqueeze(0)
            x = interpolate(x, size=self.size, mode='trilinear', align_corners=False)
            return x.squeeze(0)  # 移除batch维度
        return x

class RandomFlip:
    """随机翻转数据增强"""
    def __init__(self, p=0.5, dims=(1, 2, 3)):
        self.p = p
        self.dims = dims

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            for dim in self.dims:
                if torch.rand(1) < self.p:
                    x = torch.flip(x, [dim])
        return x

class RandomRotation90:
    """随机90度旋转"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if isinstance(x, torch.Tensor) and torch.rand(1) < self.p:
            k = torch.randint(0, 4, (1,)).item()  # 随机旋转0,1,2,3次90度
            x = torch.rot90(x, k, dims=(2, 3))  # 在H-W平面旋转
        return x

class RandomNoise:
    """添加随机噪声"""
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class RandomIntensityShift:
    """随机强度偏移"""
    def __init__(self, max_shift=0.1):
        self.max_shift = max_shift

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            shift = (torch.rand(1) * 2 - 1) * self.max_shift
            return x + shift
        return x

class CropOrPad:
    """裁剪或填充到指定大小"""
    def __init__(self, target_size, fill_value=-1000):
        self.target_size = target_size
        self.fill_value = fill_value

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            current_size = x.shape[-3:]
            diff = [t - c for t, c in zip(self.target_size, current_size)]
            
            # 需要填充的情况
            pads = []
            for d in diff[::-1]:  # reverse for pytorch padding format
                if d > 0:
                    pads.extend([d//2, d - d//2])
                else:
                    pads.extend([0, 0])
            if any(p > 0 for p in pads):
                x = torch.nn.functional.pad(x, pads, value=self.fill_value)
            
            # 需要裁剪的情况
            slices = []
            for t, c, d in zip(self.target_size, current_size, diff):
                if d < 0:
                    start = (-d) // 2
                    slices.append(slice(start, start + t))
                else:
                    slices.append(slice(None))
            if any(d < 0 for d in diff):
                x = x[..., slices[0], slices[1], slices[2]]
                
        return x

# 预定义的transform组合
def get_train_transform(augment=True):
    """获取训练集transform"""
    transforms = [
        NormalizeHU(),
        ToTensor(),
        AddChannel(),
    ]
    
    if augment:
        transforms.extend([
            RandomFlip(),
            RandomRotation90(),
            RandomNoise(std=0.05),
            RandomIntensityShift(max_shift=0.1),
        ])
    
    return Compose(transforms)

def get_val_transform():
    """获取验证集transform"""
    return Compose([
        NormalizeHU(),
        ToTensor(),
        AddChannel(),
    ])
