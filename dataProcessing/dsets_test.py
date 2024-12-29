import unittest
import torch
import numpy as np
from dsets import LunaDataset, getCt
from transforms import TransformBuilder, create_transform

class TestContrastMethods(unittest.TestCase):
    def setUp(self):
        self.dataset = LunaDataset(
            val_stride=10,
            isValSet_bool=False
        )
        
    def test_contrast_methods(self):
        """比较两种不同的对比度增强方法"""
        print("\n=== 比较不同的对比度增强方法 ===")
        
        # 获取一个样本
        sample_data, label, series_uid, center = self.dataset[0]
        
        # 方法1: 从0开始构建新的处理方法
        def custom_read_ct_with_contrast(series_uid):
            """在数据读取时就进行对比度增强"""
            ct = getCt(series_uid)
            # 在原始数据上直接进行对比度增强
            enhanced = np.copy(ct.hu_a)
            # 对比度增强处理
            min_val, max_val = np.percentile(enhanced, [5, 95])
            enhanced = np.clip(enhanced, min_val, max_val)
            enhanced = (enhanced - min_val) / (max_val - min_val)
            ct.hu_a = enhanced
            return ct
            
        # 使用方法1处理数据
        ct_enhanced = custom_read_ct_with_contrast(series_uid)
        data_method1 = torch.tensor(ct_enhanced.hu_a).float()
        
        # 方法2: 在现有transform基础上添加处理
        transform_method2 = TransformBuilder()\
            .add_normalize()\
            .add_window()\
            .add_custom(
                lambda x: torch.clamp(
                    (x - x.mean()) * 1.5 + x.mean(),  # 增强对比度
                    -1, 1
                ),
                "contrast_enhance"
            )\
            .build()
            
        # 使用方法2处理数据
        data_method2 = transform_method2(sample_data)
        
        # 打印比较结果
        print("\n方法1 (数据读取时增强):")
        print(f"范围: [{data_method1.min():.3f}, {data_method1.max():.3f}]")
        print(f"均值: {data_method1.mean():.3f}")
        print(f"标准差: {data_method1.std():.3f}")
        
        print("\n方法2 (Transform链增强):")
        print(f"范围: [{data_method2.min():.3f}, {data_method2.max():.3f}]")
        print(f"均值: {data_method2.mean():.3f}")
        print(f"标准差: {data_method2.std():.3f}")
        
        # 显示处理效果的不同
        print("\n两种方法的主要区别:")
        print("1. 方法1在数据读取阶段就进行增强，影响后续所有处理")
        print("2. 方法2保持原始数据不变，在transform链中进行增强")
        
        # 创建组合方法（两种方法结合）
        transform_combined = TransformBuilder()\
            .add_custom(
                lambda x: torch.tensor(
                    np.clip(x.numpy(), 
                           np.percentile(x.numpy(), 5), 
                           np.percentile(x.numpy(), 95))
                ),
                "initial_enhance"
            )\
            .add_normalize()\
            .add_window()\
            .add_custom(
                lambda x: torch.clamp((x - x.mean()) * 1.5 + x.mean(), -1, 1),
                "final_enhance"
            )\
            .build()
            
        # 使用组合方法
        data_combined = transform_combined(sample_data)
        
        print("\n组合方法:")
        print(f"范围: [{data_combined.min():.3f}, {data_combined.max():.3f}]")
        print(f"均值: {data_combined.mean():.3f}")
        print(f"标准差: {data_combined.std():.3f}")

    def test_usage_example(self):
        """展示不同方法的使用方式"""
        print("\n=== 使用方式示例 ===")
        
        # 方法1: 修改数据读取
        def get_enhanced_ct(series_uid):
            ct = getCt(series_uid)
            # 增强处理...
            return ct
            
        # 方法2: 使用transform链
        transform = TransformBuilder()\
            .add_normalize()\
            .add_custom(lambda x: x * 1.5, "enhance")\
            .build()
            
        # 在Dataset中使用
        print("\n使用方法1:")
        print("dataset = LunaDataset()")
        print("ct = get_enhanced_ct(series_uid)")
        
        print("\n使用方法2:")
        print("dataset = LunaDataset(transform=transform)")

if __name__ == '__main__':
    unittest.main(verbosity=2)