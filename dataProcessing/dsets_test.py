import torch
from torch.utils.data import DataLoader
from dsets import LunaDataset  # 假设你的文件名是dsets.py

def test_luna_dataset():
    # 1. 创建训练集实例
    train_ds = LunaDataset(
        val_stride=10,        # 每10个样本抽1个作为验证集
        isValSet_bool=False   # 这是训练集
    )
    
    # 2. 创建验证集实例
    val_ds = LunaDataset(
        val_stride=10,
        isValSet_bool=True    # 这是验证集
    )
    
    # 3. 创建数据加载器
    train_loader = DataLoader(
        train_ds, 
        batch_size=2,         # 小批量测试
        shuffle=True          # 随机打乱
    )
    
    # 4. 测试数据加载
    for batch_idx, (data, target, series_uid, center_irc) in enumerate(train_loader):
        print(f"Batch {batch_idx}")
        print(f"Data shape: {data.shape}")           # CT数据形状
        print(f"Target shape: {target.shape}")       # 标签形状
        print(f"Series UID: {series_uid}")          # CT序列ID
        print(f"Center IRC: {center_irc}")          # 中心坐标
        
        # 只测试一个批次
        if batch_idx == 0:
            break
            
    print(f"训练集大小: {len(train_ds)}")
    print(f"验证集大小: {len(val_ds)}")

if __name__ == "__main__":
    test_luna_dataset()