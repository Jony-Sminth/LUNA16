import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from dsets import LunaDataset, getCandidateInfoList, getCt, custom_collate
from transforms import (
    get_train_transform,
    get_val_transform,
    Compose,
    NormalizeHU,
    RandomFlip,
    RandomRotation90
)
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def test_dataset_basic():
    """测试数据集的基本功能"""
    log.info("Testing basic dataset functionality...")
    
    # 创建训练集实例
    train_ds = LunaDataset(
        val_stride=10,
        isValSet_bool=False
    )
    
    # 创建验证集实例
    val_ds = LunaDataset(
        val_stride=10,
        isValSet_bool=True
    )
    
    # 检查数据集大小
    log.info(f"Training dataset size: {len(train_ds)}")
    log.info(f"Validation dataset size: {len(val_ds)}")
    
    # 测试单个数据项的获取
    sample = train_ds[0]
    log.info(f"Sample data shape: {sample[0].shape}")
    log.info(f"Sample label shape: {sample[1].shape}")
    log.info(f"Sample series_uid: {sample[2]}")
    log.info(f"Sample center_irc: {sample[3]}")

def test_transforms():
    """测试数据转换功能"""
    log.info("Testing transforms functionality...")
    
    # 创建带有不同转换的数据集
    train_ds = LunaDataset(
        val_stride=10,
        isValSet_bool=False,
        transform=get_train_transform(augment=True)
    )
    
    val_ds = LunaDataset(
        val_stride=10,
        isValSet_bool=True,
        transform=get_val_transform()
    )
    
    # 测试训练集数据
    train_sample = train_ds[0]
    log.info(f"Transformed training sample shape: {train_sample[0].shape}")
    log.info(f"Transformed training sample value range: [{train_sample[0].min():.2f}, {train_sample[0].max():.2f}]")
    
    # 测试验证集数据
    val_sample = val_ds[0]
    log.info(f"Transformed validation sample shape: {val_sample[0].shape}")
    log.info(f"Transformed validation sample value range: [{val_sample[0].min():.2f}, {val_sample[0].max():.2f}]")

def test_dataloader():
    """测试数据加载器"""
    log.info("Testing dataloader functionality...")
    
    # 创建数据集
    train_ds = LunaDataset(
        val_stride=10,
        isValSet_bool=False,
        transform=get_train_transform(augment=True)
    )
    
    # 创建数据加载器，使用自定义的collate_fn
    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate  # 使用自定义的collate函数
    )
    
    # 测试一个批次
    for batch_idx, (data, target, series_uid, center_irc) in enumerate(train_loader):
        log.info(f"Batch {batch_idx}")
        log.info(f"Batch data shape: {data.shape}")
        log.info(f"Batch target shape: {target.shape}")
        log.info(f"Batch series_uid length: {len(series_uid)}")
        log.info(f"Batch center_irc shape: {center_irc.shape}")
        break  # 只测试第一个批次

def test_custom_transforms():
    """测试自定义转换组合"""
    log.info("Testing custom transforms combination...")
    
    # 创建自定义转换
    custom_transform = Compose([
        NormalizeHU(),
        RandomFlip(p=0.5),
        RandomRotation90(p=0.5)
    ])
    
    # 使用自定义转换创建数据集
    dataset = LunaDataset(
        val_stride=10,
        isValSet_bool=False,
        transform=custom_transform
    )
    
    # 测试数据
    sample = dataset[0]
    log.info(f"Custom transformed sample shape: {sample[0].shape}")
    log.info(f"Custom transformed value range: [{sample[0].min():.2f}, {sample[0].max():.2f}]")

def run_all_tests():
    """运行所有测试"""
    try:
        # 首先测试基本的数据集功能
        log.info("=== Testing Basic Dataset ===")
        test_dataset_basic()
        
        # 然后测试自定义transforms
        log.info("\n=== Testing Custom Transforms ===")
        test_custom_transforms()
        
        # 接着测试预定义transforms
        log.info("\n=== Testing Predefined Transforms ===")
        test_transforms()
        
        # 最后测试dataloader
        log.info("\n=== Testing DataLoader ===")
        test_dataloader()
        
        log.info("\nAll tests completed successfully!")
        
    except Exception as e:
        log.error(f"Error during testing: {str(e)}")
        log.error("Stack trace:", exc_info=True)  # 添加更详细的错误信息
        raise

if __name__ == "__main__":
    run_all_tests()