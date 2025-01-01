"""
checkpoints/checkpoint_manager.py
用于管理模型检查点的保存和加载
"""

import os
import json
from datetime import datetime
import torch
from pathlib import Path
from util.logconf import logging

log = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, model_name='luna_model', root_dir='checkpoints'):
        """
        初始化检查点管理器
        
        Args:
            model_name: 模型名称
            root_dir: 检查点根目录
        """
        self.model_name = model_name
        self.root_dir = root_dir
        self.version_dir = self._init_version_dir()
        
        # 创建必要的目录
        os.makedirs(self.version_dir, exist_ok=True)
        
        log.info(f"Checkpoint directory: {self.version_dir}")
        
    def _init_version_dir(self):
        """初始化版本目录"""
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 创建新的版本目录
        version_dir = os.path.join(
            self.root_dir,
            self.model_name,
            f"version_{timestamp}"
        )
        return version_dir
        
    def save_checkpoint(self, save_dict, is_best=False):
        """
        保存检查点
        
        Args:
            save_dict: 要保存的字典，包含模型状态等信息
            is_best: 是否为最佳模型
        """
        # 始终保存最新的模型
        last_path = os.path.join(self.version_dir, 'last.pth')
        torch.save(save_dict, last_path)
        log.info(f"Saved checkpoint: {last_path}")
        
        # 如果是最佳模型，则额外保存一份
        if is_best:
            best_path = os.path.join(self.version_dir, 'best.pth')
            torch.save(save_dict, best_path)
            log.info(f"Saved best model: {best_path}")
            
        # 保存训练配置
        config = {
            'epoch': save_dict['epoch'],
            'best_val_loss': save_dict['best_val_loss'],
            'metrics': save_dict['metrics'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        config_path = os.path.join(self.version_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
    def load_checkpoint(self, filename='best.pth'):
        """
        加载检查点
        
        Args:
            filename: 要加载的文件名（'best.pth' 或 'last.pth'）
            
        Returns:
            加载的检查点字典
        """
        checkpoint_path = os.path.join(self.version_dir, filename)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path)
        log.info(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint
        
    def list_checkpoints(self):
        """列出所有可用的检查点"""
        all_versions = []
        model_dir = os.path.join(self.root_dir, self.model_name)
        
        if os.path.exists(model_dir):
            for version in os.listdir(model_dir):
                version_path = os.path.join(model_dir, version)
                if os.path.isdir(version_path):
                    config_path = os.path.join(version_path, 'config.json')
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        all_versions.append({
                            'version': version,
                            'config': config
                        })
                        
        return all_versions