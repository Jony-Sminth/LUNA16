import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from checkpoint_manager import CheckpointManager

sys.path.append('../')  # 添加项目根目录到路径
from model.model import LunaModel
from dataProcessing.dsets import LunaDataset, custom_collate
from dataProcessing.transforms import TransformBuilder
from metrics import LunaMetrics
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def main():
    app = LunaTrainingApp()
    app.train()

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        # 训练参数设置保持不变
        self.epochs = 100
        self.batch_size = 32
        self.val_stride = 10
        self.lr = 0.001
        self.patience = 5
        
        # 初始化设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化转换器和数据集
        self.transform = self._get_transform()
        self.train_ds, self.val_ds = self._init_datasets()
        
        # 初始化模型
        self.model = self._init_model()
        
        # 初始化检查点管理器
        self.checkpoint_manager = CheckpointManager(
            model_name='luna_model',
            root_dir='../checkpoints'
        )
        
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=custom_collate
        )
        
        # 添加维度验证
        sample_batch = next(iter(self.train_loader))
        log.info(f"Sample batch shapes:")
        log.info(f"Input: {sample_batch[0].shape}")
        log.info(f"Label: {sample_batch[1].shape}")
        
        if sample_batch[0].dim() != 5:
            raise ValueError(f"Unexpected input dimensions: {sample_batch[0].shape}")
        
        # 初始化优化器和调度器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3)
        self.criterion = nn.CrossEntropyLoss()
        
        # 初始化指标计算器
        self.train_metrics = LunaMetrics()
        self.val_metrics = LunaMetrics()
        
    def _init_model(self):
        model = LunaModel()
        model = model.to(self.device)
        return model
        
    def _get_transform(self):
        return TransformBuilder()\
            .add_normalize()\
            .add_window(center=40, width=400)\
            .build()
            
    def _init_datasets(self):
        train_ds = LunaDataset(
            val_stride=self.val_stride,
            isValSet_bool=False,
            transform=self.transform,
        )
        
        val_ds = LunaDataset(
            val_stride=self.val_stride,
            isValSet_bool=True,
            transform=self.transform,
        )
        
        return train_ds, val_ds

    def _log_metrics(self, phase, metrics_dict, epoch):
        """记录训练指标"""
        log.info(f'Epoch {epoch} {phase}:')
        for metric_name, value in metrics_dict.items():
            log.info(f'{metric_name}: {value:.4f}')
        
    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            self.train_metrics.reset()
            
            for batch_idx, (inputs, targets, series_uid, centers) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs, prob = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                self.train_metrics.update(outputs, targets, loss)
                
                if batch_idx % 10 == 0:
                    log.info(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}')
            
            # 计算并记录训练指标
            train_metrics = self.train_metrics.compute()
            self._log_metrics('Train', train_metrics, epoch)
            
            # Validation phase
            self.model.eval()
            self.val_metrics.reset()
            
            with torch.no_grad():
                for inputs, targets, series_uid, centers in self.val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs, prob = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    self.val_metrics.update(outputs, targets, loss)
            
            # 计算并记录验证指标
            val_metrics = self.val_metrics.compute()
            self._log_metrics('Validation', val_metrics, epoch)
            
            # 学习率调整
            self.scheduler.step(val_metrics['loss'])
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'metrics': val_metrics
                }, 'best_model.pth')
                log.info('Model saved!')
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                log.info(f'Early stopping triggered after {epoch + 1} epochs')
                break

if __name__ == '__main__':
    app = LunaTrainingApp()
    app.train()