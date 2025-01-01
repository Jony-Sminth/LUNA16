"""
评估指标模块，实现了常用的医学图像评估指标
"""
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import torch

class LunaMetrics:
    def __init__(self):
        self.reset()
        
    def reset(self):
        """重置所有指标"""
        self.predictions = []
        self.targets = []
        self.running_loss = 0.0
        self.count = 0
        
    def update(self, outputs, targets, loss=None):
        """
        更新指标状态
        
        Args:
            outputs: 模型输出 (batch_size, 2)
            targets: 真实标签 (batch_size)
            loss: 当前batch的损失值
        """
        # 将输出转换为概率
        with torch.no_grad():  # 上下文管理器
            probs = torch.softmax(outputs, dim=1)
            # 获取正类的概率，注意使用detach()
            pos_probs = probs[:, 1].detach().cpu().numpy()
            # 获取真实标签
            true_labels = targets.detach().cpu().numpy()
        
        self.predictions.extend(pos_probs)
        self.targets.extend(true_labels)
        
        if loss is not None:
            self.running_loss += loss.item()  # 使用item()获取标量值
            self.count += 1
    
    def compute(self):
        """计算所有指标"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # 计算AUC
        auc = roc_auc_score(targets, predictions)
        
        # 计算平均精度(AP)
        ap = average_precision_score(targets, predictions)
        
        # 使用0.5作为阈值计算其他指标
        predicted_labels = (predictions >= 0.5).astype(int)
        
        # 计算准确率
        accuracy = (predicted_labels == targets).mean()
        
        # 计算敏感度(召回率)和特异度
        true_positives = ((predicted_labels == 1) & (targets == 1)).sum()
        true_negatives = ((predicted_labels == 0) & (targets == 0)).sum()
        false_positives = ((predicted_labels == 1) & (targets == 0)).sum()
        false_negatives = ((predicted_labels == 0) & (targets == 1)).sum()
        
        sensitivity = true_positives / (true_positives + false_negatives + 1e-7)
        specificity = true_negatives / (true_negatives + false_positives + 1e-7)
        
        # 计算精确率
        precision = true_positives / (true_positives + false_positives + 1e-7)
        
        # 计算F1分数
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-7)
        
        # 计算平均损失
        avg_loss = self.running_loss / self.count if self.count > 0 else 0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc,
            'ap': ap,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1
        }