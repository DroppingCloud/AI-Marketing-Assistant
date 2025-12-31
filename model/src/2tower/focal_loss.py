"""
Focal Loss 实现
用于处理类别不平衡问题，让模型更关注难分类样本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss: 让模型更关注难分类的样本
    
    公式: FL = -α * (1-pt)^γ * log(pt)
    - α: 类别权重
    - γ: 聚焦参数，γ越大，对易分类样本的权重降低越多
    - pt: 模型预测正确类别的概率
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 类别权重，可以是标量或长度为C的张量
            gamma: 聚焦参数，默认2.0
            reduction: 'mean' 或 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        # 处理 alpha
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.alpha = alpha
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] 模型输出的logits
            targets: [N] 真实标签 (0 to C-1)
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算 pt (模型对正确类别的预测概率)
        pt = torch.exp(-ce_loss)
        
        # 计算 Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 应用类别权重
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            # 为每个样本获取对应类别的权重
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # 聚合
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss
    结合了类别平衡和Focal Loss的优点
    
    论文: "Class-Balanced Loss Based on Effective Number of Samples"
    """
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0):
        """
        Args:
            samples_per_class: 每个类别的样本数量 [C]
            beta: 平衡参数，越接近1，重平衡越强
            gamma: Focal Loss的聚焦参数
        """
        super(ClassBalancedFocalLoss, self).__init__()
        self.gamma = gamma
        
        # 计算有效样本数
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / effective_num
        
        # 归一化权重
        weights = weights / weights.sum() * len(weights)
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        print(f"[ClassBalancedFocalLoss] 类别权重: {self.weights.numpy()}")
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] 模型输出的logits
            targets: [N] 真实标签
        """
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(inputs.device)
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算 pt
        pt = torch.exp(-ce_loss)
        
        # 计算 Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 应用类别平衡权重
        weights_t = self.weights[targets]
        balanced_focal_loss = weights_t * focal_loss
        
        return balanced_focal_loss.mean()


if __name__ == "__main__":
    print("=== Focal Loss 单元测试 ===")
    
    # 模拟数据
    batch_size = 8
    num_classes = 5
    
    # 模拟模型输出
    logits = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 1, 2, 3, 4, 4, 4, 4])  # 类别4的样本更多
    
    # 测试1: 标准 Focal Loss
    print("\n--- 测试1: 标准 Focal Loss ---")
    focal_loss = FocalLoss(gamma=2.0)
    loss1 = focal_loss(logits, targets)
    print(f"Loss: {loss1.item():.4f}")
    
    # 测试2: 带权重的 Focal Loss
    print("\n--- 测试2: 带权重的 Focal Loss ---")
    alpha = [2.0, 2.0, 2.0, 1.5, 0.5]  # 给少数类更大权重
    focal_loss_weighted = FocalLoss(alpha=alpha, gamma=2.0)
    loss2 = focal_loss_weighted(logits, targets)
    print(f"Loss: {loss2.item():.4f}")
    
    # 测试3: Class-Balanced Focal Loss
    print("\n--- 测试3: Class-Balanced Focal Loss ---")
    samples_per_class = np.array([27, 25, 50, 149, 254])  # 实际数据分布
    cb_focal_loss = ClassBalancedFocalLoss(samples_per_class, beta=0.9999, gamma=2.0)
    loss3 = cb_focal_loss(logits, targets)
    print(f"Loss: {loss3.item():.4f}")
    
    print("\n✅ Focal Loss 测试完成")

