import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from sklearn.metrics import f1_score, classification_report, confusion_matrix

from dataset import MultiModalDataset, get_transforms, ModelConfig
from model import ViralFlowModel
from focal_loss import FocalLoss, ClassBalancedFocalLoss

# ==================== Early Stopping 类 ====================
class EarlyStopping:
    """早停机制：当验证指标不再改善时提前停止训练"""
    def __init__(self, patience=5, delta=0.0, mode='min'):
        """
        Args:
            patience (int): 容忍多少个epoch不改善
            delta (float): 最小改善幅度
            mode (str): 'min' 表示指标越小越好(loss)，'max' 表示越大越好(acc/f1)
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        """
        Args:
            score: 当前epoch的评估指标
        Returns:
            bool: 是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            return False
            
        # 判断是否改善
        if self.mode == 'min':
            improved = score < self.best_score - self.delta
        else:  # mode == 'max'
            improved = score > self.best_score + self.delta
            
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"  [EarlyStopping] Triggered! Stopping training.")
                return True
        return False

# --- 训练配置 ---
class TrainConfig:
    BATCH_SIZE = 48        # 根据显存调整，显存小就改 16
    EPOCHS = 30            # 训练轮数（增加，因为有early stopping会自动停止）
    LEARNING_RATE = 5e-4   # 降低学习率，防止过拟合（从1e-3降到5e-4）
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_PATH = './best_model.pth'
    
    # 数据路径
    TRAIN_CSV = 'data/train.csv'
    VAL_CSV = 'data/val.csv'
    IMG_ROOT = '' # 填写图片所在的根目录，如果 CSV 是绝对路径则留空
    
    # Early Stopping 配置
    EARLY_STOPPING_PATIENCE = 10  # 增加耐心到10个epoch
    EARLY_STOPPING_DELTA = 0.0001  # 最小改善幅度
    
    # 渐进式解冻配置（暂时禁用，先让分类头充分训练）
    PROGRESSIVE_UNFREEZE = False  # ⚠️ 暂时禁用渐进式解冻
    UNFREEZE_EPOCH_1 = 8   # 延迟到第8个epoch（给分类头更多训练时间）
    UNFREEZE_EPOCH_2 = 15  # 第15个epoch后开始第二阶段解冻
    UNFREEZE_EPOCH_3 = 22  # 第22个epoch后开始第三阶段解冻
    
    # 类别不平衡处理策略
    USE_CLASS_BALANCED_LOSS = True    # 使用类别平衡损失函数
    USE_WEIGHTED_SAMPLER = True       # 使用加权采样器
    CB_LOSS_BETA = 0.9999              # Class-Balanced Loss的beta参数
    FOCAL_GAMMA = 2.0                  # Focal Loss的gamma参数

def calculate_class_weights(df, device):
    """
    策略核心：计算类别权重。
    样本越少的类别，权重越大。
    公式: Weight = Total_Samples / (Num_Classes * Class_Samples)
    """
    labels = df['hot_level'].map(ModelConfig.LABEL_MAP).values
    class_counts = np.bincount(labels, minlength=5)
    total_samples = len(labels)
    num_classes = 5
    
    # 为了防止某个类别为0导致除以0错误，加一个小 epsilon 或者做平滑
    weights = total_samples / (num_classes * class_counts + 1e-6)
    
    # 转为 Tensor
    weights_tensor = torch.FloatTensor(weights).to(device)
    
    print("\n[Strategy] 类别不平衡处理 - 计算出的类别权重:")
    for lvl, idx in ModelConfig.LABEL_MAP.items():
        print(f"  Level {lvl}: 样本数 {class_counts[idx]}, 权重 {weights[idx]:.4f}")
        
    return weights_tensor

def build_weighted_sampler(df):
    """
    根据 hot_level 构建 WeightedRandomSampler
    """
    labels = df["hot_level"].map(ModelConfig.LABEL_MAP).values
    class_counts = np.bincount(labels)

    # 采样策略
    class_weights = 1.0 / (class_counts + 1e-6)     # 每个样本的权重 
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, epoch=0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Train E{epoch+1}", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        mask = batch['attention_mask'].to(device, non_blocking=True)
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(input_ids, mask, images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(input_ids, mask, images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 统计
        bs = labels.size(0)
        total_loss += loss.item() * bs
        _, predicted = torch.max(outputs, 1)
        total += bs
        correct += (predicted == labels).sum().item()

        # tqdm 展示（每个 batch 刷新）
        pbar.set_postfix({
            "loss": f"{total_loss / max(1,total):.4f}",
            "acc": f"{100*correct / max(1,total):.2f}%"
        })

    avg_loss = total_loss / total
    acc = 100 * correct / total
    return avg_loss, acc



def print_confusion_matrix(y_true, y_pred, save_path=None):
    """打印并保存混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    labels = ['S', 'A', 'B', 'C', 'D']
    
    # 创建DataFrame便于查看
    cm_df = pd.DataFrame(
        cm, 
        index=[f'True_{l}' for l in labels],
        columns=[f'Pred_{l}' for l in labels]
    )
    
    print("\n  混淆矩阵 (行=真实标签, 列=预测标签):")
    print(cm_df)
    
    if save_path:
        cm_df.to_csv(save_path, encoding='utf-8-sig')
        print(f"  [INFO] 混淆矩阵已保存到: {save_path}")
    
    return cm_df

def validate(model, dataloader, criterion, device, epoch=0):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Val   E{epoch+1}", leave=False)

    all_preds = [] 
    all_labels = []

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            mask = batch['attention_mask'].to(device, non_blocking=True)
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            # 验证阶段也可以用 AMP（不影响结果）
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(input_ids, mask, images)
                loss = criterion(outputs, labels)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            _, predicted = torch.max(outputs, 1)
            total += bs
            correct += (predicted == labels).sum().item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # tqdm 实时显示
            pbar.set_postfix({
                "loss": f"{total_loss / max(1, total):.4f}",
                "acc": f"{100 * correct / max(1, total):.2f}%"
            })

    # 合并所有batch的预测和标签
    all_preds_concat = np.concatenate(all_preds)
    all_labels_concat = np.concatenate(all_labels)
    
    avg_loss = total_loss / total
    acc = 100 * correct / total
    val_macro_f1 = f1_score(all_labels_concat, all_preds_concat, average='macro')

    return avg_loss, acc, val_macro_f1, all_labels_concat, all_preds_concat


def main():
    print(f"Using Device: {TrainConfig.DEVICE}")

    # TensorBoard
    writer = SummaryWriter(log_dir="runs/viralflow_experiment")   # 训练日志
    
    # 1. 初始化 Tokenizer
    tokenizer = BertTokenizer.from_pretrained(ModelConfig.BERT_PATH, local_files_only=True)
    
    # 2. 加载数据
    print("正在加载数据集...")
    train_df = pd.read_csv(TrainConfig.TRAIN_CSV)
    val_df = pd.read_csv(TrainConfig.VAL_CSV)
    
    train_dataset = MultiModalDataset(train_df, TrainConfig.IMG_ROOT, tokenizer, transform=get_transforms('train'))
    val_dataset = MultiModalDataset(val_df, TrainConfig.IMG_ROOT, tokenizer, transform=get_transforms('val'))

    print(f"Train 数据集大小: {len(train_dataset)}")
    print(f"Val 数据集大小: {len(val_dataset)}")
    
    # 构建训练DataLoader（可选使用加权采样）
    if TrainConfig.USE_WEIGHTED_SAMPLER:
        print("[INFO] 使用 WeightedRandomSampler 处理类别不平衡")
        train_sampler = build_weighted_sampler(train_df)
        train_loader = DataLoader(
            train_dataset,
            batch_size=TrainConfig.BATCH_SIZE,
            sampler=train_sampler,  # 使用sampler时不能shuffle
            num_workers=4,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=TrainConfig.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TrainConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=4,          # 先用 4，不够再试 8
        pin_memory=True,
        prefetch_factor=2
    )
    
    # 3. 初始化模型
    model = ViralFlowModel(
        num_classes=5, 
        freeze_backbone=True,
        use_attention_fusion=True  # 启用注意力融合机制
    )
    model.to(TrainConfig.DEVICE)
    
    # 4. 定义损失函数 (应用类别权重)
    if TrainConfig.USE_CLASS_BALANCED_LOSS:
        # 统计每个类别的样本数
        labels = train_df['hot_level'].map(ModelConfig.LABEL_MAP).values
        samples_per_class = np.bincount(labels, minlength=5)
        
        print(f"\n[INFO] 训练集类别分布:")
        for lvl, idx in ModelConfig.LABEL_MAP.items():
            print(f"  {lvl}: {samples_per_class[idx]} 样本 ({samples_per_class[idx]/len(labels)*100:.2f}%)")
        
        # 使用 Class-Balanced Focal Loss
        criterion = ClassBalancedFocalLoss(
            samples_per_class=samples_per_class,
            beta=TrainConfig.CB_LOSS_BETA,
            gamma=TrainConfig.FOCAL_GAMMA
        )
        print(f"[INFO] 使用 ClassBalancedFocalLoss (beta={TrainConfig.CB_LOSS_BETA}, gamma={TrainConfig.FOCAL_GAMMA})")
    else:
        # 传统交叉熵损失
        class_weights = calculate_class_weights(train_df, TrainConfig.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        print("[INFO] 使用加权交叉熵损失")
    
    # 5. 定义优化器
    # 仅优化 classifier 部分的参数 (因为 backbone 被冻结了)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=TrainConfig.LEARNING_RATE)
    
    # 5.5. 学习率调度器
    # 使用 ReduceLROnPlateau：验证loss不再下降时自动降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # 监控指标越小越好
        factor=0.5,           # 学习率衰减倍数
        patience=3,           # 3个epoch不改善就降低LR
        min_lr=1e-6           # 最小学习率
    )
    print("[INFO] 学习率调度器: ReduceLROnPlateau (factor=0.5, patience=3)")
    
    # 6. 训练循环
    print("\n=== 开始训练 ===")
    best_val_loss = float('inf')
    
    # 初始化 Early Stopping
    early_stopping = EarlyStopping(
        patience=TrainConfig.EARLY_STOPPING_PATIENCE, 
        delta=TrainConfig.EARLY_STOPPING_DELTA,
        mode='min'  # 监控 val_loss，越小越好
    )
    
    for epoch in range(TrainConfig.EPOCHS):
        start_time = time.time()
        
        # 渐进式解冻策略
        if TrainConfig.PROGRESSIVE_UNFREEZE:
            unfreeze_stage = 0
            if epoch == TrainConfig.UNFREEZE_EPOCH_1:
                unfreeze_stage = 1
            elif epoch == TrainConfig.UNFREEZE_EPOCH_2:
                unfreeze_stage = 2
            elif epoch == TrainConfig.UNFREEZE_EPOCH_3:
                unfreeze_stage = 3
            
            if unfreeze_stage > 0:
                print(f"\n{'='*60}")
                print(f"[Epoch {epoch+1}] 触发渐进式解冻 - Stage {unfreeze_stage}")
                print(f"{'='*60}")
                model.unfreeze_progressively(stage=unfreeze_stage)
                
                # 重新构建优化器（包含新解冻的参数）
                # 对解冻的参数使用更小的学习率
                trainable_params = []
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        # 分类器用原始学习率，骨干网络用0.1倍学习率
                        if 'classifier' in name:
                            trainable_params.append({'params': param, 'lr': TrainConfig.LEARNING_RATE})
                        else:
                            trainable_params.append({'params': param, 'lr': TrainConfig.LEARNING_RATE * 0.1})
                
                optimizer = optim.AdamW(trainable_params)
                
                # 重新初始化学习率调度器
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
                )
                print(f"[INFO] 优化器已更新，骨干网络学习率设为 {TrainConfig.LEARNING_RATE * 0.1:.2e}")
        
        # Train Step
        scaler = torch.amp.GradScaler("cuda", enabled=(TrainConfig.DEVICE == "cuda"))
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, TrainConfig.DEVICE,
            scaler=scaler if TrainConfig.DEVICE == "cuda" else None,
            epoch=epoch
        )
        
        # Validation Step
        val_loss, val_acc, val_macro_f1, y_true, y_pred = validate(
            model, val_loader, criterion, TrainConfig.DEVICE, epoch=epoch
        )
  
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch [{epoch+1}/{TrainConfig.EPOCHS}] | Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}% | Val Macro-F1: {val_macro_f1:.4f}")
        
        # 每5个epoch或最后一个epoch打印详细报告
        if (epoch + 1) % 5 == 0 or (epoch + 1) == TrainConfig.EPOCHS:
            print("\n  --- 详细分类报告 ---")
            target_names = ['S', 'A', 'B', 'C', 'D']
            report = classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0)
            print(report)
        
        # Model Checkpoint (保存最佳模型)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), TrainConfig.SAVE_PATH)
            print("  >>> Best Model Saved! <<<")
        
        # TensorBoard logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        # 学习率调度
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)  # 根据验证loss调整学习率
        new_lr = optimizer.param_groups[0]["lr"]
        
        # 手动打印学习率变化（因为verbose参数已被移除）
        if old_lr != new_lr:
            print(f"  [LR Scheduler] 学习率从 {old_lr:.2e} 降至 {new_lr:.2e}")
        
        # 记录学习率到 TensorBoard
        writer.add_scalar("LR", new_lr, epoch)
        
        # Early Stopping 检查
        if early_stopping(val_loss):
            print(f"\n[INFO] Early Stopping triggered at epoch {epoch+1}")
            break
    
    # 训练结束后，加载最佳模型并生成最终评估报告
    print("\n" + "="*60)
    print("训练完成！正在生成最终评估报告...")
    print("="*60)
    
    # 加载最佳模型
    model.load_state_dict(torch.load(TrainConfig.SAVE_PATH, map_location=TrainConfig.DEVICE))
    
    # 在验证集上做最终评估
    print("\n在验证集上评估最佳模型...")
    final_val_loss, final_val_acc, final_val_f1, final_y_true, final_y_pred = validate(
        model, val_loader, criterion, TrainConfig.DEVICE, epoch=-1
    )
    
    print(f"\n最终验证结果:")
    print(f"  Loss: {final_val_loss:.4f}")
    print(f"  Accuracy: {final_val_acc:.2f}%")
    print(f"  Macro-F1: {final_val_f1:.4f}")
    
    # 打印详细分类报告
    print("\n最终分类报告:")
    target_names = ['S', 'A', 'B', 'C', 'D']
    report = classification_report(final_y_true, final_y_pred, target_names=target_names, digits=4, zero_division=0)
    print(report)
    
    # 保存混淆矩阵
    cm_path = TrainConfig.SAVE_PATH.replace('.pth', '_confusion_matrix.csv')
    print_confusion_matrix(final_y_true, final_y_pred, save_path=cm_path)
    
    print(f"\n✅ 最佳模型已保存至: {TrainConfig.SAVE_PATH}")
    writer.close()

if __name__ == "__main__":
    main()