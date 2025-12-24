import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from dataset import MultiModalDataset, get_transforms, ModelConfig
from model import ViralFlowModel

# --- 训练配置 ---
class TrainConfig:
    BATCH_SIZE = 32        # 根据显存调整，显存小就改 16
    EPOCHS = 10            # 训练轮数
    LEARNING_RATE = 1e-3   # 初始学习率 (因为冻结了骨干，可以用稍微大一点的LR)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_PATH = './best_model.pth'
    
    # 数据路径
    TRAIN_CSV = 'data/train.csv'
    VAL_CSV = 'data/val.csv'
    IMG_ROOT = '' # 填写图片所在的根目录，如果 CSV 是绝对路径则留空

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

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, mask, images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # tqdm 实时显示
        pbar.set_postfix({
            "loss": f"{total_loss / max(1, total):.4f}",
            "acc": f"{100 * correct / max(1, total):.2f}%"
        })

    avg_loss = total_loss / len(dataloader)
    acc = 100 * correct / total
    return avg_loss, acc

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Val", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, mask, images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                "loss": f"{total_loss / max(1, total):.4f}",
                "acc": f"{100 * correct / max(1, total):.2f}%"
            })

    avg_loss = total_loss / len(dataloader)
    acc = 100 * correct / total
    return avg_loss, acc

def main():
    print(f"Using Device: {TrainConfig.DEVICE}")

    # TensorBoard
    writer = SummaryWriter(log_dir="runs/viralflow_experiment")   # 训练日志
    
    # 1. 初始化 Tokenizer
    tokenizer = BertTokenizer.from_pretrained(ModelConfig.BERT_PATH)
    
    # 2. 加载数据
    print("正在加载数据集...")
    train_df = pd.read_csv(TrainConfig.TRAIN_CSV)
    val_df = pd.read_csv(TrainConfig.VAL_CSV)
    
    train_dataset = MultiModalDataset(train_df, TrainConfig.IMG_ROOT, tokenizer, transform=get_transforms('train'))
    val_dataset = MultiModalDataset(val_df, TrainConfig.IMG_ROOT, tokenizer, transform=get_transforms('val'))
    
    train_loader = DataLoader(train_dataset, batch_size=TrainConfig.BATCH_SIZE, shuffle=True, num_workers=0) # Windows下num_workers建议为0
    val_loader = DataLoader(val_dataset, batch_size=TrainConfig.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 3. 初始化模型
    model = ViralFlowModel(num_classes=5, freeze_backbone=True)
    model.to(TrainConfig.DEVICE)
    
    # 4. 定义损失函数 (应用类别权重)
    class_weights = calculate_class_weights(train_df, TrainConfig.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 5. 定义优化器
    # 仅优化 classifier 部分的参数 (因为 backbone 被冻结了)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=TrainConfig.LEARNING_RATE)
    
    # 6. 训练循环
    print("\n=== 开始训练 ===")
    best_val_loss = float('inf')
    
    for epoch in range(TrainConfig.EPOCHS):
        start_time = time.time()
        
        # Train Step
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, TrainConfig.DEVICE)
        
        # Validation Step
        val_loss, val_acc = validate(model, val_loader, criterion, TrainConfig.DEVICE)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1}/{TrainConfig.EPOCHS}] | Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        
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

        # （可选）记录学习率
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", current_lr, epoch)
            
    print("\n训练完成。最佳模型已保存。")
    writer.close()

if __name__ == "__main__":
    main()