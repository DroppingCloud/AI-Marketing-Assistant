# src/predict.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset import MultiModalDataset, get_transforms, ModelConfig
from model import ViralFlowModel

# 你训练时保存的权重路径
CKPT_PATH = "./best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 显示顺序（和 LABEL_MAP 对齐）
LABELS = ["S", "A", "B", "C", "D"]
ID2LABEL = {v: k for k, v in ModelConfig.LABEL_MAP.items()}

def load_model():
    print(f"[INFO] Loading model weights from: {CKPT_PATH}")
    model = ViralFlowModel(
        num_classes=5, 
        freeze_backbone=True,
        use_attention_fusion=True  # 必须与训练时保持一致！
    )
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    print("[INFO] Model loaded. Device:", DEVICE)
    return model

def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)  # rows=true, cols=pred
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def _classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> str:
    # 纯 numpy 实现，避免依赖 sklearn（DSW 环境更稳）
    # 输出每类 precision/recall/f1 + macro/micro
    eps = 1e-12
    num_classes = len(labels)
    cm = _confusion_matrix(y_true, y_pred, num_classes=num_classes)

    lines = []
    lines.append("Class  Precision  Recall  F1  Support")
    supports = cm.sum(axis=1)

    precisions, recalls, f1s = [], [], []
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        lines.append(f"{lab:>5}  {prec:9.4f}  {rec:6.4f}  {f1:6.4f}  {supports[i]:7d}")

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    # micro：整体 TP/FP/FN
    total_tp = np.trace(cm)
    total = cm.sum()
    micro_acc = total_tp / (total + eps)

    lines.append("")
    lines.append(f"MacroAvg  {macro_p:9.4f}  {macro_r:6.4f}  {macro_f1:6.4f}  {int(total):7d}")
    lines.append(f"MicroAcc  {micro_acc:9.4f}")
    return "\n".join(lines)

@torch.no_grad()
def predict_csv(input_csv: str, output_csv: str, img_root: str = "", batch_size: int = 96, num_workers: int = 4):
    print(f"[INFO] Reading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"[INFO] Test Samples: {len(df)}")

    has_label = "hot_level" in df.columns
    if has_label:
        print("[INFO] Found ground-truth labels: hot_level -> will compute accuracy & confusion matrix.")
    else:
        print("[WARN] No hot_level in CSV -> will only output predictions (no accuracy/confusion matrix).")

    # tokenizer（离线）
    tokenizer = BertTokenizer.from_pretrained(ModelConfig.BERT_PATH, local_files_only=True)

    # Dataset（验证/测试 transform，不用 train 的随机增强）
    ds = MultiModalDataset(df, img_root, tokenizer, transform=get_transforms("val"), mode="test")

    # DataLoader（推理加速）
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )

    model = load_model()

    all_preds = []
    all_probs = []
    all_true = []  # 如果有标签就填

    pbar = tqdm(dl, desc="Predict", leave=True)
    seen = 0

    for batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        images = batch["image"].to(DEVICE, non_blocking=True)

        if has_label:
            labels = batch["label"].to(DEVICE, non_blocking=True)
            all_true.append(labels.cpu().numpy())

        # AMP 推理（更快）
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            logits = model(input_ids, mask, images)  # [B, 5]
        probs = F.softmax(logits, dim=1)              # [B, 5]

        pred_idx = probs.argmax(dim=1).cpu().numpy()
        pred_label = [ID2LABEL[int(i)] for i in pred_idx]

        all_preds.extend(pred_label)
        all_probs.extend(probs.cpu().numpy().tolist())

        seen += len(pred_label)
        # tqdm 信息
        pbar.set_postfix({"seen": seen, "it/s": f"{pbar.format_dict['rate']:.2f}" if pbar.format_dict.get("rate") else "-"})

    # 保存预测结果
    out = df.copy()
    out["pred_hot_level"] = all_preds
    for i, lbl in ID2LABEL.items():
        out[f"prob_{lbl}"] = [p[i] for p in all_probs]

    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Prediction saved to: {output_csv}")

    # 评估：Accuracy + Confusion Matrix + Report
    if has_label:
        y_true = np.concatenate(all_true)
        # df 的 hot_level -> idx（确保和 dataset 一致）
        # 注意：dataset 里 label 是从 row['hot_level'] 映射的，所以这里也用同一映射
        y_pred = np.array([ModelConfig.LABEL_MAP[x] for x in all_preds], dtype=int)

        acc = float((y_true == y_pred).mean())
        print("\n=== Test Metrics ===")
        print(f"Accuracy: {acc*100:.2f}%  (correct {int((y_true==y_pred).sum())}/{len(y_true)})")

        cm = _confusion_matrix(y_true, y_pred, num_classes=len(LABELS))
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print(cm_df)

        cm_path = os.path.splitext(output_csv)[0] + "_confusion_matrix.csv"
        cm_df.to_csv(cm_path, index=True, encoding="utf-8-sig")
        print(f"[OK] Confusion matrix saved to: {cm_path}")

        report = _classification_report(y_true, y_pred, LABELS)
        print("\nClassification Report:")
        print(report)

        report_path = os.path.splitext(output_csv)[0] + "_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"[OK] Report saved to: {report_path}")

if __name__ == "__main__":
    # 示例：对 test.csv 预测 +（若有标签则评估）
    predict_csv("data/test.csv", "data/test_pred.csv", img_root="", batch_size=96, num_workers=4)
