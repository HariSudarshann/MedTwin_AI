"""
MediTwin AI — Chest X-Ray (ResNet-18) Imaging Model
Trains on Kaggle Chest X-Ray Pneumonia dataset.
Exports: trained weights, evaluation metrics, Grad-CAM support.
"""

import sys
import os
import json
import io
import base64
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
from torchvision import models, datasets

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Device Configuration ─────────────────────────────────────────────────────
DEVICE = torch.device("cpu")  # CPU for laptop compatibility; GPU will be auto-detected
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    logger.info("Running on CPU (no GPU detected)")


# ── Data Transforms ───────────────────────────────────────────────────────────
def get_transforms(split: str) -> transforms.Compose:
    """Returns augmentation transforms for train/val/test splits."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build ResNet-18 with custom classification head."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model.to(DEVICE)


def train_epoch(model, loader, criterion, optimizer) -> Tuple[float, float]:
    """Single training epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate model; returns loss, accuracy, all predictions, all labels."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


def train_model(data_dir: Path, save_path: Path,
                epochs: int = 10, lr: float = 1e-4, batch_size: int = 16) -> Dict[str, Any]:
    """
    Full training pipeline for ResNet-18 on Chest X-Ray Pneumonia dataset.
    Returns metrics dict with accuracy, precision, recall, F1, ROC-AUC.
    """
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score, precision_score, recall_score, f1_score)

    logger.info(f"Training ResNet-18 from: {data_dir}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = datasets.ImageFolder(data_dir / "train", transform=get_transforms("train"))
    val_dataset   = datasets.ImageFolder(data_dir / "val",   transform=get_transforms("val"))
    test_dataset  = datasets.ImageFolder(data_dir / "test",  transform=get_transforms("test"))

    # Handle class imbalance with weighted sampler
    class_counts = np.bincount(train_dataset.targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

    logger.info(f"Dataset: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    logger.info(f"Classes: {train_dataset.class_to_idx}")

    # ── Model, Criterion, Optimizer ───────────────────────────────────────────
    model = build_model(num_classes=2, pretrained=True)
    # Weighted loss for class imbalance
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight.item()]).to(DEVICE)
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training Loop ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, _, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        logger.info(f"Epoch {epoch:02d}/{epochs} | "
                    f"Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | "
                    f"Val Loss={vl_loss:.4f} Acc={vl_acc:.4f} | "
                    f"Time={elapsed:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Test Evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    _, test_acc, test_preds, test_labels, test_probs = evaluate(model, test_loader, criterion)

    # Map class indices to names (ImageFolder sorts alphabetically)
    class_names = train_dataset.classes  # ['NORMAL', 'PNEUMONIA']

    precision   = precision_score(test_labels, test_preds, zero_division=0)
    recall      = recall_score(test_labels, test_preds, zero_division=0)
    f1          = f1_score(test_labels, test_preds, zero_division=0)
    roc_auc     = roc_auc_score(test_labels, test_probs)
    cm          = confusion_matrix(test_labels, test_preds).tolist()
    report      = classification_report(test_labels, test_preds, target_names=class_names)

    logger.info(f"\nTest Accuracy : {test_acc:.4f}")
    logger.info(f"Precision     : {precision:.4f}")
    logger.info(f"Recall        : {recall:.4f}")
    logger.info(f"F1 Score      : {f1:.4f}")
    logger.info(f"ROC-AUC       : {roc_auc:.4f}")
    logger.info(f"\nClassification Report:\n{report}")

    # ── Save Model ────────────────────────────────────────────────────────────
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "class_to_idx": train_dataset.class_to_idx,
        "classes": class_names,
        "architecture": "resnet18",
        "metrics": {
            "test_accuracy": test_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
        }
    }, save_path)
    logger.info(f"Model saved to: {save_path}")

    # ── Training Curve ────────────────────────────────────────────────────────
    _plot_training_history(history, save_path.parent)
    _plot_confusion_matrix(cm, class_names, save_path.parent)
    _plot_roc_curve(test_labels, test_probs, roc_auc, save_path.parent)

    metrics = {
        "test_accuracy": round(test_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "history": history,
    }

    # Save metrics JSON
    with open(save_path.parent / "xray_metrics.json", "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "classification_report"}, f, indent=2)

    return metrics


def _plot_training_history(history: dict, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train")
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train")
    ax2.plot(epochs, history["val_acc"],   "r-o", label="Val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "xray_training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(cm: list, class_names: list, out_dir: Path):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix — Chest X-Ray")
    plt.tight_layout()
    plt.savefig(out_dir / "xray_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_roc_curve(labels, probs, auc_val, out_dir: Path):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4F86F7", lw=2, label=f"ROC-AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Chest X-Ray Classifier")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "xray_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM Implementation
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.
    Hooks into the last convolutional layer of ResNet-18.
    """
    def __init__(self, model: nn.Module, target_layer: str = "layer4"):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_hooks(target_layer)

    def _register_hooks(self, target_layer: str):
        layer = dict(self.model.named_modules())[target_layer]
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap (H×W float array in [0,1])."""
        self.model.eval()
        image_tensor = image_tensor.to(DEVICE)
        output = self.model(image_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay_gradcam(original_image: Image.Image, cam: np.ndarray,
                    alpha: float = 0.5) -> Tuple[Image.Image, str]:
    """
    Overlay Grad-CAM heatmap on original image.
    Returns: PIL Image + base64-encoded PNG string.
    """
    import cv2
    # Resize CAM to image size
    orig_w, orig_h = original_image.size
    cam_resized = np.uint8(255 * cam)
    cam_resized = cv2.resize(cam_resized, (orig_w, orig_h))
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    orig_arr = np.array(original_image.convert("RGB"))
    blended = (alpha * heatmap + (1 - alpha) * orig_arr).astype(np.uint8)
    result_img = Image.fromarray(blended)

    # Encode to base64
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return result_img, b64


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
class XRayClassifier:
    """
    Loads trained ResNet-18 and provides inference + Grad-CAM.
    """
    def __init__(self, model_path: Path):
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        self.classes = checkpoint.get("classes", ["NORMAL", "PNEUMONIA"])
        self.class_to_idx = checkpoint.get("class_to_idx", {"NORMAL": 0, "PNEUMONIA": 1})
        self.metrics = checkpoint.get("metrics", {})
        self.model = build_model(num_classes=len(self.classes), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE).eval()
        self.gradcam = GradCAM(self.model, target_layer="layer4")
        self.transform = get_transforms("test")
        logger.info(f"XRayClassifier loaded from {model_path}")

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict class + confidence for a PIL Image.
        Returns: prediction dict with class, confidence, probabilities, gradcam_b64.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess
        tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        tensor.requires_grad_(True)

        with torch.no_grad():
            logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_class = self.classes[pred_idx]
        confidence = probs[pred_idx].item()

        # Grad-CAM (needs grad)
        tensor_grad = self.transform(image).unsqueeze(0).to(DEVICE)
        cam = self.gradcam.generate(tensor_grad, class_idx=pred_idx)
        _, gradcam_b64 = overlay_gradcam(image, cam)

        return {
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                self.classes[i]: round(probs[i].item(), 4)
                for i in range(len(self.classes))
            },
            "gradcam_b64": gradcam_b64,
            "cam_array": cam.tolist(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point for training
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ResNet-18 on Chest X-Ray dataset")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Root folder with train/val/test subdirectories")
    parser.add_argument("--save-path", type=Path, default=Path("models/resnet18_xray.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    metrics = train_model(args.data_dir, args.save_path,
                          epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("classification_report", "history", "cam_array")}, indent=2))
