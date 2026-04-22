# ============================================================
# Brain Tumor Classification: ResNet50 vs EfficientNetB0 vs ViT-B/16
# Dataset: Brain Tumor MRI Dataset by Masoud Nickparvar (Kaggle)
# ============================================================
# KAGGLE SETUP:
# 1. Add dataset: masoudnickparvar/brain-tumor-mri-dataset
# 2. Settings → Accelerator → GPU T4 x2
# 3. pip install timm transformers grad-cam -q
# ============================================================

# ─────────────────────────────────────────
# SECTION 0: INSTALL & IMPORTS
# ─────────────────────────────────────────

import subprocess
subprocess.run(["pip", "install", "timm", "transformers", "grad-cam", "-q"])

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR

import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize

# ── KEY FIX: import ViTConfig alongside ViTForImageClassification ──
from transformers import ViTForImageClassification, ViTConfig


# ─────────────────────────────────────────
# SECTION 1: CONFIGURATION
# ─────────────────────────────────────────

import glob as _glob

def _find_dir(name):
    """Search common Kaggle mount points for a folder named `name`."""
    candidates = _glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not find '{name}' under /kaggle/input/. "
        f"Dirs found: {_glob.glob('/kaggle/input/**', recursive=False)}"
    )

CONFIG = {
    "train_dir"    : _find_dir("Training"),
    "test_dir"     : _find_dir("Testing"),
    "save_dir"     : "/kaggle/working/results",
    "num_classes"  : 4,
    "class_names"  : ["glioma", "meningioma", "notumor", "pituitary"],
    "display_names": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
    "img_size"     : 224,
    "batch_size"   : 32,
    "val_split"    : 0.2,
    "total_epochs" : 20,
    "lr_cnn"       : 1e-3,
    "lr_vit"       : 1e-4,
    "weight_decay" : 1e-4,
    "seed"         : 42,
    "device"       : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

print("=" * 60)
print(f"  Device     : {CONFIG['device']}")
print(f"  GPU        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"  Classes    : {CONFIG['class_names']}")
print(f"  Epochs     : {CONFIG['total_epochs']}")
print("=" * 60)


# ─────────────────────────────────────────
# SECTION 2: DATA LOADING & TRANSFORMS
# ─────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

full_train_dataset = datasets.ImageFolder(CONFIG["train_dir"], transform=train_transforms)
test_dataset       = datasets.ImageFolder(CONFIG["test_dir"],  transform=test_transforms)

val_size   = int(len(full_train_dataset) * CONFIG["val_split"])
train_size = len(full_train_dataset) - val_size

train_subset, val_subset = random_split(
    full_train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(CONFIG["seed"])
)

val_dataset_clean = datasets.ImageFolder(CONFIG["train_dir"], transform=test_transforms)
val_dataset       = Subset(val_dataset_clean, val_subset.indices)

train_loader = DataLoader(train_subset, batch_size=CONFIG["batch_size"],
                          shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,  batch_size=CONFIG["batch_size"],
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                          shuffle=False, num_workers=2, pin_memory=True)

print(f"\nDataset Sizes:")
print(f"  Train      : {train_size}")
print(f"  Validation : {val_size}")
print(f"  Test       : {len(test_dataset)}")


# ─────────────────────────────────────────
# SECTION 3: MODEL DEFINITIONS
# ─────────────────────────────────────────

def get_resnet50(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


def get_efficientnet_b0(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


def get_vit_b16(num_classes):
    """
    KEY FIX: Build ViT using ViTConfig with output_attentions=True baked in.
    This ensures attention tensors are always returned during both training
    and inference, without needing to pass output_attentions=True per call.
    Using from_pretrained on the config object loads the pretrained weights correctly.
    """
    vit_config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
    vit_config.num_labels        = num_classes
    vit_config.output_attentions = True   # ← baked into config permanently
    vit_config.id2label          = {i: c for i, c in enumerate(CONFIG["class_names"])}
    vit_config.label2id          = {c: i for i, c in enumerate(CONFIG["class_names"])}

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        config=vit_config,
        ignore_mismatched_sizes=True,
    )
    return model


# ─────────────────────────────────────────
# SECTION 4: TRAINING ENGINE
# ─────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, is_vit=False):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # For ViT: .logits extracts just the class scores from the full output object
        outputs = model(pixel_values=images).logits if is_vit else model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct      += outputs.max(1)[1].eq(labels).sum().item()
        total        += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device, is_vit=False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs   = model(pixel_values=images).logits if is_vit else model(images)
            loss      = criterion(outputs, labels)
            probs     = torch.softmax(outputs, dim=1)
            predicted = outputs.max(1)[1]
            running_loss += loss.item() * images.size(0)
            correct      += predicted.eq(labels).sum().item()
            total        += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return (
        running_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


def train_model(model_name, model, train_loader, val_loader,
                lr, total_epochs, device, is_vit=False):
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}
    best_val_acc     = 0.0
    best_model_state = None
    start_time       = time.time()

    for epoch in range(1, total_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, is_vit)
        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device, is_vit)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  Epoch [{epoch:02d}/{total_epochs}] "
              f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    total_time = time.time() - start_time
    print(f"\n  ⏱️  Total Training Time : {total_time/60:.2f} min")
    print(f"  🏆 Best Val Accuracy   : {best_val_acc:.4f}")

    model.load_state_dict(best_model_state)

    # Save .pth
    pth_path = os.path.join(CONFIG["save_dir"], f"{model_name.replace('/', '-')}.pth")
    torch.save(best_model_state, pth_path)
    print(f"  💾 Saved model weights: {pth_path}")

    return model, history, total_time


# ─────────────────────────────────────────
# SECTION 5: METRICS COMPUTATION
# ─────────────────────────────────────────

def compute_metrics(model, test_loader, device, is_vit=False):
    criterion = nn.CrossEntropyLoss()
    _, test_acc, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device, is_vit)

    labels_bin = label_binarize(test_labels, classes=list(range(CONFIG["num_classes"])))
    try:
        auc_score = roc_auc_score(labels_bin, test_probs, multi_class="ovr", average="macro")
    except Exception as e:
        print(f"  AUC warning: {e}")
        auc_score = float("nan")

    f1   = f1_score(test_labels, test_preds, average="macro")
    prec = precision_score(test_labels, test_preds, average="macro", zero_division=0)
    rec  = recall_score(test_labels, test_preds, average="macro", zero_division=0)

    print(f"  ✅ Acc: {test_acc:.4f} | F1: {f1:.4f} | "
          f"Prec: {prec:.4f} | Rec: {rec:.4f} | AUC: {auc_score:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(test_labels, test_preds,
                                 target_names=CONFIG["display_names"], digits=4))

    return {
        "accuracy"  : test_acc,
        "f1_score"  : f1,
        "precision" : prec,
        "recall"    : rec,
        "auc"       : auc_score,
        "preds"     : test_preds,
        "labels"    : test_labels,
        "probs"     : test_probs,
    }


# ─────────────────────────────────────────
# SECTION 6: PLOT — TRAINING CURVES
# ─────────────────────────────────────────

COLORS = {
    "ResNet50"       : "#4A90E2",
    "EfficientNetB0" : "#E25C4A",
    "ViT-B/16"       : "#50C878",
}

def plot_combined_loss_curves(all_histories):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training & Validation Loss — All Models (20 Epochs)",
                 fontsize=15, fontweight="bold")

    for ax, (model_name, history) in zip(axes, all_histories.items()):
        color = COLORS[model_name]
        x = range(1, len(history["train_loss"]) + 1)
        ax.plot(x, history["train_loss"], color=color, linewidth=2.5, label="Train Loss")
        ax.plot(x, history["val_loss"],   color=color, linewidth=2.5,
                linestyle="--", alpha=0.75, label="Val Loss")
        ax.set_title(model_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(CONFIG["save_dir"], "loss_curves_all_models.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")


def plot_combined_accuracy_curves(all_histories):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training & Validation Accuracy — All Models (20 Epochs)",
                 fontsize=15, fontweight="bold")

    for ax, (model_name, history) in zip(axes, all_histories.items()):
        color = COLORS[model_name]
        x = range(1, len(history["train_acc"]) + 1)
        ax.plot(x, [v * 100 for v in history["train_acc"]], color=color,
                linewidth=2.5, label="Train Acc")
        ax.plot(x, [v * 100 for v in history["val_acc"]], color=color,
                linewidth=2.5, linestyle="--", alpha=0.75, label="Val Acc")
        best_epoch = int(np.argmax(history["val_acc"])) + 1
        best_acc   = max(history["val_acc"]) * 100
        ax.axvline(x=best_epoch, color="#E74C3C", linestyle=":", linewidth=1.5,
                   label=f"Best E{best_epoch} ({best_acc:.1f}%)")
        ax.set_title(model_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(CONFIG["save_dir"], "accuracy_curves_all_models.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# SECTION 7: PLOT — CONFUSION MATRICES
# ─────────────────────────────────────────

def plot_combined_confusion_matrices(all_results):
    """Raw count confusion matrices — all 3 models in one figure."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("Confusion Matrices — All Models (Raw Counts)",
                 fontsize=15, fontweight="bold")

    for ax, (model_name, result) in zip(axes, all_results.items()):
        cm = confusion_matrix(result["labels"], result["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CONFIG["display_names"],
                    yticklabels=CONFIG["display_names"],
                    ax=ax, linewidths=0.5, annot_kws={"size": 11})
        ax.set_title(model_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

    plt.tight_layout()
    path = os.path.join(CONFIG["save_dir"], "confusion_matrices_all_models.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# SECTION 8: PLOT — AUC-ROC
# ─────────────────────────────────────────

def plot_roc_curves_per_model(all_results):
    """One ROC figure per model, all 4 classes + macro avg."""
    palette = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]

    for model_name, result in all_results.items():
        labels_bin = label_binarize(result["labels"],
                                    classes=list(range(CONFIG["num_classes"])))
        fig, ax = plt.subplots(figsize=(9, 7))

        for i, cls_name in enumerate(CONFIG["display_names"]):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], result["probs"][:, i])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=palette[i], linewidth=2.5,
                    label=f"{cls_name} (AUC = {roc_auc:.3f})")

        try:
            macro_auc = roc_auc_score(labels_bin, result["probs"],
                                      multi_class="ovr", average="macro")
        except Exception:
            macro_auc = float("nan")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Classifier")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=13)
        ax.set_title(f"{model_name} — ROC Curves\n(Macro AUC = {macro_auc:.3f})",
                     fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        safe = model_name.replace("/", "-")
        path = os.path.join(CONFIG["save_dir"], f"roc_{safe}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Saved: {path}")


# ─────────────────────────────────────────
# SECTION 9: PLOT — METRICS TABLE
# ─────────────────────────────────────────

def plot_metrics_table(all_results):
    """Bar chart + printed table comparing F1, Accuracy, Precision, Recall."""
    model_names   = list(all_results.keys())
    metrics_keys  = ["accuracy", "f1_score", "precision", "recall"]
    metric_labels = ["Accuracy", "F1-Score", "Precision", "Recall"]
    x      = np.arange(len(model_names))
    width  = 0.2
    bar_colors = ["#4A90E2", "#E25C4A", "#50C878", "#F5A623"]

    fig, ax = plt.subplots(figsize=(13, 7))
    for i, (key, label) in enumerate(zip(metrics_keys, metric_labels)):
        values = [all_results[m][key] for m in model_names]
        bars   = ax.bar(x + i * width, values, width, label=label,
                        color=bar_colors[i], alpha=0.88,
                        edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Performance Comparison — Accuracy, F1, Precision, Recall",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(CONFIG["save_dir"], "metrics_comparison_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")

    print("\n" + "=" * 72)
    print("  METRICS COMPARISON TABLE")
    print("=" * 72)
    print(f"  {'Model':<18} {'Accuracy':>10} {'F1-Score':>10} "
          f"{'Precision':>10} {'Recall':>10}")
    print(f"  {'-'*64}")
    for m in model_names:
        r = all_results[m]
        print(f"  {m:<18} {r['accuracy']:>10.4f} {r['f1_score']:>10.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>10.4f}")
    print("=" * 72)


# ─────────────────────────────────────────
# SECTION 10: GRAD-CAM (ResNet50 & EfficientNetB0)
# ─────────────────────────────────────────

INV_NORMALIZE = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)


def tensor_to_rgb(img_tensor):
    img = INV_NORMALIZE(img_tensor).permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1).astype(np.float32)


def visualize_gradcam(model, model_name, target_layer, test_dataset, device,
                      num_samples=8):
    """GradCAM visualization — 2 samples per class, 3-column layout."""
    model.eval()

    class_indices = {i: [] for i in range(CONFIG["num_classes"])}
    for idx, (_, label) in enumerate(test_dataset):
        if len(class_indices[label]) < 2:
            class_indices[label].append(idx)
        if all(len(v) >= 2 for v in class_indices.values()):
            break

    sample_indices = [idx for v in class_indices.values() for idx in v][:num_samples]
    cam = GradCAM(model=model, target_layers=[target_layer])

    fig, axes = plt.subplots(num_samples, 3, figsize=(14, num_samples * 3.8))
    fig.suptitle(f"{model_name} — GradCAM Visualization\n"
                 f"(Original | GradCAM Heatmap | Overlay)",
                 fontsize=15, fontweight="bold", y=1.01)

    col_titles = ["Original MRI", "GradCAM Heatmap", "Overlay"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight="bold")

    for row, idx in enumerate(sample_indices):
        img_tensor, true_label = test_dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred   = int(probs.argmax())

        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=[ClassifierOutputTarget(pred)])[0]
        rgb_img   = tensor_to_rgb(img_tensor)
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        heatmap   = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap   = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        is_correct  = (pred == true_label)
        frame_color = "#27AE60" if is_correct else "#E74C3C"
        verdict     = "✓" if is_correct else "✗"

        axes[row][0].imshow(rgb_img)
        axes[row][0].set_ylabel(
            f"True: {CONFIG['display_names'][true_label]}", fontsize=9, fontweight="bold")
        axes[row][1].imshow(heatmap)
        axes[row][2].imshow(cam_image)
        axes[row][2].set_xlabel(
            f"Pred: {CONFIG['display_names'][pred]} "
            f"({probs[pred]*100:.1f}%) {verdict}",
            fontsize=9, fontweight="bold",
            color="#27AE60" if is_correct else "#E74C3C")

        for ax in axes[row]:
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(frame_color)
                spine.set_linewidth(2.5)

    plt.tight_layout()
    safe = model_name.replace("/", "-")
    path = os.path.join(CONFIG["save_dir"], f"gradcam_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# SECTION 11: ViT ATTENTION MAPS
# ─────────────────────────────────────────

def resize_att(att_map, size):
    return np.array(
        Image.fromarray((att_map * 255).astype(np.uint8)).resize(
            (size, size), Image.BILINEAR)) / 255.0


def get_last_layer_attention(model, img_tensor, device):
    """
    Returns (14×14) normalised CLS attention from the LAST transformer layer.
    Works because output_attentions=True is baked into the model config.
    """
    model.eval()
    with torch.no_grad():
        out = model(pixel_values=img_tensor.unsqueeze(0).to(device))

    layer_attns = out.attentions
    if layer_attns is None or len(layer_attns) == 0:
        raise RuntimeError(
            "out.attentions is None — model was not built with "
            "ViTConfig(output_attentions=True). Re-run get_vit_b16()."
        )

    last_attn = layer_attns[-1]            # (1, 12, 197, 197)
    avg_heads = last_attn[0].mean(dim=0)   # (197, 197)
    cls_attn  = avg_heads[0, 1:].cpu()    # (196,)

    att_map = cls_attn.reshape(14, 14).numpy()
    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
    return att_map


def get_all_layer_attentions(model, img_tensor, device):
    """Returns list of 12 × (14×14) normalised maps, one per transformer layer."""
    model.eval()
    with torch.no_grad():
        out = model(pixel_values=img_tensor.unsqueeze(0).to(device))

    layer_attns = out.attentions
    if layer_attns is None:
        raise RuntimeError("out.attentions is None — check model config.")

    maps = []
    for layer_attn in layer_attns:
        avg_heads = layer_attn[0].mean(dim=0)
        cls_attn  = avg_heads[0, 1:].cpu().numpy()
        att_map   = cls_attn.reshape(14, 14)
        att_map   = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
        maps.append(att_map)
    return maps


def visualize_vit_attention(model, test_dataset, device, num_samples=8):
    """2 samples per class — Original | Attention Map | Overlay | Confidence."""
    model.eval()

    class_indices = {i: [] for i in range(CONFIG["num_classes"])}
    for idx, (_, label) in enumerate(test_dataset):
        if len(class_indices[label]) < 2:
            class_indices[label].append(idx)
        if all(len(v) >= 2 for v in class_indices.values()):
            break

    sample_indices = [idx for v in class_indices.values() for idx in v][:num_samples]

    fig, axes = plt.subplots(num_samples, 4, figsize=(22, num_samples * 4.2))
    fig.suptitle("ViT-B/16 — Attention Map Visualization\n"
                 "(Original MRI  |  Attention Map  |  Overlay  |  Class Confidence)",
                 fontsize=15, fontweight="bold", y=1.01)

    col_titles = ["Original MRI", "Attention Map (Last Layer)", "Overlay", "Confidence"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight="bold")

    for row, idx in enumerate(sample_indices):
        img_tensor, true_label = test_dataset[idx]
        att_map = get_last_layer_attention(model, img_tensor, device)

        with torch.no_grad():
            out        = model(pixel_values=img_tensor.unsqueeze(0).to(device))
            probs      = torch.softmax(out.logits, dim=1)[0].cpu().numpy()
            pred_label = int(probs.argmax())
            confidence = probs[pred_label] * 100

        display_img = tensor_to_rgb(img_tensor)
        att_resized = resize_att(att_map, CONFIG["img_size"])
        heatmap     = plt.cm.jet(att_resized)[:, :, :3]
        overlay     = display_img * 0.55 + heatmap * 0.45

        is_correct  = (pred_label == true_label)
        frame_color = "#27AE60" if is_correct else "#E74C3C"
        verdict     = "✓ CORRECT" if is_correct else "✗ WRONG"

        axes[row][0].imshow(display_img)
        axes[row][0].set_ylabel(
            f"True: {CONFIG['display_names'][true_label]}",
            fontsize=9, fontweight="bold")

        im = axes[row][1].imshow(att_resized, cmap="jet", vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[row][1], fraction=0.046, pad=0.04)

        axes[row][2].imshow(overlay)
        axes[row][2].set_xlabel("Red = high attention", fontsize=8, style="italic")

        bar_colors_conf = ["#E74C3C" if i == pred_label else "#BDC3C7"
                           for i in range(CONFIG["num_classes"])]
        axes[row][3].barh(CONFIG["display_names"], probs * 100,
                          color=bar_colors_conf, edgecolor="white", linewidth=1.2)
        axes[row][3].set_xlim(0, 100)
        axes[row][3].set_xlabel("Confidence (%)", fontsize=9)
        axes[row][3].axvline(x=50, color="gray", linestyle="--", alpha=0.5)
        axes[row][3].set_title(
            f"Pred: {CONFIG['display_names'][pred_label]} ({confidence:.1f}%)\n{verdict}",
            fontsize=9, fontweight="bold",
            color="#27AE60" if is_correct else "#E74C3C")

        for ax in axes[row][:3]:
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(frame_color)
                spine.set_linewidth(2.5)

    plt.tight_layout()
    path = os.path.join(CONFIG["save_dir"], "vit_attention_maps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")


def visualize_vit_attention_all_layers(model, test_dataset, device, sample_idx=0):
    """All 12 transformer layers for one sample."""
    model.eval()
    img_tensor, true_label = test_dataset[sample_idx]
    all_maps = get_all_layer_attentions(model, img_tensor, device)

    with torch.no_grad():
        out   = model(pixel_values=img_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(out.logits, dim=1)[0].cpu().numpy()
    pred_label  = int(probs.argmax())
    display_img = tensor_to_rgb(img_tensor)

    fig = plt.figure(figsize=(26, 10))
    fig.suptitle(
        f"ViT-B/16 — Attention Across All 12 Layers\n"
        f"True: {CONFIG['display_names'][true_label]}  |  "
        f"Pred: {CONFIG['display_names'][pred_label]} ({probs[pred_label]*100:.1f}%)",
        fontsize=13, fontweight="bold"
    )

    ax_orig = fig.add_subplot(2, 7, 1)
    ax_orig.imshow(display_img); ax_orig.set_title("Original", fontsize=11, fontweight="bold")
    ax_orig.axis("off")

    positions = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
    cmaps = ["Blues", "Purples", "Greens", "Oranges",
             "Reds",  "YlOrBr",  "BuGn",   "RdPu",
             "YlGnBu","hot",     "cool",    "jet"]

    for i, (att_map, pos, cmap) in enumerate(zip(all_maps, positions, cmaps)):
        att_resized = resize_att(att_map, CONFIG["img_size"])
        ax = fig.add_subplot(2, 7, pos)
        ax.imshow(att_resized, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"Layer {i+1}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(CONFIG["save_dir"], "vit_attention_all_layers.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")


def visualize_attention_heads(model, test_dataset, device, sample_idx=0):
    """All 12 heads from the last transformer layer for one sample."""
    model.eval()
    img_tensor, true_label = test_dataset[sample_idx]

    with torch.no_grad():
        out  = model(pixel_values=img_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(out.logits, dim=1)[0].cpu().numpy()

    pred_label  = int(probs.argmax())
    last_attn   = out.attentions[-1][0]   # (12, 197, 197)
    display_img = tensor_to_rgb(img_tensor)

    fig, axes = plt.subplots(3, 5, figsize=(22, 14))
    fig.suptitle(
        f"ViT-B/16 — Last Layer: All 12 Attention Heads\n"
        f"True: {CONFIG['display_names'][true_label]}  |  "
        f"Pred: {CONFIG['display_names'][pred_label]} ({probs[pred_label]*100:.1f}%)",
        fontsize=14, fontweight="bold"
    )

    axes[0][0].imshow(display_img)
    axes[0][0].set_title("Original MRI", fontsize=11, fontweight="bold")
    axes[0][0].axis("off")

    all_axes = axes.flatten()
    for h in range(12):
        cls_attn = last_attn[h, 0, 1:].cpu().numpy()
        att_map  = cls_attn.reshape(14, 14)
        att_map  = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
        att_resized = resize_att(att_map, CONFIG["img_size"])
        ax = all_axes[h + 1]
        ax.imshow(att_resized, cmap="jet", vmin=0, vmax=1)
        ax.set_title(f"Head {h+1}", fontsize=10)
        ax.axis("off")

    avg_att = last_attn.mean(dim=0)[0, 1:].cpu().numpy()
    avg_map = avg_att.reshape(14, 14)
    avg_map = (avg_map - avg_map.min()) / (avg_map.max() - avg_map.min() + 1e-8)
    all_axes[13].imshow(resize_att(avg_map, CONFIG["img_size"]), cmap="hot", vmin=0, vmax=1)
    all_axes[13].set_title("Avg (all heads)", fontsize=10, fontweight="bold")
    all_axes[13].axis("off")
    all_axes[14].axis("off")

    plt.tight_layout()
    path = os.path.join(CONFIG["save_dir"], "vit_attention_heads.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# SECTION 12: MAIN TRAINING PIPELINE
# ─────────────────────────────────────────

all_results    = {}
all_histories  = {}
trained_models = {}

# ── 12.1  ResNet50 ──────────────────────
print("\n>>> STEP 1/3 — ResNet50")
resnet_model = get_resnet50(CONFIG["num_classes"])
resnet_model, resnet_history, resnet_time = train_model(
    "ResNet50", resnet_model, train_loader, val_loader,
    CONFIG["lr_cnn"], CONFIG["total_epochs"], CONFIG["device"])

resnet_metrics = compute_metrics(resnet_model, test_loader, CONFIG["device"])
all_results["ResNet50"]    = resnet_metrics
all_histories["ResNet50"]  = resnet_history
trained_models["ResNet50"] = resnet_model


# ── 12.2  EfficientNetB0 ────────────────
print("\n>>> STEP 2/3 — EfficientNetB0")
effnet_model = get_efficientnet_b0(CONFIG["num_classes"])
effnet_model, effnet_history, effnet_time = train_model(
    "EfficientNetB0", effnet_model, train_loader, val_loader,
    CONFIG["lr_cnn"], CONFIG["total_epochs"], CONFIG["device"])

effnet_metrics = compute_metrics(effnet_model, test_loader, CONFIG["device"])
all_results["EfficientNetB0"]    = effnet_metrics
all_histories["EfficientNetB0"]  = effnet_history
trained_models["EfficientNetB0"] = effnet_model


# ── 12.3  ViT-B/16 ──────────────────────
print("\n>>> STEP 3/3 — ViT-B/16")
vit_model = get_vit_b16(CONFIG["num_classes"])   # ← uses ViTConfig fix
vit_model, vit_history, vit_time = train_model(
    "ViT-B/16", vit_model, train_loader, val_loader,
    CONFIG["lr_vit"], CONFIG["total_epochs"], CONFIG["device"], is_vit=True)

vit_metrics = compute_metrics(vit_model, test_loader, CONFIG["device"], is_vit=True)
all_results["ViT-B/16"]    = vit_metrics
all_histories["ViT-B/16"]  = vit_history
trained_models["ViT-B/16"] = vit_model


# ─────────────────────────────────────────
# SECTION 13: GENERATE ALL PLOTS
# ─────────────────────────────────────────

print("\n" + "=" * 60)
print("  GENERATING PLOTS")
print("=" * 60)

# 1. Loss curves
plot_combined_loss_curves(all_histories)

# 2. Accuracy curves
plot_combined_accuracy_curves(all_histories)

# 3. Confusion matrices
plot_combined_confusion_matrices(all_results)

# 4. ROC curves
plot_roc_curves_per_model(all_results)

# 5. Metrics comparison
plot_metrics_table(all_results)

# 6. GradCAM — ResNet50
print("\n  Generating GradCAM for ResNet50...")
resnet_target_layer = trained_models["ResNet50"].layer4[-1]
visualize_gradcam(trained_models["ResNet50"], "ResNet50",
                  resnet_target_layer, test_dataset, CONFIG["device"])

# 7. GradCAM — EfficientNetB0
print("\n  Generating GradCAM for EfficientNetB0...")
effnet_target_layer = trained_models["EfficientNetB0"].features[-1]
visualize_gradcam(trained_models["EfficientNetB0"], "EfficientNetB0",
                  effnet_target_layer, test_dataset, CONFIG["device"])

# 8. ViT Attention Maps (2 per class)
print("\n  Generating ViT Attention Maps...")
visualize_vit_attention(trained_models["ViT-B/16"], test_dataset, CONFIG["device"])

# 9. ViT All-layer attention (1 sample)
print("\n  Generating ViT All-Layer Attention Maps...")
visualize_vit_attention_all_layers(trained_models["ViT-B/16"], test_dataset,
                                   CONFIG["device"], sample_idx=0)

# 10. ViT Head diversity (last layer)
print("\n  Generating ViT Attention Heads...")
visualize_attention_heads(trained_models["ViT-B/16"], test_dataset,
                          CONFIG["device"], sample_idx=0)


# ─────────────────────────────────────────
# SECTION 14: SAVE RESULTS CSV + TIMING
# ─────────────────────────────────────────

rows = []
for model_name, r in all_results.items():
    rows.append({
        "Model"     : model_name,
        "Accuracy"  : round(r["accuracy"]  * 100, 2),
        "F1_Score"  : round(r["f1_score"],         4),
        "Precision" : round(r["precision"],         4),
        "Recall"    : round(r["recall"],            4),
        "AUC"       : round(r["auc"],               4),
    })

df       = pd.DataFrame(rows)
csv_path = os.path.join(CONFIG["save_dir"], "results_summary.csv")
df.to_csv(csv_path, index=False)
print(f"\n  Results CSV saved: {csv_path}")
print(df.to_string(index=False))

print("\n  TRAINING TIME COMPARISON")
print(f"  {'Model':<18} {'Time (min)':>12}")
print(f"  {'-'*32}")
for name, t in zip(["ResNet50", "EfficientNetB0", "ViT-B/16"],
                   [resnet_time, effnet_time, vit_time]):
    print(f"  {name:<18} {t/60:>11.2f}")

print("\n" + "=" * 60)
print("  ✅ ALL DONE!")
print(f"  📁 Outputs → {CONFIG['save_dir']}")
print("  📦 Download from Kaggle Output panel")
print("=" * 60)