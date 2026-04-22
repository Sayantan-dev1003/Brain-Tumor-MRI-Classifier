"""
Brain Tumor MRI Classifier — Streamlit App
Supports ResNet50, EfficientNetB0, ViT-B/16
Place .pth files inside  models/  before running.
"""

import os
import io
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

from torchvision import transforms, models
from transformers import ViTForImageClassification

# ─── optional grad-cam (CNN models only) ────────────────────
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

# ────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────

NUM_CLASSES   = 4
CLASS_NAMES   = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE      = 224
DEVICE        = torch.device("cpu")          # keep CPU for local inference

MODEL_FILES = {
    "ResNet50"       : "models/ResNet50.pth",
    "EfficientNetB0" : "models/EfficientNetB0.pth",
    "ViT-B/16"       : "models/ViT-B-16.pth",
}

CLASS_COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]

PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

INV_NORMALIZE = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)

# ────────────────────────────────────────────────────────────
# MODEL BUILDERS
# ────────────────────────────────────────────────────────────

def build_resnet50():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, NUM_CLASSES),
    )
    return model


def build_efficientnet_b0():
    model = models.efficientnet_b0(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, NUM_CLASSES),
    )
    return model


def build_vit_b16():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
        output_attentions=True,   # ← must be set in config at load time
    )
    return model


BUILDERS = {
    "ResNet50"       : build_resnet50,
    "EfficientNetB0" : build_efficientnet_b0,
    "ViT-B/16"       : build_vit_b16,
}

# ────────────────────────────────────────────────────────────
# LOAD MODEL (cached)
# ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    pth_path = MODEL_FILES[model_name]
    if not os.path.exists(pth_path):
        return None, f"Weight file not found: `{pth_path}`"
    try:
        model = BUILDERS[model_name]()
        state = torch.load(pth_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

# ────────────────────────────────────────────────────────────
# INFERENCE
# ────────────────────────────────────────────────────────────

def run_inference(model, img_tensor, is_vit=False):
    inp = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(pixel_values=inp).logits if is_vit else model(inp)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return pred, probs

# ────────────────────────────────────────────────────────────
# GRAD-CAM
# ────────────────────────────────────────────────────────────

def get_gradcam_overlay(model, model_name, img_tensor, pred_class):
    if not GRADCAM_AVAILABLE:
        return None
    if model_name == "ResNet50":
        target_layer = model.layer4[-1]
    else:  # EfficientNetB0
        target_layer = model.features[-1]

    cam_obj  = GradCAM(model=model, target_layers=[target_layer])
    inp      = img_tensor.unsqueeze(0).to(DEVICE)
    grayscale = cam_obj(input_tensor=inp,
                        targets=[ClassifierOutputTarget(pred_class)])[0]

    rgb_img   = INV_NORMALIZE(img_tensor).permute(1, 2, 0).numpy()
    rgb_img   = np.clip(rgb_img, 0, 1).astype(np.float32)
    overlay   = show_cam_on_image(rgb_img, grayscale, use_rgb=True)

    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return rgb_img, heatmap, overlay

# ────────────────────────────────────────────────────────────
# ViT ATTENTION ROLLOUT
# ────────────────────────────────────────────────────────────

def get_vit_attention(model, img_tensor):
    """
    Last-layer CLS attention averaged over heads.
    Much sharper than rollout — rollout collapses to near-zero across 12 layers.
    """
    inp = img_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(pixel_values=inp, output_attentions=True)
    layer_attns = out.attentions
    if not layer_attns:
        return None

    # Last layer: (1, 12, 197, 197)
    last_attn = layer_attns[-1]
    avg_heads = last_attn[0].mean(dim=0)   # (197, 197)
    cls_attn  = avg_heads[0, 1:].cpu()     # (196,)

    gs      = int(cls_attn.shape[0] ** 0.5)
    att_map = cls_attn.reshape(gs, gs).numpy()
    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
    return att_map


def resize_att(att_map, size=IMG_SIZE):
    return np.array(
        Image.fromarray((att_map * 255).astype(np.uint8)).resize(
            (size, size), Image.BILINEAR)) / 255.0

# ────────────────────────────────────────────────────────────
# FIGURE HELPERS
# ────────────────────────────────────────────────────────────

def confidence_bar_figure(probs):
    fig, ax = plt.subplots(figsize=(5, 3))
    bar_colors = [CLASS_COLORS[i] for i in range(NUM_CLASSES)]
    bars = ax.barh(DISPLAY_NAMES, probs * 100,
                   color=bar_colors, edgecolor="white", linewidth=1)
    for bar, val in zip(bars, probs * 100):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_title("Class Probabilities", fontsize=11, fontweight="bold")
    ax.axvline(50, color="gray", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def explainability_figure_cnn(rgb_img, heatmap, overlay):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = ["Original MRI", "GradCAM Heatmap", "Overlay"]
    imgs   = [rgb_img, heatmap, overlay]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.tight_layout()
    return fig


def explainability_figure_vit(img_tensor, att_map):
    rgb_img   = INV_NORMALIZE(img_tensor).permute(1, 2, 0).numpy()
    rgb_img   = np.clip(rgb_img, 0, 1).astype(np.float32)
    att_res   = resize_att(att_map, IMG_SIZE)
    heatmap   = plt.cm.jet(att_res)[:, :, :3].astype(np.float32)
    overlay   = rgb_img * 0.55 + heatmap * 0.45
    overlay   = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = ["Original MRI", "Attention Map", "Overlay"]
    imgs   = [rgb_img, att_res, overlay]
    cmaps  = [None, "jet", None]
    for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
        ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle("ViT Attention Rollout  (Red = high attention)",
                 fontsize=10, style="italic")
    fig.tight_layout()
    return fig

# ────────────────────────────────────────────────────────────
# PAGE LAYOUT
# ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .pred-box {
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .pred-correct {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .pred-tumor {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .metric-card {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #2c3e7a;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #666;
    }
    .stImage > img {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown('<p class="main-header">🧠 Brain Tumor MRI Classifier</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Deep Learning comparison — ResNet50 · EfficientNetB0 · ViT-B/16</p>',
            unsafe_allow_html=True)
st.divider()

# ────────────────────────────────────────────────────────────
# SIDEBAR — Model Selection + Info
# ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Select Model",
        list(MODEL_FILES.keys()),
        index=0,
    )

    show_explain = st.toggle("Show Explainability Map", value=True)

    st.divider()
    st.markdown("### 📋 Classes")
    for name, color in zip(DISPLAY_NAMES, CLASS_COLORS):
        st.markdown(
            f'<span style="background:{color};color:white;padding:2px 10px;'
            f'border-radius:5px;font-size:0.85rem;">{name}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    st.divider()
    st.markdown("### ℹ️ Model Info")
    info = {
        "ResNet50"       : "50-layer ResNet with skip connections. Fast & accurate.",
        "EfficientNetB0" : "Compound-scaled CNN. Best accuracy/efficiency trade-off.",
        "ViT-B/16"       : "Vision Transformer, 16×16 patches. Captures global context.",
    }
    st.info(info[model_choice])

    if not GRADCAM_AVAILABLE:
        st.warning("Install `grad-cam` for GradCAM maps:\n`pip install grad-cam`")

# ────────────────────────────────────────────────────────────
# MAIN — Upload + Predict
# ────────────────────────────────────────────────────────────

col_upload, col_result = st.columns([1, 1.5], gap="large")

with col_upload:
    st.subheader("📤 Upload MRI Image")
    uploaded = st.file_uploader(
        "Supported formats: JPG, PNG, JPEG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="Uploaded MRI", use_container_width=True)

        # Load model
        with st.spinner(f"Loading {model_choice}…"):
            model, err = load_model(model_choice)

        if err:
            st.error(f"❌ Could not load model: {err}")
            st.stop()

        img_tensor = PREPROCESS(pil_img)
        is_vit = (model_choice == "ViT-B/16")

        with st.spinner("Running inference…"):
            pred, probs = run_inference(model, img_tensor, is_vit)

        with col_result:
            st.subheader("🔍 Prediction")

            # Prediction badge
            pred_name   = DISPLAY_NAMES[pred]
            box_class   = "pred-correct" if pred == 2 else "pred-tumor"
            emoji       = "✅" if pred == 2 else "⚠️"
            st.markdown(
                f'<div class="pred-box {box_class}">'
                f'{emoji} Predicted: <strong>{pred_name}</strong> '
                f'({probs[pred]*100:.1f}% confidence)</div>',
                unsafe_allow_html=True,
            )

            # Metric cards
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-val">{probs[pred]*100:.1f}%</div>'
                    f'<div class="metric-label">Top Confidence</div></div>',
                    unsafe_allow_html=True)
            with m_col2:
                entropy = -np.sum(probs * np.log(probs + 1e-9))
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-val">{entropy:.2f}</div>'
                    f'<div class="metric-label">Entropy</div></div>',
                    unsafe_allow_html=True)
            with m_col3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-val">{model_choice}</div>'
                    f'<div class="metric-label">Model Used</div></div>',
                    unsafe_allow_html=True)

            st.divider()

            # Confidence bars
            st.pyplot(confidence_bar_figure(probs), use_container_width=True)

    else:
        with col_result:
            st.info("Upload an MRI image on the left to get started.")

# ────────────────────────────────────────────────────────────
# EXPLAINABILITY SECTION
# ────────────────────────────────────────────────────────────

if uploaded and show_explain and "model" in dir() and model is not None:
    st.divider()
    st.subheader("🔬 Explainability")

    if is_vit:
        with st.spinner("Computing ViT Attention Rollout…"):
            att_map = get_vit_attention(model, img_tensor)
        if att_map is not None:
            st.pyplot(explainability_figure_vit(img_tensor, att_map),
                      use_container_width=True)
        else:
            st.warning("Could not generate attention maps.")
    else:
        if GRADCAM_AVAILABLE:
            with st.spinner("Computing GradCAM…"):
                result = get_gradcam_overlay(model, model_choice, img_tensor, pred)
            if result:
                rgb_img, heatmap, overlay = result
                st.pyplot(explainability_figure_cnn(rgb_img, heatmap, overlay),
                          use_container_width=True)
        else:
            st.warning("Install `grad-cam` (`pip install grad-cam`) to see GradCAM maps.")

# ────────────────────────────────────────────────────────────
# FOOTER
# ────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<center><small>CV Course Project · Vision Transformers in Medical Imaging</small></center>",
    unsafe_allow_html=True,
)