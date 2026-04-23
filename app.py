# ============================================================
# app.py — Brain Tumor Classification Streamlit App
# Models: ResNet50 | EfficientNetB0 | ViT-B/16
# ============================================================
# SETUP:
#   pip install streamlit torch torchvision transformers pillow
#                numpy matplotlib opencv-python-headless
#
# Place your trained .pth files under a models/ subfolder:
#   models/ResNet50.pth
#   models/EfficientNetB0.pth
#   models/ViT-B-16.pth
#
# Run:
#   streamlit run app.py
# ============================================================

import os
import io
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torchvision import transforms, models
try:
    from transformers import ViTForImageClassification, ViTConfig
except ImportError:
    from transformers.models.vit.modeling_vit import ViTForImageClassification
    from transformers.models.vit.configuration_vit import ViTConfig

import streamlit as st

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

CLASS_NAMES   = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
NUM_CLASSES   = 4
IMG_SIZE      = 224
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_COLORS = {
    "Glioma"     : "#E74C3C",
    "Meningioma" : "#3498DB",
    "No Tumor"   : "#27AE60",
    "Pituitary"  : "#F39C12",
}

CLASS_INFO = {
    "Glioma": {
        "desc": "A tumor that starts in the glial cells of the brain or spine. "
                "Gliomas make up about 30% of all brain tumors.",
        "severity": "High",
        "color": "#E74C3C",
    },
    "Meningioma": {
        "desc": "A tumor that arises from the meninges. Usually benign and slow-growing. "
                "Most common type of primary brain tumor.",
        "severity": "Medium",
        "color": "#3498DB",
    },
    "No Tumor": {
        "desc": "No evidence of tumor detected in the MRI scan. "
                "Normal brain tissue observed.",
        "severity": "None",
        "color": "#27AE60",
    },
    "Pituitary": {
        "desc": "A tumor in the pituitary gland at the base of the brain. "
                "Usually benign and treatable.",
        "severity": "Low–Medium",
        "color": "#F39C12",
    },
}

_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PTH = {
    "ResNet50"       : os.path.join(_BASE, "models", "ResNet50.pth"),
    "EfficientNetB0" : os.path.join(_BASE, "models", "EfficientNetB0.pth"),
    "ViT-B/16"       : os.path.join(_BASE, "models", "ViT-B-16.pth"),
}

INV_NORMALIZE = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────
# CSS STYLING
# ─────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #ffffff; font-size: 2.4rem; font-weight: 700; margin: 0; }
    .main-header p  { color: #a0aec0; font-size: 1rem; margin: 0.5rem 0 0 0; }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; line-height: 1; }
    .metric-card .label {
        font-size: 0.8rem; color: #718096; margin-top: 4px;
        font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;
    }

    .prediction-banner {
        border-radius: 12px; padding: 1.5rem 2rem; margin: 1rem 0;
        font-size: 1.4rem; font-weight: 700; text-align: center;
        color: white; box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }

    .info-box {
        background: #f8fafc; border-left: 4px solid #4A90E2;
        border-radius: 0 8px 8px 0; padding: 1rem 1.2rem;
        margin: 0.5rem 0; font-size: 0.9rem; color: #2d3748;
    }

    .img-card {
        border: 2px solid #e2e8f0; border-radius: 12px;
        padding: 0.75rem; background: #fafafa;
        margin-bottom: 1rem;
    }

    .severity-badge {
        display: inline-block; padding: 3px 10px;
        border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    }
    .severity-high   { background: #fed7d7; color: #c53030; }
    .severity-medium { background: #bee3f8; color: #2b6cb0; }
    .severity-low    { background: #c6f6d5; color: #276749; }
    .severity-none   { background: #c6f6d5; color: #276749; }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 1.8rem; font-weight: 600;
        font-size: 1rem; width: 100%;
    }
    .stButton > button:hover { opacity: 0.9; }
    hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# MODEL LOADERS  (cached)
# ─────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_resnet50(pth_path):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, NUM_CLASSES)
    )
    state = torch.load(pth_path, map_location=DEVICE)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


@st.cache_resource(show_spinner=False)
def load_efficientnet(pth_path):
    model = models.efficientnet_b0(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, NUM_CLASSES)
    )
    state = torch.load(pth_path, map_location=DEVICE)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


@st.cache_resource(show_spinner=False)
def load_vit(pth_path):
    vit_config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
    vit_config.num_labels        = NUM_CLASSES
    vit_config.output_attentions = True
    vit_config.id2label          = {i: c for i, c in enumerate(CLASS_NAMES)}
    vit_config.label2id          = {c: i for i, c in enumerate(CLASS_NAMES)}
    model = ViTForImageClassification(vit_config)
    state = torch.load(pth_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def load_model(model_name):
    pth = MODEL_PTH[model_name]
    if not os.path.exists(pth):
        return None
    if model_name == "ResNet50":
        return load_resnet50(pth)
    elif model_name == "EfficientNetB0":
        return load_efficientnet(pth)
    else:
        return load_vit(pth)


# ─────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────

def predict(model, img_tensor, model_name):
    """Returns (pred_idx, probs_array, inference_ms)."""
    model.eval()
    inp = img_tensor.unsqueeze(0).to(DEVICE)
    t0  = time.time()
    with torch.no_grad():
        out   = model(pixel_values=inp).logits if model_name == "ViT-B/16" else model(inp)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    ms = (time.time() - t0) * 1000
    return int(probs.argmax()), probs, ms


# ─────────────────────────────────────────
# GRAD-CAM  (ResNet50 & EfficientNetB0)
# ─────────────────────────────────────────

def get_gradcam_target_layer(model, model_name):
    """Return the last conv layer suitable for Grad-CAM."""
    if model_name == "ResNet50":
        return model.layer4[-1].conv3
    elif model_name == "EfficientNetB0":
        # Last conv block in features
        return model.features[-1][0]
    return None


def compute_gradcam(model, img_tensor, model_name, pred_idx):
    """
    Returns a (H, W) float32 numpy heatmap in [0,1].
    Uses hooks so no model modification is needed.
    """
    target_layer = get_gradcam_target_layer(model, model_name)
    if target_layer is None:
        return None

    activations, gradients = [], []

    def fwd_hook(_, __, output):
        activations.append(output.detach())

    def bwd_hook(_, __, grad_output):
        gradients.append(grad_output[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.eval()
    inp = img_tensor.unsqueeze(0).to(DEVICE).requires_grad_(False)
    # Need grad for backward
    inp = inp.clone().detach().requires_grad_(True)

    out = model(inp)
    score = out[0, pred_idx]
    model.zero_grad()
    score.backward()

    h1.remove()
    h2.remove()

    if not activations or not gradients:
        return None

    act  = activations[0][0]   # (C, H, W)
    grad = gradients[0][0]     # (C, H, W)

    weights = grad.mean(dim=(1, 2), keepdim=True)   # (C, 1, 1)
    cam     = torch.relu((weights * act).sum(dim=0)) # (H, W)
    cam     = cam.cpu().numpy()

    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    return cam.astype(np.float32)


def plot_gradcam(img_tensor, cam, model_name, pred_name, confidence):
    """3-panel figure: original | heatmap | overlay."""
    rgb = INV_NORMALIZE(img_tensor).permute(1, 2, 0).cpu().numpy()
    rgb = np.clip(rgb, 0, 1).astype(np.float32)

    heatmap = plt.cm.jet(cam)[:, :, :3].astype(np.float32)
    overlay = rgb * 0.55 + heatmap * 0.45
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    fig.suptitle(
        f"{model_name} — Grad-CAM  |  Prediction: {pred_name}  ({confidence:.1f}%)",
        fontsize=12, fontweight="bold"
    )
    axes[0].imshow(rgb);           axes[0].set_title("Original MRI")
    im = axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[2].imshow(overlay);       axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# ViT ATTENTION HELPERS
# ─────────────────────────────────────────

def resize_att(att_map, size=IMG_SIZE):
    return np.array(
        Image.fromarray((att_map * 255).astype(np.uint8)).resize(
            (size, size), Image.BILINEAR)
    ) / 255.0


def tensor_to_rgb(img_tensor):
    img = INV_NORMALIZE(img_tensor).permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1).astype(np.float32)


def get_last_layer_attention(vit_model, img_tensor):
    vit_model.eval()
    with torch.no_grad():
        out = vit_model(pixel_values=img_tensor.unsqueeze(0).to(DEVICE))
    layer_attns = out.attentions
    if layer_attns is None or len(layer_attns) == 0:
        return None
    last_attn = layer_attns[-1]
    avg_heads = last_attn[0].mean(dim=0)
    cls_attn  = avg_heads[0, 1:].cpu()
    att_map   = cls_attn.reshape(14, 14).numpy()
    att_map   = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
    return att_map


def get_all_layer_attentions(vit_model, img_tensor):
    vit_model.eval()
    with torch.no_grad():
        out = vit_model(pixel_values=img_tensor.unsqueeze(0).to(DEVICE))
    layer_attns = out.attentions
    if layer_attns is None:
        return None
    maps = []
    for la in layer_attns:
        avg = la[0].mean(dim=0)
        cls = avg[0, 1:].cpu().numpy().reshape(14, 14)
        cls = (cls - cls.min()) / (cls.max() - cls.min() + 1e-8)
        maps.append(cls)
    return maps


def plot_vit_attention(img_tensor, vit_model, pred_name, confidence):
    att_map = get_last_layer_attention(vit_model, img_tensor)
    if att_map is None:
        return None
    display_img = tensor_to_rgb(img_tensor)
    att_resized = resize_att(att_map)
    heatmap     = plt.cm.jet(att_resized)[:, :, :3]
    overlay     = display_img * 0.55 + heatmap * 0.45

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    fig.suptitle(
        f"ViT-B/16 — Attention Map (Last Layer)  |  Prediction: {pred_name}  ({confidence:.1f}%)",
        fontsize=12, fontweight="bold"
    )
    axes[0].imshow(display_img); axes[0].set_title("Original MRI")
    im = axes[1].imshow(att_resized, cmap="jet", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("Attention Map")
    axes[2].imshow(np.clip(overlay, 0, 1)); axes[2].set_title("Overlay")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    return fig


def plot_all_layers(img_tensor, vit_model):
    all_maps = get_all_layer_attentions(vit_model, img_tensor)
    if all_maps is None:
        return None
    display_img = tensor_to_rgb(img_tensor)
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("ViT-B/16 — Attention Across All 12 Layers",
                 fontsize=13, fontweight="bold")
    ax0 = fig.add_subplot(2, 7, 1)
    ax0.imshow(display_img); ax0.set_title("Original", fontsize=10, fontweight="bold")
    ax0.axis("off")
    positions = [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
    cmaps = ["Blues", "Purples", "Greens", "Oranges",
             "Reds",  "YlOrBr",  "BuGn",   "RdPu",
             "YlGnBu","hot",     "cool",    "jet"]
    for i, (att_map, pos, cmap) in enumerate(zip(all_maps, positions, cmaps)):
        ax = fig.add_subplot(2, 7, pos)
        ax.imshow(resize_att(att_map), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"Layer {i+1}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    return fig


def plot_attention_heads(img_tensor, vit_model):
    vit_model.eval()
    with torch.no_grad():
        out = vit_model(pixel_values=img_tensor.unsqueeze(0).to(DEVICE))
    if out.attentions is None:
        return None
    last_attn   = out.attentions[-1][0]   # (12, 197, 197)
    display_img = tensor_to_rgb(img_tensor)

    fig, axes = plt.subplots(3, 5, figsize=(16, 11))
    fig.suptitle("ViT-B/16 — All 12 Attention Heads (Last Layer)",
                 fontsize=13, fontweight="bold")
    axes[0][0].imshow(display_img)
    axes[0][0].set_title("Original", fontsize=10, fontweight="bold")
    axes[0][0].axis("off")
    all_axes = axes.flatten()
    for h in range(12):
        cls = last_attn[h, 0, 1:].cpu().numpy().reshape(14, 14)
        cls = (cls - cls.min()) / (cls.max() - cls.min() + 1e-8)
        all_axes[h + 1].imshow(resize_att(cls), cmap="jet", vmin=0, vmax=1)
        all_axes[h + 1].set_title(f"Head {h+1}", fontsize=9)
        all_axes[h + 1].axis("off")
    avg = last_attn.mean(dim=0)[0, 1:].cpu().numpy().reshape(14, 14)
    avg = (avg - avg.min()) / (avg.max() - avg.min() + 1e-8)
    all_axes[13].imshow(resize_att(avg), cmap="hot", vmin=0, vmax=1)
    all_axes[13].set_title("Avg (all heads)", fontsize=9, fontweight="bold")
    all_axes[13].axis("off")
    all_axes[14].axis("off")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# CONFIDENCE BAR CHART
# ─────────────────────────────────────────

def plot_confidence_bar(probs, pred_idx):
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = [CLASS_COLORS[n] if i == pred_idx else "#CBD5E0"
              for i, n in enumerate(DISPLAY_NAMES)]
    bars = ax.barh(DISPLAY_NAMES, probs * 100, color=colors,
                   edgecolor="white", linewidth=1.2, height=0.55)
    for bar, val in zip(bars, probs * 100):
        ax.text(min(val + 1, 98), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.4)
    ax.set_title("Class Confidence", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Brain Tumor Classifier")
    st.markdown("---")

    st.markdown("### ⚙️ Model Selection")
    selected_model = st.radio(
        "Choose a model:",
        ["ResNet50", "EfficientNetB0", "ViT-B/16"],
        index=0,
        help="Select one model for prediction."
    )

    st.markdown("---")

    # Show explainability options depending on selected model
    if selected_model == "ViT-B/16":
        st.markdown("### 🔍 ViT Explainability")
        show_last_layer = st.checkbox("Attention Map (Last Layer)", value=True)
        show_all_layers = st.checkbox("All 12 Layer Maps",          value=False)
        show_all_heads  = st.checkbox("All 12 Attention Heads",     value=False)
    else:
        st.markdown("### 🔍 Grad-CAM Explainability")
        st.info("Grad-CAM heatmap will be shown automatically for the predicted class.")
        show_last_layer = show_all_layers = show_all_heads = False

    st.markdown("---")
    st.markdown("### 📋 Class Reference")
    for cls, info in CLASS_INFO.items():
        with st.expander(cls):
            st.markdown(f"**Severity:** `{info['severity']}`")
            st.markdown(info["desc"])

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#718096; text-align:center'>"
        "For research & educational use only.<br>"
        "Not a substitute for medical diagnosis."
        "</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🧠 Brain Tumor Classification</h1>
    <p>ResNet50 · EfficientNetB0 · ViT-B/16 &nbsp;|&nbsp; MRI Scan Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────
col_upload, col_info = st.columns([1, 1])

with col_upload:
    st.markdown("### 📤 Upload MRI Scans")
    uploaded_files = st.file_uploader(
        "Supported formats: JPG, PNG, JPEG",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

with col_info:
    st.markdown("### ℹ️ How to Use")
    st.markdown("""
    <div class="info-box">
    1. Upload one or more brain MRI scan images<br>
    2. Select a model from the sidebar<br>
    3. Click <strong>Classify All Scans</strong><br>
    4. View predictions, confidence scores &amp; explainability maps
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="border-color:#27AE60;">
    <strong>Tumor Classes:</strong> Glioma · Meningioma · Pituitary · No Tumor<br>
    <strong>Explainability:</strong> Grad-CAM for CNN models · Attention maps for ViT
    </div>
    """, unsafe_allow_html=True)

if not uploaded_files:
    st.info("👆 Upload one or more MRI scans to begin.")
    st.stop()

# ── Classify button ───────────────────────
st.markdown(f"**{len(uploaded_files)} image(s) uploaded** · Model: `{selected_model}`")
st.markdown("---")

run = st.button("🔬 Classify All Scans", width="stretch")
if not run:
    # Show thumbnail previews while waiting
    cols = st.columns(min(len(uploaded_files), 4))
    for i, uf in enumerate(uploaded_files):
        with cols[i % 4]:
            st.image(Image.open(uf).convert("RGB"),
                     caption=uf.name, width=300)
    st.stop()

# ─────────────────────────────────────────
# LOAD MODEL ONCE
# ─────────────────────────────────────────

with st.spinner(f"Loading **{selected_model}** weights…"):
    model_obj = load_model(selected_model)

if model_obj is None:
    st.error(
        f"❌ Weight file not found for **{selected_model}**: "
        f"`{MODEL_PTH[selected_model]}`\n\n"
        "Please ensure the `.pth` file is inside the `models/` folder."
    )
    st.stop()

st.success(f"✅ {selected_model} loaded on `{DEVICE}`")
st.markdown("---")

# ─────────────────────────────────────────
# RUN INFERENCE — ONE IMAGE AT A TIME
# ─────────────────────────────────────────

st.markdown("## 🎯 Classification Results")

sev_map = {"High": "high", "Medium": "medium", "Low–Medium": "low", "None": "none"}

for img_idx, uploaded in enumerate(uploaded_files):

    pil_img    = Image.open(uploaded).convert("RGB")
    img_tensor = PREPROCESS(pil_img)

    st.markdown(f"### 🖼️ Image {img_idx + 1} — `{uploaded.name}`")

    # ── Image preview + metadata ──────────────
    col_img, col_meta = st.columns([1, 2])
    with col_img:
        st.image(pil_img, caption=uploaded.name, width=400)
    with col_meta:
        w, h = pil_img.size
        st.markdown(f"""
        | Property | Value |
        |---|---|
        | Filename | `{uploaded.name}` |
        | Dimensions | `{w} × {h} px` |
        | Preprocessed | `{IMG_SIZE} × {IMG_SIZE} px` |
        | Model | `{selected_model}` |
        """)

    # ── Inference ─────────────────────────────
    with st.spinner(f"Running inference on {uploaded.name}…"):
        pred_idx, probs, ms = predict(model_obj, img_tensor, selected_model)

    pred_name  = DISPLAY_NAMES[pred_idx]
    color      = CLASS_COLORS[pred_name]
    info       = CLASS_INFO[pred_name]
    confidence = probs[pred_idx] * 100
    sev_cls    = sev_map.get(info["severity"], "none")

    # ── Prediction banner ─────────────────────
    st.markdown(
        f'<div class="prediction-banner" style="background:{color};">'
        f'Predicted: {pred_name} &nbsp;·&nbsp; {confidence:.1f}% confidence'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Metric cards ──────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(
        f'<div class="metric-card">'
        f'<div class="value" style="color:{color}">{pred_name}</div>'
        f'<div class="label">Prediction</div></div>',
        unsafe_allow_html=True
    )
    mc2.markdown(
        f'<div class="metric-card">'
        f'<div class="value" style="color:{color}">{confidence:.1f}%</div>'
        f'<div class="label">Confidence</div></div>',
        unsafe_allow_html=True
    )
    mc3.markdown(
        f'<div class="metric-card">'
        f'<div class="value">{ms:.0f} ms</div>'
        f'<div class="label">Inference Time</div></div>',
        unsafe_allow_html=True
    )
    mc4.markdown(
        f'<div class="metric-card">'
        f'<div class="value"><span class="severity-badge severity-{sev_cls}">'
        f'{info["severity"]}</span></div>'
        f'<div class="label">Severity</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confidence bar + class info ───────────
    col_bar, col_cls = st.columns([1, 1])
    with col_bar:
        fig_bar = plot_confidence_bar(probs, pred_idx)
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    with col_cls:
        st.markdown(f"#### 🔬 About: {pred_name}")
        st.markdown(
            f'<div class="info-box" style="border-color:{color};">'
            f'{info["desc"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown("**All class probabilities:**")
        st.dataframe(
            {
                "Class"      : DISPLAY_NAMES,
                "Probability": [f"{p*100:.2f}%" for p in probs],
                "Score"      : [f"{p:.4f}" for p in probs],
            },
            use_container_width=True,
            hide_index=True
        )

    # ── Explainability ────────────────────────
    st.markdown("#### 🔭 Explainability")

    if selected_model in ("ResNet50", "EfficientNetB0"):
        # ── Grad-CAM ──────────────────────────
        with st.spinner("Generating Grad-CAM heatmap…"):
            cam = compute_gradcam(model_obj, img_tensor, selected_model, pred_idx)
        if cam is not None:
            fig_cam = plot_gradcam(img_tensor, cam, selected_model, pred_name, confidence)
            st.pyplot(fig_cam, use_container_width=True)
            plt.close(fig_cam)
        else:
            st.warning("Grad-CAM could not be generated for this image.")

    else:
        # ── ViT Attention ─────────────────────
        any_vit = show_last_layer or show_all_layers or show_all_heads
        if not any_vit:
            st.info("Enable at least one ViT explainability option in the sidebar.")
        else:
            vit_tabs = st.tabs([
                "🗺️ Last Layer Map",
                "📊 All 12 Layers",
                "🧩 All 12 Heads",
            ])

            with vit_tabs[0]:
                if show_last_layer:
                    with st.spinner("Generating attention map…"):
                        fig_att = plot_vit_attention(img_tensor, model_obj,
                                                     pred_name, confidence)
                    if fig_att:
                        st.pyplot(fig_att, use_container_width=True)
                        plt.close(fig_att)
                    else:
                        st.warning("Attention maps unavailable.")
                else:
                    st.info("Enable 'Attention Map (Last Layer)' in the sidebar.")

            with vit_tabs[1]:
                if show_all_layers:
                    with st.spinner("Generating all-layer maps…"):
                        fig_layers = plot_all_layers(img_tensor, model_obj)
                    if fig_layers:
                        st.pyplot(fig_layers, use_container_width=True)
                        plt.close(fig_layers)
                    else:
                        st.warning("Layer attention maps unavailable.")
                else:
                    st.info("Enable 'All 12 Layer Maps' in the sidebar.")

            with vit_tabs[2]:
                if show_all_heads:
                    with st.spinner("Generating all head maps…"):
                        fig_heads = plot_attention_heads(img_tensor, model_obj)
                    if fig_heads:
                        st.pyplot(fig_heads, use_container_width=True)
                        plt.close(fig_heads)
                    else:
                        st.warning("Head attention maps unavailable.")
                else:
                    st.info("Enable 'All 12 Attention Heads' in the sidebar.")

    st.markdown("---")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────

st.markdown("""
<hr>
<div style='text-align:center; color:#718096; font-size:0.8rem; padding: 1rem 0;'>
    🧠 Brain Tumor Classifier · ResNet50 · EfficientNetB0 · ViT-B/16<br>
    For research and educational purposes only · Not a substitute for medical diagnosis
</div>
""", unsafe_allow_html=True)