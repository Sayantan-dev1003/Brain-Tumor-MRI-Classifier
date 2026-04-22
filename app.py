# ============================================================
# app.py — Brain Tumor Classification Streamlit App
# Models: ResNet50 | EfficientNetB0 | ViT-B/16
# ============================================================
# SETUP:
#   pip install streamlit torch torchvision transformers pillow
#                numpy matplotlib opencv-python-headless
#
# Place your trained .pth files in the same folder as app.py:
#   ResNet50.pth
#   EfficientNetB0.pth
#   ViT-B-16.pth
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
import matplotlib.patches as mpatches
from PIL import Image
import cv2
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import ViTForImageClassification, ViTConfig

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

CLASS_NAMES    = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES  = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
NUM_CLASSES    = 4
IMG_SIZE       = 224
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

MODEL_PTH = {
    "ResNet50"       : "ResNet50.pth",
    "EfficientNetB0" : "EfficientNetB0.pth",
    "ViT-B/16"       : "ViT-B-16.pth",
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
    .main-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #718096;
        margin-top: 4px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .prediction-banner {
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        font-size: 1.4rem;
        font-weight: 700;
        text-align: center;
        color: white;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }

    .info-box {
        background: #f8fafc;
        border-left: 4px solid #4A90E2;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #2d3748;
    }

    .severity-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .severity-high   { background: #fed7d7; color: #c53030; }
    .severity-medium { background: #bee3f8; color: #2b6cb0; }
    .severity-low    { background: #c6f6d5; color: #276749; }
    .severity-none   { background: #c6f6d5; color: #276749; }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; }

    .sidebar-section {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .tab-content { padding: 1rem 0; }
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
    model = model.to(DEVICE).eval()
    return model


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
    model = model.to(DEVICE).eval()
    return model


@st.cache_resource(show_spinner=False)
def load_vit(pth_path):
    """Load ViT with output_attentions=True baked into config."""
    vit_config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
    vit_config.num_labels        = NUM_CLASSES
    vit_config.output_attentions = True
    vit_config.id2label          = {i: c for i, c in enumerate(CLASS_NAMES)}
    vit_config.label2id          = {c: i for i, c in enumerate(CLASS_NAMES)}

    model = ViTForImageClassification(vit_config)
    state = torch.load(pth_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model = model.to(DEVICE).eval()
    return model


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
# ViT ATTENTION HELPERS
# ─────────────────────────────────────────

def resize_att(att_map, size=IMG_SIZE):
    return np.array(
        Image.fromarray((att_map * 255).astype(np.uint8)).resize(
            (size, size), Image.BILINEAR)
    ) / 255.0


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


def tensor_to_rgb(img_tensor):
    img = INV_NORMALIZE(img_tensor).permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1).astype(np.float32)


# ─────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────

def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


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


def plot_vit_attention(img_tensor, vit_model):
    att_map = get_last_layer_attention(vit_model, img_tensor)
    if att_map is None:
        return None
    display_img = tensor_to_rgb(img_tensor)
    att_resized = resize_att(att_map)
    heatmap     = plt.cm.jet(att_resized)[:, :, :3]
    overlay     = display_img * 0.55 + heatmap * 0.45

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("ViT-B/16 — Attention Map (Last Layer)", fontsize=12, fontweight="bold")

    axes[0].imshow(display_img); axes[0].set_title("Original MRI")
    im = axes[1].imshow(att_resized, cmap="jet", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("Attention Map")
    axes[2].imshow(overlay); axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    return fig


def plot_all_layers(img_tensor, vit_model):
    all_maps = get_all_layer_attentions(vit_model, img_tensor)
    if all_maps is None:
        return None
    display_img = tensor_to_rgb(img_tensor)

    fig = plt.figure(figsize=(24, 9))
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

    fig, axes = plt.subplots(3, 5, figsize=(20, 13))
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
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Brain Tumor Classifier")
    st.markdown("---")

    st.markdown("### ⚙️ Model Selection")
    selected_models = st.multiselect(
        "Choose model(s) to run:",
        ["ResNet50", "EfficientNetB0", "ViT-B/16"],
        default=["ViT-B/16"],
        help="Select one or more models for prediction."
    )

    st.markdown("---")
    st.markdown("### 🔍 ViT Explainability")
    show_last_layer = st.checkbox("Attention Map (Last Layer)", value=True)
    show_all_layers = st.checkbox("All 12 Layer Maps", value=False)
    show_all_heads  = st.checkbox("All 12 Attention Heads", value=False)

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
    st.markdown("### 📤 Upload MRI Scan")
    uploaded = st.file_uploader(
        "Supported formats: JPG, PNG, JPEG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with col_info:
    st.markdown("### ℹ️ How to Use")
    st.markdown("""
    <div class="info-box">
    1. Upload a brain MRI scan image<br>
    2. Select model(s) from the sidebar<br>
    3. Click <strong>Classify</strong><br>
    4. View predictions, confidence scores, and (for ViT) attention maps
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box" style="border-color:#27AE60;">
    <strong>Tumor Classes:</strong> Glioma · Meningioma · Pituitary · No Tumor
    </div>
    """, unsafe_allow_html=True)

if not uploaded:
    st.info("👆 Upload an MRI scan to begin.")
    st.stop()

# ── Show uploaded image ───────────────────
pil_img    = Image.open(uploaded).convert("RGB")
img_tensor = PREPROCESS(pil_img)

col_img, col_meta = st.columns([1, 2])
with col_img:
    st.image(pil_img, caption="Uploaded MRI Scan", use_container_width=True)
with col_meta:
    st.markdown("### 📊 Image Details")
    w, h = pil_img.size
    st.markdown(f"""
    | Property | Value |
    |---|---|
    | Filename | `{uploaded.name}` |
    | Dimensions | `{w} × {h} px` |
    | Preprocessed | `{IMG_SIZE} × {IMG_SIZE} px` |
    | Models selected | `{', '.join(selected_models) if selected_models else 'None'}` |
    """)

st.markdown("---")

# ── Classify button ───────────────────────
if not selected_models:
    st.warning("Please select at least one model from the sidebar.")
    st.stop()

run = st.button("🔬 Classify MRI Scan", use_container_width=True)
if not run:
    st.stop()

# ─────────────────────────────────────────
# RUN INFERENCE
# ─────────────────────────────────────────

st.markdown("## 🎯 Classification Results")

results_store = {}   # model_name → {pred_idx, probs, ms, model_obj}

progress = st.progress(0)
status   = st.empty()

for i, model_name in enumerate(selected_models):
    status.info(f"⏳ Loading & running **{model_name}**...")
    model_obj = load_model(model_name)

    if model_obj is None:
        st.error(f"❌ Weight file not found for **{model_name}**: "
                 f"`{MODEL_PTH[model_name]}`")
        progress.progress((i + 1) / len(selected_models))
        continue

    pred_idx, probs, ms = predict(model_obj, img_tensor, model_name)
    results_store[model_name] = {
        "pred_idx"  : pred_idx,
        "probs"     : probs,
        "ms"        : ms,
        "model_obj" : model_obj,
    }
    progress.progress((i + 1) / len(selected_models))

status.empty()
progress.empty()

if not results_store:
    st.error("No models could be loaded. Please check that .pth files are present.")
    st.stop()

# ─────────────────────────────────────────
# DISPLAY RESULTS PER MODEL
# ─────────────────────────────────────────

for model_name, res in results_store.items():
    pred_idx  = res["pred_idx"]
    probs     = res["probs"]
    ms        = res["ms"]
    pred_name = DISPLAY_NAMES[pred_idx]
    color     = CLASS_COLORS[pred_name]
    info      = CLASS_INFO[pred_name]
    confidence= probs[pred_idx] * 100

    # Severity badge CSS class
    sev_map = {"High": "high", "Medium": "medium",
               "Low–Medium": "low", "None": "none"}
    sev_cls = sev_map.get(info["severity"], "none")

    st.markdown(f"### 🤖 {model_name}")

    # Prediction banner
    st.markdown(
        f'<div class="prediction-banner" style="background:{color};">'
        f'Predicted: {pred_name} &nbsp;·&nbsp; {confidence:.1f}% confidence'
        f'</div>',
        unsafe_allow_html=True
    )

    # Metric cards
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

    # Confidence bar + class info
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
        prob_data = {
            "Class"      : DISPLAY_NAMES,
            "Probability": [f"{p*100:.2f}%" for p in probs],
            "Score"      : [f"{p:.4f}" for p in probs],
        }
        st.dataframe(prob_data, use_container_width=True, hide_index=True)

    # ── ViT Explainability ──────────────────
    if model_name == "ViT-B/16":
        vit_obj = res["model_obj"]
        any_vit = show_last_layer or show_all_layers or show_all_heads

        if any_vit:
            st.markdown("#### 🔭 ViT Attention Explainability")
            vit_tabs = st.tabs([
                "🗺️ Last Layer Map",
                "📊 All 12 Layers",
                "🧩 All 12 Heads",
            ])

            with vit_tabs[0]:
                if show_last_layer:
                    with st.spinner("Generating attention map..."):
                        fig_att = plot_vit_attention(img_tensor, vit_obj)
                    if fig_att:
                        st.pyplot(fig_att, use_container_width=True)
                        plt.close(fig_att)
                    else:
                        st.warning("Attention maps unavailable.")
                else:
                    st.info("Enable 'Attention Map (Last Layer)' in the sidebar.")

            with vit_tabs[1]:
                if show_all_layers:
                    with st.spinner("Generating all-layer maps..."):
                        fig_layers = plot_all_layers(img_tensor, vit_obj)
                    if fig_layers:
                        st.pyplot(fig_layers, use_container_width=True)
                        plt.close(fig_layers)
                    else:
                        st.warning("Layer attention maps unavailable.")
                else:
                    st.info("Enable 'All 12 Layer Maps' in the sidebar.")

            with vit_tabs[2]:
                if show_all_heads:
                    with st.spinner("Generating all head maps..."):
                        fig_heads = plot_attention_heads(img_tensor, vit_obj)
                    if fig_heads:
                        st.pyplot(fig_heads, use_container_width=True)
                        plt.close(fig_heads)
                    else:
                        st.warning("Head attention maps unavailable.")
                else:
                    st.info("Enable 'All 12 Attention Heads' in the sidebar.")

    st.markdown("---")

# ─────────────────────────────────────────
# MULTI-MODEL COMPARISON (if >1 model run)
# ─────────────────────────────────────────

if len(results_store) > 1:
    st.markdown("## 📊 Multi-Model Comparison")

    model_names  = list(results_store.keys())
    pred_classes = [DISPLAY_NAMES[results_store[m]["pred_idx"]] for m in model_names]
    confidences  = [results_store[m]["probs"][results_store[m]["pred_idx"]] * 100
                    for m in model_names]
    times        = [results_store[m]["ms"] for m in model_names]

    # Agreement check
    unique_preds = set(pred_classes)
    if len(unique_preds) == 1:
        st.success(f"✅ All models agree: **{pred_classes[0]}**")
    else:
        st.warning(f"⚠️ Models disagree: {', '.join([f'{m}→{p}' for m, p in zip(model_names, pred_classes)])}")

    # Summary table
    comp_df = {
        "Model"      : model_names,
        "Prediction" : pred_classes,
        "Confidence" : [f"{c:.2f}%" for c in confidences],
        "Time (ms)"  : [f"{t:.1f}" for t in times],
    }
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Grouped bar — confidence per class per model
    fig_comp, ax = plt.subplots(figsize=(12, 5))
    x     = np.arange(NUM_CLASSES)
    width = 0.8 / len(model_names)
    model_colors = ["#4A90E2", "#E25C4A", "#50C878"]

    for i, m_name in enumerate(model_names):
        probs  = results_store[m_name]["probs"]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, probs * 100, width,
                        label=m_name, color=model_colors[i % 3],
                        alpha=0.85, edgecolor="white")
        for bar in bars:
            if bar.get_height() > 3:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{bar.get_height():.1f}",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_NAMES, fontsize=11)
    ax.set_ylabel("Confidence (%)", fontsize=12)
    ax.set_title("Model Confidence Comparison Across All Classes",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_comp, use_container_width=True)
    plt.close(fig_comp)

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