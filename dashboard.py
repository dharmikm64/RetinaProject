"""
IDRiD Retinal Analysis Dashboard

Two-page Streamlit app:
  Page 1 - Dataset Overview : summary stats + 4 pre-generated charts
  Page 2 - Analyze an Image : upload image -> predict DR grade + confidence
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Make sure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from data_pipline import PROJECT_DIR
from analysis import GRADE_LABELS, GRADE_ORDER
from classifier import predict

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="IDRiD Retinal Analysis",
    page_icon="\U0001f441",   # eye emoji
    layout="wide",
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

CHART_FILES = {
    "grade_distribution"   : "grade_distribution.png",
    "exudate_vs_grade"     : "exudate_vs_grade.png",
    "sample_overlays"      : "sample_overlays.png",
    "exudate_presence_rate": "exudate_presence_rate.png",
}

CHART_TITLES = {
    "grade_distribution"   : "Grade Distribution",
    "exudate_vs_grade"     : "Hard Exudate Coverage vs Grade",
    "sample_overlays"      : "Sample Retinal Images per Grade",
    "exudate_presence_rate": "Hard Exudate Presence Rate",
}

GRADE_CONTEXT = {
    0: "No diabetic retinopathy detected. No immediate DR-related intervention required.",
    1: "Mild NPDR — microaneurysms only. Monitor annually.",
    2: "Moderate NPDR — some haemorrhages and/or hard exudates. Closer follow-up advised.",
    3: "Severe NPDR — extensive vessel damage. Refer to retinal specialist.",
    4: "Proliferative DR — new vessel growth detected. Urgent specialist referral.",
}


@st.cache_data(show_spinner="Loading dataset...")
def _load_df():
    """Load master DataFrame (CSV only — no image/mask loading for speed)."""
    import pandas as pd
    csv_path = PROJECT_DIR / "processed_dataset.csv"
    if not csv_path.exists():
        st.error(f"processed_dataset.csv not found at {PROJECT_DIR}")
        st.stop()
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Page 1 — Dataset Overview
# ---------------------------------------------------------------------------

def page_overview():
    st.title("IDRiD Dataset Overview")

    df = _load_df()

    # ---- Summary metrics ---------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Images",  len(df))
    c2.metric("Train Images",  int((df["split"] == "train").sum()))
    c3.metric("Test Images",   int((df["split"] == "test").sum()))
    c4.metric("With Seg Masks", int(df["has_masks"].sum()))
    c5.metric("Grades",        df["retinopathy_grade"].nunique())

    st.divider()

    # ---- Row 1: two bar charts side-by-side --------------------------------
    col_a, col_b = st.columns(2)
    for col, key in zip([col_a, col_b], ["grade_distribution", "exudate_vs_grade"]):
        png = PROJECT_DIR / CHART_FILES[key]
        col.subheader(CHART_TITLES[key])
        if png.exists():
            col.image(str(png), use_container_width=True)
        else:
            col.warning(f"{CHART_FILES[key]} not found — run analysis.py first.")

    st.divider()

    # ---- Row 2: full-width sample overlay grid ----------------------------
    key = "sample_overlays"
    png = PROJECT_DIR / CHART_FILES[key]
    st.subheader(CHART_TITLES[key])
    if png.exists():
        st.image(str(png), use_container_width=True)
    else:
        st.warning(f"{CHART_FILES[key]} not found — run analysis.py first.")

    st.divider()

    # ---- Row 3: presence rate + grade table --------------------------------
    col_left, col_right = st.columns([3, 2])

    key = "exudate_presence_rate"
    png = PROJECT_DIR / CHART_FILES[key]
    col_left.subheader(CHART_TITLES[key])
    if png.exists():
        col_left.image(str(png), use_container_width=True)
    else:
        col_left.warning(f"{CHART_FILES[key]} not found — run analysis.py first.")

    col_right.subheader("Grade Breakdown")
    grade_counts = (
        df.groupby("retinopathy_grade")
        .size()
        .reindex(GRADE_ORDER, fill_value=0)
    )
    rows = []
    for g in GRADE_ORDER:
        rows.append({
            "Grade": f"Grade {g}",
            "Label": GRADE_LABELS[g],
            "Images": int(grade_counts[g]),
        })
    import pandas as pd
    col_right.dataframe(pd.DataFrame(rows).set_index("Grade"), use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2 — Analyze an Image
# ---------------------------------------------------------------------------

def page_analyze():
    st.title("Analyze a Retinal Image")
    st.write(
        "Upload a retinal fundus photograph. "
        "The EfficientNet-B0 classifier will predict the DR grade."
    )

    uploaded = st.file_uploader("Choose a retinal image (JPG or PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image above to begin.")
        return

    # Load and resize
    img_pil = Image.open(uploaded).convert("RGB")
    img_arr = np.array(img_pil.resize((512, 512), Image.LANCZOS))

    col_img, col_result = st.columns([1, 1])

    col_img.subheader("Uploaded Image")
    col_img.image(img_arr, use_container_width=True)

    with st.spinner("Running classifier..."):
        grade, confidence = predict(img_arr)

    col_result.subheader("Prediction")

    grade_colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF5722", "#F44336"]
    color = grade_colors[grade]

    col_result.markdown(
        f"<h2 style='color:{color}'>Grade {grade} — {GRADE_LABELS[grade]}</h2>",
        unsafe_allow_html=True,
    )
    col_result.metric("Confidence", f"{confidence:.1f}%")
    col_result.divider()
    col_result.markdown(f"**Clinical context:** {GRADE_CONTEXT[grade]}")

    # ---- Confidence bar across all grades ---------------------------------
    col_result.divider()
    col_result.subheader("Grade Confidence Breakdown")

    import torch
    import torchvision.transforms as transforms
    from classifier import _model_cache, _transform, MODEL_PATH, NUM_CLASSES
    from torchvision import models

    # Re-run forward pass to get full softmax probabilities
    # (predict() only returns the top grade — we need all 5 probs for the bar)
    try:
        from classifier import _model_cache, _transform

        model = _model_cache  # already loaded by predict()
        if model is not None:
            device = next(model.parameters()).device
            t = _transform(img_pil.resize((224, 224), Image.LANCZOS))
            with torch.no_grad():
                logits = model(t.unsqueeze(0).to(device))
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

            import pandas as pd
            prob_df = pd.DataFrame({
                "Grade": [f"Grade {g}: {GRADE_LABELS[g]}" for g in GRADE_ORDER],
                "Probability (%)": [float(p) * 100 for p in probs],
            })
            col_result.bar_chart(prob_df.set_index("Grade"))
    except Exception:
        pass  # silently skip if model not available


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.sidebar.title("IDRiD Analysis")
    st.sidebar.markdown("Diabetic Retinopathy grading tool built on the IDRiD dataset.")
    st.sidebar.divider()

    page = st.sidebar.radio("Navigation", ["Dataset Overview", "Analyze an Image"])

    if page == "Dataset Overview":
        page_overview()
    else:
        page_analyze()


if __name__ == "__main__":
    main()
