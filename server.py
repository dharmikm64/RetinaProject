"""
Flask backend for the IDRiD retinal analysis dashboard.

Endpoints:
  GET  /                      -> serves dashboard.html
  GET  /api/stats             -> dataset summary JSON
  GET  /api/chart/<name>      -> pre-generated PNG charts
  POST /api/predict           -> upload image, returns grade + probs JSON
  POST /api/progression       -> multi-visit progression analysis
"""

import io
import base64
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from data_pipline import PROJECT_DIR
from classifier import predict

app  = Flask(__name__)
CORS(app)

@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()
    return jsonify({'error': str(e), 'type': type(e).__name__}), 500

@app.errorhandler(400)
def handle_400(e):
    return jsonify({'error': str(e)}), 400

VALID_CHARTS = {
    'grade_distribution',
    'exudate_vs_grade',
    'sample_overlays',
    'exudate_presence_rate',
}

GRADE_LABELS = {
    0: 'No DR',
    1: 'Mild DR',
    2: 'Moderate DR',
    3: 'Severe DR',
    4: 'Proliferative DR',
}

GRADE_CONTEXT = {
    0: 'No diabetic retinopathy detected. No immediate DR-related intervention required.',
    1: 'Mild NPDR — microaneurysms only. Monitor annually.',
    2: 'Moderate NPDR — some haemorrhages and/or hard exudates. Closer follow-up advised.',
    3: 'Severe NPDR — extensive vessel damage. Refer to retinal specialist.',
    4: 'Proliferative DR — new vessel growth detected. Urgent specialist referral.',
}




def _auto_mask(img_arr):
    """
    Auto-detect hard exudate regions from a retinal image when no mask is provided.
    Hard exudates appear as bright yellowish-white patches on a darker retinal background.
    Uses the green channel (exudates are bright in green) + CLAHE + adaptive thresholding.
    """
    import cv2

    # --- Step 1: isolate the retinal disc (exclude black border) ---
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    _, retina = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    # keep only the largest connected region (the eye)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(retina, connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        retina = (lbl == biggest).astype(np.uint8) * 255

    # --- Step 2: enhance green channel contrast inside the retina ---
    green = img_arr[:, :, 1].copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)

    # --- Step 3: threshold the top 8% brightest retinal pixels ---
    retinal_pixels = enhanced[retina > 0]
    if retinal_pixels.size == 0:
        return np.zeros((512, 512), dtype=np.uint8)
    thresh = float(np.percentile(retinal_pixels, 92))
    bright = ((enhanced > thresh) & (retina > 0)).astype(np.uint8) * 255

    # --- Step 4: remove the optic disc (the single largest bright blob) ---
    n2, lbl2, stats2, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n2 > 1:
        biggest2 = 1 + int(np.argmax(stats2[1:, cv2.CC_STAT_AREA]))
        if stats2[biggest2, cv2.CC_STAT_AREA] > 800:
            bright[lbl2 == biggest2] = 0

    # --- Step 5: morphological cleanup ---
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,  k)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k)

    return (bright > 0).astype(np.uint8)

@app.route('/')
def index():
    return send_file(str(PROJECT_DIR / 'dashboard.html'))


@app.route('/api/stats')
def stats():
    df = pd.read_csv(PROJECT_DIR / 'processed_dataset.csv')
    grade_counts = (
        df.groupby('retinopathy_grade').size()
        .reindex([0, 1, 2, 3, 4], fill_value=0)
        .to_dict()
    )
    return jsonify({
        'total'      : len(df),
        'train'      : int((df['split'] == 'train').sum()),
        'test'       : int((df['split'] == 'test').sum()),
        'with_masks' : int(df['has_masks'].sum()),
        'grade_counts': {str(k): int(v) for k, v in grade_counts.items()},
        'grade_labels': {str(k): v for k, v in GRADE_LABELS.items()},
    })


@app.route('/api/chart/<name>')
def chart(name):
    if name not in VALID_CHARTS:
        return jsonify({'error': 'unknown chart'}), 404
    path = PROJECT_DIR / f'{name}.png'
    if not path.exists():
        return jsonify({'error': 'chart not generated — run analysis.py first'}), 404
    return send_file(str(path), mimetype='image/png')


@app.route('/api/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'no image field in request'}), 400

    img_pil = Image.open(request.files['image'].stream).convert('RGB')
    img_arr = np.array(img_pil.resize((512, 512), Image.LANCZOS))

    grade, confidence = predict(img_arr)

    probs = [0.0] * 5
    try:
        import torch
        from classifier import _model_cache, _transform

        model = _model_cache
        if model is not None:
            device = next(model.parameters()).device
            t      = _transform(img_pil.resize((224, 224), Image.LANCZOS))
            with torch.no_grad():
                logits = model(t.unsqueeze(0).to(device))
                p      = torch.softmax(logits, dim=1)[0].cpu().numpy()
                probs  = [round(float(x) * 100, 2) for x in p]
    except Exception:
        probs[grade] = round(confidence, 2)

    return jsonify({
        'grade'      : grade,
        'label'      : GRADE_LABELS[grade],
        'confidence' : round(confidence, 1),
        'probs'      : probs,
        'context'    : GRADE_CONTEXT[grade],
    })


@app.route('/api/progression', methods=['POST'])
def progression_route():
    from progression import analyze_progression

    # Parse how many visits were submitted
    try:
        visit_count = int(request.form.get('visit_count', 0))
    except (ValueError, TypeError):
        return jsonify({'error': 'invalid visit_count'}), 400

    if visit_count < 2:
        return jsonify({'error': 'at least 2 visits are required'}), 400

    visits = []
    for i in range(visit_count):
        date_str = request.form.get(f'date_{i}', '').strip()
        if not date_str:
            return jsonify({'error': f'missing date for visit {i + 1}'}), 400

        img_file = request.files.get(f'image_{i}')
        if not img_file:
            return jsonify({'error': f'missing image for visit {i + 1}'}), 400

        img_pil = Image.open(img_file.stream).convert('RGB')
        img_arr = np.array(img_pil.resize((512, 512), Image.LANCZOS))

        # If a mask is provided, use it; otherwise auto-detect exudates from the image
        mask_file = request.files.get(f'mask_{i}')
        if mask_file:
            mask_pil = Image.open(mask_file.stream).convert('L')
            mask_arr = (np.array(mask_pil.resize((512, 512), Image.NEAREST)) > 127).astype(np.uint8)
        else:
            mask_arr = _auto_mask(img_arr)

        visits.append({'date': date_str, 'image': img_arr, 'mask': mask_arr})

    # Optional fovea center
    fovea_center = None
    try:
        fx = int(request.form.get('fovea_x', '').strip())
        fy = int(request.form.get('fovea_y', '').strip())
        fovea_center = (fx, fy)
    except (ValueError, TypeError):
        pass   # falls back to image center (256, 256) inside analyze_progression

    try:
        report, figures = analyze_progression(visits, fovea_center=fovea_center)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'analyze_progression failed: {e}', 'type': type(e).__name__}), 500

    # Encode all figures as base64 PNG so the browser can display them inline
    figures_b64 = {}
    for name, fig in figures.items():
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        figures_b64[name] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

    return jsonify({'report': report, 'figures': figures_b64})


if __name__ == '__main__':
    print('[IDRiD] Starting server at http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)
