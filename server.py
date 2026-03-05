"""
Flask backend for the IDRiD retinal analysis dashboard.

Endpoints:
  GET  /                      -> serves dashboard.html
  GET  /api/stats             -> dataset summary JSON
  GET  /api/chart/<name>      -> pre-generated PNG charts
  POST /api/predict           -> upload image, returns grade + probs JSON
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from data_pipline import PROJECT_DIR
from classifier import predict

app  = Flask(__name__)
CORS(app)

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

    # Full softmax probabilities for the probability bar chart
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


if __name__ == '__main__':
    print('[IDRiD] Starting server at http://localhost:5000')
    app.run(host='0.0.0.0', port=5000, debug=False)
