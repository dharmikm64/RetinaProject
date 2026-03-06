"""
IDRiD Retinal Exudate Progression Analysis

Analyzes a series of retinal visits from a single patient and tracks
how hard exudate coverage changes over time.

Public API:
    analyze_progression(visits, fovea_center=None)
        -> (report: dict, figures: dict)
"""

import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from data_pipline import PROJECT_DIR


# -- Constants ---------------------------------------------------

IMAGE_SIZE   = 512
TOTAL_PX     = IMAGE_SIZE * IMAGE_SIZE
THREAT_RADIUS = 100          # pixels from fovea to flag as macular threat
STABLE_SLOPE  = 0.1          # % per visit; below this = "stable"


# -- Helpers -----------------------------------------------------

def _parse_date(s):
    return datetime.strptime(s, '%Y-%m-%d')

def _label(dt):
    return dt.strftime('%b %Y')

def _style(ax):
    ax.spines[['top', 'right']].set_visible(False)

def _coverage(mask):
    """Return exudate coverage as a percentage of total image area."""
    return float((mask > 0).sum()) / TOTAL_PX * 100.0

def _overlay(base_img, mask_a, mask_b=None, color_a=(1, 1, 0), color_b=None, alpha=0.45):
    """
    Blend colored mask overlays onto base_img (float32 0-1).
    mask_a gets color_a, mask_b gets color_b.
    """
    out = base_img.copy()
    for mask, color in [(mask_a, color_a), (mask_b, color_b)]:
        if mask is None or color is None:
            continue
        m = mask.astype(bool)
        for ch, val in enumerate(color):
            out[m, ch] = out[m, ch] * (1 - alpha) + val * alpha
    return np.clip(out, 0, 1)

def _threat_clusters(mask, fovea_xy):
    """
    Run connected components on mask.
    Returns (num_labels, labels, centroids, threat_flags) where
    threat_flags[i] is True if cluster i+1 centroid is within THREAT_RADIUS of fovea.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    fx, fy = fovea_xy
    threat_flags = []
    for lid in range(1, num_labels):   # skip background label 0
        cx, cy = centroids[lid]
        dist = np.sqrt((cx - fx) ** 2 + (cy - fy) ** 2)
        threat_flags.append(dist < THREAT_RADIUS)
    return num_labels, labels, centroids, threat_flags


# -- Main function -----------------------------------------------

def analyze_progression(visits, fovea_center=None):
    """
    Analyze exudate progression across multiple patient visits.

    Parameters
    ----------
    visits : list of dict
        Each entry: {'date': 'YYYY-MM-DD', 'image': np.array (512,512,3),
                                            'mask':  np.array (512,512) binary}
        Visits are sorted by date internally.
    fovea_center : tuple (x, y) or None
        Pixel coordinate of the fovea. Defaults to image center (256, 256).

    Returns
    -------
    report  : dict  (see docstring for keys)
    figures : dict  {name: matplotlib.figure.Figure}
    """

    # -- Sort visits oldest -> newest ----------------------------
    visits = sorted(visits, key=lambda v: _parse_date(v['date']))
    n      = len(visits)
    dates  = [_parse_date(v['date']) for v in visits]
    labels = [_label(d) for d in dates]

    if fovea_center is None:
        fovea_center = (IMAGE_SIZE // 2, IMAGE_SIZE // 2)

    figures = {}

    # ============================================================
    # 1. Exudate coverage trend over time
    # ============================================================

    coverages = [_coverage(v['mask']) for v in visits]

    # Fit a line to coverage vs visit index to determine trend
    x_idx = np.arange(n, dtype=float)
    slope  = float(np.polyfit(x_idx, coverages, 1)[0]) if n > 1 else 0.0
    if abs(slope) < STABLE_SLOPE:
        trend = 'stable'
    elif slope > 0:
        trend = 'increasing'
    else:
        trend = 'decreasing'

    trend_colors = {'increasing': '#F44336', 'decreasing': '#4CAF50', 'stable': '#4f8ef7'}

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(labels, coverages, marker='o', color=trend_colors[trend],
            linewidth=2.5, markersize=9, zorder=3)
    ax.fill_between(labels, coverages, alpha=0.12, color=trend_colors[trend])
    for i, (lbl, cov) in enumerate(zip(labels, coverages)):
        ax.annotate(f'{cov:.2f}%', (lbl, cov),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=9, color=trend_colors[trend])
    ax.set_xlabel('Visit Date', labelpad=8)
    ax.set_ylabel('Exudate Coverage (%)')
    ax.set_title(f'Hard Exudate Coverage Over Time  \u2014  Trend: {trend.capitalize()}',
                 fontsize=13)
    _style(ax)
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / 'progression_trend.png', dpi=150, bbox_inches='tight')
    figures['progression_trend'] = fig

    # ============================================================
    # 2. New vs resolved exudates between consecutive visits
    # ============================================================

    new_fracs      = []
    resolved_fracs = []

    for i in range(n - 1):
        older = visits[i]['mask'].astype(bool)
        newer = visits[i + 1]['mask'].astype(bool)
        img   = visits[i + 1]['image'].astype(np.float32) / 255.0

        # Pixel-level change maps
        new_px        = newer & ~older    # appeared since last visit
        resolved_px   = ~newer & older   # cleared since last visit
        persistent_px = newer & older    # still present

        new_fracs.append(float(new_px.sum()) / TOTAL_PX * 100.0)
        resolved_fracs.append(float(resolved_px.sum()) / TOTAL_PX * 100.0)

        # Build overlay: persistent=yellow, new=red, resolved=green
        out = img.copy()
        alpha = 0.45
        for mask, r, g, b in [
            (persistent_px, 1, 1, 0),   # yellow
            (new_px,        1, 0, 0),   # red
            (resolved_px,   0, 1, 0),   # green
        ]:
            m = mask
            out[m, 0] = out[m, 0] * (1 - alpha) + r * alpha
            out[m, 1] = out[m, 1] * (1 - alpha) + g * alpha
            out[m, 2] = out[m, 2] * (1 - alpha) + b * alpha
        out = np.clip(out, 0, 1)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(out)
        ax.set_title(
            f'Exudate Changes: {labels[i]} \u2192 {labels[i + 1]}\n'
            f'New (red)   Resolved (green)   Persistent (yellow)',
            fontsize=10,
        )
        ax.axis('off')
        legend = [
            Patch(color='red',    label=f'New ({new_px.sum()} px, {new_fracs[-1]:.2f}%)'),
            Patch(color='green',  label=f'Resolved ({resolved_px.sum()} px, {resolved_fracs[-1]:.2f}%)'),
            Patch(color='yellow', label=f'Persistent ({persistent_px.sum()} px)'),
        ]
        ax.legend(handles=legend, loc='lower right', fontsize=8, framealpha=0.85)
        fig.tight_layout()
        name = f'progression_diff_visit{i + 1}_visit{i + 2}'
        fig.savefig(PROJECT_DIR / f'{name}.png', dpi=150, bbox_inches='tight')
        figures[name] = fig

    # ============================================================
    # 3. Fovea proximity analysis per visit
    # ============================================================

    macular_threats = []
    fx, fy = fovea_center

    for i, visit in enumerate(visits):
        img = visit['image'].astype(np.float32) / 255.0
        out = img.copy()
        alpha = 0.5

        num_labels, label_map, centroids, threat_flags = _threat_clusters(
            visit['mask'], fovea_center
        )

        any_threat = any(threat_flags)
        macular_threats.append(any_threat)

        # Colorize each cluster: red if threat, yellow otherwise
        for lid, is_threat in enumerate(threat_flags, start=1):
            m = (label_map == lid)
            if is_threat:
                out[m, 0] = out[m, 0] * (1 - alpha) + alpha   # R
                out[m, 1] = out[m, 1] * (1 - alpha)            # G off
                out[m, 2] = out[m, 2] * (1 - alpha)            # B off
            else:
                out[m, 0] = out[m, 0] * (1 - alpha) + alpha   # R
                out[m, 1] = out[m, 1] * (1 - alpha) + alpha   # G (yellow)
        out = np.clip(out, 0, 1)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(out)

        # Draw fovea threat radius circle
        circle = plt.Circle((fx, fy), THREAT_RADIUS,
                             color='cyan', fill=False, linewidth=2, linestyle='--')
        ax.add_patch(circle)
        ax.plot(fx, fy, '+', color='cyan', markersize=14, markeredgewidth=2.5)

        threat_str   = 'MACULAR THREAT DETECTED' if any_threat else 'No macular threat'
        title_color  = '#F44336' if any_threat else '#e8eaf0'
        ax.set_title(f'Fovea Proximity \u2014 {labels[i]}\n{threat_str}',
                     fontsize=10, color=title_color)
        ax.axis('off')
        legend = [
            Patch(color='red',    label='Macular threat cluster'),
            Patch(color='yellow', label='Non-threatening cluster'),
            Patch(facecolor='none', edgecolor='cyan', linewidth=1.5,
                  label=f'Threat zone ({THREAT_RADIUS} px radius)'),
        ]
        ax.legend(handles=legend, loc='lower right', fontsize=8, framealpha=0.85)
        fig.tight_layout()
        name = f'fovea_proximity_visit{i + 1}'
        fig.savefig(PROJECT_DIR / f'{name}.png', dpi=150, bbox_inches='tight')
        figures[name] = fig

    # ============================================================
    # 4. Plain English summary per consecutive pair
    # ============================================================

    pair_summaries = []

    for i in range(n - 1):
        d1  = dates[i].strftime('%B %Y')
        d2  = dates[i + 1].strftime('%B %Y')
        delta     = coverages[i + 1] - coverages[i]
        direction = 'increased' if delta >= 0 else 'decreased'

        # Count threat clusters in the newer visit
        _, _, _, threat_flags = _threat_clusters(visits[i + 1]['mask'], fovea_center)
        n_threats = sum(threat_flags)

        line = (
            f'Between {d1} and {d2}, exudate coverage {direction} by '
            f'{abs(delta):.1f}% and new exudates appeared in '
            f'{new_fracs[i]:.1f}% of the image area. '
            f'{n_threats} cluster{"s" if n_threats != 1 else ""} '
            f'{"are" if n_threats != 1 else "is"} within the macular threat zone.'
        )
        print(line)
        pair_summaries.append(line)

    overall = (
        f'Overall trend across {n} visit{"s" if n != 1 else ""}: '
        f'exudate coverage is {trend}. '
        f'Coverage ranged from {min(coverages):.1f}% to {max(coverages):.1f}% '
        f'across the observation period.'
    )
    print(overall)

    # ============================================================
    # 5. Summary report dict
    # ============================================================

    report = {
        'coverage_over_time'      : coverages,
        'trend'                   : trend,
        'new_exudate_fractions'   : new_fracs,
        'resolved_exudate_fractions': resolved_fracs,
        'macular_threat_per_visit': macular_threats,
        'summary_text'            : ' '.join(pair_summaries + [overall]),
    }

    return report, figures


# -- Demo --------------------------------------------------------

if __name__ == '__main__':
    print('[demo] Generating synthetic patient visits...')
    rng  = np.random.default_rng(42)
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    demo_visits = []
    for idx, date_str in enumerate(['2021-03-01', '2022-06-15', '2023-09-10']):
        # Simulate progressive exudate growth by adding blobs each visit
        n_blobs = 6 + idx * 4
        for _ in range(n_blobs):
            cx = int(rng.integers(40, IMAGE_SIZE - 40))
            cy = int(rng.integers(40, IMAGE_SIZE - 40))
            r  = int(rng.integers(8, 28))
            cv2.circle(mask, (cx, cy), r, 1, -1)

        # Synthetic retinal image: dark reddish background
        img = rng.integers(30, 120, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        img[..., 0] = np.clip(img[..., 0] + 60, 0, 255)   # red channel boost

        demo_visits.append({'date': date_str, 'image': img, 'mask': mask.copy()})

    report, figures = analyze_progression(demo_visits, fovea_center=(256, 256))

    print('\n--- Report Summary ---')
    print(f"Trend            : {report['trend']}")
    print(f"Coverage/visit   : {[f'{c:.2f}%' for c in report['coverage_over_time']]}")
    print(f"New exudate fracs: {[f'{f:.2f}%' for f in report['new_exudate_fractions']]}")
    print(f"Macular threats  : {report['macular_threat_per_visit']}")
    print(f"Figures saved    : {list(figures.keys())}")
    print('\n[complete] progression.py demo done.')
