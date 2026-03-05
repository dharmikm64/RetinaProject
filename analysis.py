'''
IDRiD Retinal Exudate Analysis

Generates 4 diagnostic charts from the IDRiD dataset.
Imports processed data from data_pipline.py.
'''

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import tifffile

from data_pipline import (
    PROJECT_DIR,
    GRADING_TRAIN_IMGS,
    GRADING_TEST_IMGS,
    SEG_MASKS_DIR,
    LESION_TYPES,
    build_master_dataset,
)


# -- Constants ---------------------------------------------------

DISPLAY_SIZE = (512, 512)
GRADE_ORDER  = [0, 1, 2, 3, 4]
GRADE_COLORS = ['#4CAF50', '#8BC34A', '#FFC107', '#FF5722', '#F44336']
GRADE_LABELS = {
    0: 'No DR',
    1: 'Mild DR',
    2: 'Moderate DR',
    3: 'Severe DR',
    4: 'Proliferative DR',
}


# -- Data loader -------------------------------------------------

def load_dataset():
    '''Load the IDRiD dataset. Returns (df, images, masks).'''
    csv_path = PROJECT_DIR / 'processed_dataset.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f'[loaded] master CSV ({len(df)} rows)')
    else:
        df = build_master_dataset()

    images = {}
    for _, row in df.iterrows():
        folder   = GRADING_TRAIN_IMGS if row['split'] == 'train' else GRADING_TEST_IMGS
        img_id   = row['image_id']
        img_path = folder / f'{img_id}.jpg'
        if img_path.exists():
            img = Image.open(img_path).convert('RGB').resize(DISPLAY_SIZE, Image.LANCZOS)
            images[img_id] = np.array(img)
    print(f'[loaded] {len(images)} retinal images')

    masks = {}
    folder_name, suffix = LESION_TYPES['hard_exudates']
    glob_pat = f'*{suffix}.tif'
    for subdir in ['a. Training Set', 'b. Testing Set']:
        mask_dir = SEG_MASKS_DIR / subdir / folder_name
        if not mask_dir.exists():
            continue
        for mask_file in sorted(mask_dir.glob(glob_pat)):
            seg_id      = mask_file.stem.replace(suffix, '')
            prefix, num = seg_id.rsplit('_', 1)
            grading_id  = f'{prefix}_{int(num):03d}'
            raw_mask    = tifffile.imread(str(mask_file))
            mask_img    = Image.fromarray((raw_mask > 0).astype(np.uint8) * 255)
            mask_resized = np.array(mask_img.resize(DISPLAY_SIZE, Image.NEAREST))
            masks[grading_id] = (mask_resized > 127).astype(np.uint8)
    print(f'[loaded] {len(masks)} hard exudate masks')

    return df, images, masks

# -- Chart helpers -----------------------------------------------

def _style_ax(ax):
    ax.spines[['top', 'right']].set_visible(False)


# -- Chart 1: Grade distribution ---------------------------------

def _chart_grade_distribution(df):
    '''Bar chart: image count per DR grade (all 516 images).'''
    grade_counts = (
        df.groupby('retinopathy_grade').size().reindex(GRADE_ORDER, fill_value=0)
    )
    tick_labels = [f'Grade {g}\n{GRADE_LABELS[g]}' for g in GRADE_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        GRADE_ORDER, grade_counts.values,
        color=GRADE_COLORS, edgecolor='white', linewidth=0.8,
    )
    for bar, count in zip(bars, grade_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(count), ha='center', va='bottom', fontsize=11, fontweight='bold',
        )
    ax.set_xticks(GRADE_ORDER)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_xlabel('Diabetic Retinopathy Grade', labelpad=8)
    ax.set_ylabel('Image Count')
    ax.set_title(f'Grade Distribution \u2014 IDRiD Dataset (n={len(df)})', fontsize=13)
    ax.set_ylim(0, grade_counts.max() * 1.18)
    _style_ax(ax)
    fig.tight_layout()
    return fig


# -- Chart 2: Mean exudate coverage vs grade ---------------------

def _chart_exudate_vs_grade(df):
    '''Bar chart: mean hard exudate area % per grade with std error bars (seg subset).'''
    seg   = df[df['has_masks'] & df['hard_exudates_area_pct'].notna()].copy()
    stats = (
        seg.groupby('retinopathy_grade')['hard_exudates_area_pct']
        .agg(['mean', 'std', 'count'])
        .reindex(GRADE_ORDER, fill_value=0)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        stats.index, stats['mean'],
        yerr=stats['std'],
        color=GRADE_COLORS, edgecolor='white', linewidth=0.8,
        capsize=5, error_kw={'elinewidth': 1.5, 'ecolor': '#555'},
    )
    for grade, row_s in stats.iterrows():
        cnt = row_s['count']
        if cnt > 0:
            y_top = row_s['mean'] + row_s['std'] + 0.08
            ax.text(grade, y_top, f'n={int(cnt)}',
                    ha='center', va='bottom', fontsize=9, color='#333')
    ax.set_xticks(GRADE_ORDER)
    ax.set_xticklabels([f'Grade {g}' for g in GRADE_ORDER])
    ax.set_xlabel('Diabetic Retinopathy Grade', labelpad=8)
    ax.set_ylabel('Mean Hard Exudate Coverage (%)')
    ax.set_title('Hard Exudate Coverage vs DR Grade\n(segmentation subset only)', fontsize=13)
    _style_ax(ax)
    fig.tight_layout()
    return fig

# -- Chart 3: Sample overlay grid --------------------------------

def _chart_sample_overlays(df, images, masks):
    '''For each DR grade, show one retinal image with yellow exudate overlay.'''
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, grade in zip(axes, GRADE_ORDER):
        candidates = df[
            (df['retinopathy_grade'] == grade) &
            df['image_id'].isin(images)
        ]
        if candidates.empty:
            ax.set_visible(False)
            continue
        img_id   = candidates.iloc[0]['image_id']
        img      = images[img_id].copy().astype(np.float32) / 255.0
        has_mask = img_id in masks
        if has_mask:
            yellow = np.zeros_like(img)
            yellow[..., 0] = 1.0
            yellow[..., 1] = 1.0
            alpha  = masks[img_id][..., np.newaxis].astype(np.float32) * 0.30
            img    = img * (1 - alpha) + yellow * alpha
        ax.imshow(np.clip(img, 0, 1))
        suffix = ' (+ mask)' if has_mask else ''
        ax.set_title(f'Grade {grade}: {GRADE_LABELS[grade]}{suffix}', fontsize=9)
        ax.axis('off')
    fig.suptitle(
        'Sample Retinal Images per DR Grade  \u2014  yellow = hard exudate overlay',
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    return fig

# -- Chart 4: Exudate presence rate per grade --------------------

def _chart_exudate_presence_rate(df):
    '''Bar chart: pct of images with masks that have any hard exudates per grade.'''
    seg      = df[df['has_masks'] & df['has_hard_exudates'].notna()].copy()
    presence = (
        seg.groupby('retinopathy_grade')['has_hard_exudates']
        .agg(lambda x: x.sum() / len(x) * 100)
        .reindex(GRADE_ORDER, fill_value=0)
    )
    counts = (
        seg.groupby('retinopathy_grade').size()
        .reindex(GRADE_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        GRADE_ORDER, presence.values,
        color=GRADE_COLORS, edgecolor='white', linewidth=0.8,
    )
    for bar, pct, grade in zip(bars, presence.values, GRADE_ORDER):
        n     = counts[grade]
        label = f'{pct:.0f}%' if n > 0 else '\u2014'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            label, ha='center', va='bottom', fontsize=10, fontweight='bold',
        )
        if n > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, -5,
                f'n={n}', ha='center', va='top', fontsize=8, color='#666',
            )
    ax.set_xticks(GRADE_ORDER)
    ax.set_xticklabels([f'Grade {g}' for g in GRADE_ORDER])
    ax.set_ylim(-8, 115)
    ax.set_xlabel('Diabetic Retinopathy Grade', labelpad=8)
    ax.set_ylabel('% of Images with Any Hard Exudates')
    ax.set_title('Hard Exudate Presence Rate by DR Grade\n(segmentation subset only)', fontsize=13)
    _style_ax(ax)
    fig.tight_layout()
    return fig

# -- Main analysis function --------------------------------------

def run_analysis(df, images, masks):
    '''
    Generate all 4 analysis charts. Returns dict {chart_name: Figure}.

    Parameters
    ----------
    df     : master DataFrame from load_dataset()
    images : dict {image_id: np.array (512,512,3)}
    masks  : dict {image_id: np.array (512,512) binary}
    '''
    tasks = {
        'grade_distribution'   : (_chart_grade_distribution,   (df,)),
        'exudate_vs_grade'     : (_chart_exudate_vs_grade,      (df,)),
        'sample_overlays'      : (_chart_sample_overlays,       (df, images, masks)),
        'exudate_presence_rate': (_chart_exudate_presence_rate, (df,)),
    }
    figures = {}
    for name, (fn, args) in tasks.items():
        fig = fn(*args)
        figures[name] = fig
        out_path = PROJECT_DIR / f'{name}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'[saved] {out_path.name}')
    return figures


# -- Entry point -------------------------------------------------

if __name__ == '__main__':
    df, images, masks = load_dataset()
    run_analysis(df, images, masks)
    print('[complete] Analysis done.')