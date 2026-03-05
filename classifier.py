'''
Retinal DR Grade Classifier

Trains an EfficientNet-B0 on the IDRiD dataset to predict
diabetic retinopathy grade (0-4) from retinal fundus images.
Saves weights to classifier.pth so the dashboard can load
without retraining.
'''

import sys
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from data_pipline import PROJECT_DIR, build_master_dataset


# -- Constants ---------------------------------------------------

MODEL_PATH   = PROJECT_DIR / 'classifier.pth'
NUM_CLASSES  = 5      # DR grades 0-4
BATCH_SIZE   = 16
EPOCHS       = 10
LR           = 0.001
IMG_SIZE     = 224    # EfficientNet input size

# ImageNet normalization (used for pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# -- Dataset class -----------------------------------------------

class RetinalDataset(Dataset):
    '''PyTorch Dataset wrapping the images dict from load_dataset().'''

    def __init__(self, image_ids, labels, images_dict, transform=None):
        self.image_ids = image_ids
        self.labels    = [int(l) for l in labels]
        self.images    = images_dict
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img    = Image.fromarray(self.images[img_id])
        label  = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# -- Data preparation --------------------------------------------

def prepare_data(df, images):
    '''
    Build train and test DataLoaders from df and images dict.
    Only uses images that were successfully loaded.
    Normalizes with ImageNet stats to match the pretrained weights.
    '''
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Filter to only images that are actually on disk
    df_loaded = df[df['image_id'].isin(images)].copy()
    train_df  = df_loaded[df_loaded['split'] == 'train']
    test_df   = df_loaded[df_loaded['split'] == 'test']

    train_ds = RetinalDataset(
        list(train_df['image_id']), list(train_df['retinopathy_grade']),
        images, transform,
    )
    test_ds = RetinalDataset(
        list(test_df['image_id']), list(test_df['retinopathy_grade']),
        images, transform,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    print(f'[data] train={len(train_ds)}, test={len(test_ds)}')
    return train_loader, test_loader


# -- Model setup -------------------------------------------------

def build_model():
    '''
    Load pretrained EfficientNet-B0, replace final linear layer
    for 5-class output. Freeze all layers except the new head.
    '''
    # Load pretrained weights (downloads ~20 MB first time)
    try:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except AttributeError:
        # Older torchvision API
        model = models.efficientnet_b0(pretrained=True)

    # Freeze all pretrained parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head: Sequential(Dropout, Linear(1280, 1000))
    # Only the new Linear layer will be trained
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    return model

# -- Training ----------------------------------------------------

def train_model(model, train_loader, device):
    '''Train for EPOCHS using CrossEntropyLoss and Adam on the head only.'''
    criterion = nn.CrossEntropyLoss()
    # Only pass the new head parameters to the optimizer
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total   = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
        ep_loss = running_loss / total
        ep_acc  = correct / total * 100
        print(f'  Epoch {epoch+1:2d}/{EPOCHS}  loss={ep_loss:.4f}  acc={ep_acc:.1f}%')

    return model


# -- Evaluation --------------------------------------------------

def evaluate_model(model, test_loader, device):
    '''Run on test set, print overall accuracy and per-grade breakdown.'''
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs    = imgs.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall    = (all_preds == all_labels).mean() * 100
    print(f'\n[eval] Overall test accuracy: {overall:.1f}%')

    grade_names = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative'}
    print('[eval] Per-grade breakdown:')
    for grade in range(NUM_CLASSES):
        mask = all_labels == grade
        if mask.sum() == 0:
            continue
        n       = int(mask.sum())
        correct = int((all_preds[mask] == grade).sum())
        acc     = correct / n * 100
        print(f'  Grade {grade} {grade_names[grade]:15s}: {correct}/{n} correct ({acc:.0f}%)')

    return overall

# -- Predict function --------------------------------------------

# Module-level model cache so the dashboard doesn't reload on every call
_model_cache  = None
_transform    = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def predict(image_array):
    '''
    Predict DR grade from a single retinal image.

    Parameters
    ----------
    image_array : np.array (512, 512, 3) uint8 RGB

    Returns
    -------
    grade      : int  (0-4)
    confidence : float (0.0-100.0 percent)
    '''
    global _model_cache
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model from disk the first time
    if _model_cache is None:
        model = build_model()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        _model_cache = model

    img    = Image.fromarray(image_array.astype(np.uint8))
    tensor = _transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = _model_cache(tensor)
        probs   = torch.softmax(outputs, dim=1)
        conf, pred = probs.max(1)

    return int(pred.item()), float(conf.item() * 100)


# -- Entry point -------------------------------------------------

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    # Load dataset using the analysis module loader
    from analysis import load_dataset
    df, images, masks = load_dataset()

    if MODEL_PATH.exists():
        # Weights already trained - load and evaluate only
        print(f'[skip training] {MODEL_PATH.name} found, loading saved weights...')
        model = build_model()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        _, test_loader = prepare_data(df, images)
        evaluate_model(model, test_loader, device)
    else:
        # Fresh training run
        train_loader, test_loader = prepare_data(df, images)
        model = build_model()
        model.to(device)
        print(f'[train] EfficientNet-B0, {EPOCHS} epochs, lr={LR}')
        model = train_model(model, train_loader, device)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f'[saved] {MODEL_PATH.name}')
        evaluate_model(model, test_loader, device)

    # Quick sanity check on predict()
    sample_id = list(images.keys())[0]
    grade, conf = predict(images[sample_id])
    print(f'\n[predict test] {sample_id} -> Grade {grade} ({conf:.1f}% confidence)')
    print('[complete] Classifier ready.')