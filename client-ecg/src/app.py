import os
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk

from fastapi import UploadFile, File, Form, APIRouter
from fastapi.responses import JSONResponse

# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(tags=["ECG Medical API"])

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model" / "model_wide_deep_pure_FIXED.pth"

UPLOAD_DIR.mkdir(exist_ok=True)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class WideDeepModel(nn.Module):

    def __init__(self, num_wide_features=32, num_classes=5):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=14, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=14, padding=7),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=10, padding=5),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=10, padding=5),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=10, padding=5),
            nn.BatchNorm1d(512), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(512, 512, kernel_size=10, padding=5),
            nn.BatchNorm1d(512), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.cnn_to_transformer = nn.Linear(512, 256)

        self.deep_fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64)
        )

        self.wide_fc = nn.Sequential(
            nn.Linear(num_wide_features, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32)
        )

        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, signal, wide_features):
        x = self.conv_layers(signal).squeeze(-1)
        x = self.cnn_to_transformer(x).unsqueeze(1)
        x = self.transformer(x).mean(dim=1)
        deep_out = self.deep_fc(x)
        wide_out = self.wide_fc(wide_features)
        combined = torch.cat([deep_out, wide_out], dim=1)
        return self.fusion(combined)

# =============================================================================
# LOAD MODEL
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WideDeepModel(num_wide_features=32, num_classes=5)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_DESCRIPTIONS = {
    'NORM': 'ECG Normal',
    'MI': 'Infarctus du Myocarde',
    'STTC': 'Changements ST/T',
    'CD': 'Troubles de Conduction',
    'HYP': 'Hypertrophie'
}

# =============================================================================
# ECG PROCESSING
# =============================================================================

def load_ecg_from_file(record_path):
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal.T
    return signal.astype(np.float32)


def clean_ecg_signal(signal, sampling_rate=100):
    cleaned = np.zeros_like(signal)

    for lead_idx in range(12):
        lead = signal[lead_idx, :]

        if np.isnan(lead).any():
            lead = pd.Series(lead).interpolate().fillna(0).values

        try:
            lead_clean = nk.ecg_clean(lead, sampling_rate=sampling_rate, method='neurokit')
        except:
            lead_clean = lead

        mean_val = np.mean(lead_clean)
        std_val = np.std(lead_clean)
        if std_val > 1e-6:
            lead_clean = (lead_clean - mean_val) / std_val
        else:
            lead_clean = lead_clean - mean_val

        cleaned[lead_idx, :] = lead_clean

    return cleaned.astype(np.float32)


def extract_wide_features_from_signal(signal):
    features = []

    lead_II = signal[1, :]

    try:
        signals_df, info = nk.ecg_process(lead_II, sampling_rate=100)
        hr = signals_df['ECG_Rate'].mean()
        if np.isnan(hr):
            hr = 70
        features.append(hr / 100)
        rr_interval = 60 / hr if hr > 0 else 0.85
        features.append(rr_interval)
    except:
        features.append(0.70)
        features.append(0.85)

    for lead_idx in [0, 1, 5, 6, 7, 8]:
        lead = signal[lead_idx, :]
        features.append(np.mean(lead))
        features.append(np.std(lead))
        features.append(np.max(lead) - np.min(lead))

    while len(features) < 32:
        features.append(0.0)

    return np.array(features[:32], dtype=np.float32)


def predict_ecg(signal, wide_features):
    if signal.ndim == 2:
        signal = signal[np.newaxis, ...]
    if wide_features.ndim == 1:
        wide_features = wide_features[np.newaxis, ...]

    signal_t = torch.from_numpy(signal).float().to(DEVICE)
    wide_t = torch.from_numpy(wide_features).float().to(DEVICE)

    with torch.no_grad():
        logits = model(signal_t, wide_t)
        probs = torch.sigmoid(logits).cpu().numpy()

    return probs[0]

# =============================================================================
# API ENDPOINT
# =============================================================================

@router.post("/predict")
async def predict(
    first_name: str = Form(...),
    last_name: str = Form(...),
    sex: str = Form(...),
    age: int = Form(...),
    weight: float = Form(...),
    height: float = Form(...),
    symptoms: str = Form(None),
    dat_file: UploadFile = File(...),
    hea_file: UploadFile = File(...)
):

    base_name = dat_file.filename.replace(".dat", "")
    record_dir = UPLOAD_DIR / base_name
    record_dir.mkdir(exist_ok=True)

    dat_path = record_dir / dat_file.filename
    hea_path = record_dir / hea_file.filename

    with open(dat_path, "wb") as f:
        shutil.copyfileobj(dat_file.file, f)

    with open(hea_path, "wb") as f:
        shutil.copyfileobj(hea_file.file, f)

    record_path = str(record_dir / base_name)

    try:
        signal_raw = load_ecg_from_file(record_path)
        signal_clean = clean_ecg_signal(signal_raw)
        wide_features = extract_wide_features_from_signal(signal_clean)

        probs = predict_ecg(signal_clean, wide_features)

        results = []
        for i, p in enumerate(probs):
            results.append({
                "class": CLASS_NAMES[i],
                "description": CLASS_DESCRIPTIONS[CLASS_NAMES[i]],
                "probability": float(p)
            })

        detected = [CLASS_NAMES[i] for i, p in enumerate(probs) if p >= 0.5]

        return {
            "patient": {
                "first_name": first_name,
                "last_name": last_name,
                "sex": sex,
                "age": age,
                "weight": weight,
                "height": height,
                "symptoms": symptoms
            },
            "results": results,
            "detected_pathologies": detected
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
