"""
model_loader.py
--------------
Model and scaler loading utilities for AU detection.
"""
import os
import joblib
from typing import Dict, Tuple, Any

all_models: Dict[str, Dict[str, Any]] = {}
all_scalers: Dict[str, Dict[str, Any]] = {}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
AUS_STR = [f"AU{n}" for n in AUS]
FOLDER_NAMES = [
    "results_landmark_binary",
    "results_hog_binary",
    "results_landmark_multiclass",
    "results_hog_multiclass"
]

def load_models_and_scalers(folder_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load all models and scalers for a given method folder (e.g., HOG binary).
    Args:
        folder_path: Absolute path to model folder
    Returns:
        Tuple of dictionaries (models, scalers)
    """
    models: Dict[str, Any] = {}
    scalers: Dict[str, Any] = {}
    for au in AUS_STR:
        # model
        model_path = os.path.join(folder_path, f"{au}_model.pkl")
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found for {au} in {folder_path}")
            continue
        try:
            models[au] = joblib.load(model_path)
        except Exception as e:
            print(f"❌ Failed to load model for {au} in {folder_path}: {e}")
            continue
        # scaler
        scaler_path = os.path.join(folder_path, f"{au}_scaler.pkl")
        if os.path.exists(scaler_path):
            try:
                scalers[au] = joblib.load(scaler_path)
            except Exception as e:
                print(f"❌ Failed to load scaler for {au} in {folder_path}: {e}")
        else:
            print(f"⚠️ Scaler not found for {au} in {folder_path}")
    return models, scalers

# Load all model/scaler sets into dictionaries
for folder_name in FOLDER_NAMES:
    folder_path = os.path.join(SCRIPT_DIR, folder_name)
    if not os.path.isdir(folder_path):
        print(f"⚠️ Model folder not found: {folder_path}")
        all_models[folder_name] = {}
        all_scalers[folder_name] = {}
        continue
    models, scalers = load_models_and_scalers(folder_path)
    all_models[folder_name] = models
    all_scalers[folder_name] = scalers
