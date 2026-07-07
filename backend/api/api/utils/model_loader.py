import os
import tempfile
from typing import Optional

import numpy as np
import joblib
import onnxruntime as ort

from .supabase_client import download_model

_MODEL_CACHE: dict[str, tuple] = {}
_BEST_MODEL_NAME = "best_model"
_FEATURE_NAMES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

_LOCAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")


def load_best_model() -> tuple:
    return load_model(_BEST_MODEL_NAME)


def load_model(model_name: str) -> tuple:
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    session = None
    scaler = None

    onnx_path = os.path.join(_LOCAL_DIR, "best_model.onnx")
    scaler_path = os.path.join(_LOCAL_DIR, "scaler.pkl")

    if os.path.exists(onnx_path):
        session = ort.InferenceSession(onnx_path)
    else:
        model_bytes = download_model("best_model.onnx")
        if model_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                tmp.write(model_bytes)
                tmp_path = tmp.name
            try:
                session = ort.InferenceSession(tmp_path)
            finally:
                os.unlink(tmp_path)

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler_bytes = download_model("scaler.pkl")
        if scaler_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                tmp.write(scaler_bytes)
                tmp_path = tmp.name
            try:
                scaler = joblib.load(tmp_path)
            finally:
                os.unlink(tmp_path)

    if session is None:
        raise FileNotFoundError(f"Model '{model_name}' not found locally or in Supabase Storage")

    _MODEL_CACHE[model_name] = (session, scaler)
    return session, scaler


def preprocess(features: list[float], scaler) -> np.ndarray:
    arr = np.array(features, dtype=np.float32).reshape(1, -1)
    if scaler is not None:
        arr = scaler.transform(arr)
    return arr


def predict(model, features: list[float], scaler=None) -> float:
    processed = preprocess(features, scaler)
    input_name = model.get_inputs()[0].name
    pred = model.run(None, {input_name: processed.astype(np.float32)})
    return float(pred[0][0][0])


def get_model_info(model_name: Optional[str] = None) -> dict:
    models_info = {
        "MLP Baseline": {
            "name": "MLP Baseline",
            "type": "Classic",
            "params": {"layers": [64, 32]},
            "metrics": {"rmse": 0.5406, "mae": 0.4060, "r2": 0.7770},
        },
        "Deep MLP": {
            "name": "Deep MLP",
            "type": "Classic",
            "params": {"layers": [128, 64, 32, 16], "dropout": 0.3},
            "metrics": {"rmse": 0.7130, "mae": 0.5321, "r2": 0.6121},
        },
        "Residual MLP": {
            "name": "Residual MLP",
            "type": "Classic",
            "params": {"blocks": 3, "units": 128},
            "metrics": {"rmse": 0.6093, "mae": 0.4550, "r2": 0.7167},
        },
        "CNN-LSTM": {
            "name": "CNN-LSTM",
            "type": "Hybrid",
            "params": {"conv_filters": 64, "lstm_units": 32},
            "metrics": {"rmse": 0.5230, "mae": 0.3931, "r2": 0.7913},
        },
        "Autoencoder-MLP": {
            "name": "Autoencoder-MLP",
            "type": "Hybrid",
            "params": {"encoding_dim": 4},
            "metrics": {"rmse": 0.5751, "mae": 0.4374, "r2": 0.7476},
        },
    }
    if model_name:
        return models_info.get(model_name, {})
    return models_info


def clear_cache() -> None:
    _MODEL_CACHE.clear()
