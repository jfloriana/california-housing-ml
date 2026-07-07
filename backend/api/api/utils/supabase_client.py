import os
import io
import json
from typing import Optional
from functools import lru_cache

import numpy as np

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

_SUPABASE_URL: Optional[str] = None
_SUPABASE_ANON_KEY: Optional[str] = None
_SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None
_CLIENT: Optional[Client] = None
_SERVICE_CLIENT: Optional[Client] = None


def _get_env_or_fallback(key: str, fallback: str) -> str:
    return os.environ.get(key, fallback)


SUPABASE_URL = _get_env_or_fallback("SUPABASE_URL", "")
SUPABASE_ANON_KEY = _get_env_or_fallback("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = _get_env_or_fallback("SUPABASE_SERVICE_ROLE_KEY", "")


def get_client() -> Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return _CLIENT


def get_service_client() -> Client:
    global _SERVICE_CLIENT
    if _SERVICE_CLIENT is None:
        _SERVICE_CLIENT = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return _SERVICE_CLIENT


def upload_model(
    model_name: str,
    model_bytes: bytes,
    bucket: str = "models",
) -> str:
    client = get_service_client()
    file_path = f"models/{model_name}/{model_name}.h5"
    client.storage.from_(bucket).upload(
        file_path,
        model_bytes,
        {"content-type": "application/octet-stream", "upsert": "true"},
    )
    public_url = client.storage.from_(bucket).get_public_url(file_path)
    return public_url


def download_model(model_name: str, bucket: str = "models") -> Optional[bytes]:
    client = get_service_client()
    file_path = f"models/{model_name}/{model_name}.h5"
    try:
        res = client.storage.from_(bucket).download(file_path)
        return res
    except Exception:
        return None


def upload_report(
    report_name: str,
    report_bytes: bytes,
    content_type: str,
    bucket: str = "reports",
) -> str:
    client = get_service_client()
    file_path = f"reports/{report_name}"
    client.storage.from_(bucket).upload(
        file_path,
        report_bytes,
        {"content-type": content_type, "upsert": "true"},
    )
    public_url = client.storage.from_(bucket).get_public_url(file_path)
    return public_url


def download_report(report_name: str, bucket: str = "reports") -> Optional[bytes]:
    client = get_service_client()
    file_path = f"reports/{report_name}"
    try:
        res = client.storage.from_(bucket).download(file_path)
        return res
    except Exception:
        return None


def store_training_metrics(
    model_name: str,
    model_type: str,
    metrics: dict,
) -> dict:
    client = get_service_client()
    data = {
        "model_name": model_name,
        "model_type": model_type,
        "mse": metrics.get("mse"),
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "r2": metrics.get("r2"),
        "training_time_sec": metrics.get("training_time_sec"),
        "num_params": metrics.get("num_params"),
    }
    result = client.table("training_metrics").insert(data).execute()
    return result.data[0] if result.data else {}


def get_training_metrics(model_name: Optional[str] = None) -> list[dict]:
    client = get_client()
    query = client.table("training_metrics").select("*")
    if model_name:
        query = query.eq("model_name", model_name)
    result = query.execute()
    return result.data if result.data else []


def store_cv_results(model_name: str, n_folds: int, scores: list[float]) -> dict:
    client = get_service_client()
    mean_score = float(np.mean(scores)) if scores else 0.0
    std_score = float(np.std(scores)) if scores else 0.0
    data = {
        "model_name": model_name,
        "n_folds": n_folds,
        "fold_scores": json.dumps(scores),
        "mean_score": mean_score,
        "std_score": std_score,
    }
    result = client.table("cross_validation_results").insert(data).execute()
    return result.data[0] if result.data else {}


def get_cv_results(model_name: Optional[str] = None) -> list[dict]:
    client = get_client()
    query = client.table("cross_validation_results").select("*")
    if model_name:
        query = query.eq("model_name", model_name)
    result = query.execute()
    return result.data if result.data else []


def store_hyperparameter_results(
    model_name: str,
    params: dict,
    best_mse: float,
    best_r2: float,
) -> dict:
    client = get_service_client()
    data = {
        "model_name": model_name,
        "params": json.dumps(params),
        "best_mse": best_mse,
        "best_r2": best_r2,
    }
    result = client.table("hyperparameter_results").insert(data).execute()
    return result.data[0] if result.data else {}


def get_hyperparameter_results(model_name: Optional[str] = None) -> list[dict]:
    client = get_client()
    query = client.table("hyperparameter_results").select("*")
    if model_name:
        query = query.eq("model_name", model_name)
    result = query.execute()
    return result.data if result.data else []


def log_prediction(
    input_data: dict,
    prediction: float,
    model_used: str,
    user_id: Optional[str] = None,
) -> dict:
    client = get_service_client()
    data = {
        "user_id": user_id,
        "input_data": json.dumps(input_data),
        "prediction": prediction,
        "model_used": model_used,
    }
    result = client.table("predictions_log").insert(data).execute()
    return result.data[0] if result.data else {}
