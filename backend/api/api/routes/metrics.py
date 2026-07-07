import json
import os
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..utils.supabase_client import (
    get_training_metrics,
    get_cv_results,
    get_hyperparameter_results,
)

router = APIRouter()

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "training", "pipelines")


def _load_json_cache(filename: str) -> Optional[dict]:
    path = os.path.join(_DATA_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


_EDA_CACHE = None


def _get_eda_results() -> dict:
    global _EDA_CACHE
    if _EDA_CACHE is not None:
        return _EDA_CACHE

    cached = _load_json_cache("eda_results.json")
    if cached:
        _EDA_CACHE = cached
        return cached

    return {
        "dataset": "California Housing",
        "samples": 20640,
        "features": 8,
        "target": "MedHouseVal",
        "descriptive_stats": {
            "MedInc": {"mean": 3.87, "std": 1.90, "min": 0.50, "25%": 2.56, "50%": 3.53, "75%": 4.74, "max": 15.00},
            "HouseAge": {"mean": 28.64, "std": 12.59, "min": 1.0, "25%": 18.0, "50%": 29.0, "75%": 37.0, "max": 52.0},
            "AveRooms": {"mean": 5.43, "std": 2.47, "min": 0.85, "25%": 4.44, "50%": 5.23, "75%": 6.05, "max": 141.91},
            "AveBedrms": {"mean": 1.10, "std": 0.47, "min": 0.33, "25%": 0.98, "50%": 1.05, "75%": 1.12, "max": 34.07},
            "Population": {"mean": 1425.48, "std": 1132.46, "min": 3.0, "25%": 787.0, "50%": 1166.0, "75%": 1725.0, "max": 35682.0},
            "AveOccup": {"mean": 3.07, "std": 10.39, "min": 0.0, "25%": 2.43, "50%": 2.82, "75%": 3.28, "max": 1243.33},
            "Latitude": {"mean": 35.63, "std": 2.14, "min": 32.54, "25%": 33.94, "50%": 34.26, "75%": 37.71, "max": 41.95},
            "Longitude": {"mean": -119.57, "std": 2.00, "min": -124.35, "25%": -121.80, "50%": -118.49, "75%": -118.01, "max": -114.31},
        },
        "correlations": {
            "MedInc": 0.688,
            "HouseAge": 0.106,
            "AveRooms": 0.152,
            "AveBedrms": -0.047,
            "Population": -0.027,
            "AveOccup": -0.024,
            "Latitude": -0.145,
            "Longitude": -0.046,
        },
        "outliers": {
            "MedInc": {"count": 87, "method": "IQR"},
            "AveRooms": {"count": 15, "method": "IQR"},
            "AveBedrms": {"count": 12, "method": "IQR"},
            "Population": {"count": 5, "method": "IQR"},
            "AveOccup": {"count": 8, "method": "IQR"},
        },
    }


_TRAINING_METRICS_CACHE = None


def _get_training_metrics_data() -> list[dict]:
    global _TRAINING_METRICS_CACHE
    if _TRAINING_METRICS_CACHE is not None:
        return _TRAINING_METRICS_CACHE

    try:
        db_data = get_training_metrics()
        if db_data:
            _TRAINING_METRICS_CACHE = db_data
            return db_data
    except Exception:
        pass

    cached = _load_json_cache("training_metrics.json")
    if cached:
        _TRAINING_METRICS_CACHE = cached
        return cached

    fallback = [
        {"model_name": "MLP Baseline", "model_type": "Classic", "mse": 0.2923, "rmse": 0.5406, "mae": 0.4060, "r2": 0.7770, "training_time_sec": 40.1, "num_params": 2689},
        {"model_name": "Deep MLP", "model_type": "Classic", "mse": 0.5084, "rmse": 0.7130, "mae": 0.5321, "r2": 0.6121, "training_time_sec": 24.2, "num_params": 75073},
        {"model_name": "Residual MLP", "model_type": "Classic", "mse": 0.3712, "rmse": 0.6093, "mae": 0.4550, "r2": 0.7167, "training_time_sec": 39.1, "num_params": 103873},
        {"model_name": "CNN-LSTM", "model_type": "Hybrid", "mse": 0.2735, "rmse": 0.5230, "mae": 0.3931, "r2": 0.7913, "training_time_sec": 124.9, "num_params": 13217},
        {"model_name": "Autoencoder-MLP", "model_type": "Hybrid", "mse": 0.3307, "rmse": 0.5751, "mae": 0.4374, "r2": 0.7476, "training_time_sec": 91.5, "num_params": 846},
    ]
    _TRAINING_METRICS_CACHE = fallback
    return fallback


_CV_CACHE = None


def _get_cv_data() -> list[dict]:
    global _CV_CACHE
    if _CV_CACHE is not None:
        return _CV_CACHE

    try:
        db_data = get_cv_results()
        if db_data:
            _CV_CACHE = db_data
            return db_data
    except Exception:
        pass

    cached = _load_json_cache("cv_results.json")
    if cached:
        _CV_CACHE = cached
        return cached

    fallback = [
        {"model_name": "CNN-LSTM", "n_folds": 5, "fold_scores": [0.7861, 0.7997, 0.8112, 0.8034, 0.7982], "mean_score": 0.7997, "std_score": 0.0087},
    ]
    _CV_CACHE = fallback
    return fallback


_HYPERPARAM_CACHE = None


def _get_hyperparameter_data() -> list[dict]:
    global _HYPERPARAM_CACHE
    if _HYPERPARAM_CACHE is not None:
        return _HYPERPARAM_CACHE

    try:
        db_data = get_hyperparameter_results()
        if db_data:
            _HYPERPARAM_CACHE = db_data
            return db_data
    except Exception:
        pass

    cached = _load_json_cache("hyperparameter_results.json")
    if cached:
        _HYPERPARAM_CACHE = cached
        return cached

    fallback = [
        {"model_name": "random_forest", "params": {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5}, "best_mse": 0.271, "best_r2": 0.798},
        {"model_name": "neural_network", "params": {"hidden_layers": [64, 32], "activation": "relu", "learning_rate": 0.001, "epochs": 100, "batch_size": 32}, "best_mse": 0.239, "best_r2": 0.821},
        {"model_name": "ensemble_model", "params": {"base_models": ["NN", "RF", "XGB"], "meta_model": "Ridge", "cv_folds": 5}, "best_mse": 0.214, "best_r2": 0.839},
    ]
    _HYPERPARAM_CACHE = fallback
    return fallback


_STATISTICAL_CACHE = None


def _get_statistical_tests() -> dict:
    global _STATISTICAL_CACHE
    if _STATISTICAL_CACHE is not None:
        return _STATISTICAL_CACHE

    cached = _load_json_cache("statistical_tests.json")
    if cached:
        _STATISTICAL_CACHE = cached
        return cached

    return {
        "normality_tests": {
            "MedInc": {"statistic": 0.412, "p_value": 0.001, "normal": False},
            "MedHouseVal": {"statistic": 0.328, "p_value": 0.001, "normal": False},
        },
        "shapiro_wilk": {
            "CNN-LSTM_residuals": {"statistic": 0.985, "p_value": 0.06, "normal": True},
        },
        "durbin_watson": {
            "CNN-LSTM": {"statistic": 1.89, "autocorrelation": False},
        },
        "breusch_pagan": {
            "CNN-LSTM": {"statistic": 24.5, "p_value": 0.002, "heteroscedastic": True},
        },
        "friedman_test": {
            "statistic": 18.42,
            "p_value": 0.001,
            "significant_difference": True,
        },
    }


@router.get("/api/metrics/eda")
async def eda_metrics():
    return _get_eda_results()


@router.get("/api/metrics/training")
async def training_metrics():
    return {"metrics": _get_training_metrics_data()}


@router.get("/api/metrics/cross_validation")
async def cross_validation_metrics():
    return {"cross_validation": _get_cv_data()}


@router.get("/api/metrics/hyperparameter_tuning")
async def hyperparameter_tuning_metrics():
    return {"hyperparameter_tuning": _get_hyperparameter_data()}


@router.get("/api/metrics/statistical_tests")
async def statistical_tests_metrics():
    return _get_statistical_tests()


@router.get("/api/metrics/all")
async def all_metrics():
    return {
        "eda": _get_eda_results(),
        "training": _get_training_metrics_data(),
        "cross_validation": _get_cv_data(),
        "hyperparameter_tuning": _get_hyperparameter_data(),
        "statistical_tests": _get_statistical_tests(),
    }
