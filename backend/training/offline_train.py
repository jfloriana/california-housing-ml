#!/usr/bin/env python3
"""
Offline Training Pipeline for California Housing
Trains all 5 neural network models, evaluates, selects best model,
and saves everything needed for the API and dashboard.
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.models import (
    build_mlp,
    build_deep_mlp,
    build_residual_mlp,
    build_cnn_lstm,
    build_autoencoder_mlp,
)

from training.pipelines.eda import run_eda
from training.pipelines.train import run_training
from training.pipelines.cross_validation import run_cross_validation
from training.pipelines.hyperparameter_tuning import run_hyperparameter_tuning
from training.pipelines.statistical_tests import run_statistical_tests, run_statistical_tests_on_residuals


def serialize_for_json(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import pandas.api.types as ptypes
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(serialize_for_json(v) for v in obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return serialize_for_json(obj.to_dict(orient="split"))
    elif isinstance(obj, pd.Series):
        return serialize_for_json(obj.to_dict())
    elif hasattr(obj, "__name__"):
        return str(obj.__name__)
    elif hasattr(obj, "name"):
        return str(obj.name)
    return str(obj)


def main():
    print("=" * 60)
    print("CALIFORNIA HOUSING - OFFLINE TRAINING PIPELINE")
    print("=" * 60)
    
    LANG = "es"  # Default, can be changed
    OUTPUT_DIR = Path(__file__).parent / "outputs"
    PLOTS_DIR = OUTPUT_DIR / "plots"
    PIPELINES_DIR = Path(__file__).parent / "pipelines"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    total_start = time.time()
    
    # Step 1: Load data
    print("\n[1/6] Loading dataset...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.columns = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                   "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"]
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    
    print(f"  Dataset loaded: {df.shape[0]} samples, {df.shape[1]-1} features")
    
    # Step 2: Split
    print("\n[2/6] Splitting dataset (70/15/15)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/0.85, random_state=42
    )
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Step 3: Scale
    print("\n[3/6] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, str(OUTPUT_DIR / "scaler.pkl"))
    print(f"  Scaler saved to {OUTPUT_DIR / 'scaler.pkl'}")
    
    # Step 4: EDA
    print("\n[4/6] Running EDA...")
    eda_start = time.time()
    eda_results = run_eda(
        save_plots_dir=str(PLOTS_DIR),
        lang=LANG,
    )
    with open(PIPELINES_DIR / "eda_results.json", "w", encoding="utf-8") as f:
        json.dump(serialize_for_json(eda_results), f, ensure_ascii=False, indent=2)
    print(f"  EDA completed in {time.time()-eda_start:.1f}s")
    print(f"  Plots saved to {PLOTS_DIR}")
    
    # Step 5: Train all models
    print("\n[5/6] Training all 5 models...")
    model_builders = {
        "MLP Baseline": build_mlp,
        "Deep MLP": build_deep_mlp,
        "Residual MLP": build_residual_mlp,
        "CNN-LSTM": build_cnn_lstm,
        "Autoencoder-MLP": build_autoencoder_mlp,
    }
    
    training_results = run_training(
        model_builders=model_builders,
        X_train=X_train_scaled,
        y_train=y_train.values,
        X_val=X_val_scaled,
        y_val=y_val.values,
        X_test=X_test_scaled,
        y_test=y_test.values,
        epochs=100,
        batch_size=32,
        lang=LANG,
        save_dir=str(OUTPUT_DIR),
    )
    
    # Save training metrics
    with open(PIPELINES_DIR / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(serialize_for_json(training_results.get("comparison_table", [])), f, ensure_ascii=False, indent=2)
    
    # Also save full results
    with open(PIPELINES_DIR / "training_full_results.json", "w", encoding="utf-8") as f:
        json.dump(serialize_for_json(training_results), f, ensure_ascii=False, indent=2)
    
    print(f"  Training completed")
    print(f"  Best model: {training_results.get('best_model_name', 'N/A')}")
    print(f"  Best R²: {training_results.get('comparison_table', [{}])[0].get('r2', 'N/A') if training_results.get('comparison_table') else 'N/A'}")
    
    # Step 6: Cross-validation on best model
    print("\n[6/6] Running additional analyses...")
    
    # Cross-validation
    best_model_name_cv = training_results.get("best_model_name", "MLP Baseline")
    best_builder = model_builders.get(best_model_name_cv, build_mlp)
    
    cv_results = run_cross_validation(
        model_builder=best_builder,
        X=np.vstack([X_train_scaled, X_val_scaled]),
        y=np.concatenate([y_train.values, y_val.values]),
        n_folds=5,
        epochs=50,
        batch_size=32,
        lang=LANG,
        save_dir=str(OUTPUT_DIR),
    )
    with open(PIPELINES_DIR / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(serialize_for_json(cv_results), f, ensure_ascii=False, indent=2)
    print(f"  Cross-validation completed: R² = {cv_results.get('mean_r2', 'N/A'):.4f} ± {cv_results.get('std_r2', 'N/A'):.4f}")
    
    # Hyperparameter tuning (on best model - simplified, only 10 trials)
    # Collect all predictions for statistical tests
    all_predictions = {}
    individual_results = training_results.get("individual_results", {})
    for model_name, result in individual_results.items():
        if "predictions" in result:
            all_predictions[model_name] = np.array(result["predictions"])
    
    # Statistical tests
    if all_predictions:
        stats_results = run_statistical_tests(
            models_results=all_predictions,
            y_true=y_test.values,
            lang=LANG,
            save_dir=str(OUTPUT_DIR),
        )
        with open(PIPELINES_DIR / "statistical_tests.json", "w", encoding="utf-8") as f:
            json.dump(serialize_for_json(stats_results), f, ensure_ascii=False, indent=2)
        print(f"  Statistical tests completed")
    
    # Final summary
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETED IN {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Best model: {OUTPUT_DIR / 'best_model.h5'}")
    print(f"  Scaler: {OUTPUT_DIR / 'scaler.pkl'}")
    print(f"  EDA results: {PIPELINES_DIR / 'eda_results.json'}")
    print(f"  Training metrics: {PIPELINES_DIR / 'training_metrics.json'}")
    print(f"  CV results: {PIPELINES_DIR / 'cv_results.json'}")
    print(f"  Statistical tests: {PIPELINES_DIR / 'statistical_tests.json'}")
    print(f"  Plots: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
