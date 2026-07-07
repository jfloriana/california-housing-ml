from typing import Optional

from fastapi import APIRouter, HTTPException

from ..utils.model_loader import get_model_info
from ..utils.supabase_client import get_training_metrics

router = APIRouter()


@router.get("/api/models")
async def list_models():
    models_info = get_model_info()
    result = []
    for key, info in models_info.items():
        result.append({
            "id": key,
            "name": info["name"],
            "type": info["type"],
            "params": info["params"],
            "metrics": info["metrics"],
        })
    return {"models": result, "total": len(result)}


@router.get("/api/models/comparison")
async def compare_models():
    models_info = get_model_info()
    comparison = []
    for key, info in models_info.items():
        comparison.append({
            "id": key,
            "name": info["name"],
            "type": info["type"],
            "rmse": info["metrics"].get("rmse"),
            "mae": info["metrics"].get("mae"),
            "r2": info["metrics"].get("r2"),
            "params": info["params"],
        })
    return {"comparison": comparison, "total": len(comparison)}


@router.get("/api/models/{model_name:path}")
async def get_model(model_name: str):
    models_info = get_model_info()
    if model_name not in models_info:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    info = models_info[model_name]

    db_metrics = []
    try:
        db_metrics = get_training_metrics(model_name)
    except Exception:
        pass

    return {
        "id": model_name,
        "name": info["name"],
        "type": info["type"],
        "params": info["params"],
        "metrics": info["metrics"],
        "db_metrics": db_metrics,
    }
