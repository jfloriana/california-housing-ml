import io
import csv
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel, Field

from ..utils.model_loader import load_best_model, load_model, predict, get_model_info
from ..utils.supabase_client import log_prediction

router = APIRouter()


class PredictionInput(BaseModel):
    MedInc: float = Field(..., description="Median income in block group")
    HouseAge: float = Field(..., description="Median house age")
    AveRooms: float = Field(..., description="Average rooms per household")
    AveBedrms: float = Field(..., description="Average bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average household occupancy")
    Latitude: float = Field(..., description="Latitude")
    Longitude: float = Field(..., description="Longitude")


class PredictionOutput(BaseModel):
    prediction: float
    model_used: str
    model_info: dict


class BatchPredictionOutput(BaseModel):
    predictions: list[dict]
    total: int
    model_used: str


@router.post("/api/predict", response_model=PredictionOutput)
async def predict_endpoint(input_data: PredictionInput):
    try:
        model, scaler = load_best_model()
        features = [
            input_data.MedInc,
            input_data.HouseAge,
            input_data.AveRooms,
            input_data.AveBedrms,
            input_data.Population,
            input_data.AveOccup,
            input_data.Latitude,
            input_data.Longitude,
        ]
        result = predict(model, features, scaler)
        model_name = "CNN-LSTM"
        model_info = get_model_info(model_name)

        try:
            log_prediction(
                input_data=input_data.model_dump(),
                prediction=result,
                model_used=model_name,
            )
        except Exception:
            pass

        return PredictionOutput(
            prediction=result,
            model_used=model_name,
            model_info=model_info,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/api/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    try:
        content = await file.read()
        text = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))

        if not reader.fieldnames:
            raise HTTPException(status_code=400, detail="CSV file has no headers")

        required = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {', '.join(missing)}",
            )

        model, scaler = load_best_model()
        model_name = "CNN-LSTM"
        model_info = get_model_info(model_name)

        results = []
        for row in reader:
            try:
                features = [float(row[c]) for c in required]
                pred = predict(model, features, scaler)
                results.append({"input": {c: float(row[c]) for c in required}, "prediction": pred})
            except (ValueError, KeyError) as e:
                results.append({"input": {c: row.get(c, "") for c in required}, "error": str(e)})

        return BatchPredictionOutput(
            predictions=results,
            total=len(results),
            model_used=model_name,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
