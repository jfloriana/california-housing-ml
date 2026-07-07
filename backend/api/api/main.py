import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

load_dotenv()

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from .routes import predict, models, reports, metrics, auth
from .utils.supabase_client import get_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        client = get_client()
        client.from_("training_metrics").select("count", count="exact").limit(1).execute()
        print("[startup] Supabase connection OK")
    except Exception as e:
        print(f"[startup] Supabase connection failed: {e}")
    yield


app = FastAPI(
    title="California Housing ML API",
    description="API for California Housing price prediction using neural networks",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(models.router)
app.include_router(reports.router)
app.include_router(metrics.router)
app.include_router(auth.router)


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/api/health")
async def health():
    supabase_ok = False
    try:
        client = get_client()
        client.from_("training_metrics").select("count", count="exact").limit(1).execute()
        supabase_ok = True
    except Exception:
        pass
    return {
        "status": "healthy",
        "version": "1.0.0",
        "supabase_connected": supabase_ok,
    }


handler = app
