import os
import requests
from typing import Optional

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-delta-three-96.vercel.app")

def get_eda_results():
    response = requests.get(f"{API_BASE_URL}/api/metrics/eda")
    response.raise_for_status()
    return response.json()

def get_training_metrics():
    response = requests.get(f"{API_BASE_URL}/api/metrics/training")
    response.raise_for_status()
    return response.json()

def get_cross_validation():
    response = requests.get(f"{API_BASE_URL}/api/metrics/cross_validation")
    response.raise_for_status()
    return response.json()

def get_hyperparameter_tuning():
    response = requests.get(f"{API_BASE_URL}/api/metrics/hyperparameter_tuning")
    response.raise_for_status()
    return response.json()

def get_statistical_tests():
    response = requests.get(f"{API_BASE_URL}/api/metrics/statistical_tests")
    response.raise_for_status()
    return response.json()

def get_all_metrics():
    response = requests.get(f"{API_BASE_URL}/api/metrics/all")
    response.raise_for_status()
    return response.json()

def get_models():
    response = requests.get(f"{API_BASE_URL}/api/models")
    response.raise_for_status()
    return response.json()

def get_model_comparison():
    response = requests.get(f"{API_BASE_URL}/api/models/comparison")
    response.raise_for_status()
    return response.json()

def predict(features: dict):
    response = requests.post(f"{API_BASE_URL}/api/predict", json=features)
    response.raise_for_status()
    return response.json()

def signup(email: str, password: str):
    response = requests.post(f"{API_BASE_URL}/api/auth/signup", json={"email": email, "password": password})
    response.raise_for_status()
    return response.json()

def signin(email: str, password: str):
    response = requests.post(f"{API_BASE_URL}/api/auth/signin", json={"email": email, "password": password})
    response.raise_for_status()
    return response.json()

def get_preferences(token: str):
    response = requests.get(f"{API_BASE_URL}/api/auth/preferences", headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    return response.json()

def update_preferences(token: str, language: str):
    response = requests.put(f"{API_BASE_URL}/api/auth/preferences", json={"language": language}, headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    return response.json()

def verify_token(token: str):
    response = requests.post(f"{API_BASE_URL}/api/auth/verify", headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    return response.json()

def generate_report_url(format: str = "pdf", language: str = "es"):
    return f"{API_BASE_URL}/api/reports/generate?format={format}&language={language}"
