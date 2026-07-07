#!/usr/bin/env python3
"""
Genera un dataset financiero sintético realista para entrenar modelos de pronóstico.
Salida: dataset_financiero.csv
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n_days = 2000

# Random walk for price
returns = np.random.normal(0.0005, 0.015, n_days)
price = 100 * np.exp(np.cumsum(returns))

# Generate OHLCV
close = price
open_p = close * (1 + np.random.normal(0, 0.005, n_days))
high = np.maximum(open_p, close) * (1 + np.abs(np.random.normal(0, 0.008, n_days)))
low = np.minimum(open_p, close) * (1 - np.abs(np.random.normal(0, 0.008, n_days)))
volume = np.random.lognormal(14, 0.8, n_days).astype(int)

# Technical indicators
def sma(x, w): return pd.Series(x).rolling(w).mean().values
def ema(x, w): return pd.Series(x).ewm(span=w, adjust=False).mean().values
def rsi(x, w=14):
    delta = np.diff(x)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(w).mean().values
    avg_loss = pd.Series(loss).rolling(w).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    result = 100 - (100 / (1 + rs))
    return np.concatenate([[np.nan], result])  # pad to match length

close_sma_10 = sma(close, 10)
close_sma_50 = sma(close, 50)
close_ema_20 = ema(close, 20)
rsi_14 = rsi(close)
volume_sma_20 = sma(volume, 20)

# Volatility (20-day)
volatility = pd.Series(close).pct_change().rolling(20).std().values * np.sqrt(252)

# Target: next day close price
target_price = pd.Series(close).shift(-1).values  # next day price

df = pd.DataFrame({
    "Open": np.round(open_p, 2),
    "High": np.round(high, 2),
    "Low": np.round(low, 2),
    "Close": np.round(close, 2),
    "Volume": volume,
    "SMA_10": np.round(close_sma_10, 2),
    "SMA_50": np.round(close_sma_50, 2),
    "EMA_20": np.round(close_ema_20, 2),
    "RSI_14": np.round(rsi_14, 2),
    "Volume_SMA_20": np.round(volume_sma_20, 0),
    "Volatility": np.round(volatility, 4),
    "Target_Next_Close": np.round(target_price, 2),
})

# Drop rows with NaN (from rolling windows)
df = df.iloc[60:].reset_index(drop=True)

df.to_csv("dataset_financiero.csv", index=False)
print(f"Dataset generado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Columnas: {list(df.columns)}")
print("\nUsa 'Target_Next_Close' como variable objetivo (target) en la app.")
