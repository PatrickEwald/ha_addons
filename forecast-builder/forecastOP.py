#!/usr/bin/env python3
"""
forecastOP.py  –  Minimal Forecast Runner

* holt historische Auslastungs-Daten (und optionale Features) aus InfluxDB
* trainiert ein LightGBM-Modell
* erstellt für die heutigen Öffnungszeiten alle 10 Minuten eine Prognose
* schreibt NUR diese Prognosewerte (Spalten y_pred / pred_*) wieder in InfluxDB

Keine Plots, CSVs, MQTT oder SHAP-Analysen mehr.
"""

import argparse
from datetime import datetime
from typing import List
import json
import os
import sys

import numpy as np
import pandas as pd
import pytz
import requests
from influxdb import InfluxDBClient
from lightgbm import LGBMRegressor

import holidays

# ────────────────────────────────────
# Konfiguration
# ────────────────────────────────────
CONFIG = {
    # Influx — Datenquelle
    "influx_host": "192.168.0.123",
    "influx_url": "http://192.168.0.123:8086/query",
    "influx_db": "homeassistant",
    "influx_auth": (None, None),  # kein Default mehr, muss geladen werden
    # Queries
    "query": """
SELECT mean("value") FROM "autogen"."Personen"
  WHERE "entity_id"='ff_darmstadt_auslastung' AND time >= '2025-05-10T00:00:00Z'
  GROUP BY time(10m) fill(null);
""",
    "temp_query": """
SELECT mean("value") FROM "autogen"."°C"
  WHERE "entity_id"='openweathermap_feels_like_temperature'
  AND time >= '2025-05-10T00:00:00Z'
  GROUP BY time(10m) fill(null);
""",
    # Zeitzone und Öffnungszeiten
    "tz": "Europe/Berlin",
    "working_hours": {
        "Mon": "06:00-22:00",
        "Tue": "06:00-22:00",
        "Wed": "06:00-22:00",
        "Thu": "06:00-22:00",
        "Fri": "06:00-22:00",
        "Sat": "09:00-20:00",
        "Sun": "09:00-20:00",
    },
    # ML-Hyperparameter
    "model_params": {"n_estimators": 400, "learning_rate": 0.05, "random_state": 42},
    # Feature-Schalter (Default-Werte)
    "features": {
        "use_lags": False,
        "use_bridge_day": False,
        "use_temperature": False,
    },
    # Ziel-Messung in InfluxDB
    "out_measurement": "forecast_ff",
}

def load_addon_config():
    """Lädt die Addon-Konfiguration aus /data/options.json, zwingend inkl. Influx-User und Passwort."""
    options_path = "/data/options.json"
    if not os.path.exists(options_path):
        print("❌ Fehler: /data/options.json nicht gefunden. Bitte konfigurieren.")
        sys.exit(1)

    with open(options_path, "r") as f:
        options = json.load(f)

    # Feature Flags setzen
    for key in ["use_temperature", "use_bridge_day", "use_lags"]:
        if key in options and isinstance(options[key], bool):
            CONFIG["features"][key] = options[key]

    # Influx User/Pass prüfen und setzen
    user = options.get("influx_user")
    passwd = options.get("influx_password")
    if not user or not passwd:
        print("❌ Fehler: influx_user und influx_password müssen in der Addon-Konfiguration gesetzt sein!")
        sys.exit(1)
    CONFIG["influx_auth"] = (user, passwd)

    print(f"⚙️ Geladene Addon-Features: {CONFIG['features']}")
    print(f"⚙️ Influx Auth: (user=***, password=***) geladen.")

# Konfig laden - direkt beim Start
load_addon_config()

# ────────────────────────────────────
# Influx-Helpers
# ────────────────────────────────────
def influx_query(q: str) -> pd.DataFrame:
    """Execute query & return DataFrame(time,value) in local TZ."""
    resp = requests.post(
        CONFIG["influx_url"],
        data={"db": CONFIG["influx_db"], "q": q, "epoch": "ms"},
        auth=CONFIG["influx_auth"],
        timeout=20,
    )
    resp.raise_for_status()
    series = resp.json()["results"][0]["series"][0]
    df = pd.DataFrame(series["values"], columns=["epoch_ms", "value"])
    df["ds_utc"] = pd.to_datetime(df["epoch_ms"], unit="ms", utc=True)
    df["ds"] = df["ds_utc"].dt.tz_convert(CONFIG["tz"]).dt.tz_localize(None)
    return df[["ds", "value"]]

# ────────────────────────────────────
# Feature Engineering
# ────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["ds"].dt.dayofweek
    df["time_minutes"] = df["ds"].dt.hour * 60 + df["ds"].dt.minute
    df["sin_time"] = np.sin(2 * np.pi * df["time_minutes"] / (24 * 60))
    df["cos_time"] = np.cos(2 * np.pi * df["time_minutes"] / (24 * 60))
    df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["is_holiday"] = df["ds"].dt.date.isin(holidays.Germany()).astype(int)
    if CONFIG["features"]["use_bridge_day"]:
        prev = df["ds"].dt.date - pd.Timedelta(days=1)
        next_ = df["ds"].dt.date + pd.Timedelta(days=1)
        df["is_bridge_day"] = (
            ((df["dow"] == 4) & prev.isin(holidays.Germany())) |
            ((df["dow"] == 0) & next_.isin(holidays.Germany()))
        ).astype(int)
    return df

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.difference(["y"])
    df[num_cols] = df[num_cols].interpolate(method="linear").bfill().ffill()
    return df

def make_future_df(day: datetime.date, tz) -> pd.DataFrame:
    open_str, close_str = CONFIG["working_hours"][day.strftime("%a")].split("-")
    open_dt = tz.localize(datetime.combine(day, datetime.strptime(open_str, "%H:%M").time()))
    close_dt = tz.localize(datetime.combine(day, datetime.strptime(close_str, "%H:%M").time()))
    future_times = pd.date_range(start=open_dt, end=close_dt, freq="10min", tz=tz.zone).tz_localize(None)
    fdf = pd.DataFrame({"ds": future_times})
    return add_time_features(fdf)

def get_feature_cols() -> List[str]:
    base = ["sin_time", "cos_time", "sin_dow", "cos_dow", "is_holiday"]
    if CONFIG["features"]["use_bridge_day"]:
        base.append("is_bridge_day")
    if CONFIG["features"]["use_temperature"]:
        base.append("temp_feels_like")
    if CONFIG["features"]["use_lags"]:
        base.append("y_lag_7d")
    return base

def train_and_predict(target_date: datetime):
    tz = pytz.timezone(CONFIG["tz"])

    # 1) Daten holen
    df = influx_query(CONFIG["query"]).rename(columns={"value": "y"})
    if CONFIG["features"]["use_temperature"]:
        temp = influx_query(CONFIG["temp_query"]).rename(columns={"value": "temp_feels_like"})
        df = pd.merge(df, temp, on="ds", how="left")
    df = add_time_features(df).sort_values("ds")

    if CONFIG["features"]["use_lags"]:
        df["y_lag_7d"] = df["y"].shift(1008)

    df = fill_missing_values(df)

    feature_cols = get_feature_cols()
    df_train = df[feature_cols + ["y"]]

    model = LGBMRegressor(**CONFIG["model_params"]).fit(df_train[feature_cols], df_train["y"])

    future_df = make_future_df(target_date.date(), tz)
    if CONFIG["features"]["use_temperature"]:
        future_df["temp_feels_like"] = np.nan
    if CONFIG["features"]["use_lags"]:
        future_df["y_lag_7d"] = np.nan

    future_df["y_pred"] = model.predict(future_df[feature_cols])
    return future_df[["ds", "y_pred"]]

def write_forecast_to_influx(df: pd.DataFrame):
    client = InfluxDBClient(
        host=CONFIG["influx_host"],
        port=CONFIG.get("influx_port", 8086),
        username=CONFIG["influx_auth"][0],
        password=CONFIG["influx_auth"][1],
        database=CONFIG["influx_db"],
    )
    tz_berlin = pytz.timezone(CONFIG["tz"])
    points = []
    for _, row in df.iterrows():
        ts = tz_berlin.localize(row["ds"]).astimezone(pytz.utc)
        points.append({
            "measurement": CONFIG["out_measurement"],
            "time": ts.isoformat(),
            "fields": {"y_pred": float(row["y_pred"])},
        })
    client.write_points(points, time_precision="s")
    print(f"✅ {len(points)} Prognose-Punkte geschrieben → {CONFIG['out_measurement']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, help="YYYY-MM-DD (Default=today)")
    args = parser.parse_args()

    tz = pytz.timezone(CONFIG["tz"])
    target_dt = tz.localize(datetime.strptime(args.date, "%Y-%m-%d")) if args.date else datetime.now(tz)

    forecast_df = train_and_predict(target_dt)
    write_forecast_to_influx(forecast_df)

if __name__ == "__main__":
    main()