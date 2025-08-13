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
    "hum_query": """
SELECT mean("value") FROM "autogen"."%"
  WHERE "entity_id"='openweathermap_humidity'
  AND time >= '2025-05-10T00:00:00Z'
  GROUP BY time(10m) fill(null);
""",
    "press_query": """
SELECT mean("value") FROM "autogen"."hPa"
  WHERE "entity_id"='openweathermap_pressure'
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
    # ML-Hyperparameter (LightGBM) – identisch zum Feature-Tester
    "model_params": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "n_jobs": -1,
        "random_state": 42
    },
    # Feature-Schalter (Default-Werte) – per /data/options.json überschreibbar
    "features": {
        "use_weather": True,      # Temperatur, Luftfeuchte, Luftdruck
        "use_bridge_day": True,   # Brückentag-Feature
        "use_lags": True,         # Lag-Features
        "use_rolling": True,      # Rolling-Window-Features
        "use_trends": True        # Trend-Features
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

    # Feature Flags setzen (um neue Features erweitert)
    for key in ["use_weather", "use_bridge_day", "use_lags", "use_rolling", "use_trends"]:
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
    """Zeit- und Kalender-Features (wie im Feature-Tester)"""
    df = df.copy()
    
    # Basis Zeit-Features
    df["dow"] = df["ds"].dt.dayofweek
    df["hour"] = df["ds"].dt.hour
    df["time_minutes"] = df["hour"] * 60 + df["ds"].dt.minute
    df["sin_time"] = np.sin(2 * np.pi * df["time_minutes"] / (24 * 60))
    df["cos_time"] = np.cos(2 * np.pi * df["time_minutes"] / (24 * 60))
    df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7)

    # Feiertage (Bundesland Hessen)
    df["is_holiday"] = df["ds"].dt.date.isin(holidays.Germany(prov="HE")).astype(int)
    
    # Bridge Day Feature
    if CONFIG["features"]["use_bridge_day"]:
        prev = df["ds"].dt.date - pd.Timedelta(days=1)
        next_ = df["ds"].dt.date + pd.Timedelta(days=1)
        df["is_bridge_day"] = (
            ((df["dow"] == 4) & prev.isin(holidays.Germany(prov="HE"))) |
            ((df["dow"] == 0) & next_.isin(holidays.Germany(prov="HE")))
        ).astype(int)
    
    # Erweiterte Zeit-Features
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_monday"] = (df["dow"] == 0).astype(int)
    df["is_friday"] = (df["dow"] == 4).astype(int)
    
    # Tageszeit-Kategorien
    df["is_early_morning"] = ((df["hour"] >= 6) & (df["hour"] < 9)).astype(int)
    df["is_morning"] = ((df["hour"] >= 9) & (df["hour"] < 12)).astype(int)
    df["is_lunch"] = ((df["hour"] >= 12) & (df["hour"] < 14)).astype(int)
    df["is_afternoon"] = ((df["hour"] >= 14) & (df["hour"] < 17)).astype(int)
    df["is_evening"] = ((df["hour"] >= 17) & (df["hour"] < 20)).astype(int)
    df["is_late"] = ((df["hour"] >= 20)).astype(int)
    
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag-Features"""
    if not CONFIG["features"]["use_lags"]:
        return df
        
    df = df.copy()
    # 10-Minuten-Raster: 1h=6, 4h=24, 1d=144, 7d=1008
    lag_configs = {
        "y_lag_1h": 6,
        "y_lag_4h": 24,
        "y_lag_1d": 144,
        "y_lag_7d": 1008,
    }
    for lag_name, periods in lag_configs.items():
        df[lag_name] = df["y"].shift(periods)
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling-Window-Features"""
    if not CONFIG["features"]["use_rolling"]:
        return df
        
    df = df.copy()
    windows = {"1h": 6, "4h": 24, "12h": 72}
    for window_name, periods in windows.items():
        df[f"y_rolling_mean_{window_name}"] = df["y"].rolling(periods, min_periods=1).mean()
        df[f"y_rolling_std_{window_name}"] = df["y"].rolling(periods, min_periods=1).std()
        df[f"y_rolling_max_{window_name}"] = df["y"].rolling(periods, min_periods=1).max()
        df[f"y_rolling_min_{window_name}"] = df["y"].rolling(periods, min_periods=1).min()
    return df

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trend-Features"""
    if not CONFIG["features"]["use_trends"]:
        return df
        
    df = df.copy()
    # absolute Änderungen
    df["y_diff_1h"] = df["y"] - df["y"].shift(6)
    df["y_diff_4h"] = df["y"] - df["y"].shift(24)
    df["y_diff_1d"] = df["y"] - df["y"].shift(144)
    # prozentuale Änderungen
    df["y_pct_change_1h"] = df["y"].pct_change(6)
    df["y_pct_change_4h"] = df["y"].pct_change(24)
    df["y_pct_change_1d"] = df["y"].pct_change(144)
    # kategoriale Richtungen
    df["trend_direction_1h"] = np.where(df["y_diff_1h"] > 0, 1, np.where(df["y_diff_1h"] < 0, -1, 0))
    df["trend_direction_4h"] = np.where(df["y_diff_4h"] > 0, 1, np.where(df["y_diff_4h"] < 0, -1, 0))
    # Stärke
    df["trend_strength_1h"] = np.abs(df["y_pct_change_1h"])
    df["trend_strength_4h"] = np.abs(df["y_pct_change_4h"])
    return df

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Analog Feature-Tester: nur EXOGENE Features interpolieren, keine Ziel-abgeleiteten."""
    all_num = df.select_dtypes(include=[np.number]).columns.difference(["y"])
    target_derived = [c for c in all_num if c.startswith(("y_lag_", "y_rolling_", "y_diff_", "y_pct_", "trend_"))]
    exogenous = [c for c in all_num if c not in target_derived]
    df[exogenous] = df[exogenous].interpolate(method="linear").bfill().ffill()
    return df

def make_future_df(day: datetime.date, tz) -> pd.DataFrame:
    """Erstellt Future DataFrame mit allen Features"""
    open_str, close_str = CONFIG["working_hours"][day.strftime("%a")].split("-")
    open_dt = tz.localize(datetime.combine(day, datetime.strptime(open_str, "%H:%M").time()))
    close_dt = tz.localize(datetime.combine(day, datetime.strptime(close_str, "%H:%M").time()))
    future_times = pd.date_range(start=open_dt, end=close_dt, freq="10min", tz=tz.zone).tz_localize(None)
    fdf = pd.DataFrame({"ds": future_times})
    # Zeit-Features hinzufügen
    fdf = add_time_features(fdf)
    return fdf

def get_feature_cols() -> List[str]:
    """Erweiterte Feature-Liste (identisch zum Feature-Tester full, per Schalter aktivierbar)."""
    base = [
        "hour", "dow", "time_minutes",
        "sin_time", "cos_time", "sin_dow", "cos_dow", "is_holiday",
        "is_weekend", "is_monday", "is_friday",
        "is_early_morning", "is_morning", "is_lunch",
        "is_afternoon", "is_evening", "is_late",
    ]
    if CONFIG["features"].get("use_bridge_day", False):
        base.append("is_bridge_day")
    if CONFIG["features"].get("use_weather", False):
        base.extend(["temperature", "humidity", "pressure"])
    if CONFIG["features"].get("use_lags", False):
        base.extend(["y_lag_1h", "y_lag_4h", "y_lag_1d", "y_lag_7d"])
    if CONFIG["features"].get("use_rolling", False):
        base.extend([
            "y_rolling_mean_1h", "y_rolling_std_1h", "y_rolling_max_1h", "y_rolling_min_1h",
            "y_rolling_mean_4h", "y_rolling_std_4h", "y_rolling_max_4h", "y_rolling_min_4h",
            "y_rolling_mean_12h", "y_rolling_std_12h", "y_rolling_max_12h", "y_rolling_min_12h",
        ])
    if CONFIG["features"].get("use_trends", False):
        base.extend([
            "y_diff_1h", "y_diff_4h", "y_diff_1d",
            "y_pct_change_1h", "y_pct_change_4h", "y_pct_change_1d",
            "trend_direction_1h", "trend_direction_4h",
            "trend_strength_1h", "trend_strength_4h",
        ])
    return base

def train_and_predict(target_date: datetime):
    """Training & Prediction (Feature-Engineering analog Feature-Tester full)"""
    tz = pytz.timezone(CONFIG["tz"])

    # 1) Daten holen
    print("📥 Lade Auslastungsdaten...")
    df = influx_query(CONFIG["query"]).rename(columns={"value": "y"})
    
    if CONFIG["features"].get("use_weather", False):
        print("🌡️ Lade Wetterdaten (Temp/Humidity/Pressure)...")
        try:
            temp = influx_query(CONFIG["temp_query"]).rename(columns={"value": "temperature"})
            df = pd.merge(df, temp, on="ds", how="left")
        except Exception:
            print("⚠️ Temperaturdaten nicht verfügbar")
        try:
            hum = influx_query(CONFIG["hum_query"]).rename(columns={"value": "humidity"})
            df = pd.merge(df, hum, on="ds", how="left")
        except Exception:
            print("⚠️ Feuchtigkeitsdaten nicht verfügbar")
        try:
            press = influx_query(CONFIG["press_query"]).rename(columns={"value": "pressure"})
            df = pd.merge(df, press, on="ds", how="left")
        except Exception:
            print("⚠️ Luftdruckdaten nicht verfügbar")
    
    print(f"📊 Dataset: {len(df)} Datenpunkte")
    
    # 2) Features hinzufügen
    print("🔧 Erstelle Features...")
    df = add_time_features(df).sort_values("ds")
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)
    
    # 3) Fehlende Werte behandeln (nur exogene)
    df = fill_missing_values(df)

    # 4) Feature-Spalten bestimmen
    feature_cols = get_feature_cols()
    print(f"📈 Verwende {len(feature_cols)} Features: {feature_cols[:5]}..." + 
          (f" und {len(feature_cols)-5} weitere" if len(feature_cols) > 5 else ""))
    
    # 5) Training-Daten vorbereiten (nur Zeilen ohne NaN in Features)
    df_train = df[feature_cols + ["y"]].dropna()
    print(f"🎯 Training mit {len(df_train)} sauberen Datenpunkten")
    
    if len(df_train) < 100:
        print("⚠️ Warnung: Wenig Trainingsdaten verfügbar!")

    # 6) Modell trainieren
    print("🤖 Trainiere LightGBM Modell...")
    model = LGBMRegressor(**CONFIG["model_params"])
    model.fit(df_train[feature_cols], df_train["y"])

    # 7) Future DataFrame erstellen
    print(f"🔮 Erstelle Prognosen für {target_date.date()}...")
    future_df = make_future_df(target_date.date(), tz)
    
    # 8) Fehlende Features für Future DataFrame setzen (exogene bleiben NaN)
    if CONFIG["features"].get("use_weather", False):
        for col in ["temperature", "humidity", "pressure"]:
            if col not in future_df.columns:
                future_df[col] = np.nan

    if CONFIG["features"].get("use_lags", False):
        for lag_col in ["y_lag_1h", "y_lag_4h", "y_lag_1d", "y_lag_7d"]:
            future_df[lag_col] = np.nan

    if CONFIG["features"].get("use_rolling", False):
        for col in [
            "y_rolling_mean_1h", "y_rolling_std_1h", "y_rolling_max_1h", "y_rolling_min_1h",
            "y_rolling_mean_4h", "y_rolling_std_4h", "y_rolling_max_4h", "y_rolling_min_4h",
            "y_rolling_mean_12h", "y_rolling_std_12h", "y_rolling_max_12h", "y_rolling_min_12h",
        ]:
            future_df[col] = np.nan

    if CONFIG["features"].get("use_trends", False):
        for col in [
            "y_diff_1h", "y_diff_4h", "y_diff_1d",
            "y_pct_change_1h", "y_pct_change_4h", "y_pct_change_1d",
            "trend_direction_1h", "trend_direction_4h",
            "trend_strength_1h", "trend_strength_4h",
        ]:
            future_df[col] = np.nan
    
    # 9) Fehlende Werte in Future DataFrame behandeln (nur exogene)
    future_df = fill_missing_values(future_df)

    # 10) Vorhersagen machen
    future_df["y_pred"] = model.predict(future_df[feature_cols])
    
    print(f"✅ {len(future_df)} Prognosen erstellt")
    print(f"📊 Prognose-Bereich: {future_df['y_pred'].min():.1f} - {future_df['y_pred'].max():.1f}")
    
    return future_df[["ds", "y_pred"]]

def write_forecast_to_influx(df: pd.DataFrame):
    """Schreibt Prognosen in InfluxDB"""
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

    print(f"🚀 Starte Forecast für {target_dt.date()}")
    forecast_df = train_and_predict(target_dt)
    write_forecast_to_influx(forecast_df)
    print("🎉 Forecast abgeschlossen!")

if __name__ == "__main__":
    main()