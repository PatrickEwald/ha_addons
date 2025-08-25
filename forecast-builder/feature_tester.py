#!/usr/bin/env python3
"""
feature_tester.py — Minimal (Baseline only)

• Lädt Auslastungsdaten aus InfluxDB
• Nutzt NUR baseline-Features (Zeit/Kalender) + LightGBM
• Erstellt eine Vorhersage für heute (06–22 Uhr, konfigurierbar)
• Exportiert ausschließlich einen interaktiven Plot als eigenständige HTML-Datei

Beispiel:
  python feature_tester.py --config config.json --today --start-hour 6 --end-hour 22 --plot-html-file today_forecast.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import pytz
import requests
import holidays
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Helper / Config
# ------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "influx_host": "192.168.0.123",
    "influx_port": 8086,
    "influx_url": "http://192.168.0.123:8086/query",
    "influx_db": "homeassistant",
    "influx_user": "homeassistant",
    "influx_password": "assi",

    "queries": {
        "occupancy": """
            SELECT mean("value") FROM "autogen"."Personen"
            WHERE "entity_id"='ff_darmstadt_auslastung'
            AND time >= '2025-05-10T00:00:00Z'
            GROUP BY time(10m) fill(null);
        """,
    },

    "timezone": "Europe/Berlin",
    "model_params": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 8,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    },
}

BASELINE_FEATURES: List[str] = [
    "hour", "dow", "time_minutes",
    "sin_time", "cos_time", "sin_dow", "cos_dow",
    "is_weekend", "is_monday", "is_friday",
    "is_early_morning", "is_morning", "is_lunch",
    "is_afternoon", "is_evening", "is_late",
    "is_holiday", "is_bridge_day",
]

# ------------------------------------------------------------
# Core class
# ------------------------------------------------------------

class BaselineForecaster:
    def __init__(self, config_path: Optional[str]):
        self.config = self._load_config(config_path)

    def _load_config(self, path: Optional[str]) -> Dict[str, Any]:
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deepcopy
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    user_cfg = json.load(f)
                cfg = _deep_merge(cfg, user_cfg)
                print(f"✅ Konfiguration geladen aus: {path}")
            except Exception as e:
                print(f"⚠️  Fehler beim Laden der Config: {e}. Verwende Defaults.")
        # Fallback für Credentials via ENV
        cfg["influx_user"] = cfg.get("influx_user") or os.getenv("INFLUX_USER")
        cfg["influx_password"] = cfg.get("influx_password") or os.getenv("INFLUX_PASSWORD")
        if not cfg.get("influx_user") or not cfg.get("influx_password"):
            print("❌ InfluxDB-Credentials fehlen (in Datei oder ENV setzen).")
            sys.exit(1)
        return cfg

    # -------------------------- Data -------------------------
    def influx_query(self, query: str) -> pd.DataFrame:
        try:
            resp = requests.post(
                self.config["influx_url"],
                data={"db": self.config["influx_db"], "q": query, "epoch": "ms"},
                auth=(self.config["influx_user"], self.config["influx_password"]),
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data.get("results") or not data["results"][0].get("series"):
                return pd.DataFrame(columns=["ds", "y"])
            series = data["results"][0]["series"][0]
            df = pd.DataFrame(series["values"], columns=["epoch_ms", "value"])
            df["ds_utc"] = pd.to_datetime(df["epoch_ms"], unit="ms", utc=True)
            df["ds"] = df["ds_utc"].dt.tz_convert(self.config["timezone"]).dt.tz_localize(None)
            df = df.drop(columns=["epoch_ms", "ds_utc"]).rename(columns={"value": "y"})
            return df.dropna()
        except Exception as e:
            print(f"❌ Fehler bei InfluxDB Query: {e}")
            return pd.DataFrame(columns=["ds", "y"])

    def load_data(self) -> pd.DataFrame:
        print("📥 Lade Daten aus InfluxDB…")
        q = self.config["queries"]["occupancy"]
        df = self.influx_query(q)
        if df.empty:
            print("❌ Keine Auslastungsdaten gefunden!")
            sys.exit(1)
        df = df.sort_values("ds").reset_index(drop=True)
        print(f"✅ Auslastungsdaten: {len(df)} Punkte | Range: {df['ds'].min()} → {df['ds'].max()}")
        return df

    # -------------------------- Features ----------------------
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["dow"] = out["ds"].dt.dayofweek
        out["hour"] = out["ds"].dt.hour
        out["time_minutes"] = out["hour"] * 60 + out["ds"].dt.minute
        out["sin_time"] = np.sin(2 * np.pi * out["time_minutes"] / (24 * 60))
        out["cos_time"] = np.cos(2 * np.pi * out["time_minutes"] / (24 * 60))
        out["sin_dow"] = np.sin(2 * np.pi * out["dow"] / 7)
        out["cos_dow"] = np.cos(2 * np.pi * out["dow"] / 7)
        out["is_weekend"] = (out["dow"] >= 5).astype(int)
        out["is_monday"] = (out["dow"] == 0).astype(int)
        out["is_friday"] = (out["dow"] == 4).astype(int)
        out["is_early_morning"] = ((out["hour"] >= 6) & (out["hour"] < 9)).astype(int)
        out["is_morning"] = ((out["hour"] >= 9) & (out["hour"] < 12)).astype(int)
        out["is_lunch"] = ((out["hour"] >= 12) & (out["hour"] < 14)).astype(int)
        out["is_afternoon"] = ((out["hour"] >= 14) & (out["hour"] < 17)).astype(int)
        out["is_evening"] = ((out["hour"] >= 17) & (out["hour"] < 20)).astype(int)
        out["is_late"] = (out["hour"] >= 20).astype(int)
        de_he = holidays.Germany(prov="HE")
        out["is_holiday"] = out["ds"].dt.date.isin(de_he).astype(int)
        prev_day = out["ds"].dt.date - timedelta(days=1)
        next_day = out["ds"].dt.date + timedelta(days=1)
        out["is_bridge_day"] = (
            ((out["dow"] == 4) & prev_day.isin(de_he)) |
            ((out["dow"] == 0) & next_day.isin(de_he))
        ).astype(int)
        return out

    # -------------------------- Forecast ----------------------
    def make_future_df(self, target_date: date, start_hour: int, end_hour: int) -> pd.DataFrame:
        tz = pytz.timezone(self.config.get("timezone", "Europe/Berlin"))
        start_dt = tz.localize(datetime(target_date.year, target_date.month, target_date.day, start_hour, 0))
        end_dt = tz.localize(datetime(target_date.year, target_date.month, target_date.day, end_hour, 0))
        idx = pd.date_range(start=start_dt, end=end_dt, freq="10min", tz=tz).tz_localize(None)
        fdf = pd.DataFrame({"ds": idx})
        return self.add_time_features(fdf)

    def forecast_today(self, df: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
        tz = pytz.timezone(self.config.get("timezone", "Europe/Berlin"))
        target = datetime.now(tz).date()
        # Train (baseline-only)
        train = self.add_time_features(df.copy())
        X = train[BASELINE_FEATURES].dropna()
        y = train.loc[X.index, "y"]
        if len(X) < 100:
            print("⚠️  Sehr wenige Trainingsdaten – Prognose kann ungenau sein.")
        model = LGBMRegressor(**self.config["model_params"])
        model.fit(X, y)
        # Future
        future = self.make_future_df(target, start_hour, end_hour)
        y_pred = model.predict(future[BASELINE_FEATURES])
        y_pred = np.clip(y_pred, 0, None)
        return pd.DataFrame({"ds": future["ds"], "y_pred": y_pred})

    # -------------------------- Plot (HTML) -------------------
    def export_interactive_html(self, fc_df: pd.DataFrame, save_path: str) -> Optional[str]:
        if fc_df is None or fc_df.empty:
            print("❌ Keine Prognosedaten zum Plotten vorhanden.")
            return None
        df = fc_df.copy().sort_values("ds")
        ts_ms = (pd.to_datetime(df["ds"]).astype("int64") // 10**6).tolist()
        data = [[int(t), float(v)] for t, v in zip(ts_ms, df["y_pred"].tolist())]
        series_json = json.dumps([{"name": "Forecast heute", "data": data}])
        html = f"""
<!doctype html>
<html lang='de'>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>Forecast heute</title>
  <script src='https://cdn.jsdelivr.net/npm/apexcharts'></script>
  <style>
    body{{margin:0;background:#0b0e14;color:#e6e6e6;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial}}
    .card{{max-width:1200px;margin:24px auto;background:#121826;border-radius:16px;padding:18px;box-shadow:0 10px 30px rgba(0,0,0,.35)}}
    h1{{margin:0 0 8px;font-size:20px;font-weight:600}}
  </style>
</head>
<body>
  <div class='card'>
    <h1>Prognose heute (Baseline)</h1>
    <div id='chart'></div>
  </div>
  <script>
    const series = {series_json};
    const options = {{
      chart: {{ type: 'line', height: 520, toolbar: {{ show: true }}, zoom: {{ enabled: true }} }},
      theme: {{ mode: 'dark' }},
      stroke: {{ width: 3 }},
      xaxis: {{ type: 'datetime' }},
      yaxis: {{ min: 0 }},
      tooltip: {{ shared: true, x: {{ format: 'dd.MM.yyyy HH:mm' }} }},
      legend: {{ show: true }},
      series
    }};
    new ApexCharts(document.querySelector('#chart'), options).render();
  </script>
</body>
</html>
"""
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"🌐 Interaktive HTML gespeichert: {save_path}")
            return save_path
        except Exception as e:
            print(f"❌ Fehler beim Speichern der HTML: {e}")
            return None

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Baseline-Only Forecast + interaktiver Plot")
    p.add_argument("--config", type=str, help="Pfad zur JSON-Konfigurationsdatei")
    p.add_argument("--today", action="store_true", help="Erzeuge Vorhersage für heute (Standard)")
    p.add_argument("--start-hour", type=int, default=6, help="Startstunde (default: 6)")
    p.add_argument("--end-hour", type=int, default=22, help="Endstunde (default: 22)")
    p.add_argument("--plot-html-file", type=str, default="today_forecast.html", help="Zieldatei für HTML")
    args = p.parse_args()

    try:
        fc = BaselineForecaster(args.config)
        df = fc.load_data()
        fc_df = fc.forecast_today(df, start_hour=args.start_hour, end_hour=args.end_hour)
        fc.export_interactive_html(fc_df, args.plot_html_file)
    except KeyboardInterrupt:
        print("\n⏹️  Abgebrochen")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()