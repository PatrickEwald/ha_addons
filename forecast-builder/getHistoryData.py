import json, ssl
try:
    import websocket  # from websocket-client
    HAS_WS = True
except Exception:
    HAS_WS = False
import datetime as dt
import pandas as pd
import numpy as np
from math import pi
try:
    import plotly.graph_objs as go
    from plotly.offline import plot as plotly_plot
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
try:
    import holidays as _hol
    HAS_HOL = True
except Exception:
    HAS_HOL = False

HA_BASE_URL = "http://192.168.0.123:8123"
HA_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJiM2MzNTY0NGUwOWM0MmI1ODhiYzI4MDdkZDlkZTk3OCIsImlhdCI6MTc1NTA5NjE0OSwiZXhwIjoyMDcwNDU2MTQ5fQ.1z2ZP_eZ2MZkibbqUSMLGpMv67wuPZp_6wofq6iXei8"
HA_ENTITY = "sensor.ff_darmstadt_auslastung"

# Fallback auf Long‑Term‑Statistics (History/Statistics API via WebSocket)
USE_STATISTICS_FALLBACK: bool = True   # nutzt Statistics, wenn States leer sind
STAT_PERIOD: str = "5minute"          # 5minute | hourly | daily

LOCAL_TZ = "Europe/Berlin"
HTML_PATH = "ff_history.html"

# Öffnungszeiten (lokale Zeit)
WORKING_HOURS = {
    "Mon": "06:00-22:00",
    "Tue": "06:00-22:00",
    "Wed": "06:00-22:00",
    "Thu": "06:00-22:00",
    "Fri": "06:00-22:00",
    "Sat": "09:00-20:00",
    "Sun": "09:00-20:00",
}
FORECAST_HTML = "ff_forecast_today.html"

# Lag-Konfiguration
USE_LAGS: bool = True
LAG_DAYS: list[int] = [7, 14]

def _ws_url_from_http(base: str) -> str:
    b = base.rstrip('/')
    if b.startswith('https://'):
        return 'wss://' + b.split('https://',1)[1] + '/api/websocket'
    if b.startswith('http://'):
        return 'ws://' + b.split('http://',1)[1] + '/api/websocket'
    return 'ws://' + b + '/api/websocket'

def ha_history_chunk(start_iso: str, end_iso: str, entity_id: str) -> pd.DataFrame:
    """Lädt States via REST-API (history) und gibt DataFrame mit Spalten ts,value,state,entity_id zurück."""
    import requests
    url = f"{HA_BASE_URL}/api/history/period/{start_iso}"
    headers = {"Authorization": f"Bearer {HA_TOKEN}"}
    params = {"end_time": end_iso, "filter_entity_id": entity_id}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data or not isinstance(data, list) or len(data)==0:
            return pd.DataFrame(columns=["ts","value","state","entity_id"])
        records = data[0]
        df = pd.DataFrame(records)
        df["ts"] = pd.to_datetime(df["last_updated"], utc=True, errors="coerce")
        df["value"] = pd.to_numeric(df["state"], errors="coerce")
        df["state"] = df["state"]
        df["entity_id"] = entity_id
        return df[["ts","value","state","entity_id"]].dropna(subset=["ts"])
    except Exception:
        return pd.DataFrame(columns=["ts","value","state","entity_id"])

def ha_statistics_chunk_ws(start_iso: str, end_iso: str, entity_id: str, period: str = STAT_PERIOD) -> pd.DataFrame:
    """Lädt Long‑Term‑Statistics via WebSocket und gibt DataFrame mit Spalten ts,value,state,entity_id zurück."""
    if not HAS_WS:
        return pd.DataFrame(columns=["ts","value","state","entity_id"])
    ws_url = _ws_url_from_http(HA_BASE_URL)
    try:
        ws = websocket.create_connection(ws_url, sslopt={"cert_reqs": ssl.CERT_NONE})
        ws.recv()  # hello
        ws.send(json.dumps({"type":"auth","access_token": HA_TOKEN}))
        hello = json.loads(ws.recv())
        if hello.get("type") != "auth_ok":
            ws.close(); return pd.DataFrame(columns=["ts","value","state","entity_id"])
        msg_id = 1
        req = {
            "id": msg_id,
            "type": "recorder/statistics_during_period",
            "start_time": start_iso,
            "end_time": end_iso,
            "statistic_ids": [entity_id],
            "period": period,
        }
        ws.send(json.dumps(req))
        resp = json.loads(ws.recv()); ws.close()
        if not resp.get("success"):
            return pd.DataFrame(columns=["ts","value","state","entity_id"])
        payload = resp.get("result", {}).get(entity_id, [])
    except Exception:
        return pd.DataFrame(columns=["ts","value","state","entity_id"])
    if not payload:
        return pd.DataFrame(columns=["ts","value","state","entity_id"])
    df = pd.DataFrame(payload)
    val_col = 'mean' if 'mean' in df.columns else ('state' if 'state' in df.columns else None)
    if val_col is None:
        return pd.DataFrame(columns=["ts","value","state","entity_id"])
    df["ts"] = pd.to_datetime(df["start"], format="ISO8601", utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df[val_col], errors="coerce")
    df["state"] = df["value"].astype(str)
    df["entity_id"] = entity_id
    return df[["ts","value","state","entity_id"]].dropna(subset=["ts"])  

def load_history(start_iso: str, end_iso: str, entity_id: str) -> pd.DataFrame:
    parts = []
    start = pd.to_datetime(start_iso)
    end = pd.to_datetime(end_iso)
    max_chunk_days = 30
    while start < end:
        chunk_end = min(start + pd.Timedelta(days=max_chunk_days), end)
        s_iso = start.isoformat()
        e_iso = chunk_end.isoformat()
        df_chunk = ha_history_chunk(s_iso, e_iso, entity_id)
        parts.append(df_chunk)
        print(f"  {s_iso} → {e_iso} (STATES): {len(df_chunk)} Punkte")
        # Fallback: wenn States leer, versuche Statistics (5‑min, hourly, daily)
        if USE_STATISTICS_FALLBACK and (df_chunk is None or df_chunk.empty):
            df_stats = ha_statistics_chunk_ws(s_iso, e_iso, entity_id, STAT_PERIOD)
            parts.append(df_stats)
            print(f"  {s_iso} → {e_iso} (STAT {STAT_PERIOD}): {len(df_stats)} Punkte")
        start = chunk_end
    # Leere Teile vor dem concat entfernen (vermeidet FutureWarning)
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame(columns=["ts","value","state","entity_id"])
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    return df

def add_time_features(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    local = df["ts"].dt.tz_convert(tz)
    df["weekday"] = local.dt.weekday
    df["minute_of_day"] = local.dt.hour * 60 + local.dt.minute
    df["slot_10"] = (df["minute_of_day"] // 10) * 10
    return df

def enrich_features(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    local = df["ts"].dt.tz_convert(tz)
    df["hour"] = local.dt.hour
    df["minute"] = local.dt.minute
    # zyklische Kodierung
    df["hour_sin"] = (2 * pi * df["hour"] / 24).apply(lambda x: np.sin(x))
    df["hour_cos"] = (2 * pi * df["hour"] / 24).apply(lambda x: np.cos(x))
    df["min_sin"]  = (2 * pi * df["minute"] / 60).apply(lambda x: np.sin(x))
    df["min_cos"]  = (2 * pi * df["minute"] / 60).apply(lambda x: np.cos(x))
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    # Feiertage (Hessen/DE), falls Bibliothek vorhanden
    if HAS_HOL:
        try:
            try:
                hols = _hol.country_holidays("DE", subdiv="HE")
            except Exception:
                hols = _hol.country_holidays("DE")
            df["is_holiday"] = local.dt.date.map(lambda d: 1 if d in hols else 0)
        except Exception:
            df["is_holiday"] = 0
    else:
        df["is_holiday"] = 0
    return df

def add_lag_features(df: pd.DataFrame, tz: str, lag_days: list[int]) -> pd.DataFrame:
    """Fügt lag_{d}d je (Datum, slot_10) hinzu – letzter Wert pro Slot/Tag."""
    if df is None or df.empty or not lag_days:
        return df
    x = df.copy()
    local = x["ts"].dt.tz_convert(tz)
    x["date_ts"] = pd.to_datetime(local.dt.date)
    x = x.sort_values(["date_ts", "slot_10", "ts"])  # letzter Wert pro Slot nehmen
    daily_slot = x.groupby(["date_ts", "slot_10"], as_index=False)["value"].last()
    for d in lag_days:
        lag_map = daily_slot.copy()
        lag_map["date_ts"] = lag_map["date_ts"] + pd.Timedelta(days=d)
        lag_map = lag_map.rename(columns={"value": f"lag_{d}d"})
        daily_slot = daily_slot.merge(lag_map, on=["date_ts", "slot_10"], how="left")
    lag_cols = [c for c in daily_slot.columns if c.startswith("lag_")]
    out = x.merge(daily_slot[["date_ts", "slot_10"] + lag_cols], on=["date_ts", "slot_10"], how="left")
    return out

def _parse_hhmm(s: str) -> tuple[int,int]:
    h, m = s.split(":")
    return int(h), int(m)

def get_today_open_close(tz: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    now_local = pd.Timestamp.now(tz)
    idx_to_key = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    key = idx_to_key[now_local.weekday()]
    spec = WORKING_HOURS.get(key)
    if not spec or "-" not in spec:
        return None, None
    start_s, end_s = spec.split("-")
    sh, sm = _parse_hhmm(start_s)
    eh, em = _parse_hhmm(end_s)
    open_dt = now_local.replace(hour=sh, minute=sm, second=0, microsecond=0)
    close_dt = now_local.replace(hour=eh, minute=em, second=0, microsecond=0)
    return open_dt, close_dt

def build_baseline_model(feat_df: pd.DataFrame) -> dict:
    """Median je (weekday, slot_10) mit Fallbacks."""
    model = {}
    if feat_df is None or feat_df.empty:
        return model
    train = feat_df.dropna(subset=["value"])  # nur gültige Werte
    if train.empty:
        return model
    # globaler Median
    global_med = float(train["value"].median())
    model[(None, None)] = global_med
    # Slot-Median über alle Wochentage
    for s, v in train.groupby("slot_10")["value"].median().items():
        model[(None, int(s))] = float(v)
    # (weekday, slot)
    for (w, s), v in train.groupby(["weekday","slot_10"])["value"].median().items():
        model[(int(w), int(s))] = float(v)
    return model

def predict_today(model: dict, tz: str) -> pd.DataFrame:
    open_dt, close_dt = get_today_open_close(tz)
    if open_dt is None or close_dt is None or close_dt <= open_dt:
        return pd.DataFrame(columns=["ts","ts_local","pred","pred_smooth"])
    # Zeitraster: alle 10 Minuten innerhalb der Öffnungszeiten
    times = []
    t = open_dt
    while t <= close_dt:
        times.append(t)
        t = t + pd.Timedelta(minutes=10)
    weekday = open_dt.weekday()
    rows = []
    global_med = model.get((None, None), 0.0)
    for t in times:
        minute_of_day = t.hour * 60 + t.minute
        slot_10 = (minute_of_day // 10) * 10
        y = model.get((weekday, slot_10))
        if y is None:
            y = model.get((None, slot_10), global_med)
        rows.append({
            "ts": pd.Timestamp(t).tz_convert("UTC"),
            "ts_local": pd.Timestamp(t),
            "pred": float(y)
        })
    pred_df = pd.DataFrame(rows)
    pred_df["pred_smooth"] = pred_df["pred"].rolling(3, min_periods=1, center=True).median()
    return pred_df

def build_predict_lgbm(train_df: pd.DataFrame, tz: str, lag_days: list[int], slot_lag_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Trainiert LightGBM auf historischen Punkten und liefert Prognose (10‑Minuten‑Raster, innerhalb Öffnungszeiten).
    Nutzt optionale Lag-Features. Liefert Spalten ts_local, ts, pred_ml, pred_ml_smooth.
    """
    if not HAS_LGBM:
        return pd.DataFrame()
    x = train_df.dropna(subset=["value"]).copy()
    if len(x) < 200:
        return pd.DataFrame()
    feat_cols = [
        "weekday","slot_10","hour","minute","hour_sin","hour_cos","min_sin","min_cos","is_weekend","is_holiday"
    ]
    lag_cols = [f"lag_{d}d" for d in lag_days if f"lag_{d}d" in train_df.columns]
    feat_cols = feat_cols + lag_cols
    X = x[feat_cols].copy()
    y = x["value"].astype(float)
    # fehlende Lags robust füllen
    for c in lag_cols:
        if c in X.columns:
            med = X[c].median(skipna=True)
            if pd.isna(med):
                med = 0.0
            X[c] = X[c].fillna(med)
    model = LGBMRegressor(n_estimators=400, learning_rate=0.05, max_depth=-1, subsample=0.9, colsample_bytree=0.9, random_state=42)
    model.fit(X, y)
    # Grid für heute
    open_dt, close_dt = get_today_open_close(tz)
    if open_dt is None or close_dt is None or close_dt <= open_dt:
        return pd.DataFrame()
    times = []
    t = open_dt
    while t <= close_dt:
        times.append(t)
        t = t + pd.Timedelta(minutes=10)
    rows = []
    for t in times:
        wd = t.weekday()
        mod = t.hour * 60 + t.minute
        s10 = (mod // 10) * 10
        rows.append({
            "ts_local": pd.Timestamp(t),
            "weekday": wd,
            "slot_10": s10,
            "hour": t.hour,
            "minute": t.minute,
            "hour_sin": np.sin(2 * pi * t.hour / 24),
            "hour_cos": np.cos(2 * pi * t.hour / 24),
            "min_sin": np.sin(2 * pi * t.minute / 60),
            "min_cos": np.cos(2 * pi * t.minute / 60),
            "is_weekend": 1 if wd in (5, 6) else 0,
            "is_holiday": 0,
        })
    grid = pd.DataFrame(rows)
    # Feiertage im Grid markieren (optional)
    if HAS_HOL:
        try:
            try:
                hols = _hol.country_holidays("DE", subdiv="HE")
            except Exception:
                hols = _hol.country_holidays("DE")
            grid["is_holiday"] = grid["ts_local"].dt.date.map(lambda d: 1 if d in hols else 0)
        except Exception:
            pass
    # Lags auf Grid mergen
    grid["date_ts"] = pd.to_datetime(grid["ts_local"].dt.date)
    if slot_lag_table is not None and lag_cols:
        merge_cols = ["date_ts","slot_10"]
        grid = grid.merge(slot_lag_table[merge_cols + lag_cols], on=merge_cols, how="left")
        for c in lag_cols:
            if c in grid.columns:
                med = grid[c].median(skipna=True)
                if pd.isna(med):
                    med = 0.0
                grid[c] = grid[c].fillna(med)
    preds = model.predict(grid[feat_cols])
    out = grid[["ts_local"]].copy()
    out["ts"] = out["ts_local"].dt.tz_convert("UTC")
    out["pred_ml"] = preds
    out["pred_ml_smooth"] = pd.Series(preds).rolling(3, min_periods=1, center=True).median()
    return out

def fetch_today_actuals(entity_id: str, tz: str) -> pd.DataFrame:
    open_dt, close_dt = get_today_open_close(tz)
    if open_dt is None:
        return pd.DataFrame(columns=["ts","value","state","entity_id","ts_local"]) 
    now_local = pd.Timestamp.now(tz)
    end_dt = min(close_dt if close_dt is not None else now_local, now_local)
    df = ha_history_chunk(open_dt.isoformat(), end_dt.isoformat(), entity_id)
    if df.empty and USE_STATISTICS_FALLBACK:
        df = ha_statistics_chunk_ws(open_dt.isoformat(), end_dt.isoformat(), entity_id, STAT_PERIOD)
    if df.empty:
        return df
    df = df.copy()
    df["ts_local"] = df["ts"].dt.tz_convert(tz)
    # Nur valide Werte
    df = df.dropna(subset=["value"]).sort_values("ts_local")
    # Auf Öffnungszeiten und bis jetzt clippen
    if open_dt is not None:
        df = df[df["ts_local"] >= open_dt]
    if end_dt is not None:
        df = df[df["ts_local"] <= end_dt]
    return df

def make_forecast_plot_html(pred_df: pd.DataFrame, act_df: pd.DataFrame, html_path: str, title: str):
    if not HAS_PLOTLY:
        print("Plotly ist nicht installiert (pip install plotly) – überspringe Forecast-Plot.")
        return
    if pred_df is None or pred_df.empty:
        print("Keine Vorhersagedaten vorhanden.")
        return
    traces = []

    # Öffnungszeiten für heutiges Fenster
    open_dt, close_dt = get_today_open_close(LOCAL_TZ)

    # --- Prognose vorbereiten ---
    pred_plot = pred_df.copy()
    # Nur valide Zeilen
    for col in ["pred", "pred_smooth", "ts_local"]:
        if col not in pred_plot.columns:
            pred_plot[col] = np.nan
    pred_plot = pred_plot.dropna(subset=["ts_local"]).copy()
    # Clipping auf Öffnungszeiten
    if open_dt is not None and close_dt is not None:
        pred_plot = pred_plot[(pred_plot["ts_local"] >= open_dt) & (pred_plot["ts_local"] <= close_dt)]
    # Sicherstellen, dass y numerisch ist
    for col in ["pred", "pred_smooth"]:
        pred_plot[col] = pd.to_numeric(pred_plot[col], errors="coerce")

    if pred_plot.empty or (pred_plot[["pred", "pred_smooth"]].dropna(how="all").empty):
        print("Warnung: Keine Prognosepunkte im Plotfenster gefunden.")
    else:
        # Prognose (geglättet)
        if not pred_plot["pred_smooth"].dropna().empty:
            traces.append(go.Scatter(
                x=pred_plot["ts_local"], y=pred_plot["pred_smooth"], mode="lines", name="Prognose (geglättet)"
            ))
        # Prognose (roh)
        if not pred_plot["pred"].dropna().empty:
            traces.append(go.Scatter(
                x=pred_plot["ts_local"], y=pred_plot["pred"], mode="lines", name="Prognose (roh)", line=dict(dash="dot")
            ))

    # Optional: Prognose mit Lags (ML)
    y_ml = None
    if pred_df is not None and not pred_df.empty:
        if "pred_ml_smooth" in pred_df.columns and pred_df["pred_ml_smooth"].notna().any():
            y_ml = pd.to_numeric(pred_df["pred_ml_smooth"], errors="coerce")
        elif "pred_ml" in pred_df.columns and pred_df["pred_ml"].notna().any():
            y_ml = pd.to_numeric(pred_df["pred_ml"], errors="coerce")
    if y_ml is not None:
        ml_plot = pred_df[["ts_local"]].copy()
        ml_plot["y_ml"] = y_ml
        ml_plot = ml_plot.dropna(subset=["ts_local", "y_ml"]) 
        if open_dt is not None and close_dt is not None:
            ml_plot = ml_plot[(ml_plot["ts_local"] >= open_dt) & (ml_plot["ts_local"] <= close_dt)]
        if not ml_plot.empty:
            traces.append(go.Scatter(
                x=ml_plot["ts_local"], y=ml_plot["y_ml"], mode="lines", name="Prognose mit Lags"
            ))

    # --- Ist heute vorbereiten ---
    if act_df is not None and not act_df.empty:
        act_plot = act_df.dropna(subset=["value"]).copy()
        if open_dt is not None and close_dt is not None:
            act_plot = act_plot[(act_plot["ts_local"] >= open_dt) & (act_plot["ts_local"] <= close_dt)]
        if not act_plot.empty:
            traces.append(go.Scatter(
                x=act_plot["ts_local"], y=act_plot["value"], mode="lines+markers", name="Ist heute"
            ))
        else:
            print("Hinweis: Keine heutigen Ist-Punkte im Plotfenster.")
    else:
        print("Hinweis: act_df leer – keine Istlinie.")

    # Achsen & Layout
    layout_kwargs = dict(
        title=title,
        xaxis=dict(title="Zeit"),
        yaxis=dict(title="Personen", rangemode="tozero"),
        hovermode="x unified",
    )
    if open_dt is not None and close_dt is not None:
        layout_kwargs["xaxis"]["range"] = [open_dt, close_dt]

    if not traces:
        print("Warnung: Es wurden keine Datenreihen gezeichnet (leere traces).")

    fig = go.Figure(data=traces, layout=go.Layout(**layout_kwargs))
    plotly_plot(fig, filename=html_path, auto_open=False, include_plotlyjs="cdn")
    print(f"Plot gespeichert → {html_path}")

def make_history_plot_html(df: pd.DataFrame, html_path: str, title: str = "FF Darmstadt – Historie"):
    """Erstellt einen interaktiven Plot der Historie und speichert als HTML."""
    if not HAS_PLOTLY:
        print("Plotly ist nicht installiert (pip install plotly) – überspringe Plot.")
        return
    if df is None or df.empty:
        print("Keine Daten für Plot vorhanden.")
        return
    df = df.copy()
    # lokale Zeit für X-Achse
    df["ts_local"] = df["ts"].dt.tz_convert(LOCAL_TZ)
    # optional leicht glätten: 10‑Minuten Mittelwert, ohne Lücken zu füllen
    s = df.set_index("ts_local")["value"].dropna().resample("10min").mean()
    if s.empty:
        print("Keine validen Werte für den Historien-Plot gefunden (nur NaNs).")
        return

    traces = [
        go.Scatter(x=s.index, y=s.values, mode="lines", name="Belegung (Ø 10 min)")
    ]
    layout = go.Layout(
        title=title,
        xaxis=dict(title="Zeit"),
        yaxis=dict(title="Personen"),
        hovermode="x unified",
    )
    fig = go.Figure(data=traces, layout=layout)
    plotly_plot(fig, filename=html_path, auto_open=False, include_plotlyjs="cdn")
    print(f"Plot gespeichert → {html_path}")

if __name__ == "__main__":
    data = load_history(
        "2025-05-10T00:00:00+02:00",
        "2025-08-13T23:59:59+02:00",
        HA_ENTITY
    )
    print(len(data), "Punkte")
    make_history_plot_html(data, HTML_PATH, title="FF Darmstadt – Historie (States + Statistics)")
    # Forecast für HEUTE erzeugen (Baseline + optional ML mit Lags)
    feat = add_time_features(data, LOCAL_TZ)
    feat = enrich_features(feat, LOCAL_TZ)
    slot_lag_table = None
    if USE_LAGS and LAG_DAYS:
      feat = add_lag_features(feat, LOCAL_TZ, LAG_DAYS)
      slot_lag_table = feat.drop_duplicates(["date_ts","slot_10"]).loc[:, ["date_ts","slot_10"] + [f"lag_{d}d" for d in LAG_DAYS if f"lag_{d}d" in feat.columns]]
    model = build_baseline_model(feat)
    pred_today = predict_today(model, LOCAL_TZ)
    # ML-Forecast
    pred_ml = build_predict_lgbm(feat, LOCAL_TZ, LAG_DAYS if USE_LAGS else [], slot_lag_table)
    if pred_ml is not None and not pred_ml.empty:
        pred_today = pred_today.merge(pred_ml[["ts_local","pred_ml","pred_ml_smooth"]], on="ts_local", how="left")
    # Heutige Istwerte laden (zur Überlagerung)
    act_today = fetch_today_actuals(HA_ENTITY, LOCAL_TZ)
    print(f"Heutige Ist-Punkte: {0 if act_today is None else len(act_today)}")
    print(f"Vorhersage-Punkte: {0 if pred_today is None else len(pred_today)}  (min={pred_today['pred'].min() if 'pred' in pred_today.columns else 'n/a'}, max={pred_today['pred'].max() if 'pred' in pred_today.columns else 'n/a'})")
    if 'pred_ml' in pred_today.columns:
        print(f"ML-Punkte: {pred_today['pred_ml'].notna().sum()} von {len(pred_today)}")
    # Forecast-Plot speichern
    make_forecast_plot_html(pred_today, act_today, FORECAST_HTML, title="FF Darmstadt – Vorhersage heute")