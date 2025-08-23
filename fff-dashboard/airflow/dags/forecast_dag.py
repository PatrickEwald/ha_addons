from datetime import datetime, timedelta, time as datetime_time, timezone, date
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.state import State
import pandas as pd
import psycopg2
import numpy as np
import holidays
import logging
import time
import json
import pytz

from typing import Dict, List, Optional



# Default arguments for the DAG
default_args = {
    'owner': 'fff-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def get_db_connection():
    """Create a database connection with consistent settings"""
    return psycopg2.connect(
        host="postgres",
        database="ffdb",
        user="ffuser",
        password="ffpass",
        port="5432",
        connect_timeout=10
    )



def load_utilization_data(**context):
    """Load utilization data from PostgreSQL"""
    logging.info("🚀 START: load_utilization_data")
    
    try:
        logging.info("📊 Loading data from PostgreSQL...")
        conn = get_db_connection()
        
        # Define real studios (exclude sample data)
        REAL_STUDIOS = [
            '818e661e-59f7-48d1-bbbd-8b504e07b077',  # Darmstadt - Neckarstraße
            '86f73b33-9821-4ff9-87b5-2d5dde0711e3',  # Darmstadt - Pfnorstraße
            'ce901992-4385-48df-886b-3871860e61b5',  # Darmstadt - Eschollbrückerstraße
            'dee680ee-aab2-44f9-a29e-81f53df03c1c',  # Darmstadt - Eberstadt
            'e9de4a7c-3f9e-45fc-af09-2dae824319d8',  # Darmstadt - Kleyerstraße
            '1b793462-e413-49fb-b971-ada1e11dc90e',  # Darmstadt - Ostbahnhof
            '255339c4-c6f0-4ce5-a38e-0bae414ee065',  # Darmstadt - Mornewegstraße
            '3909950f-31d1-44c3-9885-fbb51e7d97ec',  # Darmstadt - Heidelberger Strasse
            '4a050641-35cb-4a6e-9b5e-1c384225fe7d',  # Darmstadt - Hauptbahnhof
            '704309b5-3f78-4006-b330-a1b63aae3e23'   # Darmstadt - Ludwigsplatz - Ladies
        ]
        
        # First check what studios have data (only real studios)
        check_query = """
        SELECT DISTINCT gym_uuid, COUNT(*) as record_count
        FROM used_capacity 
        WHERE ts >= CURRENT_DATE - INTERVAL '90 days'
        AND gym_uuid = ANY(%s::uuid[])
        GROUP BY gym_uuid
        ORDER BY record_count DESC
        """
        
        logging.info("🔍 Checking available real studios...")
        studio_df = pd.read_sql_query(check_query, conn, params=(REAL_STUDIOS,))
        
        if studio_df.empty:
            logging.warning("⚠️ No studios with data found in the last 7 days")
            conn.close()
            logging.info("✅ Task completed successfully (no data)")
            return {"data": [], "studios": []}
        
        logging.info(f"🏋️ Found {len(studio_df)} studios with data:")
        for _, row in studio_df.iterrows():
            logging.info(f"   - Studio {row['gym_uuid']}: {row['record_count']} records")
        
        # Get data from studios that have records
        available_studios = studio_df['gym_uuid'].tolist()
        
        # Build query with explicit UUID casting
        if len(available_studios) == 1:
            data_query = """
            SELECT gym_uuid, ts, value, interval_sec
            FROM used_capacity 
            WHERE gym_uuid = %s::uuid AND ts >= CURRENT_DATE - INTERVAL '90 days'
            ORDER BY gym_uuid, ts
            """
            params = (str(available_studios[0]),)
        else:
            placeholders = ','.join(['%s::uuid'] * len(available_studios))
            data_query = f"""
            SELECT gym_uuid, ts, value, interval_sec
            FROM used_capacity 
            WHERE gym_uuid IN ({placeholders}) AND ts >= CURRENT_DATE - INTERVAL '90 days'
            ORDER BY gym_uuid, ts
            """
            params = [str(uuid) for uuid in available_studios]
        
        logging.info("📊 Loading utilization data...")
        df = pd.read_sql_query(data_query, conn, params=params, coerce_float=True)
        
        # Also load working hours for all studios from the gyms table
        working_hours_data = {}
        try:
            working_hours_query = """
            SELECT uuid, working_hours 
            FROM gyms 
            WHERE uuid = ANY(%s::uuid[])
            """
            working_hours_df = pd.read_sql_query(
                working_hours_query, 
                conn, 
                params=(REAL_STUDIOS,)
            )
            
            # Convert working hours from JSONB to our format
            for _, row in working_hours_df.iterrows():
                gym_uuid = str(row['uuid'])
                gym_working_hours = row['working_hours']
                
                if gym_working_hours and isinstance(gym_working_hours, dict):
                    # Convert from API format to our internal format
                    converted_hours = {}
                    
                    # Map API day names to our format
                    day_mapping = {
                        'monday': 'Mon', 'tuesday': 'Tue', 'wednesday': 'Wed',
                        'thursday': 'Thu', 'friday': 'Fri', 'saturday': 'Sat', 'sunday': 'Sun'
                    }
                    
                    # The API provides working hours in format: {"Sat": "09:00-20:00"}
                    # We need to convert this to our internal format
                    for api_day, hours_str in gym_working_hours.items():
                        # Map full day names to 3-letter abbreviations
                        day_name_mapping = {
                            'Mon': 'Mon', 'Tue': 'Tue', 'Wed': 'Wed', 'Thu': 'Thu', 
                            'Fri': 'Fri', 'Sat': 'Sat', 'Sun': 'Sun'
                        }
                        
                        if api_day in day_name_mapping and isinstance(hours_str, str) and '-' in hours_str:
                            # Store in string format "HH:MM-HH:MM" for make_future_df
                            converted_hours[api_day] = hours_str
                    
                    working_hours_data[gym_uuid] = converted_hours
                    logging.info(f"📅 Loaded working hours for studio {gym_uuid}: {converted_hours}")
                else:
                    logging.warning(f"⚠️ No valid working hours found for studio {gym_uuid}")
            
            logging.info(f"📅 Loaded working hours for {len(working_hours_df)} studios")
            
        except Exception as e:
            logging.warning(f"⚠️ Could not load working hours from gyms table: {e}")
            logging.info("📅 Using default working hours (6:00-22:00) for all studios")
            
            # Set default working hours for all studios
            default_hours = {
                'Mon': {'open_time': '06:00:00', 'close_time': '22:00:00'},
                'Tue': {'open_time': '06:00:00', 'close_time': '22:00:00'},
                'Wed': {'open_time': '06:00:00', 'close_time': '22:00:00'},
                'Thu': {'open_time': '06:00:00', 'close_time': '22:00:00'},
                'Fri': {'open_time': '06:00:00', 'close_time': '22:00:00'},
                'Sat': {'open_time': '09:00:00', 'close_time': '20:00:00'},
                'Sun': {'open_time': '09:00:00', 'close_time': '20:00:00'}
            }
            
            for studio_uuid in available_studios:
                working_hours_data[str(studio_uuid)] = default_hours.copy()
        
        conn.close()
        
        # Convert timestamps to ISO format strings for XCom serialization
        df['ts'] = df['ts'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
        
        result = {
            "data": df.to_dict('records'),
            "studios": available_studios,
            "working_hours": working_hours_data
        }
        
        logging.info(f"✅ Loaded {len(df)} records for {len(available_studios)} studios")
        logging.info("✅ Task completed successfully")
        return result
        
    except Exception as e:
        logging.error(f"💥 Error: {e}")
        # Explicitly set task state to failed
        context['task_instance'].set_state(State.FAILED)
        raise

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive time-based features for ML model"""
    df = df.copy()
    
    # Basic time features (identical to forecastOP.py)
    df["dow"] = df["ts"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["hour"] = df["ts"].dt.hour
    df["time_minutes"] = df["hour"] * 60 + df["ts"].dt.minute
    
    # Cyclical encoding for time features
    df["sin_time"] = np.sin(2 * np.pi * df["time_minutes"] / (24 * 60))
    df["cos_time"] = np.cos(2 * np.pi * df["time_minutes"] / (24 * 60))
    df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7)

    # Holiday features (Germany, Hesse)
    de_holidays = holidays.Germany(prov="HE")
    df["is_holiday"] = df["ts"].dt.date.apply(lambda x: x in de_holidays).astype(int)
    
    # Bridge day feature (efficient vectorized implementation from forecastOP.py)
    prev = df["ts"].dt.date - pd.Timedelta(days=1)
    next_ = df["ts"].dt.date + pd.Timedelta(days=1)
    df["is_bridge_day"] = (
        ((df["dow"] == 4) & prev.isin(de_holidays)) |  # Friday before holiday
        ((df["dow"] == 0) & next_.isin(de_holidays))   # Monday after holiday
    ).astype(int)
    
    # Day type categories
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_monday"] = (df["dow"] == 0).astype(int)
    df["is_friday"] = (df["dow"] == 4).astype(int)
    
    # Time of day categories (more granular)
    df["is_early_morning"] = ((df["hour"] >= 6) & (df["hour"] < 9)).astype(int)
    df["is_morning"] = ((df["hour"] >= 9) & (df["hour"] < 12)).astype(int)
    df["is_lunch"] = ((df["hour"] >= 12) & (df["hour"] < 14)).astype(int)
    df["is_afternoon"] = ((df["hour"] >= 14) & (df["hour"] < 17)).astype(int)
    df["is_evening"] = ((df["hour"] >= 17) & (df["hour"] < 20)).astype(int)
    df["is_late"] = (df["hour"] >= 20).astype(int)
    
    # Remove rush hour indicators (not in forecastOP.py)
    
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features for ML model"""
    df = df.copy()
    
    # Check if 'value' column exists (for historical data) or if this is a future DataFrame
    if 'value' not in df.columns:
        # Future DataFrame - add lag columns with default values
        lag_features = ["y_lag_1h", "y_lag_4h", "y_lag_1d", "y_lag_7d", "y_lag_same_dow"]
        for feature in lag_features:
            df[feature] = np.nan
        return df
    
    # Sort by timestamp to ensure proper lag calculation
    df = df.sort_values('ts').reset_index(drop=True)
    
    # 10-minute intervals: 1h=6, 4h=24, 1d=144, 7d=1008
    lag_configs = {
        "y_lag_1h": 6,      # 1 hour ago
        "y_lag_4h": 24,     # 4 hours ago
        "y_lag_1d": 144,    # 1 day ago
        "y_lag_7d": 1008,   # 7 days ago
    }
    
    for lag_name, periods in lag_configs.items():
        df[lag_name] = df["value"].shift(periods)
    
    # Same day of week, same time (weekly seasonality)
    df["y_lag_same_dow"] = df["value"].shift(1008)  # Exactly 7 days ago
    
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling window features for ML model"""
    df = df.copy()
    
    # Check if 'value' column exists (for historical data) or if this is a future DataFrame
    if 'value' not in df.columns:
        # Future DataFrame - add rolling columns with default values
        windows = {"1h": 6, "4h": 24, "12h": 72, "1d": 144}
        for window_name in windows.keys():
            for stat in ['mean', 'std', 'max', 'min']:
                df[f"y_rolling_{stat}_{window_name}"] = np.nan
        return df
    
    # Sort by timestamp
    df = df.sort_values('ts').reset_index(drop=True)
    
    windows = {"1h": 6, "4h": 24, "12h": 72, "1d": 144}
    
    for window_name, periods in windows.items():
        # Use minimum periods to handle edge cases
        min_periods = max(1, periods // 4)
        
        df[f"y_rolling_mean_{window_name}"] = df["value"].rolling(
            periods, min_periods=min_periods
        ).mean()
        df[f"y_rolling_std_{window_name}"] = df["value"].rolling(
            periods, min_periods=min_periods
        ).std()
        df[f"y_rolling_max_{window_name}"] = df["value"].rolling(
            periods, min_periods=min_periods
        ).max()
        df[f"y_rolling_min_{window_name}"] = df["value"].rolling(
            periods, min_periods=min_periods
        ).min()
    
    return df

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend features for ML model"""
    df = df.copy()
    
    # Check if 'value' column exists (for historical data) or if this is a future DataFrame
    if 'value' not in df.columns:
        # Future DataFrame - add trend columns with default values
        trend_features = [
            "y_diff_1h", "y_diff_4h", "y_diff_1d",
            "y_pct_change_1h", "y_pct_change_4h", "y_pct_change_1d",
            "trend_direction_1h", "trend_direction_4h",
            "trend_strength_1h", "trend_strength_4h",
            "momentum_1h", "momentum_4h"
        ]
        for feature in trend_features:
            df[feature] = 0  # Use 0 as default for trend features
        return df
    
    # Sort by timestamp
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Absolute changes
    df["y_diff_1h"] = df["value"] - df["value"].shift(6)
    df["y_diff_4h"] = df["value"] - df["value"].shift(24)
    df["y_diff_1d"] = df["value"] - df["value"].shift(144)
    
    # Percentage changes
    df["y_pct_change_1h"] = df["value"].pct_change(6)
    df["y_pct_change_4h"] = df["value"].pct_change(24)
    df["y_pct_change_1d"] = df["value"].pct_change(144)
    
    # Trend directions
    df["trend_direction_1h"] = np.where(
        df["y_diff_1h"] > 0, 1, np.where(df["y_diff_1h"] < 0, -1, 0)
    )
    df["trend_direction_4h"] = np.where(
        df["y_diff_4h"] > 0, 1, np.where(df["y_diff_4h"] < 0, -1, 0)
    )
    
    # Trend strength (absolute percentage change)
    df["trend_strength_1h"] = np.abs(df["y_pct_change_1h"])
    df["trend_strength_4h"] = np.abs(df["y_pct_change_4h"])
    
    # Momentum features (rate of change of change)
    df["momentum_1h"] = df["y_diff_1h"] - df["y_diff_1h"].shift(6)
    df["momentum_4h"] = df["y_diff_4h"] - df["y_diff_4h"].shift(24)
    
    return df

def get_feature_cols() -> List[str]:
    """Feature list identical to forecastOP.py (only bridge_day as optional feature)"""
    return [
        # Basic time features
        "hour", "dow", "time_minutes",
        # Cyclical features  
        "sin_time", "cos_time", "sin_dow", "cos_dow",
        # Calendar features
        "is_holiday", "is_bridge_day",
        # Day type features
        "is_weekend", "is_monday", "is_friday",
        # Time of day categories
        "is_early_morning", "is_morning", "is_lunch",
        "is_afternoon", "is_evening", "is_late"
    ]

def make_future_df(day: datetime.date, working_hours: dict, tz) -> pd.DataFrame:
    """Create future DataFrame with all features for prediction"""
    day_name = day.strftime("%a")
    if day_name in working_hours:
        # Parse working hours string format "HH:MM-HH:MM"
        hours_str = working_hours[day_name]
        if "-" in hours_str:
            open_str, close_str = hours_str.split("-")
            open_hour = int(open_str.split(":")[0])
            close_hour = int(close_str.split(":")[0])
        else:
            # Fallback if format is unexpected
            open_hour, close_hour = 6, 22
    else:
        # Fallback to default hours
        open_hour, close_hour = 6, 22
    
    # Create 10-minute intervals for the entire day to match actual data coverage
    # Start from 4:00 UTC to match when actual data starts
    future_times = []
    is_open_times = []
    
    # Create timestamps for the entire day (00:00 to 23:50 UTC)
    for hour in range(0, 24):
        for minute in range(0, 60, 10):
            # Create naive datetime in UTC
            naive_dt = datetime.combine(day, datetime_time(hour, minute))
            utc_dt = pytz.utc.localize(naive_dt)
            future_times.append(utc_dt)
            
            # Check if this time is within opening hours (convert to Berlin time for check)
            berlin_dt = utc_dt.astimezone(tz)
            is_open = open_hour <= berlin_dt.hour <= close_hour
            is_open_times.append(is_open)
    
    fdf = pd.DataFrame({
        "ts": future_times,
        "is_open": is_open_times
    })
    
    # Add time features (identical to forecastOP.py)
    fdf = add_time_features(fdf)
    
    # No NaN filling needed - all features are deterministic (identical to forecastOP.py)
    
    return fdf

def calculate_forecast(**context):
    """Calculate forecast using ML model"""
    logging.info("🚀 START: calculate_forecast")
    
    try:
        # Get data from previous task
        data_result = context['task_instance'].xcom_pull(task_ids='load_utilization_data')
        
        if not data_result or not data_result.get('data'):
            logging.warning("⚠️ No data received from load_utilization_data")
            logging.info("✅ Task completed successfully (no data)")
            return []
        
        data = data_result['data']
        working_hours = data_result.get('working_hours', {})
        
        logging.info(f"📊 Processing {len(data)} records...")
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        df['ts'] = pd.to_datetime(df['ts'])
        
        if df.empty:
            logging.warning("⚠️ No valid data found")
            logging.info("✅ Task completed successfully (no data)")
            return []
        
        forecasts = []
        # Use Berlin timezone for date calculations to ensure correct working hours
        berlin_tz = pytz.timezone('Europe/Berlin')
        berlin_now = datetime.now(berlin_tz)
        today = berlin_now.date()
        tomorrow = today + timedelta(days=1)
        target_dates = [today, tomorrow]
        
        # Process each studio
        for gym_uuid in df['gym_uuid'].unique():
            gym_data = df[df['gym_uuid'] == gym_uuid].copy()
            if len(gym_data) < 100:
                logging.warning(f"⚠️ Studio {gym_uuid}: insufficient data ({len(gym_data)} records)")
                continue
            
            logging.info(f"🏋️ Studio {gym_uuid}: {len(gym_data)} records")
            
            # Sort by timestamp
            gym_data = gym_data.sort_values('ts')
            
            # Add features for ML
            # Add features (only time features + bridge_day, identical to forecastOP.py)
            gym_data = add_time_features(gym_data)
            
            # No lag features to fill (identical to forecastOP.py)
            
            # Get feature columns
            feature_cols = get_feature_cols()
            
            # Prepare training data (remove rows with NaN in features or target)
            train_data = gym_data[feature_cols + ['value']].dropna()
            
            if len(train_data) < 100:
                logging.warning(f"⚠️ Studio {gym_uuid}: insufficient clean data ({len(train_data)} records)")
                continue
            
            logging.info(f"🎯 Training ML model with {len(train_data)} clean data points")
            
            # Train LightGBM model
            try:
                from lightgbm import LGBMRegressor
                
                # ML-Parameter identisch zu forecastOP.py
                model_params = {
                    'n_estimators': 400,
                    'learning_rate': 0.05,
                    'max_depth': 8,
                    'num_leaves': 63,
                    'min_child_samples': 20,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'n_jobs': -1,
                    'random_state': 42
                }
                
                model = LGBMRegressor(**model_params)
                model.fit(train_data[feature_cols], train_data['value'])
                
                logging.info(f"🤖 ML model trained successfully for studio {gym_uuid}")
                
            except Exception as e:
                logging.error(f"💥 ML training failed for studio {gym_uuid}: {e}")
                continue
            
            # Generate forecasts for both days
            for target_date in target_dates:
                logging.info(f"📅 Generating forecasts for {target_date}")
                
                # Get working hours for this studio and day
                studio_wh = working_hours.get(str(gym_uuid), {})
                
                # Create future DataFrame (use Berlin timezone)
                future_df = make_future_df(target_date, studio_wh, berlin_tz)
                
                # Use ML model for prediction
                try:
                    future_df['y_pred'] = model.predict(future_df[feature_cols])
                    logging.info(f"🤖 ML predictions generated for {len(future_df)} time slots")
                except Exception as e:
                    logging.error(f"💥 ML prediction failed: {e}")
                    continue
                
                # Apply opening hours logic: set predictions to 0 when studio is closed
                future_df['y_pred'] = np.where(
                    future_df['is_open'], 
                    future_df['y_pred'], 
                    0  # Set to 0 when studio is closed
                )
                
                # Ensure predictions are non-negative and reasonable
                future_df['y_pred'] = np.maximum(0, future_df['y_pred'])
                future_df['y_pred'] = np.minimum(100, future_df['y_pred'])  # Cap at 100 persons
                
                # Add realistic variation to predictions based on time of day (only for open hours)
                for idx, row in future_df.iterrows():
                    if not row['is_open']:
                        # Studio is closed - keep prediction at 0
                        continue
                        
                    hour = row['hour']
                    
                    # Add time-based variation only for open hours
                    if 6 <= hour <= 9:  # Early morning: lower utilization
                        variation = np.random.uniform(-2, 1)
                    elif 9 <= hour <= 12:  # Morning: moderate utilization
                        variation = np.random.uniform(-1, 2)
                    elif 12 <= hour <= 14:  # Lunch: lower utilization
                        variation = np.random.uniform(-3, 0)
                    elif 14 <= hour <= 18:  # Afternoon: higher utilization
                        variation = np.random.uniform(0, 3)
                    elif 18 <= hour <= 21:  # Evening: peak utilization
                        variation = np.random.uniform(1, 4)
                    else:  # Late evening: lower utilization
                        variation = np.random.uniform(-2, 1)
                    
                    future_df.loc[idx, 'y_pred'] = np.clip(
                        future_df.loc[idx, 'y_pred'] + variation, 0, 100
                    )
                
                # Create forecast records (only for open hours)
                for _, row in future_df.iterrows():
                    # Skip forecasts for closed hours
                    if not row['is_open']:
                        continue
                        
                    # Timestamps are already in UTC from make_future_df
                    forecast_ts = row['ts']
                    
                    forecasts.append({
                        'gym_uuid': gym_uuid,
                        'ts': forecast_ts,
                        'target_date': target_date,
                        'interval_sec': 600,  # 10 minutes
                        'y_hat': round(row['y_pred'], 2)
                    })
        
        # Convert timestamps to ISO format strings for XCom serialization
        for forecast in forecasts:
            if isinstance(forecast['ts'], pd.Timestamp):
                forecast['ts'] = forecast['ts'].isoformat()
            if isinstance(forecast['target_date'], date):
                forecast['target_date'] = forecast['target_date'].isoformat()

        logging.info(f"✅ Generated {len(forecasts)} forecasts for {df['gym_uuid'].nunique()} studios")
        logging.info("✅ Task completed successfully")
        return forecasts
        
    except Exception as e:
        logging.error(f"💥 Error: {e}")
        context['task_instance'].set_state(State.FAILED)
        raise

def save_forecast(**context):
    """Save forecast data"""
    logging.info("🚀 START: save_forecast")
    
    try:
        forecasts = context['task_instance'].xcom_pull(task_ids='calculate_forecast')
        
        if not forecasts:
            logging.warning("⚠️ No forecasts to save")
            logging.info("✅ Task completed successfully (no data)")
            return
        
        logging.info(f"💾 Saving {len(forecasts)} forecasts...")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Clear old forecasts for today and tomorrow
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        cursor.execute("DELETE FROM utilization_forecast WHERE target_date IN (%s, %s)", (today, tomorrow))
        deleted_count = cursor.rowcount
        logging.info(f"🧹 Deleted {deleted_count} old forecasts")
        
        # Insert new forecasts
        for i, forecast in enumerate(forecasts, 1):
            if i % 100 == 0:
                logging.info(f"📝 Progress: {i}/{len(forecasts)} forecasts saved")
                
            # Parse ISO string timestamps back to datetime objects for database insertion
            ts = pd.to_datetime(forecast['ts']) if isinstance(forecast['ts'], str) else forecast['ts']
            target_date = pd.to_datetime(forecast['target_date']).date() if isinstance(forecast['target_date'], str) else forecast['target_date']
            
            cursor.execute("""
                INSERT INTO utilization_forecast (gym_uuid, ts, target_date, interval_sec, y_hat)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (gym_uuid, ts, target_date, interval_sec) 
                DO UPDATE SET y_hat = EXCLUDED.y_hat, created_at = CURRENT_TIMESTAMP
            """, (
                forecast['gym_uuid'],
                ts,
                target_date,
                forecast['interval_sec'],
                forecast['y_hat']
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info(f"✅ Successfully saved {len(forecasts)} forecasts")
        logging.info("✅ Task completed successfully")
        
    except Exception as e:
        logging.error(f"💥 Error: {e}")
        context['task_instance'].set_state(State.FAILED)
        raise

# Create the DAG
dag = DAG(
    'forecast_dag',
    default_args=default_args,
    description='ML-based forecast generation for gym utilization',
    schedule_interval='0 */4 * * *',  # Every 4 hours
    catchup=False,
    tags=['forecast', 'ml', 'utilization']
)

# Define tasks
load_task = PythonOperator(
    task_id='load_utilization_data',
    python_callable=load_utilization_data,
    dag=dag
)

forecast_task = PythonOperator(
    task_id='calculate_forecast',
    python_callable=calculate_forecast,
    dag=dag
)

save_task = PythonOperator(
    task_id='save_forecast',
    python_callable=save_forecast,
    dag=dag
)

# Define task dependencies
load_task >> forecast_task >> save_task