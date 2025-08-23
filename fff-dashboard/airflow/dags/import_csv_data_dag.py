from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
import psycopg2
import logging
import os

# Default arguments for the DAG
default_args = {
    'owner': 'fff-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
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

def import_csv_data(**context):
    """Import data from data.csv into the database"""
    try:
        # Studio UUID for Darmstadt - Ostbahnhof
        studio_uuid = '1b793462-e413-49fb-b971-ada1e11dc90e'
        csv_path = '/opt/airflow/dags/data.csv'
        
        logging.info(f"📥 Starting CSV import for studio {studio_uuid}")
        
        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        
        # Read CSV file
        logging.info(f"📊 Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Check columns
        expected_columns = ['time', 'Personen.value']
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"Expected columns {expected_columns}, got {df.columns.tolist()}")
        
        logging.info(f"📊 Loaded {len(df)} rows from CSV")
        
        # Clean and prepare data
        df = df.rename(columns={'Personen.value': 'value'})
        df = df.dropna()  # Remove rows with missing values
        
        # Parse timestamps
        df['ts'] = pd.to_datetime(df['time'])
        df = df.drop('time', axis=1)
        
        # Add studio UUID
        df['gym_uuid'] = studio_uuid
        
        # Convert to 10-minute intervals by grouping
        df['ts_10min'] = df['ts'].dt.floor('10min')
        df_aggregated = df.groupby(['gym_uuid', 'ts_10min']).agg({
            'value': 'mean'  # Take mean of values in each 10-minute window
        }).reset_index()
        df_aggregated = df_aggregated.rename(columns={'ts_10min': 'ts'})
        
        logging.info(f"📊 Aggregated to {len(df_aggregated)} 10-minute intervals")
        
        # Connect to database
        logging.info("🔌 Connecting to database...")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Clear existing data for this studio
        logging.info(f"🗑️ Clearing existing data for studio {studio_uuid}")
        cursor.execute("""
            DELETE FROM used_capacity 
            WHERE gym_uuid = %s::uuid AND ts >= %s AND ts <= %s
        """, (studio_uuid, df_aggregated['ts'].min(), df_aggregated['ts'].max()))
        
        deleted_rows = cursor.rowcount
        logging.info(f"🗑️ Deleted {deleted_rows} existing rows")
        
        # Insert new data
        logging.info("💾 Inserting new data...")
        insert_query = """
            INSERT INTO used_capacity (gym_uuid, ts, value, interval_sec)
            VALUES (%s::uuid, %s, %s, 600)
        """
        
        # Prepare data for insertion
        insert_data = [
            (row['gym_uuid'], row['ts'], float(row['value']))
            for _, row in df_aggregated.iterrows()
        ]
        
        cursor.executemany(insert_query, insert_data)
        inserted_rows = cursor.rowcount
        
        # Commit changes
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info(f"✅ Successfully imported {inserted_rows} records for studio {studio_uuid}")
        logging.info(f"📅 Data range: {df_aggregated['ts'].min()} to {df_aggregated['ts'].max()}")
        
        # Return summary
        return {
            'studio_uuid': studio_uuid,
            'total_csv_rows': len(df),
            'aggregated_intervals': len(df_aggregated),
            'inserted_rows': inserted_rows,
            'deleted_rows': deleted_rows,
            'date_range': {
                'start': df_aggregated['ts'].min().isoformat(),
                'end': df_aggregated['ts'].max().isoformat()
            }
        }
        
    except Exception as e:
        logging.error(f"💥 CSV import failed: {e}")
        raise

def verify_import(**context):
    """Verify the imported data"""
    try:
        studio_uuid = '1b793462-e413-49fb-b971-ada1e11dc90e'
        
        logging.info(f"🔍 Verifying import for studio {studio_uuid}")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Count total records
        cursor.execute("""
            SELECT COUNT(*), MIN(ts), MAX(ts), AVG(value)
            FROM used_capacity 
            WHERE gym_uuid = %s::uuid
        """, (studio_uuid,))
        
        count, min_date, max_date, avg_persons = cursor.fetchone()
        
        # Get sample records
        cursor.execute("""
            SELECT ts, value
            FROM used_capacity 
            WHERE gym_uuid = %s::uuid
            ORDER BY ts
            LIMIT 5
        """, (studio_uuid,))
        
        sample_records = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        logging.info(f"✅ Verification complete:")
        logging.info(f"   📊 Total records: {count}")
        logging.info(f"   📅 Date range: {min_date} to {max_date}")
        logging.info(f"   👥 Average persons: {avg_persons:.2f}")
        logging.info(f"   📝 Sample records: {sample_records}")
        
        return {
            'total_records': count,
            'date_range': {'start': min_date.isoformat(), 'end': max_date.isoformat()},
            'average_persons': float(avg_persons) if avg_persons else 0,
            'sample_records': [(r[0].isoformat(), float(r[1])) for r in sample_records]
        }
        
    except Exception as e:
        logging.error(f"💥 Verification failed: {e}")
        raise

# Create DAG
dag = DAG(
    'import_csv_data',
    default_args=default_args,
    description='Import historical data from data.csv for Darmstadt Ostbahnhof studio',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['import', 'historical-data', 'csv']
)

# Define tasks
import_task = PythonOperator(
    task_id='import_csv_data',
    python_callable=import_csv_data,
    dag=dag,
)

verify_task = PythonOperator(
    task_id='verify_import',
    python_callable=verify_import,
    dag=dag,
)

# Set task dependencies
import_task >> verify_task
