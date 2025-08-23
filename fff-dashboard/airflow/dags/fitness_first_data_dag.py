from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import json
from typing import Dict, List, Any

# Default arguments for the DAG
default_args = {
    'owner': 'fff-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=1),
}

def fetch_fitness_first_data(**context):
    """Fetch data from Fitness First API"""
    try:
        # API configuration
        url = "https://fitnessfirst.netpulse.com/np/locations/v1.0/gym-chains/31f47b4e-781e-402f-9697-d4c0243c50bd/location-details"
        cookies = {
            'JSESSIONID': 'YTIzYTkzNDUtYmM0MS00YWY3LWFiOTAtODQxNjdlOGIzMDk2'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'de-DE,de;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://fitnessfirst.netpulse.com/',
        }
        
        logging.info(f"Fetching data from: {url}")
        
        # Make the request
        response = requests.get(url, cookies=cookies, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        logging.info(f"Successfully fetched data. Response status: {response.status_code}")
        logging.info(f"Data structure: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        return data
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise

def parse_and_save_data(**context):
    """Parse the API response and save to database - ONLY Darmstadt studios"""
    try:
        # Get data from previous task
        api_data = context['task_instance'].xcom_pull(task_ids='fetch_fitness_first_data')
        
        if not api_data:
            logging.warning("No API data received")
            return
        
        logging.info(f"Processing API data: {type(api_data)}")
        
        # Database connection
        conn = psycopg2.connect(
            host="postgres",
            database="ffdb",
            user="ffuser",
            password="ffpass",
            port="5432"
        )
        
        cursor = conn.cursor()
        
        # Get current timestamp
        current_ts = datetime.now()
        
        # Parse the data structure (adjust based on actual API response)
        if isinstance(api_data, dict) and 'locations' in api_data:
            locations = api_data['locations']
        elif isinstance(api_data, list):
            locations = api_data
        else:
            # If we don't know the structure, log it and try to process anyway
            logging.warning(f"Unknown API data structure: {type(api_data)}")
            locations = [api_data] if not isinstance(api_data, list) else api_data
        
        logging.info(f"Processing {len(locations)} total locations")
        
        # Filter for Darmstadt studios only
        darmstadt_locations = []
        for location in locations:
            try:
                location_name = location.get('name') or location.get('locationName') or ''
                location_city = location.get('city') or location.get('address', {}).get('city') or ''
                
                # Check if location is in Darmstadt (case-insensitive)
                if ('darmstadt' in location_name.lower() or 
                    'darmstadt' in location_city.lower() or
                    'darms' in location_name.lower()):
                    
                    darmstadt_locations.append(location)
                    logging.info(f"Found Darmstadt studio: {location_name}")
                    
            except Exception as e:
                logging.warning(f"Error checking location {location}: {e}")
                continue
        
        logging.info(f"Filtered to {len(darmstadt_locations)} Darmstadt locations")
        
        # Process only Darmstadt locations
        for location in darmstadt_locations:
            try:
                # Extract location information
                location_id = location.get('id') or location.get('locationId') or location.get('uuid')
                location_name = location.get('name') or location.get('locationName') or 'Unknown'
                
                if not location_id:
                    logging.warning(f"No ID found for location: {location}")
                    continue
                
                # Log the first location structure for debugging
                if location_name == darmstadt_locations[0].get('name') or location_name == darmstadt_locations[0].get('locationName'):
                    logging.info(f"DEBUG: First Darmstadt location structure: {json.dumps(location, indent=2, default=str)}")
                
                # Extract capacity and utilization data
                utilization = location.get('utilization', {})
                total_capacity = utilization.get('totalCapacity') or 0
                used_capacity = utilization.get('usedCapacity') or 0
                
                # Extract working hours if available
                working_hours = location.get('workingHours') or location.get('hours') or {}
                
                # Ensure values are numbers
                try:
                    total_capacity = int(total_capacity) if total_capacity else 0
                    used_capacity = int(used_capacity) if used_capacity else 0
                except (ValueError, TypeError):
                    total_capacity = 0
                    used_capacity = 0
                
                logging.info(f"Processing studio: {location_name} - Total: {total_capacity}, Used: {used_capacity}")
                
                # Check if gym exists, if not create it
                cursor.execute(
                    "SELECT uuid FROM gyms WHERE uuid = %s",
                    (location_id,)
                )
                
                existing_gym = cursor.fetchone()
                
                if not existing_gym:
                    # Create new gym with complete information
                    cursor.execute("""
                        INSERT INTO gyms (uuid, name, timezone, working_hours, total_capacity)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (uuid) DO NOTHING
                    """, (
                        location_id,
                        location_name,
                        'Europe/Berlin',  # Darmstadt timezone
                        json.dumps(working_hours),
                        total_capacity     # total_capacity
                    ))
                    logging.info(f"Created new Darmstadt gym: {location_name} ({location_id}) with capacity {total_capacity}")
                else:
                    # Update existing gym with new capacity and working hours
                    cursor.execute("""
                        UPDATE gyms 
                        SET total_capacity = %s, working_hours = %s, updated_at = CURRENT_TIMESTAMP
                        WHERE uuid = %s
                    """, (total_capacity, json.dumps(working_hours), location_id))
                    logging.info(f"Updated existing gym: {location_name} with capacity {total_capacity}")
                
                # Save utilization data (usedCapacity as absolute number of people)
                cursor.execute("""
                    INSERT INTO used_capacity (gym_uuid, ts, interval_sec, value)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (gym_uuid, ts, interval_sec) 
                    DO UPDATE SET value = EXCLUDED.value, created_at = CURRENT_TIMESTAMP
                """, (
                    location_id,
                    current_ts,
                    300,  # 5 minutes interval
                    used_capacity  # Absolute number of people, not percentage
                ))
                
                logging.info(f"Saved utilization data for Darmstadt studio {location_name}: {used_capacity} people (capacity: {total_capacity})")
                
            except Exception as e:
                logging.error(f"Error processing Darmstadt location {location}: {e}")
                continue
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logging.info(f"Successfully processed and saved {len(darmstadt_locations)} Darmstadt studio records")
        
    except Exception as e:
        logging.error(f"Error parsing and saving data: {e}")
        raise

def log_data_summary(**context):
    """Log a summary of the data processing"""
    try:
        # Get data from previous task
        api_data = context['task_instance'].xcom_pull(task_ids='fetch_fitness_first_data')
        
        if api_data:
            logging.info(f"Data processing completed successfully at {datetime.now()}")
            
            # Log some statistics if available
            if isinstance(api_data, dict) and 'locations' in api_data:
                locations_count = len(api_data['locations'])
                logging.info(f"Processed {locations_count} locations")
            elif isinstance(api_data, list):
                logging.info(f"Processed {len(api_data)} data items")
        else:
            logging.warning("No data was processed in this run")
            
    except Exception as e:
        logging.error(f"Error logging summary: {e}")

# Create the DAG
dag = DAG(
    'fitness_first_data_dag',
    default_args=default_args,
    description='Fetch Fitness First utilization data for Darmstadt studios every 5 minutes',
    schedule_interval='*/5 * * * *',  # Every 5 minutes
    catchup=False,
    tags=['fitness-first', 'data-collection', 'utilization', 'darmstadt']
)

# Define tasks
fetch_task = PythonOperator(
    task_id='fetch_fitness_first_data',
    python_callable=fetch_fitness_first_data,
    dag=dag
)

parse_task = PythonOperator(
    task_id='parse_and_save_data',
    python_callable=parse_and_save_data,
    dag=dag
)

log_task = PythonOperator(
    task_id='log_data_summary',
    python_callable=log_data_summary,
    dag=dag
)

# Define task dependencies
fetch_task >> parse_task >> log_task
