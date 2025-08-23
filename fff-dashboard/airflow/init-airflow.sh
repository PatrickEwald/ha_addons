#!/bin/bash
set -e

echo "Starting Airflow initialization..."

echo "Initializing database..."
airflow db init

echo "Creating admin user..."
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin

echo "Admin user created successfully!"
echo "Airflow initialization completed!"
