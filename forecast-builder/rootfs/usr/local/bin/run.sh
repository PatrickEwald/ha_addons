#!/bin/bash

INTERVAL=$(jq --raw-output '.interval // 300' /data/options.json)
echo "Starte Forecast Runner mit Intervall ${INTERVAL} Sekunden"

while true; do
    current_hour=$(date +%H)
    if [ "$current_hour" -ge 6 ] && [ "$current_hour" -lt 22 ]; then
        echo "Starte Forecast-Run $(date)"
        python3 -u /opt/forecast/forecastOP.py
        echo "✅ Fertig – Warte ${INTERVAL}s"
    else
        echo "Außerhalb der Laufzeit (6-22 Uhr). Warte ${INTERVAL}s"
    fi
    sleep $INTERVAL
    INTERVAL=$(jq --raw-output '.interval // 300' /data/options.json)
    INTERVAL=${INTERVAL:-300}
done