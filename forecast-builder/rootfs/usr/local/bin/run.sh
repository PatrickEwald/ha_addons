#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'        

readonly NIGHT_INTERVAL=3600

DAY_INTERVAL=$(jq -re '.interval // 300' /data/options.json)
[[ $DAY_INTERVAL =~ ^[0-9]+$ ]] || DAY_INTERVAL=300

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

log "Starte Forecast Runner – Tagintervall ${DAY_INTERVAL}s, Nachtintervall ${NIGHT_INTERVAL}s"

trap 'log "⏹️  Beende Skript"; exit 0' SIGINT SIGTERM

while :; do
    current_hour=$((10#$(date +%H))) 

    if (( current_hour >= 6 && current_hour < 22 )); then
        DAY_INTERVAL=$(jq -re '.interval // 300' /data/options.json)
        [[ $DAY_INTERVAL =~ ^[0-9]+$ ]] || DAY_INTERVAL=300
        INTERVAL=$DAY_INTERVAL
        log "🌞 Tagbetrieb – starte Forecast (Intervall ${INTERVAL}s)"
    else
        INTERVAL=$NIGHT_INTERVAL
        log "🌙 Nachtbetrieb – starte Forecast (Intervall ${INTERVAL}s)"
    fi

    if python3 -u /opt/forecast/forecastOP.py; then
        log "✅ Forecast fertig – warte ${INTERVAL}s"
    else
        rc=$?
        log "❌ Forecast fehlgeschlagen (Exit-Code $rc)"
    fi

    sleep "$INTERVAL" & wait $!
done