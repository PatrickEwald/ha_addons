#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n'

readonly NIGHT_INTERVAL=3600

log() { printf '%s %s\n' "$(date '+%F %T')" "$*"; }

get_day_interval() {
    local val
    val=$(jq -er '.interval // 300' /data/options.json 2>/dev/null || echo 300)
    [[ $val =~ ^[0-9]+$ ]] || val=300
    echo "$val"
}

trap 'log "⏹️  Beende Skript"; kill 0' SIGINT SIGTERM

while true; do
    current_hour=$(date +%H)
    if (( 10#$current_hour >= 6 && 10#$current_hour < 22 )); then
        INTERVAL=$(get_day_interval)
        log "🌞 generiere Forecast (Intervall ${INTERVAL}s)"
        if python3 -u /opt/forecast/forecastOP.py; then
            log "✅ Forecast fertig – warte ${INTERVAL}s"
        else
            rc=$?
            log "❌ Forecast fehlgeschlagen (Exit-Code $rc)"
        fi
    else
        INTERVAL=$NIGHT_INTERVAL
        log "🌙 Nachtbetrieb – kein Forecast, warte ${INTERVAL}s"
    fi

    sleep "$INTERVAL" & wait $!
done