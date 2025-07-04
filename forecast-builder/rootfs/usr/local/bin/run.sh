#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n'

readonly NIGHT_INTERVAL=3600
readonly DAY_START_HOUR=6
readonly DAY_END_HOUR=22

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
    if (( 10#$current_hour >= DAY_START_HOUR && 10#$current_hour < DAY_END_HOUR )); then
        INTERVAL=$(get_day_interval)
        log "🌞 generiere Forecast (Intervall ${INTERVAL}s)"
        if python3 -u /opt/forecast/forecastOP.py; then
            log "✅ Forecast fertig – warte ${INTERVAL}s"
        else
            rc=$?
            log "❌ Forecast fehlgeschlagen (Exit-Code $rc)"
        fi
    else
        now_ts=$(date +%s)

        if (( 10#$current_hour < DAY_START_HOUR )); then
            target_ts=$(date -d "$(date +%F) ${DAY_START_HOUR}:00" +%s)
        else
            target_ts=$(date -d "$(date -d 'tomorrow' +%F) ${DAY_START_HOUR}:00" +%s)
        fi

        INTERVAL=$(( target_ts - now_ts ))
        
        if (( INTERVAL <= 0 )); then
            INTERVAL=$NIGHT_INTERVAL
        fi

        log "🌙 Nachtbetrieb – kein Forecast, warte ${INTERVAL}s - ${target_ts}"
    fi

    sleep "$INTERVAL" & wait $!
done