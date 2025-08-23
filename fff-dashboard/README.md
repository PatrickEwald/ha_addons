# FFF Fitness Dashboard - Home Assistant Addon

Ein Custom Addon für Home Assistant OS, das ein Fitness Studio Auslastungs-Dashboard mit ML-Vorhersagen bereitstellt.

## Features

- **Dashboard Frontend**: Nginx-basiertes Web-Interface auf Port 3003
- **FastAPI Backend**: REST-API für Datenverwaltung auf Port 8000
- **Airflow**: Workflow-Management für ML-Pipelines auf Port 8080
- **PostgreSQL**: Datenbank für alle Anwendungsdaten
- **Automatische Updates**: Daten werden alle 5 Minuten aktualisiert

## Installation in Home Assistant OS

### 1. Repository hinzufügen

1. Gehe zu **Settings** → **Add-ons** → **Add-on Store**
2. Klicke auf die drei Punkte oben rechts → **Repositories**
3. Füge folgende URL hinzu:
   ```
   https://github.com/[DEIN_USERNAME]/FFF-Docker
   ```
4. Klicke auf **Add**

### 2. Addon installieren

1. Suche nach **"FFF Fitness Dashboard"** im Add-on Store
2. Klicke auf **FFF Fitness Dashboard**
3. Klicke auf **Install**
4. Warte bis die Installation abgeschlossen ist

### 3. Konfiguration

1. Klicke auf **Configuration**
2. Überprüfe die Standardwerte:
   - **Dashboard Port**: `3003` (Dashboard Frontend)
   - **FastAPI Port**: `8001` (Backend API)
   - **Airflow Port**: `8081` (Workflow-Management)
   - **PostgreSQL Host**: `postgres`
   - **Datenbank**: `ffdb`
   - **Benutzer**: `ffuser`
   - **Passwort**: `ffpass`
3. **Wichtig**: Falls Port-Konflikte auftreten, ändere die Ports
4. Klicke auf **Save**

### 4. Addon starten

1. Klicke auf **Start**
2. Warte bis alle Services gestartet sind
3. Das Dashboard ist verfügbar unter: `http://[HA-IP]:3003`

## Verwendung

### Dashboard
- **URL**: `http://[HA-IP]:{{ dashboard_port }}`
- **Port**: Konfigurierbar (Standard: 3003)
- **Features**: Auslastungsanzeige, Tagesverlauf, Vorhersagen

### API
- **URL**: `http://[HA-IP]:{{ fastapi_port }}`
- **Port**: Konfigurierbar (Standard: 8001)
- **Endpoints**: `/api/v1/gyms`, `/api/v1/gyms/{id}/timeseries`

### Airflow
- **URL**: `http://[HA-IP]:{{ airflow_port }}`
- **Port**: Konfigurierbar (Standard: 8081)
- **Features**: DAGs verwalten, ML-Pipelines überwachen

## Ports

- **{{ dashboard_port }}**: Dashboard Frontend (Hauptzugang)
- **{{ fastapi_port }}**: FastAPI Backend (API)
- **{{ airflow_port }}**: Airflow Web UI (Workflow-Management)

**Hinweis**: Alle Ports sind konfigurierbar und können bei Konflikten geändert werden.

## Verzeichnisstruktur

```
fff-fitness-dashboard/
├── config.yaml              # Addon-Konfiguration
├── docker-compose.yml       # Service-Definitionen
├── Dockerfile.dashboard     # Dashboard-Container
├── Dockerfile.fastapi       # FastAPI-Container
├── Dockerfile.airflow       # Airflow-Container
├── nginx.conf               # Nginx-Konfiguration
├── dashboard/               # Frontend-Dateien
├── fastapi/                 # Backend-Code
├── airflow/                 # ML-Workflows und DAGs
└── README.md                # Diese Datei
```

## Troubleshooting

### Dashboard nicht erreichbar
1. Überprüfe, ob der Addon läuft
2. Prüfe die Logs im Addon
3. Stelle sicher, dass Port 3003 nicht von anderen Services belegt ist

### Datenbank-Verbindungsfehler
1. Überprüfe die PostgreSQL-Konfiguration
2. Stelle sicher, dass der PostgreSQL-Service läuft
3. Prüfe die Logs des FastAPI-Services

### Port-Konflikte
Falls Ports bereits belegt sind, ändere sie in der Addon-Konfiguration:

1. **Dashboard Port-Konflikt** (Standard: 3003):
   - Ändere auf z.B. 3004, 3005, 3010

2. **FastAPI Port-Konflikt** (Standard: 8001):
   - Ändere auf z.B. 8002, 8003, 8010

3. **Airflow Port-Konflikt** (Standard: 8081):
   - Ändere auf z.B. 8082, 8083, 8090

**Häufige Konflikte:**
- Port 8000: Home Assistant Supervisor
- Port 8080: Zigbee2MQTT, andere Addons
- Port 3000-3010: Andere Custom Addons

## Support

Bei Problemen oder Fragen:
1. Überprüfe die Addon-Logs
2. Stelle sicher, dass alle Services laufen
3. Überprüfe die Netzwerk-Konfiguration

## Lizenz

Dieses Projekt ist für den privaten Gebrauch bestimmt.
