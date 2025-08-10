# Verwende das Home Assistant Base-Image für die entsprechende Architektur
FROM ghcr.io/home-assistant/amd64-base:latest

# Python installieren
RUN apk add --no-cache python3 py3-pip

# App-Dateien
COPY app /app
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# s6-Service
COPY rootfs/ /

# Ausführbar machen
RUN chmod +x /etc/services.d/amazon/run