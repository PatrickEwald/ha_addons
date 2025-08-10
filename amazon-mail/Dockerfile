# Verwende das Home Assistant Base-Image für die entsprechende Architektur
FROM ghcr.io/home-assistant/amd64-base:latest

RUN apk add --no-cache python3 py3-pip
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY app /app
RUN pip install --no-cache-dir -r /app/requirements.txt