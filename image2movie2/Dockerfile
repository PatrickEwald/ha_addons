# Verwende das Home Assistant Base-Image für die entsprechende Architektur
FROM ghcr.io/home-assistant/amd64-base:latest

# Installiere FFmpeg und Python3
RUN apk add --no-cache ffmpeg python3

# Copy root filesystem (falls du zusätzliche Dateien ins Dateisystem kopieren möchtest)
COPY rootfs /