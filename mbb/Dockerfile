# Verwende das Home Assistant Base-Image für die entsprechende Architektur
FROM ghcr.io/home-assistant/amd64-base:latest

RUN apk add --no-cache samba samba-common-tools avahi jq

COPY rootfs/ /

RUN chmod +x /usr/local/bin/entrypoint.sh
RUN chmod +x /etc/services.d/timemachine/run