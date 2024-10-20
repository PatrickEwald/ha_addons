ARG BUILD_FROM
FROM $BUILD_FROM

# Installiere die erforderlichen Pakete für das Add-on
RUN \
  apk add --no-cache \
    python3 \
    python3-dev \
    py3-pip \
    ffmpeg && \
  pip3 install --no-cache-dir Pillow

# Setze das Arbeitsverzeichnis auf das persistente Datenverzeichnis des Add-ons
WORKDIR /data

# Kopiere die Daten für das Add-on
COPY run.sh /
RUN chmod a+x /run.sh

# Führe das Skript aus
CMD [ "/run.sh" ]