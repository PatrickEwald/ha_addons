FROM python:3.9-alpine

# Installiere FFmpeg und Pillow-Abhängigkeiten
RUN apk add --no-cache \
    ffmpeg \
    zlib-dev \
    libjpeg-turbo-dev \
    freetype-dev \
    lcms2-dev \
    openjpeg-dev \
    tiff-dev \
    tk-dev

# Installiere Pillow
RUN pip install --no-cache-dir Pillow

# Setze das Arbeitsverzeichnis
WORKDIR /data

# Kopiere die Daten für das Add-on
COPY run.sh /run.sh
RUN chmod a+x /run.sh

# Führe das Skript aus
CMD [ "/run.sh" ]