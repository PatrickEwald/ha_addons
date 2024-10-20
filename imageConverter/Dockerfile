FROM alpine:3.18

# Install FFmpeg
RUN apk add --no-cache ffmpeg bash

# Kopiere das Startskript
COPY run.sh /run.sh

# Setze die Ausführungsrechte für das Skript
RUN chmod a+x /run.sh

# Setze das Skript als Entrypoint
CMD ["/run.sh"]