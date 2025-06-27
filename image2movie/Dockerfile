# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile
FROM ghcr.io/home-assistant/amd64-base:latest

# Installiere FFmpeg
RUN apk add --no-cache ffmpeg

# Copy root filesystem
COPY rootfs /
