# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile
FROM ghcr.io/home-assistant/amd64-base:16.2.3

RUN apk add --no-cache ffmpeg

# Copy root filesystem
COPY rootfs /
