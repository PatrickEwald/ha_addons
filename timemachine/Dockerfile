FROM ghcr.io/hassio-addons/base:14.1.1

ENV LANG="C.UTF-8"

RUN apk add --no-cache samba samba-common-tools avahi jq

COPY rootfs/ /

RUN chmod +x /usr/local/bin/entrypoint.sh
RUN chmod +x /etc/services.d/timemachine/run