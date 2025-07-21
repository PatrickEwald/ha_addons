ARG BUILD_FROM
FROM $BUILD_FROM

ENV LANG="C.UTF-8"

# Install Pakete
RUN apk add --no-cache samba samba-common-tools avahi jq

# Kopiere rootfs
COPY rootfs/ /

RUN chmod +x /usr/local/bin/entrypoint.sh
RUN chmod +x /etc/services.d/timemachine/run