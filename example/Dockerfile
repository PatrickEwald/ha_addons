# https://developers.home-assistant.io/docs/add-ons/configuration#add-on-dockerfile
ARG BUILD_FROM
FROM $BUILD_FROM

RUN apk add --no-cache samba samba-common-tools avahi jq

COPY rootfs/ /

RUN chmod +x /usr/bin/my_program
RUN chmod +x /etc/services.d/example/run
