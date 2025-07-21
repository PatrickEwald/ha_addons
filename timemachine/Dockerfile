ARG BUILD_FROM
FROM $BUILD_FROM

ENV LANG="C.UTF-8"

# Installiere benötigte Pakete
RUN apk add --no-cache samba samba-common-tools avahi jq

# Kopiere Startskript und Samba-Konfiguration
COPY run.sh /run.sh
COPY smb.conf /etc/samba/smb.conf

RUN chmod +x /run.sh

CMD [ "/run.sh" ]