#!/bin/bash

set -x

USERNAME=$(jq -r '.username' /data/options.json)
PASSWORD=$(jq -r '.password' /data/options.json)

# Benutzer anlegen
adduser -D -H -s /bin/false "$USERNAME"
echo "$USERNAME:$PASSWORD" | chpasswd

# Ordner sicherstellen
mkdir -p /share/timemachine
chown "$USERNAME:$USERNAME" /share/timemachine

# Avahi starten (für Bonjour/Time Machine Detection)
avahi-daemon --daemonize

# Samba starten
smbd --foreground --no-process-group &

tail -f /dev/null