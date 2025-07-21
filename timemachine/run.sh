#!/bin/sh

USERNAME=$(jq -r '.username' /data/options.json)
PASSWORD=$(jq -r '.password' /data/options.json)

adduser -D -H -s /bin/false "$USERNAME"
echo "$USERNAME:$PASSWORD" | chpasswd

mkdir -p /share/timemachine
chown "$USERNAME":"$USERNAME" /share/timemachine

# Avahi für Bonjour-Zugriff (Time Machine erkennt das automatisch)
avahi-daemon --daemonize

# Start SMB
smbd --foreground --no-process-group