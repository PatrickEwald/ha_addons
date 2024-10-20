#!/bin/sh

# Debug-Ausgabe aktivieren
set -x  # Zeigt alle ausgeführten Befehle an
set -e  # Beende das Skript bei Fehlern

# Konvertiere JPEGs in WebP und resize auf 800x600
echo "Starte die Konvertierung von JPEGs in WebP und resize auf 800x600..."

# Schleife über alle JPEG-Dateien im /media-Verzeichnis
for img in /media/*.jpg; do
  # Überprüfe, ob die Datei existiert
  if [ -f "$img" ]; then
    echo "Verarbeite: $img"
    
    # Resize das Bild auf 800x600 und speichere es temporär
    temp_img="/media/temp_resized.jpg"
    ffmpeg -i "$img" -vf "scale=800:600" "$temp_img"
    
    echo "Erstellt temporäres Bild: $temp_img"

    # Konvertiere das temporäre Bild in WebP
    webp_img="${img%.jpg}.webp"
    ffmpeg -i "$temp_img" -quality 80 "$webp_img"
    
    echo "Erstellt: $webp_img"
    
    # Entferne das temporäre Bild
    rm "$temp_img"
    echo "Temporäres Bild gelöscht."
  else
    echo "Keine JPEG-Dateien zum Konvertieren gefunden."
  fi
done

echo "Konvertierung abgeschlossen."