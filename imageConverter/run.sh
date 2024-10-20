#!/bin/sh

# Debug-Ausgabe aktivieren
# set -x  # Zeigt alle ausgeführten Befehle an
set -e  # Beende das Skript bei Fehlern

# Verzeichnis, in dem die JPG- und WebP-Dateien liegen
directory="/media/imageToMovie"

# Konvertiere JPEGs in WebP und resize auf 800x600
echo "Starte die Konvertierung von JPEGs in WebP und resize auf 800x600..."

# Finde die höchste vorhandene WebP-Datei-Nummer
last_index=$(ls "$directory"/index*.webp 2>/dev/null | sed -E 's/.*index([0-9]+)\.webp/\1/' | sort -n | tail -1)
if [ -z "$last_index" ]; then
  last_index=0
fi

# Schleife über alle JPEG-Dateien im Verzeichnis
for img in "$directory"/*.jpg; do
  # Überprüfe, ob die Datei existiert
  if [ -f "$img" ]; then
    echo "Verarbeite: $img"
    
    # Erhöhe die Nummer für die neue WebP-Datei
    new_index=$((last_index + 1))
    last_index=$new_index
    
    # Erstelle einen temporären Dateinamen
    temp_img="$directory/temp_resized.jpg"
    
    # Resize das Bild auf 800x600 und speichere es temporär
    ffmpeg -i "$img" -vf "scale=800:600" "$temp_img"
    echo "Erstellt temporäres Bild: $temp_img"

    # Benenne die neue WebP-Datei mit der nächsten Indexnummer
    webp_img="$directory/index$new_index.webp"
    
    # Konvertiere das temporäre Bild in WebP
    ffmpeg -i "$temp_img" -quality 80 "$webp_img"
    echo "Erstellt: $webp_img"
    
    # Entferne das temporäre Bild
    rm "$temp_img"
    echo "Temporäres Bild gelöscht."

    # Lösche die originale JPEG-Datei
    rm "$img"
    echo "Originales JPEG-Bild gelöscht: $img"
  else
    echo "Keine JPEG-Dateien zum Konvertieren gefunden."
  fi
done

echo "Konvertierung abgeschlossen."

# Erstelle ein MP4-Video aus den WebP-Dateien
echo "Erstelle MP4-Video aus den WebP-Dateien..."
ffmpeg -y -framerate 25 -i "$directory/index%d.webp" -c:v libx264 -r 30 -pix_fmt yuv420p "$directory/output.mp4"
echo "MP4-Video erstellt: output.mp4"