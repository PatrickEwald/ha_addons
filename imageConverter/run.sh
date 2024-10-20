#!/bin/bash

# Pfade für Eingangs- und Ausgangsbild
INPUT_IMAGE="/media/input.jpg"
OUTPUT_IMAGE="/media/output.webp"

# FFmpeg-Befehl zur Konvertierung
ffmpeg -i "$INPUT_IMAGE" -c:v libwebp -q:v 80 "$OUTPUT_IMAGE"

echo "Bild wurde erfolgreich von JPEG zu WebP konvertiert."