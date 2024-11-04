# Datei: /config/scripts/create_video.py
import os
import sys
import json
import subprocess
from datetime import datetime

def log(message, level="INFO"):
    print(f"{level}: {message}")

def create_video(framerate, inputpath, loglevel, revert):
    current_time = datetime.now().strftime("%Y-%m-%d-%H%M")
    output_video = f'/media/timelapse-{current_time}.mp4'
    output_json = f'/media/filenames-{current_time}.json'

    filenames = [filename for filename in sorted(os.listdir(inputpath))
                 if filename.endswith(".jpg") and filename.startswith("yourcamera_")]

    if revert.lower() == 'true':
        filenames.reverse()

    with open(output_json, 'w') as json_file:
        json.dump(filenames, json_file, indent=4)

    log(f"Array der Dateinamen erfolgreich in {output_json} gespeichert.")
    log(f"Anzahl der Frames (Bilder) für das Video: {len(filenames)}")

    if not filenames:
        log("Keine Bilddateien gefunden, die dem Muster entsprechen.", level="ERROR")
        return

    # Temporäre Textdatei für die Eingabepfade erstellen
    temp_file = "/tmp/input_files.txt"  # Speichere die temporäre Datei im /tmp/ Verzeichnis
    with open(temp_file, 'w') as f:
        for filename in filenames:
            f.write(f"file '{inputpath}/{filename}'\n")

    # log(f"Inhalt der temporären Datei:\n{open(temp_file).read()}")  # Logge den Inhalt der temporären Datei

    # FFmpeg-Befehl erstellen
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-loglevel", loglevel,
        "-f", "concat",
        "-safe", "0",
        "-i", temp_file,
        "-framerate", str(framerate),
        "-s", "800x600",  # Setze die Größe auf 800x600
        "-c:v", "libx264",
        "-crf", "18",  # Bessere Qualität
        "-b:v", "1000k",  # Erhöhte Bitrate
        "-maxrate", "1200k",
        "-bufsize", "2400k",  # Erhöhte Puffergröße
        "-profile:v", "high",  # Höheres Profil für bessere Qualität
        "-level", "4.0",
        "-movflags", "+faststart",
        "-an",
        "-pix_fmt", "yuv420p"  # Setze das Pixel-Format
    ]

    ffmpeg_command.append(output_video)

    try:
        subprocess.run(ffmpeg_command, check=True)
        log(f"Video erfolgreich erstellt: {output_video}")
    except subprocess.CalledProcessError as e:
        log(f"Fehler beim Erstellen des Videos: {e}", level="ERROR")
    finally:
        # Temporäre Datei löschen
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        log("Fehlende Argumente. Erwartet: framerate, inputpath, loglevel, revert", level="ERROR")
        sys.exit(1)

    framerate = sys.argv[1]
    inputpath = sys.argv[2]
    loglevel = sys.argv[3]
    revert = sys.argv[4]

    create_video(framerate, inputpath, loglevel, revert)