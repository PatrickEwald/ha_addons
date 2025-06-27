# Datei: /config/scripts/create_video.py
import os
import sys
import json
import subprocess
from datetime import datetime

def log(message, level="INFO"):
    print(f"{level}: {message}")

def create_video(framerate, inputpath, loglevel, revert, imageFormat):
    current_time = datetime.now().strftime("%Y-%m-%d-%H%M")
    output_video = f'/media/timelapse/timelapse-{current_time}.mp4'
    output_json = f'/media/timelapse/filenames-{current_time}.json'

    filenames = [filename for filename in sorted(os.listdir(inputpath))
                 if filename.endswith(imageFormat) and filename.startswith("yourcamera_")]

    if revert.lower() == 'true':
        filenames.reverse()

    with open(output_json, 'w') as json_file:
        json.dump(filenames, json_file, indent=4)

    log(f"Array der Dateinamen erfolgreich in {output_json} gespeichert.")
    log(f"Anzahl der Frames (Bilder) für das Video: {len(filenames)}")

    if not filenames:
        log("Keine Bilddateien gefunden, die dem Muster entsprechen.", level="ERROR")
        return

    temp_file = "/tmp/input_files.txt" 
    with open(temp_file, 'w') as f:
        for filename in filenames:
            f.write(f"file '{inputpath}/{filename}'\n")

    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-loglevel", loglevel,
        "-f", "concat",
        "-safe", "0",
        "-i", temp_file,
        "-framerate", str(framerate),
        "-s", "800x600",
        "-c:v", "libx264",
        "-crf", "18",
        "-b:v", "1000k",
        "-maxrate", "1200k",
        "-bufsize", "2400k",
        "-profile:v", "high",
        "-level", "4.0",
        "-movflags", "+faststart",
        "-an",
        "-pix_fmt", "yuv420p"
    ]

    ffmpeg_command.append(output_video)

    try:
        subprocess.run(ffmpeg_command, check=True)
        log(f"Video erfolgreich erstellt: {output_video}")
    except subprocess.CalledProcessError as e:
        log(f"Fehler beim Erstellen des Videos: {e}", level="ERROR")
    finally:
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
    imageFormat = sys.argv[5]

    create_video(framerate, inputpath, loglevel, revert, imageFormat)