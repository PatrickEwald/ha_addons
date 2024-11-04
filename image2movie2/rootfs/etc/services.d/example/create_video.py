# Datei: /config/scripts/create_video.py
import os
import sys
import json
import subprocess
from datetime import datetime

def log(message, level="INFO"):
    print(f"{level}: {message}")

def check_ffmpeg():
    try:
        result = subprocess.run(["which", "ffmpeg"], stdout=subprocess.PIPE, text=True)
        ffmpeg_path = result.stdout.strip()
        if ffmpeg_path:
            log(f"FFmpeg gefunden unter: {ffmpeg_path}")
        else:
            log("FFmpeg wurde nicht gefunden. Stellen Sie sicher, dass es installiert ist.", level="ERROR")
    except Exception as e:
        log(f"Fehler beim Überprüfen von FFmpeg: {e}", level="ERROR")

def create_video(framerate, inputpath, loglevel, revert):
    current_time = datetime.now().strftime("%Y-%m-%d-%H%M")
    output_video = f'/media/timelapse-{current_time}.mp4'
    output_json = f'/media/filenames-{current_time}.json'

    filenames = []
    for filename in sorted(os.listdir(inputpath)):
        if filename.endswith(".jpg") and filename.startswith("yourcamera_"):
            filenames.append(filename)

    # Wenn revert aktiviert ist, kehre die Reihenfolge der Dateinamen um
    if revert.lower() == 'true':
        filenames.reverse()

    with open(output_json, 'w') as json_file:
        json.dump(filenames, json_file, indent=4)

    log(f"Array der Dateinamen erfolgreich in {output_json} gespeichert.")
    log(f"Anzahl der Frames (Bilder) für das Video: {len(filenames)}")

    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-loglevel", loglevel,
        "-pattern_type", "glob",
        "-framerate", str(framerate),
        "-i", f"{inputpath}/yourcamera_*.jpg",
        "-vcodec", "libx264",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-movflags", "+faststart"
    ]

    if revert.lower() == 'true':
        ffmpeg_command.extend(["-vf", "reverse"])

    ffmpeg_command.append(output_video)

    try:
        subprocess.run(ffmpeg_command, check=True)
        log(f"Video erfolgreich erstellt: {output_video}")
    except subprocess.CalledProcessError as e:
        log(f"Fehler beim Erstellen des Videos: {e}", level="ERROR")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        log("Fehlende Argumente. Erwartet: framerate, inputpath, loglevel, revert", level="ERROR")
        sys.exit(1)

    framerate = sys.argv[1]
    inputpath = sys.argv[2]
    loglevel = sys.argv[3]
    revert = sys.argv[4]

    check_ffmpeg()
    create_video(framerate, inputpath, loglevel, revert)