# Datei: /config/scripts/create_video.py
import os
import json
import subprocess

def log(message, level="INFO"):
    # Einfache Log-Funktion
    print(f"{level}: {message}")

def check_ffmpeg():
    # Überprüfe, ob FFmpeg installiert ist
    try:
        result = subprocess.run(["which", "ffmpeg"], stdout=subprocess.PIPE, text=True)
        ffmpeg_path = result.stdout.strip()
        if ffmpeg_path:
            log(f"FFmpeg gefunden unter: {ffmpeg_path}")
        else:
            log("FFmpeg wurde nicht gefunden. Stellen Sie sicher, dass es installiert ist.", level="ERROR")
    except Exception as e:
        log(f"Fehler beim Überprüfen von FFmpeg: {e}", level="ERROR")

def create_video():
    # Verzeichnisse und Ausgabedateien
    image_folder = '/media/growcamJPG'
    output_video = '/media/timelapse-webp.mp4'
    output_json = '/media/filenames.json'

    filenames = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".jpg") and filename.startswith("yourcamera_"):
            filenames.append(filename)

    # Speichere die Dateinamen in einer JSON-Datei
    with open(output_json, 'w') as json_file:
        json.dump(filenames, json_file, indent=4)

    log(f"Array der Dateinamen erfolgreich in {output_json} gespeichert.")
    log(f"Anzahl der Frames (Bilder) für das Video: {len(filenames)}")

    try:
        subprocess.run([
            "ffmpeg",
            "-pattern_type", "glob",
            "-framerate", "25",
            "-i", f"{image_folder}/yourcamera_*.jpg",
            "-vcodec", "libx264",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-movflags", "+faststart",
            output_video
        ], check=True)
        log(f"Video erfolgreich erstellt: {output_video}")
    except subprocess.CalledProcessError as e:
        log(f"Fehler beim Erstellen des Videos: {e}", level="ERROR")


if __name__ == "__main__":
    check_ffmpeg()
    create_video()