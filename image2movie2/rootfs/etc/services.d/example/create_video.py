# Datei: /config/scripts/create_video.py
import os
import json
import subprocess
import appdaemon.plugins.hass.hassapi as hass

class CreateVideo(hass.Hass):

    def initialize(self):
        # Überprüfe, ob FFmpeg verfügbar ist und gib den Pfad aus
        self.check_ffmpeg()
        # Rufe die Videoerstellungsfunktion auf, wenn die App startet
        self.create_video()

    def check_ffmpeg(self):
        try:
            result = subprocess.run(["which", "ffmpeg"], stdout=subprocess.PIPE, text=True)
            ffmpeg_path = result.stdout.strip()
            if ffmpeg_path:
                self.log(f"FFmpeg gefunden unter: {ffmpeg_path}")
            else:
                self.log("FFmpeg wurde nicht gefunden. Stellen Sie sicher, dass es installiert ist.", level="ERROR")
        except Exception as e:
            self.log(f"Fehler beim Überprüfen von FFmpeg: {e}", level="ERROR")

    def create_video(self):
        # Verzeichnisse und Ausgabedateien
        image_folder = '/media/growcamJPG'
        output_video = '/media/timelapse-webp.mp4'
        output_json = '/media/filenames.json'

        filenames = []
        for filename in sorted(os.listdir(image_folder)):
            if filename.endswith(".jpg") and filename.startswith("yourcamera_"):
                filenames.append(filename)

        with open(output_json, 'w') as json_file:
            json.dump(filenames, json_file, indent=4)

        self.log(f"Array der Dateinamen erfolgreich in {output_json} gespeichert.")
        self.log(f"Anzahl der Frames (Bilder) für das Video: {len(filenames)}")

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
            self.log(f"Video erfolgreich erstellt: {output_video}")
        except subprocess.CalledProcessError as e:
            self.log(f"Fehler beim Erstellen des Videos: {e}", level="ERROR")