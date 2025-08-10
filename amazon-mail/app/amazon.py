import datetime, os, sys

print("=== amazon.py TESTSTART ===")
print("Uhrzeit:", datetime.datetime.now().isoformat())
print("CWD:", os.getcwd())
print("ARGV:", sys.argv)
print("ENV_HAS_EMAIL_USER:", "EMAIL_USER" in os.environ)
print("Hello from inside the add-on!")
print("=== amazon.py TESTENDE ===")
