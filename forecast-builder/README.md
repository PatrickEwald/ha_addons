# Forecast Builder Add-on

Dieses Add-on für Home Assistant sagt die Auslastung eines Fitnessstudios vorher und schreibt die Ergebnisse in eine InfluxDB.

## Funktionen

- Holt historische Auslastungsdaten aus InfluxDB
- Trainiert ein LightGBM-Modell zur Vorhersage der Auslastung
- Erstellt Prognosen für die aktuellen Öffnungszeiten
- Schreibt die Vorhersagewerte zurück in InfluxDB
- Flexible Konfiguration über die Home Assistant Add-on-Oberfläche

## Konfiguration

Die Add-on-Konfiguration erlaubt folgende Optionen (siehe auch `config.yaml`):

| Option          | Typ  | Beschreibung                                                 | Standardwert |
| --------------- | ---- | ------------------------------------------------------------ | ------------ |
| interval        | int  | Intervall in Sekunden für die Skriptausführung               | 300          |
| use_temperature | bool | Temperatur als Feature einbeziehen                           | false        |
| use_bridge_day  | bool | Brückentage als Feature einbeziehen                          | false        |
| use_lags        | bool | Historische Werte (z. B. 7-Tage-Lag) als Feature einbeziehen | false        |
| influx_user     | str  | Benutzername für InfluxDB                                    | ""           |
| influx_password | str  | Passwort für InfluxDB                                        | ""           |

Die Konfiguration kann über die Home Assistant UI angepasst werden.

---
