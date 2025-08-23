# 🏋️ FFF Fitness Studio Dashboard

Ein modernes, responsives Dashboard zur Visualisierung der Auslastungsdaten und Vorhersagen von Fitness-Studios.

## ✨ Features

- **📊 Interaktive Zeitserien-Charts** mit Chart.js
- **🌙 Dark/Light Mode** Toggle
- **📱 Responsive Design** für alle Geräte
- **🔄 Real-time Updates** alle 5 Minuten
- **📅 Datumsauswahl** für historische Daten
- **🏢 Studio-Auswahl** aus verfügbaren Standorten
- **📈 Vorhersage-Visualisierung** (gestrichelte Linie)
- **📊 Übersichtskarten** mit wichtigen Metriken
- **📊 Wochentrend** mit Balkendiagrammen
- **⚡ Smooth Animations** und Transitions

## 🚀 Installation & Verwendung

### 1. Dashboard starten
```bash
# Im dashboard-Verzeichnis
cd dashboard

# Mit Python HTTP-Server (einfach)
python3 -m http.server 3000

# Oder mit Node.js (falls installiert)
npx serve .

# Oder mit PHP (falls installiert)
php -S localhost:3000
```

### 2. Im Browser öffnen
```
http://localhost:3000
```

### 3. Studio auswählen
- Wähle ein Studio aus dem Dropdown-Menü
- Wähle ein Datum aus (Standard: heute)
- Das Dashboard lädt automatisch die Daten

## 🔧 Konfiguration

### API-Endpunkte
Das Dashboard versucht, Daten von folgenden Endpunkten zu laden:

- **Studios**: `http://localhost:8000/api/v1/gyms`
- **Zeitserien**: `http://localhost:8000/api/v1/gyms/{uuid}/timeseries?date={date}`

### Fallback-Daten
Falls die API nicht verfügbar ist, werden Fallback-Daten verwendet:
- 3 Beispiel-Studios
- Mock-Vorhersagen basierend auf Tageszeiten

## 📊 Dashboard-Komponenten

### Übersichtskarten
- **Datum**: Aktuell ausgewähltes Datum
- **Durchschnitt**: Durchschnittliche Auslastung
- **Spitzenzeit**: Zeitpunkt der höchsten Auslastung
- **Öffnungszeit**: Studio-Öffnungszeiten

### Hauptchart
- **Blaue Linie**: Tatsächliche Auslastung
- **Grüne gestrichelte Linie**: Vorhersage (optional)
- **X-Achse**: Zeit (10-Minuten-Intervalle)
- **Y-Achse**: Auslastung in Prozent

### Wochentrend
- **7 Balken**: Montag bis Sonntag
- **Höhe**: Durchschnittliche Auslastung pro Tag
- **Animation**: Smooth Transitions

## 🎨 Anpassungen

### Theme ändern
- Klicke auf den Sonne/Mond-Button in der oberen rechten Ecke
- Das Theme wird im localStorage gespeichert

### Chart-Optionen
- **Vorhersage anzeigen**: Zeigt/versteckt die Vorhersage-Linie
- **Öffnungszeiten**: Zeigt/versteckt Öffnungszeiten-Markierungen

### Auto-Update
- Daten werden alle 5 Minuten automatisch aktualisiert
- Updates werden pausiert, wenn der Tab nicht sichtbar ist

## 🛠️ Technische Details

### Technologie-Stack
- **HTML5**: Struktur
- **Tailwind CSS**: Moderne UI-Komponenten
- **Chart.js**: Interaktive Charts
- **Vanilla JavaScript**: API-Integration & Logik

### Browser-Unterstützung
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

### Performance
- **Lazy Loading**: Daten werden nur bei Bedarf geladen
- **Efficient Updates**: Nur geänderte Chart-Daten werden aktualisiert
- **Memory Management**: Charts werden korrekt zerstört und neu erstellt

## 🔍 Troubleshooting

### Dashboard lädt nicht
1. Prüfe, ob der HTTP-Server läuft
2. Öffne die Browser-Entwicklertools (F12)
3. Schaue in der Konsole nach Fehlermeldungen

### Keine Daten angezeigt
1. Prüfe, ob die FastAPI läuft (Port 8000)
2. Prüfe die API-Endpunkte
3. Das Dashboard verwendet Fallback-Daten, falls die API nicht verfügbar ist

### Charts werden nicht angezeigt
1. Prüfe, ob Chart.js geladen wurde
2. Prüfe die Browser-Konsole auf JavaScript-Fehler
3. Stelle sicher, dass ein Studio ausgewählt ist

## 📝 Entwicklung

### Neue Features hinzufügen
1. Bearbeite `script.js` für neue Funktionalität
2. Aktualisiere `index.html` für neue UI-Elemente
3. Füge neue Styles in `styles.css` hinzu

### API-Integration erweitern
1. Erweitere die `fetch*` Methoden in der `FFFDashboard` Klasse
2. Füge neue Datenverarbeitung in `processData` hinzu
3. Aktualisiere die Chart-Darstellung entsprechend

## 🤝 Beitragen

1. Fork das Repository
2. Erstelle einen Feature-Branch
3. Mache deine Änderungen
4. Teste gründlich
5. Erstelle einen Pull Request

## 📄 Lizenz

Dieses Projekt ist für den internen Gebrauch der FFF Fitness Studios bestimmt.

---

**Viel Spaß mit dem Dashboard! 🎉**
