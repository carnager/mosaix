# Aktionen, Skripte und Stapelverarbeitung

Alles läuft über dieselben Befehle wie die Agenten-Schnittstelle (`docs/agenten.md`).
Ordner: `~/.local/share/imagine/actions/` (bzw. `$XDG_DATA_HOME/imagine/actions/`).

## Aktionen (Makros)

*Datei › Aktionen › Aufnahme starten*, dann ganz normal arbeiten, *Aufnahme beenden …*
und einen Namen vergeben. Aufgezeichnet werden:

- Filter (auch Smart-Filter und Plugins), Korrektur-Ebenen und ihre Änderungen, Ebenenstile, RAW-Entwicklung
- Deckkraft, Fläche, Füllmethode, Hindurchwirken, Ebene löschen
- Neue Ebene, Duplizieren, Gruppieren/Auflösen, Maske hinzufügen/anwenden, Rastern
- Auswahl: Rechteck, Ellipse, Lasso (mit Hinzufügen/Abziehen/Schnittmenge), Alles, Aufheben, Umkehren, Weiche Kante
- Füllen (Alt+Rücktaste), Löschen der Auswahl, Verschieben, Arbeitsfläche, Bildgröße, Rückgängig/Wiederholen

Nicht aufgezeichnet: Pinselstriche und Werkzeugwechsel. Schritte enthalten keine
Ebenen-IDs – sie wirken auf die jeweils aktive Ebene, also auf jedem Bild.

Eine Aktion ist eine Textdatei `NAME.jsonl`, ein Befehl pro Zeile (`#` = Kommentar),
von Hand oder von einem Agenten schreibbar:

```jsonl
# Web-Look
{"cmd": "filter_apply", "filter": {"kind": "plugin", "name": "Vignette", "values": {"amount": -60}}}
{"cmd": "adjustment_add", "adjustment": {"kind": "brightness_contrast", "brightness": 5, "contrast": 20}}
{"cmd": "image_resize", "width": 1600}
```

Abspielen: Klick im Menü (jeder Schritt ein Rückgängig-Schritt, Abbruch beim ersten
Fehler mit Meldung), Agenten: `action_list`, `action_run`.

## Skripte

Jede **ausführbare** Datei im Ordner erscheint ebenfalls unter *Datei › Aktionen* und
wird mit `IMAGINE_SOCKET` (dieses Fenster) und `IMAGINE` (Programmpfad) gestartet;
ihre Ausgabe landet im Protokoll. Beliebige Sprache, z. B.:

```sh
#!/bin/sh
# Aktive Ebene halb durchsichtig
echo '{"cmd": "layer_set", "opacity": 0.5}' | "$IMAGINE" run --attach -
```

Für Logik (Schleifen über Ebenen usw.): `document_info` liefert den Ebenenbaum als JSON;
mit Python etwa `subprocess.run([imagine, "run", "--attach", "-"], input=…)` oder direkt
über den Unix-Socket (JSON-Zeilen, eine Antwort pro Zeile).

## Stapelverarbeitung

```sh
imagine batch "Web Look" ~/Bilder/*.jpg --out ~/web --format webp
```

Öffnet jede Datei (auch RAW, PSD), spielt die Aktion ab und exportiert. Ohne `--out`
landet alles in `imagine-batch/` neben den Eingaben, ohne `--format` im selben Format
(PNG, wenn das Eingabeformat nicht exportierbar ist, z. B. RAW).
