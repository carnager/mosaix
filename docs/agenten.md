# imagine für Agenten

imagine lässt sich von KI-Agenten genauso bedienen wie von Menschen. Alle Zugänge teilen dieselbe Befehlsschicht (`crates/imagine-app/src/agent/command.rs`). Jede Änderung ist **ein Undo-Schritt**, egal ob sie vom Agenten oder von dir kommt.

| Zugang | Wofür | Aufruf |
|---|---|---|
| **MCP-Server, eigene Engine** | Agent arbeitet allein, ohne Fenster | `imagine mcp [DATEI]` |
| **MCP-Server, live** | Agent steuert dein offenes Fenster, du siehst zu und kannst eingreifen | `imagine mcp --attach` |
| **JSON-Lines-Skript** | Agenten mit Shell-Zugriff, Batch, Tests | `imagine run SKRIPT.jsonl` / `… --attach -` |

## Einrichten (Claude Code)

```sh
# live: der Agent arbeitet in deinem laufenden imagine-Fenster
claude mcp add imagine -- /pfad/zu/imagine mcp --attach

# oder headless mit eigener Engine
claude mcp add imagine-headless -- /pfad/zu/imagine mcp
```

Alternativ als `.mcp.json` im Projekt:

```json
{ "mcpServers": { "imagine": { "command": "/pfad/zu/imagine", "args": ["mcp", "--attach"] } } }
```

Jedes imagine-Fenster lauscht auf `$XDG_RUNTIME_DIR/imagine/<pid>.sock` (nur für deinen Benutzer zugänglich). `--attach` verbindet sich mit dem zuletzt gestarteten Fenster; `--socket PFAD` oder `IMAGINE_SOCKET` wählen ein bestimmtes.

## Konventionen

- Koordinaten in **Dokumentpixeln**, Ursprung oben links
- Farben in **sRGB**: `"#rrggbb"`, `"#rrggbbaa"` oder `[r, g, b(, a)]` mit 0–255
- `layer` ist optional und meint sonst die aktive Ebene. Ebenen-IDs bleiben stabil, auch über Speichern und Laden hinweg.
- Fehler kommen als lesbare Meldung zurück (MCP: `isError`, Skript: `"ok": false`). Tippfehler in Parametern werden abgelehnt, nicht ignoriert.
- Malst du gerade selbst, lehnt imagine verändernde Agentenbefehle kurz ab („the user is painting right now“).

## Werkzeuge

| Werkzeug | Zweck |
|---|---|
| `document_new`, `document_open`, `document_info` | Dokument anlegen/öffnen; Struktur, Ebenenbaum, Verlauf abfragen. Kamera-RAW (CR2/CR3, NEF, ARW, RAF, ORF, RW2, DNG …) wird szenenlinear entwickelt; `raw: {"white_balance": {"mode": "as_shot"\|"auto"\|"daylight"} oder {"mode": "custom", "temperature": 5200, "tint": 10}, "exposure": 0.5}` |
| `raw_info` | Kamera, Objektiv, Aufnahmedaten und Weißabgleich (Temperatur/Tönung) einer RAW-Datei, ohne sie zu öffnen |
| `raw_set` | RAW-Ebene neu entwickeln (`white_balance`, `exposure`, `highlights`: `reconstruct`/`clip`, `noise`: `{luminance, color}` 0–100, `lens`: Objektivkorrektur an/aus, `tone`: `{contrast, highlights, shadows, vibrance, saturation}` −100…100). Geöffnete und mit `place_image` platzierte RAW-Dateien bleiben RAW-Ebenen (Datei steckt im Dokument, Transformieren verlustfrei); `layer_rasterize` macht Pixel daraus |
| `layer_add`, `layer_duplicate`, `layer_delete`, `layer_select`, `layer_move`, `layer_group`, `layer_ungroup` | Ebenenverwaltung |
| `layer_set` | Name, Deckkraft, Füllmethode (`multiply`, `screen`, `overlay`, … 19 Stück), Sichtbarkeit |
| `set_blend_space` | `srgb` (Photoshop-artig) oder `linear` |
| `paint_stroke` | Pinselstrich entlang einer Punktliste, optional mit Druck pro Punkt; Größe, Härte, Deckkraft, Fluss, Radierer; Spitzen (`tip`: round/chalk/bristles/spray/rough/charcoal), Winkel, Rundheit, Zufall (`size_jitter`, `angle_jitter`, `scatter`, `flow_jitter`), `follow_direction`; `smudge: 0…1` verwischt statt zu malen; `target: "mask"` malt in die Ebenenmaske |
| `select_rect`, `select_ellipse`, `select_polygon` | Auswahl mit Kantenglättung; `mode`: `replace`, `add`, `subtract`, `intersect` |
| `select_all`, `select_none`, `select_invert`, `select_feather` | Auswahl verwalten, weiche Kante |
| `fill`, `clear`, `gradient` | Auswahl (oder ganze Ebene) füllen, löschen, Verlauf linear/radial; Ebene oder Maske |
| `layer_mask_add`, `layer_mask_delete`, `layer_mask_set`, `layer_mask_apply` | Ebenenmasken: anlegen (aus Auswahl), löschen, ein/aus, in Pixel übernehmen |
| `layer_translate` | ganze Ebene samt Maske um ganze Pixel verschieben (ignoriert die Auswahl) |
| `layer_transform` | skalieren, drehen (Grad, im Uhrzeigersinn), verschieben um einen Drehpunkt (Standard: Mitte des Inhalts); mit Auswahl nur die ausgewählten Pixel; `interp` bicubic/bilinear/nearest |
| `layer_bounds` | pixelgenaue Grenzen des sichtbaren Inhalts einer Ebene |
| `filter_apply` mit `smart: true`, `smart_filter_set` | Smart-Filter: nicht-destruktiv an der Ebene, ändern, ein/aus, entfernen, in die Pixel übernehmen (`apply`); `document_info` listet sie unter `smart_filters` |
| `liquify` | Verflüssigen mit Strichen (push entlang eines Pfads, reconstruct, twirl_cw/ccw, pucker, bloat), ein Undo-Schritt |
| `paint_stroke` mit `clone_offset` | Kopierstempel: kopiert pixelgenau von Punkt + Versatz; `clone_all_layers` nimmt das ganze Bild als Quelle; mit `heal: true` Reparaturpinsel (Farbe/Helligkeit an die Umgebung angeglichen); `spot_heal: true` Bereichsreparatur ohne Quelle |
| `content_aware_fill` | füllt die Auswahl inhaltsbasiert (Objekte entfernen); `all_layers` lernt aus dem ganzen Bild |
| `text_add`, `shape_add`, `vector_set`, `layer_rasterize` | Text- und Formebenen anlegen, über ihre Beschreibung ändern (JSON-Merge, Farben auch als `#rrggbb`) und in Pixel umwandeln |
| `layer_effects_set` | Ebenenstile setzen: Kontur, Schlagschatten, Schatten/Schein innen und außen, Farbüberlagerung, Kante & Relief |
| `histogram` | Histogramm je Kanal mit Mittelwert, Median, 1./99. Perzentil und Clipping, wahlweise von dem, was unter einer Ebene liegt |
| `crop`, `canvas_size`, `image_resize` | Zuschneiden (darf über den Rand hinausgehen), Arbeitsfläche mit Anker, Bildgröße; Pixel außerhalb bleiben erhalten |
| `select_magic_wand`, `select_color_range` | Zauberstab (Toleranz 0–255, zusammenhängend oder nicht, `sample` layer/image) und Farbbereich (Farbe oder Punkt, Unschärfe 0–200) |
| `fill_rect`, `clear_rect` | Rechteck füllen bzw. transparent machen (unabhängig von der Auswahl) |
| `place_image` | Bilddatei als neue Ebene an Position (x, y) |
| `sample_color` | Farbe des Gesamtbilds an einem Punkt |
| `render_preview` | **Das Ergebnis ansehen:** Gesamtbild oder Ausschnitt als PNG, optional zusätzlich als Datei (`output`); `show_selection` färbt nicht Ausgewähltes rot |
| `adjustment_add`, `adjustment_set` | Einstellungsebenen (Kurven, Tonwerte, Farbton/Sättigung, Belichtung, …) hinzufügen und ändern; mit Auswahl wird sie zur Maske |
| `filter_apply` | Filter: Gaußscher Weichzeichner, Unscharf maskieren, Hochpass, Rauschen (durch die Auswahl) |
| `undo`, `redo` | Verlauf |
| `save`, `export` | `.imagine` (mit Ebenen, inkrementell) bzw. PNG/JPEG/WebP/TIFF; `export` optional in Display P3, Adobe RGB oder Rec. 2020 (Profil eingebettet), mit 16 Bit (PNG/TIFF) und JPEG-Qualität |

Die vollständigen JSON-Schemata liefert der Server per `tools/list`.

## Beispiel: Skript

```jsonl
# Kommentare mit # oder //
{"cmd": "document_new", "width": 800, "height": 500}
{"cmd": "fill_rect", "x": 0, "y": 0, "width": 800, "height": 500, "color": "#1e2a44"}
{"cmd": "layer_add", "name": "Sonne"}
{"cmd": "paint_stroke", "points": [[600, 120]], "size": 140, "hardness": 0.4, "color": "#ffcc33"}
{"cmd": "layer_set", "blend": "screen", "opacity": 0.8}
{"cmd": "render_preview", "max_size": 512, "output": "vorschau.png"}
{"cmd": "save", "path": "nacht.imagine"}
```

```sh
imagine run nacht.jsonl            # eigene Engine
imagine run --attach nacht.jsonl   # im offenen Fenster
```

Jede Zeile erzeugt eine JSON-Ergebniszeile auf stdout. Beim ersten Fehler bricht das Skript mit Exit-Code 1 ab. Logs gehen nach stderr.

## Grenzen (Stand Phase 2b)

- Noch nicht vorhanden: Smart-Filter, Verflüssigen, Retusche-Werkzeuge (Phase 5), Text (Phase 6). Gruppen lassen sich noch nicht transformieren.
- Alle Auswahl-, Füll- und Malbefehle respektieren die aktuelle Auswahl, auch eine weiche.
- `render_preview` zeigt das Bild, nicht die Fensteransicht (kein Zoom, keine Drehung).
- `document_open`/`document_new` ersetzen im Live-Modus das offene Dokument ohne Rückfrage. Ungespeicherte Änderungen gehen dabei verloren.
