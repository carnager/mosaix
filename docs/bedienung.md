# Bedienung

Ausführliche Referenz zu Aufruf, Tastenkürzeln und Werkzeugen. Kurzüberblick: [README](../README.md).

## Bauen & Starten

```sh
cargo run --release -- bild.jpg              # Bild öffnen (PSD/PSB mit Ebenen, PNG, JPEG, WebP, TIFF, EXR, .mosaix)
cargo run --release -- --new 30000x30000     # leeres Dokument
cargo run --release -- --demo 10000x10000    # 100-MP-Testbild
cargo run --release -- in.jpg --save out.mosaix --export out.png   # ohne Fenster konvertieren
cargo run --release -- --demo 10000x10000 --bench-offscreen         # Benchmark ohne Fenster
cargo run --release --features profile -- …  # mit Tracy-Profiling
cargo test --workspace                       # inkl. Headless-GPU-Tests (Vulkan)
```

Voraussetzungen: Wayland-Compositor, Vulkan-Treiber mit Descriptor Indexing (RADV, ANV, NVIDIA).

## Sprache / Language

The interface is English; other languages come from gettext catalogs (`crates/mosaix-core/i18n/<lang>.po`, currently German). The language follows `LANGUAGE`, `LC_ALL`, `LC_MESSAGES`, `LANG` (C/POSIX → English); `"language": "de"` in the settings file overrides it. `scripts/i18n-check.py check` lists untranslated or stale texts.

## Für Agenten

mosaix ist per **MCP** und JSON-Skript steuerbar, auch live im offenen Fenster:

```sh
claude mcp add mosaix -- /pfad/zu/mosaix mcp --attach
```

Details: [docs/agenten.md](agenten.md)

## Bedienung

Ohne Datei startet Mosaix mit einem Startbildschirm: Neues Bild, Öffnen, Schnellstart-Größen und die zuletzt geöffneten Dateien mit Vorschaubildern (zwischengespeichert in `~/.cache/mosaix/thumbs`, geschrieben beim Speichern, beim Öffnen einer anderen Datei und beim Beenden). Esc oder „Leere Arbeitsfläche“ schließt ihn.

Öffnen, Speichern, Exportieren und Importieren nutzen den Dateidialog des Systems (XDG-Desktop-Portal, z. B. GTK/GNOME); Farbraum, 16 Bit und JPEG-Qualität stehen beim Exportieren direkt im Dialog. Ohne Portal, oder mit `MOSAIX_BUILTIN_DIALOG=1`, kommt der eingebaute Dialog.

Verwandte Werkzeuge teilen sich einen Platz in der Werkzeugleiste (kleines Dreieck in der Ecke): Rechtsklick oder gedrückt halten zeigt alle, der Platz merkt sich das zuletzt benutzte. Menüeinträge, die gerade nicht gehen, sind ausgegraut; der Tooltip sagt, warum.

Die Oberfläche hat eine Menüleiste, links die Werkzeugleiste und rechts Panels für Farbe, Werkzeug-Einstellungen, Ebenen (Drag & Drop, Doppelklick zum Umbenennen, Klick auf das Masken-Symbol zum Bemalen) und Verlauf (Klick springt zurück). Unten ist eine Statusleiste mit Meldungen, ein Klick darauf öffnet alle Meldungen. Strg+`K` oder Strg+Umschalt+`P` öffnet die Befehlspalette, `Tab` blendet die Oberfläche aus. Strg+`N` neues Bild, Strg+`O` öffnen (mit „Zuletzt geöffnet“), Strg+Umschalt+`S` speichern unter. Alt+Rechtsziehen ändert Pinselgröße (waagrecht) und Härte (senkrecht). Pinsel, Farbe, Werkzeug und zuletzt geöffnete Dateien merkt sich mosaix in `~/.config/mosaix/settings.json`. Alle drei Minuten wird bei ungespeicherten Änderungen eine Wiederherstellungskopie nach `~/.local/state/mosaix/autosave/` gesichert. Deine Datei bleibt dabei unberührt.

Tastenkürzel folgen Photoshop. Die Ebenenliste erscheint bis zum UI (Phase 3) im Terminal, die aktive Ebene im Fenstertitel.

| Aktion | Eingabe |
|---|---|
| Werkzeuge | `B` Pinsel, `U` Wischfinger, `E` Radierer, `M` Auswahlrechteck/-ellipse, `L` Lasso, `W` Zauberstab, `G` Verlauf linear/radial, `V` Verschieben, `C` Zuschneiden |
| Malen | linke Maustaste / Stift; Radierer-Ende des Stifts; `[` `]` Größe, Ziffern Deckkraft (Wischfinger: Stärke), `H` Härte, Umschalt+`S` Glättung, Umschalt+`B` Spitze, Umschalt+`J` Dynamik, Umschalt+`P` Palettenfarbe, `X` Vorder-/Hintergrundfarbe tauschen, `D` Schwarz/Weiß, `I` Pipette (auch Alt+Klick beim Malen; 1 px bis 31×31 gemittelt, Ebene oder alle Ebenen) |
| Kopierstempel / Reparatur | `S` Kopierstempel; `J` Bereichsreparatur (einfach über den Fleck malen, Quelle wird gesucht), nochmal `J` Reparaturpinsel. Stempel und Reparaturpinsel: Alt+Klick setzt die Quelle, dann malen. Der Reparaturpinsel gleicht beim Loslassen Farbe und Helligkeit an die Umgebung an. Optionen: alle Ebenen aufnehmen (auf leere Ebene stempeln), ausgerichtet. Ein Kreuz zeigt, woher gerade kopiert wird |
| Inhaltsbasiert füllen | Objekt großzügig auswählen, Umschalt+Rücktaste (Strg+Umschalt+Rücktaste: aus allen Ebenen lernen, z. B. auf leerer Ebene). Rechnet im Hintergrund |
| Text & Formen | `T` Text: Klick ins Bild legt Text an und man tippt direkt; Klick in einen Text setzt den Cursor (Doppelklick Wort, Dreifachklick Zeile, Umschalt erweitert, Ziehen markiert). Pfeile/Pos1/Ende (mit Strg wortweise, mit Umschalt markieren), Strg+A alles, Enter neue Zeile, Esc oder Strg+Enter beendet. Schrift, Größe, Farbe unter Eigenschaften. Umschalt+`U` Form: Rechteck/Ellipse aufziehen. Bleiben scharf und editierbar, auch nach Strg+`T`; Strg+Alt+`R` rastert |
| Zeichenstift | `P`: Klick = Ecke, Ziehen = Kurve mit Griffen, Klick auf den Startpunkt schließt (gefüllte Form), Enter/Esc beendet offen (Linie), Rücktaste nimmt den letzten Punkt weg. Auf einem Pfad: Punkte und Griffe ziehen, Alt+Klick Ecke ↔ Kurve, Alt+Ziehen Griff einzeln, `Entf` löscht den Punkt. Farbe = Pinselfarbe, danach unter Eigenschaften (Fläche/Kontur). Umschalt+`P` wechselt die Palettenfarbe |
| Ebenenstile | Rechts im Panel „Ebenenstile“: Kontur, Schlagschatten, Schatten/Schein innen und außen, Farbüberlagerung, Kante & Relief – ein/aus, einstellen, entfernen; bleiben editierbar |
| Ansicht | Mausrad schwenkt (Umschalt: seitlich), Alt+Mausrad zoomt (wie Photoshop), Alt+Umschalt+Mausrad dreht, Leertaste+Ziehen schwenken; Herauszoomen endet bei ~128 px Bildgröße, `R`+Ziehen drehen, Strg+`0` einpassen, Strg+`1` 100 %, `Q` Maskierungsmodus-Ansicht |
| Auswahl | Aufziehen mit Umschalt/Alt/beidem = hinzufügen/abziehen/Schnittmenge; Strg+`A` / Strg+`D` / Strg+Umschalt+`I`; Umschalt+`F6` weiche Kante (Dialog) |
| Lineale & Hilfslinien | Strg+`R` Lineale. Aus einem Lineal eine Hilfslinie ziehen (Umschalt: auf Teilstriche), mit dem Verschieben-Werkzeug oder Strg verschieben, zurück ins Lineal ziehen löscht sie. Ansicht › Neue Hilfslinie … für genaue Positionen. Strg+`;` ein/aus, Strg+Alt+`;` alle löschen. Hilfslinien gehören zum Dokument (Rückgängig, `.mosaix`) |
| Raster & Einrasten | Strg+`'` Raster (Ansicht › Raster einstellen: Abstand, Unterteilungen). Strg+Umschalt+`;` Einrasten: Verschieben, Auswahlrahmen, Formen, Zuschneiden und Transformieren rasten an Hilfslinien, Bildkanten, Bildmitte und (sichtbarem) Raster ein |
| Auswahl verändern | Auswahl › Verändern: erweitern, verkleinern (runde Ecken), Rand (weicher Ring um die Kante), abrunden (Ecken rund, Krümel weg). Auswahl › Farbbereich …: eine Farbe überall auswählen, weich nach Toleranz |
| Auswahl speichern | Auswahl › Auswahl speichern … legt sie als Kanal im Dokument ab (wird in `.mosaix` gespeichert und wandert bei Arbeitsfläche, Drehen und Bildgröße mit). Auswahl laden: Klick ersetzt, mit Umschalt hinzufügen, Alt abziehen, Umschalt+Alt Schnittmenge. PSD-Export nimmt die Kanäle noch nicht mit |
| Transformieren | Strg+`T`: Ecken skalieren (Umschalt frei), Kanten strecken, außen drehen (Umschalt 15°), Alt um die Mitte; Enter übernimmt, Esc bricht ab. Mit Auswahl nur die ausgewählten Pixel (auch `V`) |
| Neigen, Verzerren, Perspektive | Bearbeiten › Transformieren oder in der Optionsleiste umschalten. Strg + Ecke ziehen verzerrt frei, Strg + Kante verschiebt die Kante. Neigen: Kanten gleiten an sich entlang. Perspektive: die Nachbarecke bewegt sich gegengleich. Verzerren und Perspektive gehen nur bei Pixelebenen (Text, Form und RAW erst rastern) |
| Drehen & Spiegeln | Beim Transformieren: Winkel eintippen, 90° links/rechts, 180°, spiegeln (Panel rechts). Ebene: Strg+Alt+Pfeil links/rechts/runter, Strg+Alt+`H`/`V`; ganzes Bild: zusätzlich Umschalt (Menüs Bild und Ebene) |
| Verflüssigen | Strg+Umschalt+`X`: `W` schieben, `R` rekonstruieren, `C` Strudel (Alt: links), `S` zusammenziehen, `B` aufblähen (Alt tauscht), `[` `]` Größe; Enter übernimmt, Esc bricht ab, Strg+`Z` nimmt den letzten Strich zurück |
| Zauberstab | Klick wählt ähnliche Farben, Umschalt/Alt kombinieren, `[` `]` Toleranz, `W` nochmal: zusammenhängend ↔ überall |
| Zuschneiden | `C`, Rahmen ziehen/anpassen, über den Rand hinaus vergrößert die Arbeitsfläche, Enter / Esc. Nichts wird gelöscht |
| Füllen & Löschen | Alt+Rücktaste füllt Auswahl (bzw. Ebene), `Entf` oder Rücktaste löscht ausgewählte Pixel |
| Ebenen | Strg+Umschalt+`N` neu, Strg+`J` duplizieren, `Entf` (ohne Auswahl) löschen, Alt+`[` `]` auswählen, Strg+`[` `]` verschieben, Strg+Alt+`G` Schnittmaske (an die Ebene darunter; Pfeil in der Ebenenliste, Deckkraft/Füllmethode der Basisebene gelten für die Gruppe) |
| Ebenen-Eigenschaften | Umschalt+Ziffer Deckkraft, Umschalt+`+`/`-` Füllmethode, Strg+`,` ein/aus, Strg+`G` gruppieren, Strg+Umschalt+`G` aufheben, `F7` Blend-Raum |
| Korrekturebenen | Ebene › Neue Korrekturebene: Helligkeit/Kontrast, Tonwerte, Gradationskurven, Belichtung, Dynamik, Farbton/Sättigung, Schwarzweiß, Farbbalance (Tiefen/Mitteltöne/Lichter, Luminanz erhalten), Kanalmixer (auch monochrom), Verlaufsumsetzung, Umkehren, Tontrennung, Schwellenwert. Alle bleiben bearbeitbar und gehen als Photoshop-Korrekturebenen in PSD |
| Filter | Bild › Filter: Gaußscher Weichzeichner, Unscharf maskieren, Hochpass, Tiefen/Lichter (hellt dunkle und dämpft helle Bereiche nach ihrer Umgebung), Rauschen; auch als Smartfilter |
| Ebenenmasken | Strg+Umschalt+`M` neu (aus Auswahl), `\` Maske bemalen ↔ Pixel, Strg+Alt+`M` ein/aus, Strg+Alt+Umschalt+`M` anwenden |
| Verlauf & Datei | Strg+`Z` / Strg+Umschalt+`Z` / Strg+`Y`, Strg+`S` speichern (im Hintergrund), Strg+Umschalt+`E` exportieren (im Hintergrund), Strg+Alt+Umschalt+`E` exportieren unter (Farbraum, 16 Bit, JPEG-Qualität) |
| Zwischenablage | Strg+`C` kopiert (Text beim Bearbeiten, sonst die Pixel der Ebene in der Auswahl), Strg+Umschalt+`C` aus allen Ebenen, Strg+`X` ausschneiden, Strg+`V` einfügen (Bild → neue Ebene, an der alten Stelle wenn aus mosaix; Text → in den bearbeiteten Text bzw. als neue Textebene). Auch in Textfeldern der Oberfläche |
| Tastaturbefehle | Bearbeiten › Tastaturbefehle … (Strg+Alt+Umschalt+`K`): jeden Menübefehl neu belegen (Kürzel anklicken, Tasten drücken; Rücktaste entfernt, Esc bricht ab); ein Kürzel, das ein anderer Befehl hatte, wird dort entfernt; „Standard“ bzw. „Alle zurücksetzen“ |
| Pinselspitzen | Bearbeiten › Pinsel importieren … (Strg+Alt+Umschalt+`B`): Photoshop-Pinsel (.abr) , GIMP-Pinsel (.gbr/.gih) oder ein Bild werden zu eigenen Spitzen (Ordner `~/.local/share/mosaix/brushes`); Auswahl im Spitzen-Menü der Optionsleiste, Umschalt+`B` schaltet durch alle |
| Sonstiges | `F12` Benchmark, `F1` Hilfe, Strg+`Q` beenden |

## Aufbau

- `crates/mosaix-core` – Dokumentmodell, Füllmethoden (CPU-Referenz), Verlauf, Kachel-Mathematik, View, Pinselstrich → Dabs (ohne GPU)
- `crates/mosaix-gpu` – wgpu-Engine: Kachel-Pools, Strich-Puffer, Composite-Cache, WGSL-Shader
- `crates/mosaix-io` – natives `.mosaix`-Format (append-only), Export
- `crates/mosaix-app` – Wayland (smithay-client-toolkit), Tablet-Protokoll, Eingabe, Loader, Headless-CLI, Agenten-Schnittstelle (`src/agent/`: Befehle, MCP, Socket)
