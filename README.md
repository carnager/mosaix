# imagine

GPU-first Bildeditor für Linux/Wayland – siehe [PLAN.md](PLAN.md).

Aktueller Stand: **Phase 2a – Auswahl & Masken** (dazu Ebenen, Undo, Füllmethoden, natives Format, Agenten-Schnittstelle). Siehe Stand in PLAN.md.

## Bauen & Starten

```sh
cargo run --release -- bild.jpg              # Bild öffnen (PNG, JPEG, WebP, TIFF, EXR, .imagine)
cargo run --release -- --new 30000x30000     # leeres Dokument
cargo run --release -- --demo 10000x10000    # 100-MP-Testbild
cargo run --release -- in.jpg --save out.imagine --export out.png   # ohne Fenster konvertieren
cargo run --release -- --demo 10000x10000 --bench-offscreen         # Benchmark ohne Fenster
cargo run --release --features profile -- …  # mit Tracy-Profiling
cargo test --workspace                       # inkl. Headless-GPU-Tests (Vulkan)
```

Voraussetzungen: Wayland-Compositor, Vulkan-Treiber mit Descriptor Indexing (RADV, ANV, NVIDIA).

## Für Agenten

imagine ist per **MCP** und JSON-Skript steuerbar, auch live im offenen Fenster:

```sh
claude mcp add imagine -- /pfad/zu/imagine mcp --attach
```

Details: [docs/agenten.md](docs/agenten.md)

## Bedienung

Die Oberfläche hat eine Menüleiste, links die Werkzeugleiste und rechts Panels für Farbe, Werkzeug-Einstellungen, Ebenen (Drag & Drop, Doppelklick zum Umbenennen, Klick auf das Masken-Symbol zum Bemalen) und Verlauf (Klick springt zurück). Unten ist eine Statusleiste mit Meldungen, ein Klick darauf öffnet alle Meldungen. Strg+`K` oder Strg+Umschalt+`P` öffnet die Befehlspalette, `Tab` blendet die Oberfläche aus. Strg+`N` neues Bild, Strg+`O` öffnen (mit „Zuletzt geöffnet“), Strg+Umschalt+`S` speichern unter. Alt+Rechtsziehen ändert Pinselgröße (waagrecht) und Härte (senkrecht). Pinsel, Farbe, Werkzeug und zuletzt geöffnete Dateien merkt sich imagine in `~/.config/imagine/settings.json`. Alle drei Minuten wird bei ungespeicherten Änderungen eine Wiederherstellungskopie nach `~/.local/state/imagine/autosave/` gesichert. Deine Datei bleibt dabei unberührt.

Tastenkürzel folgen Photoshop. Die Ebenenliste erscheint bis zum UI (Phase 3) im Terminal, die aktive Ebene im Fenstertitel.

| Aktion | Eingabe |
|---|---|
| Werkzeuge | `B` Pinsel, `U` Wischfinger, `E` Radierer, `M` Auswahlrechteck/-ellipse, `L` Lasso, `W` Zauberstab, `G` Verlauf linear/radial, `V` Verschieben, `C` Zuschneiden |
| Malen | linke Maustaste / Stift; Radierer-Ende des Stifts; `[` `]` Größe, Ziffern Deckkraft (Wischfinger: Stärke), `H` Härte, Umschalt+`S` Glättung, Umschalt+`B` Spitze, Umschalt+`J` Dynamik, `P` Farbe, `X` Schwarz↔Weiß |
| Kopierstempel / Reparatur | `S` Kopierstempel; `J` Bereichsreparatur (einfach über den Fleck malen, Quelle wird gesucht), nochmal `J` Reparaturpinsel. Stempel und Reparaturpinsel: Alt+Klick setzt die Quelle, dann malen. Der Reparaturpinsel gleicht beim Loslassen Farbe und Helligkeit an die Umgebung an. Optionen: alle Ebenen aufnehmen (auf leere Ebene stempeln), ausgerichtet. Ein Kreuz zeigt, woher gerade kopiert wird |
| Inhaltsbasiert füllen | Objekt großzügig auswählen, Umschalt+Rücktaste (Strg+Umschalt+Rücktaste: aus allen Ebenen lernen, z. B. auf leerer Ebene). Rechnet im Hintergrund |
| Text & Formen | `T` Text: Klick ins Bild legt Text an, Klick auf Text wählt ihn; bearbeitet wird unter Eigenschaften. Umschalt+`U` Form: Rechteck/Ellipse aufziehen. Bleiben scharf und editierbar, auch nach Strg+`T`; Strg+Alt+`R` rastert |
| Ansicht | Leertaste+Ziehen schwenken, Mausrad zoomen, `R`+Ziehen drehen, Strg+`0` einpassen, Strg+`1` 100 %, `Q` Maskierungsmodus-Ansicht |
| Auswahl | Aufziehen mit Umschalt/Alt/beidem = hinzufügen/abziehen/Schnittmenge; Strg+`A` / Strg+`D` / Strg+Umschalt+`I`; Umschalt+`F6` weiche Kante |
| Transformieren | Strg+`T`: Ecken skalieren (Umschalt frei), Kanten strecken, außen drehen (Umschalt 15°), Alt um die Mitte; Enter übernimmt, Esc bricht ab. Mit Auswahl nur die ausgewählten Pixel (auch `V`) |
| Verflüssigen | Strg+Umschalt+`X`: `W` schieben, `R` rekonstruieren, `C` Strudel (Alt: links), `S` zusammenziehen, `B` aufblähen (Alt tauscht), `[` `]` Größe; Enter übernimmt, Esc bricht ab, Strg+`Z` nimmt den letzten Strich zurück |
| Zauberstab | Klick wählt ähnliche Farben, Umschalt/Alt kombinieren, `[` `]` Toleranz, `W` nochmal: zusammenhängend ↔ überall |
| Zuschneiden | `C`, Rahmen ziehen/anpassen, über den Rand hinaus vergrößert die Arbeitsfläche, Enter / Esc. Nichts wird gelöscht |
| Füllen & Löschen | Alt+Rücktaste füllt Auswahl (bzw. Ebene), `Entf` oder Rücktaste löscht ausgewählte Pixel |
| Ebenen | Strg+Umschalt+`N` neu, Strg+`J` duplizieren, `Entf` (ohne Auswahl) löschen, Alt+`[` `]` auswählen, Strg+`[` `]` verschieben |
| Ebenen-Eigenschaften | Umschalt+Ziffer Deckkraft, Umschalt+`+`/`-` Füllmethode, Strg+`,` ein/aus, Strg+`G` gruppieren, Strg+Umschalt+`G` aufheben, `F7` Blend-Raum |
| Ebenenmasken | Strg+Umschalt+`M` neu (aus Auswahl), `\` Maske bemalen ↔ Pixel, Strg+Alt+`M` ein/aus, Strg+Alt+Umschalt+`M` anwenden |
| Verlauf & Datei | Strg+`Z` / Strg+Umschalt+`Z` / Strg+`Y`, Strg+`S` speichern (im Hintergrund), Strg+Umschalt+`E` exportieren (im Hintergrund), Strg+Alt+Umschalt+`E` exportieren unter (Farbraum, 16 Bit, JPEG-Qualität) |
| Sonstiges | `F12` Benchmark, `F1` Hilfe, Strg+`Q` beenden |

## Aufbau

- `crates/imagine-core` – Dokumentmodell, Füllmethoden (CPU-Referenz), Verlauf, Kachel-Mathematik, View, Pinselstrich → Dabs (ohne GPU)
- `crates/imagine-gpu` – wgpu-Engine: Kachel-Pools, Strich-Puffer, Composite-Cache, WGSL-Shader
- `crates/imagine-io` – natives `.imagine`-Format (append-only), Export
- `crates/imagine-app` – Wayland (smithay-client-toolkit), Tablet-Protokoll, Eingabe, Loader, Headless-CLI, Agenten-Schnittstelle (`src/agent/`: Befehle, MCP, Socket)
