# imagine – Plan für einen GPU-first Bildeditor unter Linux

> Ziel: Ein Photoshop-Ersatz, der sich **sofort** anfühlt, Spaß beim Bedienen macht
> und die GPU überall nutzt, wo es sinnvoll ist.
> Drei Säulen, gleich gewichtet: **Performance · Usability · Features.**

Zielhardware für Entwicklung: AMD RX 6700 XT (Navi 22, 12 GB VRAM, RADV/Vulkan), Wayland, Arch.

---

## 1. Leitprinzipien

1. **Der UI-Thread blockiert nie.** Alles, was > 2 ms dauern kann, läuft asynchron, ist abbrechbar und zeigt sofort eine Vorschau.
2. **Nur rechnen, was man sieht.** Dirty-Tiles + Level-of-Detail: Bei 25 % Zoom wird auf Mip-Stufe 2 komponiert (16× weniger Arbeit), Vollauflösung erst beim Reinzoomen/Export.
3. **Alles ist nicht-destruktiv, wo es geht.** Filter und Korrekturen sind Ebenen/Knoten mit Live-Parametern, keine modalen Dialoge mit „OK“-Knopf.
4. **Alles ist ein Kommando.** Jede Aktion ist ein serialisierbares Command → Undo, Makros, Scripting, Headless-Batch und Command-Palette fallen „gratis“ ab.
5. **Linux-only ist ein Feature.** Wir nutzen Wayland-Protokolle direkt (Tablet, Color-Management/HDR, Fractional Scaling, Presentation-Time) statt auf den kleinsten gemeinsamen Nenner zu zielen.
6. **Agenten sind Nutzer erster Klasse.** Jede Funktion ist auch als Befehl (MCP-Tool / JSON) erreichbar; Agenten können das Ergebnis *sehen* (gerenderte Vorschau), und sie arbeiten auf Wunsch live im selben Dokument wie der Mensch – mit gemeinsamem Undo.
7. **Messen statt glauben.** Performance-Budgets (Abschnitt 7) sind Teil der CI, Profiling (Tracy) ist ab Tag 1 eingebaut.

---

## 2. Technologie-Stack

| Bereich | Wahl | Begründung |
|---|---|---|
| Sprache | **Rust** | Speichersicherheit ohne GC-Pausen, gutes Threading (rayon), starkes GPU-Ökosystem |
| GPU-API | **wgpu** (Vulkan-Backend) mit nativen Features (`shader-f16`, `subgroup`, `push-constants`, `timestamp-query`, `texture-binding-array`) | Moderne API, WGSL-Compute, gut gewartet. Escape-Hatch: über `wgpu-hal` bzw. `ash` an rohes Vulkan, falls nötig |
| Fenster/Input | **smithay-client-toolkit** direkt auf Wayland (kein winit): `zwp_tablet_v2`, `wp_fractional_scale_v1` + `wp_viewporter`, `wp_cursor_shape`, Frame-Callbacks; später `wp_color_management_v1`, `wp_presentation` | winit liefert auf Wayland keine vollwertigen Tablet-Daten; direkter Protokollzugriff gibt volle Kontrolle (entschieden in Phase 0) |
| UI | **egui** + `egui_dock`, eigener Theme, eigene Widgets; im selben wgpu-Device wie die Canvas | Ein GPU-Kontext, keine Texturkopien zwischen Toolkit und Canvas; sehr schnell; Rerun/Graphite beweisen, dass es skaliert |
| Vektor & Text | **vello** (GPU-Compute 2D-Renderer), **parley/cosmic-text** + **swash** für Layout/Shaping | Vektor-Ebenen, Formen, Text und Pfade komplett auf der GPU |
| Farbmanagement | **lcms2** (ICC), optional **OpenColorIO** | Display-Transform wird als 3D-LUT in den finalen Shader gebacken |
| Parallelität | **rayon** (CPU-Jobs), eigener Render-Thread, `crossbeam` Channels | |
| Kompression | **zstd** / **lz4** | Tile-Swap, natives Dateiformat |
| Profiling | **tracy** (`tracing-tracy`) + GPU-Timestamp-Queries | |
| ML (optional) | **ort** (ONNX Runtime, ROCm/MIGraphX-EP) oder **burn** mit wgpu-Backend | Lokale Modelle, kein Cloud-Zwang |

**Referenzen zum Lernen (keine Basis):** *Krita* (Brush-Engine, Tablet-Handling, Tile-Speicher) und *Graphite* – Letzteres ist ein node-basierter **Vektor**-Editor, also kein Raster-Konkurrent, aber nützlich als Anschauungsmaterial für Rust + wgpu + vello in einer Desktop-App.

---

## 3. Architektur

```
┌──────────────────────────── imagine-app ────────────────────────────┐
│  Wayland/winit · Input (Tablet, Maus, Tastatur) · Keymap · Palette  │
│                                                                      │
│   ┌────────── imagine-ui (egui) ──────────┐   ┌──── Canvas-View ───┐ │
│   │ Panels · Docking · Inspector · HUDs   │   │ Pan/Zoom/Rotate    │ │
│   └───────────────┬───────────────────────┘   │ Overlays/Gizmos    │ │
│                   │ Commands                   └─────────┬──────────┘ │
└───────────────────┼──────────────────────────────────────┼────────────┘
                    ▼                                      │ View-Request (Rect, LOD)
          ┌──── imagine-core ────┐                         ▼
          │ Dokumentmodell       │──Snapshot──▶ ┌──── imagine-render ────┐
          │ (persistent, CoW)    │              │ Render-Graph (DAG)     │
          │ Commands · Undo      │              │ Dirty-Tracking · Cache │
          └──────────┬───────────┘              │ Scheduler · LOD        │
                     │                          └──────────┬─────────────┘
          ┌──────────▼───────────┐              ┌──────────▼─────────────┐
          │ imagine-tiles        │◀────────────▶│ imagine-gpu            │
          │ CPU-Tiles, Swap,     │   Upload/    │ Device, Tile-Pool      │
          │ Kompression          │   Readback   │ (Texture-Arrays), LRU  │
          └──────────────────────┘              └──────────┬─────────────┘
                                                ┌──────────▼─────────────┐
                                                │ imagine-ops (WGSL)     │
                                                │ Blend · Filter · Brush │
                                                └────────────────────────┘
   imagine-io (Formate)   imagine-color (ICC/LUT)   imagine-cli (headless)
```

### 3.1 Dokumentmodell
- Baum aus Knoten: Gruppe, Pixel-Ebene, Korrektur-Ebene, Filter-Ebene, Vektor-Ebene, Text-Ebene, Smart-Object, Maske (Pixel + Vektor), Clipping.
- **Persistente Datenstruktur** (`Arc` + Copy-on-Write auf Tile-Ebene). Ein Undo-Schritt kostet nur die tatsächlich geänderten Tiles. Snapshots für den Renderer sind O(1).
- Pro Dokument wählbare Präzision: **8 bit**, **16 bit float (Standard)**, **32 bit float**.

### 3.2 Tiles & Speicher
- Tile-Größe 256×256 (tunebar), **sparse**: leere/einfarbige Tiles werden nicht allokiert.
- GPU-Seite: Tile-Pool als große `Texture2DArray`s pro Format; Seitentabelle in einem Storage-Buffer. Shader adressieren Tiles indirekt („virtual texturing light“, ohne Sparse-Binding).
- Residency-Manager mit LRU und VRAM-Budget (z. B. 70 % von 12 GB). Nicht sichtbare Tiles wandern nach RAM (lz4), bei Druck auf Disk (zstd, mmap).
- **Mip-Pyramide pro Ebene**, inkrementell aktualisiert, damit herausgezoomte Ansichten billig sind.

### 3.3 Render-Graph
- Ebenenstapel wird zu einem DAG kompiliert; jeder Knoten = Compute-/Render-Pass.
- **Dirty-Regionen** propagieren durch den Graph (Blur vergrößert die Region um seinen Radius usw.).
- Cache für Zwischenergebnisse (z. B. Gruppen-Komposit unterhalb der aktiven Ebene) → beim Malen wird nur „Cache unten + aktive Ebene + Cache oben“ komponiert.
- **Progressive Verfeinerung:** Erst niedrige LOD sofort zeigen, dann in Vollauflösung nachrechnen. Jobs tragen eine Generation-ID und werden verworfen, wenn sich Parameter ändern (Slider ziehen ⇒ keine Rückstaus).
- Filter deklarieren, ob sie **skalierungsinvariant** sind (Blur mit skaliertem Radius) oder Vollauflösung brauchen.

### 3.4 Threads
- **UI-Thread:** Input, egui, Command-Dispatch. Nie GPU-Waits.
- **Render-Thread:** (zurückgestellt, siehe Stand Phase 2b) besitzt Queue, baut Frames, präsentiert (Mailbox/FIFO-relaxed, Frame-Pacing über `wp_presentation`).
- **Worker-Pool (rayon):** Decoding, Encoding, Kompression, CPU-Fallbacks, Histogramme.
- **IO-Thread(s):** Laden/Speichern, Autosave.

### 3.5 Farbe
- Arbeitsfarbraum: **linear** (Standard: linear Rec.2020 oder ACEScg wählbar), optional **„Photoshop-kompatibles Gamma-Blending“** – Nutzer erwarten den PS-Look bei Blend-Modi, Linear-Blending ist physikalisch korrekter. Beides anbieten, pro Dokument.
- Display-Transform (ICC des Monitors via lcms2) → 3D-LUT → letzter Shader-Pass.
- HDR/Wide-Gamut-Ausgabe über `wp_color_management_v1` (KWin/Mutter unterstützen das).
- Soft-Proofing (CMYK/Druck) als zusätzlicher LUT-Pass.

---

## 4. Feature-Katalog (nach Priorität)

### P0 – Fundament (ohne das ist es kein Ersatz)
- Öffnen/Speichern: PNG, JPEG, WebP, TIFF, AVIF, **JPEG XL**, EXR, natives Format
- Ebenen, Gruppen, Deckkraft, alle gängigen **Blend-Modi** (PS-Parität), Ebenenmasken, Clipping
- Pan / Zoom / **Canvas-Rotation**, flüssig bei jeder Dokumentgröße
- Pinsel, Radierer, Farbwähler, Füllen, Verlauf
- Auswahl: Rechteck, Ellipse, Lasso, Polygon, **Zauberstab (GPU-Floodfill)**, Farbbereich, Add/Sub/Intersect, Weiche Kante, Quick-Mask
- Freies Transformieren (Skalieren, Drehen, Verzerren, Perspektive), Zuschneiden mit Gerade-Richten
- Unendliches Undo/Redo mit History-Panel
- Kopieren/Einfügen mit Wayland-Clipboard (inkl. Bilddaten)

### P1 – Was es zum Arbeitswerkzeug macht
- **Korrektur-Ebenen (live):** Tonwert, Gradationskurven, Farbton/Sättigung, Farbbalance, Vibrance, Belichtung, Schwarzweiß, Kanalmixer, Selektive Farbe, Color Lookup (.cube), Verlaufsumsetzung, Schwellenwert, Tontrennung
- **Filter-Ebenen (live):** Gauß/Box/Bewegung/Lens-Blur (Bokeh), Unscharf maskieren, Hochpass, Rauschen hinzufügen/entfernen, Clarity/lokaler Kontrast, Verflüssigen (Liquify, Echtzeit-Displacement-Feld)
- Retusche: **Stempel, Reparaturpinsel (Poisson/Multigrid auf GPU), Bereichsreparatur, inhaltsbasiertes Füllen (PatchMatch auf GPU)**
- Pinsel-Engine v1: Druck/Neigung/Rotation-Kurven, Stabilisator, Textur-/Bild-Spitzen, Smudge/Mixer, ABR-Import
- Text-Ebenen (Shaping, OpenType-Features, Absatz-Text)
- Vektor-Formen und Pfade (Pen-Tool) über vello
- **PSD-Import/Export** (Ebenen, Masken, Blend-Modi, Korrektur-Ebenen soweit abbildbar) – entscheidend für Adoption
- Farbmanagement komplett, Soft-Proofing
- Warp, Puppet-Warp
- Ebenenstile: Schlagschatten, Kontur, Schein, Farbüberlagerung (als GPU-Effekte)

### P2 – Kür
- **RAW-Entwicklung** (`rawler`, Demosaicing auf GPU) als Camera-Raw-artiger Smart-Filter
- KI (lokal, optional): Motiv auswählen / Segmentierung (SAM-artig), generatives Füllen (Inpainting), Upscaling, Entrauschen
- Makros/Aktionen (Command-Log aufnehmen & abspielen), **Batch über CLI** (`imagine-cli run makro.toml *.jpg`)
- Plugins: **WASM** (wasmtime, sandboxed) + **eigene WGSL-Filter** als Plugin-Typ
- Scripting (Rhai oder Lua)
- Artboards, Slices/Export-As mit mehreren Größen
- Frequenztrennung als Ein-Klick-Workflow, Panorama/Focus-Stacking

### Bewusst NICHT (vorerst)
3D, Video-Timeline, Animation, Druckvorstufen-Spezialitäten. Fokus: **Foto-Bearbeitung, Compositing, Retusche, Digital Painting.**

---

## 5. Usability – „macht Spaß“

- **Command-Palette** (`Strg+K` / `Leertaste` kurz): jede Aktion, jedes Werkzeug, jede Ebene, jeder Filter durchsuchbar – mit Fuzzy-Suche und zuletzt verwendet.
- **Keymap-Presets:** Photoshop, GIMP, Krita – plus frei konfigurierbar. Umsteiger sollen in 5 Minuten produktiv sein.
- **On-Canvas statt Dialoge:** Pinselgröße/Härte per `Alt+Rechtsklick ziehen` (HUD), Transform-Gizmos, Verlauf direkt auf der Leinwand, Kurven-Korrektur durch Ziehen im Bild (Targeted Adjustment).
- **Scrubby-Slider:** Auf jedem Zahlenfeld-Label ziehen ändert den Wert; Shift = fein, Strg = grob. Werte auch als Ausdruck eingebbar (`120*2`).
- **Live-Vorschau beim Hovern:** Blend-Modi, Filter-Presets, LUTs – Vorschau ohne Klick.
- **Kein Spinner im Normalbetrieb.** Wenn etwas länger dauert: progressive Vorschau + dezente Fortschrittsanzeige, jederzeit abbrechbar.
- Flüssige Animationen (Zoom mit Trägheit, Panel-Übergänge) – aber jederzeit abschaltbar.
- Tabs + schneller Dokumentwechsel, **Crash-Recovery und Autosave** (inkrementell, kostet nichts).
- Konsistente, ehrliche Fehlermeldungen; nie Datenverlust.
- Tooltips mit Shortcut, „Was ist das?“-Hover-Erklärungen.
- Barrierearmut: skalierbares UI (fractional scaling), Kontrast-Theme, alles per Tastatur erreichbar.

---

## 6. Natives Dateiformat (`.imagine`)

- Container mit Manifest (Dokumentbaum, JSON/CBOR) + Tile-Blobs (zstd).
- **Append-only/inkrementell:** Speichern schreibt nur geänderte Tiles → Speichern großer Dateien in Millisekunden. Gelegentliches Kompaktieren im Hintergrund.
- Eingebettete Vorschau (für Dateimanager-Thumbnailer, `.thumbnailer`-Datei mitliefern).
- Versioniert, mit Vorwärts-Kompatibilitätsregeln.
- Export nach OpenRaster (`.ora`) für Interop mit Krita/GIMP.

---

## 7. Performance-Budgets (Referenz: RX 6700 XT, Ryzen-Desktop)

| Szenario | Budget |
|---|---|
| Kaltstart bis bedienbar | < 400 ms |
| 50-MP-JPEG öffnen → erste Pixel sichtbar | < 300 ms (volle Auflösung < 800 ms) |
| Pan/Zoom/Rotation, 100 Ebenen, 8K-Dokument | Bildschirm-Refreshrate (bis 165 Hz), keine Frame-Drops |
| Pinsel: Input → Pixel im Swapchain-Bild | < 1 Frame (+ Compositor) |
| Slider an Korrektur-Ebene (24 MP) | Vorschau im selben Frame |
| Gauß 100 px auf 24 MP, volle Auflösung | < 30 ms |
| Inhaltsbasiertes Füllen, 1 MP Loch | < 1 s (progressive Vorschau sofort) |
| Speichern (inkrementell) nach Pinselstrich | < 50 ms |
| RAM bei 8K × 8K × 50 Ebenen (16F) | Bleibt im Budget dank Sparse-Tiles + Swap |

Gemessen in einer **Benchmark-Suite** (criterion + GPU-Timestamps), Regressionen > 5 % brechen den Build.

---

## 8. Crate-Struktur

```
imagine/
├── crates/
│   ├── imagine-core     # Dokumentmodell, Commands, Undo (kein GPU-Code)
│   ├── imagine-tiles    # Tile-Speicher, CoW, Swap, Kompression
│   ├── imagine-gpu      # Device, Tile-Pool, Residency, Uploads
│   ├── imagine-render   # Render-Graph, Scheduler, LOD, Caches
│   ├── imagine-ops      # WGSL-Shader: Blend, Filter, Korrekturen
│   ├── imagine-brush    # Pinsel-Engine, Stroke-Verarbeitung, Stabilisator
│   ├── imagine-select   # Auswahl-Algorithmen
│   ├── imagine-color    # ICC, LUTs, Farbräume
│   ├── imagine-io       # Formate (PSD, JXL, EXR, …), natives Format
│   ├── imagine-ui       # egui-Panels, Widgets, Theme, Palette
│   ├── imagine-app      # Binary: Fenster, Wayland, Input, Glue
│   └── imagine-cli      # Headless Batch-Verarbeitung
├── shaders/             # WGSL, Hot-Reload im Debug-Build
├── benches/
└── tests/golden/        # Referenzbilder für Pixel-Tests
```

---

## 9. Qualitätssicherung

- **Golden-Image-Tests:** Jede Operation gegen Referenzbilder (mit Toleranz, ΔE-Metrik). Blend-Modi zusätzlich gegen Photoshop-Referenzexporte.
- **Fuzzing** aller Datei-Parser (`cargo-fuzz`) – PSD und RAW sind Minenfelder.
- Property-Tests für Undo/Redo (zufällige Command-Folgen → Undo-alles = Ursprungszustand).
- **Shader-Hot-Reload** und ein Debug-Overlay (Tile-Grenzen, Dirty-Regionen, Residency, Frame-Timing).
- CI: Build + Tests + Benchmarks; GPU-Tests auf lavapipe (CPU-Vulkan) in CI, echte GPU lokal.

---

## 10. Roadmap

| Phase | Dauer (grob, 1 Person) | Ergebnis |
|---|---|---|
| **0 – Spike** | 2–4 Wochen | Fenster auf Wayland, wgpu-Device, **Tablet-Input mit Druck**, sparse Tile-Canvas, riesiges Bild öffnen, Pan/Zoom/Rotate, Tracy. *Go/No-Go für den Stack.* |
| **1 – Kern** | 6–8 Wochen | Dokumentmodell, Ebenen/Gruppen, Blend-Modi, Undo, Render-Graph mit Dirty-Tracking & LOD, natives Format, PNG/JPEG/WebP |
| **2 – Malen & Auswählen** | 6–8 Wochen | Pinsel-Engine v1, Radierer, Füllen, Verlauf, Auswahl-Werkzeuge, Masken, Transform, Crop |
| **3 – UI-Politur (MVP)** | 4 Wochen | Docking-Layout, Command-Palette, Keymaps, HUDs, Scrubby-Slider, Autosave → **erstes benutzbares Release** |
| **4 – Nicht-destruktiv** | 6–8 Wochen | Korrektur- & Filter-Ebenen, Liquify, Farbmanagement komplett |
| **5 – Retusche** | 6 Wochen | Stempel, Reparaturpinsel, inhaltsbasiertes Füllen |
| **6 – Text & Vektor** | 6 Wochen | vello-Integration, Text-Ebenen, Formen, Pfade, Ebenenstile |
| **7 – Interop** | 6 Wochen | PSD Import/Export, TIFF/EXR/JXL/AVIF komplett, ABR-Pinsel |
| **8 – Erweiterbarkeit & Kür** | fortlaufend | Makros, CLI-Batch, WASM-Plugins, RAW, lokale KI |

Realistisch: **MVP nach ~5–6 Monaten**, „ernsthafter Photoshop-Ersatz für Foto/Retusche“ nach ~12–18 Monaten Vollzeit.

---

## 11. Risiken & Gegenmaßnahmen

| Risiko | Gegenmaßnahme |
|---|---|
| Tablet-Support auf Wayland (winit) unvollständig | Phase 0 klärt das zuerst; direkte `tablet-v2`-Anbindung über eine zweite Event-Queue auf derselben Wayland-Verbindung |
| wgpu-Limits (kein Sparse-Binding, f16-Storage) | Indirektes Tile-Addressing statt Sparse; `wgpu-hal`/`ash` als Ausweg für einzelne Passes |
| PSD-Kompatibilität ist endlos | Klarer Kompatibilitätsumfang, Fallback: gerasterte Ebene + Warnung, Testkorpus realer Dateien |
| egui wirkt „wie ein Dev-Tool“ | Früh in eigenes Theme/Widgets investieren; UI hinter eigener Schicht kapseln, damit ein Wechsel möglich bliebe |
| Scope-Explosion (Photoshop hat 35 Jahre Vorsprung) | Strenge Priorisierung (P0/P1/P2), Fokus auf Foto/Retusche/Painting; jede Phase endet mit etwas Benutzbarem |
| Treiberbugs (RADV ist gut, aber …) | Golden-Tests auf mehreren Treibern (RADV, NVIDIA proprietär, Intel ANV, lavapipe) |

---

## 12. Offene Entscheidungen

1. **UI-Toolkit:** egui (empfohlen, schnellster Weg, ein GPU-Kontext) vs. eigenes Retained-UI vs. Qt/QML-Shell mit Rust-Kern (bestes Tablet/IME-Handling, aber zwei Sprachen).
2. **Standard-Blending:** linear (korrekt) oder Gamma (PS-kompatibel) als Default?
3. **X11-Support:** komplett weglassen oder später via winit als Fallback?
4. **Lizenz:** GPL-3 (wie GIMP/Krita) oder MPL-2/Apache-2?

---

## Stand Phase 0 (20.11.2026)

Umgesetzt und gemessen auf RX 6700 XT / RADV / niri, 1872×2046-px-Fenster:

- Sparse Kachel-Canvas (256², RGBA16F linear, premultiplied) in einem wachsenden GPU-Texture-Array mit kompletter Mip-Kette pro Kachel; Mips per Compute nur für geänderte Kacheln
- Pinsel/Radierer per Hardware-Blending direkt in die Kacheln, Druck → Größe/Deckkraft, Glättung, gleichmäßiger Dab-Abstand
- Tablet über `tablet-v2` (Druck, Neigung, Stifttasten, Radierer-Ende), Fractional Scaling, Frame-Callback-getriebenes Rendering (Mailbox)
- Streaming-Laden: Dokumentgröße sofort, Kacheln parallel konvertiert, von der Mitte nach außen hochgeladen
- Benchmark eingebaut (`--bench`, F12)

| Messung | Ergebnis | Budget |
|---|---|---|
| 100 MP (10000², 1600 Kacheln), Zoom/Rotation/Pan | 144 fps (= Monitor), CPU Ø 0,07–0,2 ms/Frame | Refreshrate ✅ |
| 48-MP-JPEG (38 MB, Worst-Case-Rauschen) | dekodiert 271 ms, komplett 354 ms | < 300 / < 800 ms ✅ |

**Bekannte Grenzen (bewusst, für Phase 1):** ein einziges Texture-Array → max. 2048 Kacheln (~130 MP voll belegt; leere Dokumente beliebig groß); keine Ebenen, kein Undo; Strich-Deckkraft akkumuliert (kein Stroke-Buffer); Rendering noch im UI-Thread; Tablet-Druck erst mit echtem Stift verifizieren.

## Stand Phase 1 (20.11.2026) – Engine

Umgesetzt (Bedienung noch per Tastatur, Ebenenliste im Terminal – das UI kommt in Phase 3):

- **Dokumentmodell:** Ebenen, Gruppen (isoliert, beliebig tief bis 7), Deckkraft, Sichtbarkeit, 19 Füllmethoden (W3C/Photoshop-Formeln) mit CPU-Referenz; GPU weicht max. 7,6·10⁻⁴ ab
- **Blend-Raum pro Dokument:** sRGB (Photoshop-kompatibel, Standard) oder linear – F7. *Offene Entscheidung 2 damit vorläufig entschieden.*
- **Undo/Redo auf der GPU:** Dokument-Schnappschüsse mit geteilten, ref-gezählten Kacheln; ein Schritt kostet nur die geänderten Kacheln; Ebene duplizieren kostet nichts
- **Strich-Puffer:** Fluss baut innerhalb eines Strichs auf, Deckkraft begrenzt ihn (wie Photoshop); Radierer über denselben Mechanismus
- **Kachel-Pools in Blöcken** (256 Kacheln/Block, Binding-Arrays): kein Kopieren beim Wachsen, bis 16 384 Ebenen- + 16 384 Bildkacheln – das 130-MP-Limit aus Phase 0 ist weg
- **Composite-Cache mit Signaturen:** pro Kachel wird das Ebenen-Programm gehasht; nur Kacheln mit geänderter Signatur werden neu zusammengesetzt – sichtbare zuerst, im Budget pro Frame
- **Natives Format `.imagine`:** zstd-Kacheln + JSON-Manifest, append-only (Speichern schreibt nur geänderte Kacheln), Wiederherstellung nach Absturz beim Anhängen, Kompaktierung bei viel totem Platz
- **Export** PNG/JPEG/WebP/TIFF; **Headless-CLI** (`--save`, `--export`, `--bench-offscreen`)

| Messung (RX 6700 XT) | Ergebnis |
|---|---|
| Darstellung 100 MP / 192 MP, 2560×1440, inkl. GPU-Wartezeit | Ø 0,14 / 0,17 ms pro Frame |
| Füllmethode ändern → 100 MP, 2 Ebenen komplett neu zusammensetzen (inkl. Mips) | 10–11 ms |
| 100 MP `.imagine` speichern (komplett) / laden | 457 ms / 96 ms |
| inkrementell speichern nach Pinselstrich | nur berührte Kacheln (Test: 1 von 16) |
| PNG → .imagine → PNG | bitgenau identisch |

**Bekannte Grenzen:** kein UI (Phase 3), kein Datei-Dialog (Speichern neben die Quelldatei); Speichern läuft noch im UI-Thread (erstes Speichern 100 MP ≈ 0,5 s Pause, danach inkrementell); Pinsel-Kanten werden linear geblendet; Pass-Through-Gruppen fehlen; Kacheln liegen komplett im VRAM (Auslagerung = Residency-Manager, später); Fenster-Benchmark braucht ein sichtbares Fenster (niri drosselt verdeckte Fenster) – dafür gibt es `--bench-offscreen`.

## Agenten-Schnittstelle (seit 20.11.2026)

Querschnittsthema neben den Phasen; Details in [docs/agenten.md](docs/agenten.md).

- Gemeinsame Befehlsschicht (22 Befehle: Dokument, Ebenen, Malen, Füllen, Bild platzieren, Farbe abfragen, Vorschau, Verlauf, Speichern/Export) mit strikter Parameterprüfung
- **MCP-Server** über stdio (`imagine mcp`), Vorschau als Bild-Content
- **Live-Modus** `--attach`: Unix-Socket pro Fenster, Agent und Mensch teilen Dokument und Undo-Verlauf
- **JSON-Lines-Skripte** (`imagine run`) für Shell-Agenten und Batch
- Getestet: Protokoll-Handshake, Tool↔Befehl-Konsistenz, kompletter Agenten-Ablauf gegen die echte GPU-Engine, Live-Steuerung eines Fensters, Aufräumen verwaister Sockets

**Regel für alle folgenden Phasen:** Jedes neue Werkzeug bekommt sofort seinen Befehl und sein MCP-Tool. Die Tastaturkürzel der App laufen mittelfristig ebenfalls über die Befehlsschicht, dann gibt es Makros und Aufzeichnung gratis.

## Stand Phase 2a (20.11.2026) – Auswahl & Masken

- **Masken-Infrastruktur:** eigener R8-Kachel-Pool; einheitliche Kacheln bleiben `Solid` (kein Speicher) – eine Rechteckauswahl über 100 MP belegt nur die Randkacheln
- **Auswahl:** Rechteck, Ellipse, Lasso/Polygon mit Kantenglättung (4× vertikal, analytisch horizontal); hinzufügen/abziehen/Schnittmenge (Photoshop-Modifier), alles/nichts/umkehren, weiche Kante (3× Box ≈ Gauß); Ameisenlinie über eine Bildschirm-Maske (immer 1 px scharf, animiert), Maskierungsmodus-Ansicht
- **Auswahl begrenzt alles:** Pinsel (schon in der Live-Vorschau), Füllen, Löschen, Verlauf – auch weiche Auswahlkanten
- **Ebenenmasken** für Ebenen und Gruppen: anlegen (aus Auswahl), bemalen (Live-Vorschau), ein/aus, anwenden; im Zusammensetz-Programm, Masken-Tile mit Wert 0 überspringt die Ebene ganz
- **Apply-Kernel:** ein GPU-Pfad für Strich übernehmen, Füllen, Verlauf (sRGB-Interpolation wie Photoshop), Löschen, Verschieben (Pixel + Maske), Maske anwenden – immer als neue Kacheln, also ein Undo-Schritt
- **Werkzeuge im Fenster:** M/L/G/V/B/E, Overlays für Rahmen, Lasso, Verlaufslinie und Verschiebe-Umriss
- **Speichern im Hintergrund:** Schnappschuss + asynchroner GPU-Readback, Komprimieren/Schreiben im Worker-Thread; Weiterarbeiten währenddessen
- **Format v2:** Ebenenmasken und Auswahl werden gespeichert (v1-Dateien bleiben lesbar)
- **Agenten:** 15 neue Befehle (Auswahl, Füllen, Verlauf, Masken, Verschieben), Vorschau mit Auswahl-Einfärbung

| Messung (100 MP, RX 6700 XT, headless) | Ergebnis |
|---|---|
| Füllen / Verlauf (1600 Kacheln) | 27 ms / 19 ms |
| Ellipse auswählen / Rechteck abziehen | 49 ms / 15 ms |
| Weiche Kante 40 px | 428 ms (CPU) |
| Ebene samt Maske verschieben | 27 ms |
| Darstellung (Regression) | Ø 0,14 ms/Frame, unverändert |

**Bekannte Grenzen:** ~~Verschieben zeigt beim Ziehen nur den Umriss~~ (inzwischen Live-Vorschau des Inhalts); weiche Kante und Auswahl-Kombination laufen auf der CPU; Inhalte, die über den Rand geschoben werden, werden abgeschnitten; Ameisen und Werkzeug-Overlays sind nur per Offscreen-Test abgesichert (niri zeigt das Testfenster nicht an) – bitte von Hand ausprobieren.

## Stand Phase 2b (20.11.2026) – Transformieren, Zauberstab, Pinsel

- **Warp-Kernel:** affines Resampling ganzer Ebenen auf der GPU (nächster Nachbar / bilinear / bikubisch Catmull-Rom; beim Verkleinern Supersampling). Quellkacheln werden über eine Nachschlagetabelle gefunden, Zielkacheln über einen exakten Parallelogramm-Kachel-Test bestimmt. Ersetzt den alten Verschiebe-Pfad.
- **Schwebende Auswahl:** Mit Auswahl bewegen/transformieren V und Strg+T nur die ausgewählten Pixel. Die Auswahl wandert mit, das Loch bleibt zurück (wie Photoshop).
- **Live-Vorschau:** Beim Verschieben und Transformieren wird nur der sichtbare Bereich (plus Rand) neu berechnet, ohne Undo-Schritt. Bei Ansichtswechsel wird nachgerechnet.
- **Inhalt außerhalb der Leinwand bleibt erhalten** (wird nur nicht angezeigt). Zuschneiden ist damit verlustfrei, das Vergrößern der Arbeitsfläche holt alles zurück. Zuschneiden auf Kachelgrenzen nummeriert nur Kacheln um.
- **Frei transformieren (Strg+T):** Rahmen mit 8 Anfassern. Ecken skalieren proportional (Umschalt = frei), Kanten strecken, Alt = um die Mitte, außen drehen (Umschalt = 15°), innen verschieben, Enter/Esc. Inhaltsgrenzen pixelgenau per GPU-Reduktion.
- **Zuschneiden (C)**, Arbeitsfläche und Bildgröße (Agenten), alle Ebenen und Masken.
- **Zauberstab (W) und Farbbereich:** GPU-Ähnlichkeitskarte (aktive Ebene oder Gesamtbild), Scanline-Füllung auf der CPU mit weichem 1-px-Rand, Toleranz wie Photoshop in 8-Bit-Stufen.
- **Pinsel-Engine v1:** fünf prozedurale Textur-Spitzen (Kreide, Borsten, Spray, rau, Kohle) mit Mipmaps; Winkel, Rundheit, Richtung folgen; Zufall für Größe, Winkel, Fluss und Streuung (deterministisch pro Strich).
- **Wischfinger (U):** Strich-Kacheln starten als Kopie der Ebene. Je Tupfer nimmt ein 128²-Puffer Farbe auf und wird durch die Spitze gestempelt (Tupfer nacheinander, voneinander abhängig).
- **UI bleibt bedienbar:** weiche Kante, Zauberstab-Füllung und Export-Kodierung laufen in Worker-Threads, der Export liest asynchron zurück. Ergebnisse auf veralteter Auswahl werden verworfen.
- **Agenten:** `layer_transform`, `layer_bounds`, `crop`, `canvas_size`, `image_resize`, `select_magic_wand`, `select_color_range`; `paint_stroke` mit Spitzen, Dynamik und `smudge`.

| Messung (100 MP, RX 6700 XT, headless) | Ergebnis |
|---|---|
| Drehen 17° bikubisch | ≈ 40 ms |
| Skalieren ×1,3 bikubisch (→ 169 MP) | ≈ 73 ms |
| Ganzpixliges Verschieben (exakt) | ≈ 20 ms |
| Zuschneiden außerhalb / auf Kachelgrenzen | ≈ 20 ms / < 1 ms |
| Inhaltsgrenzen (GPU-Reduktion) | ≈ 19 ms |
| Zauberstab zusammenhängend / überall | 120–190 ms / 70 ms |
| Farbbereich | 70 ms |

**Entscheidung Render-Thread:** Ein eigener Render-Thread hätte das eigentliche Problem nicht gelöst. Die GPU-Arbeit ist bereits asynchron, und Eingaben laufen ohnehin über den UI-Thread. Gehakt hat es bei schwerer CPU-Arbeit auf dem UI-Thread, die jetzt in Worker-Threads läuft. Ein echter Render-Thread wird wieder interessant, wenn egui (Phase 3) Layout-Zeit kostet.

**Bekannte Grenzen:**
- Die Vorschau beim Transformieren rechnet bei 100 MP in eingepasster Ansicht alle sichtbaren Kacheln in voller Auflösung, das sind etwa 40 ms je Änderung. Eine Vorschau in reduzierter Auflösung fehlt noch.
- ~~Gruppen lassen sich noch nicht transformieren.~~ Erledigt 2026-11-20: Verschieben und Frei transformieren wirken auf ganze Gruppen (alle Ebenen, Masken, Text/Formen bleiben editierbar).
- Der Wischfinger nimmt nur von der aktiven Ebene auf.
- Pinselspitzen sind fest eingebaut (eigene Spitzen und ABR kommen mit Phase 7).
- Neue Fensterwerkzeuge (Strg+T, C, W, U, Umschalt+B, J) sind nur über Engine-, Agenten- und Geometrie-Tests abgesichert und brauchen den Handtest.

## Stand Phase 3 (20.11.2026) – Oberfläche (MVP)

- **egui** (0.37, egui-wgpu auf derselben wgpu-30-Instanz), ohne winit: Wayland-Eingaben (Maus, Stift, Tastatur, Modifier, Tastenwiederholung) werden selbst übersetzt. Die Oberfläche wird nach der Leinwand im selben Frame gezeichnet.
- **Eingabe-Routing:** Klicks über Panels oder Fenstern gehen an die Oberfläche, auf der Leinwand begonnene Striche bleiben bei der Leinwand. Auch der Stift bedient die Oberfläche. Textfelder und die Palette bekommen die Tastatur, sonst gelten die Tastenkürzel.
- **Layout:** Menüleiste (alle Befehle mit Kürzeln), Werkzeugleiste mit selbst gezeichneten Vektor-Symbolen, rechte Seitenleiste mit Farbwähler (eigenes HSV-Feld, Farbfelder, Hex), kontextabhängigen Werkzeug-Einstellungen (Pinsel inkl. Spitze und Dynamik, Wischfinger, Zauberstab, Verlauf), Ebenen (Sichtbarkeit, Füllmethode, Deckkraft mit einem Undo-Schritt pro Ziehen, Drag & Drop inkl. in Gruppen, Umbenennen, Masken-Symbol) und Verlaufsliste mit Sprüngen. Statusleiste mit Zoom, Größe, Cursor-Position, Hintergrund-Arbeit und der letzten Meldung (alle `tracing`-Meldungen landen im Meldungsfenster).
- **Befehlspalette** (Strg+K / Strg+Umschalt+P) und Menüs nutzen eine Befehlsliste, die die Tastenkürzel abspielt. Menü, Palette und Tastatur können so nicht auseinanderlaufen.
- **Dialoge:** Arbeitsfläche (mit Anker) und Bildgröße (Seitenverhältnis).
- **Automatisches Sichern:** alle 3 min bei Änderungen eine vollständige Kopie im Hintergrund nach `~/.local/state/imagine/autosave/`. Die eigene Datei und das inkrementelle Speichern bleiben unberührt. Nach dem Speichern bzw. Beenden wird die Kopie gelöscht, verwaiste Kopien werden beim Start gemeldet.
- **Einpassen** berücksichtigt die Panels. `Tab` blendet die Oberfläche aus.
- **Kosten:** etwa 0,4 ms pro UI-Durchlauf inkl. GPU. Ohne Eingabe und ohne Modelländerung wird das letzte UI-Bild wiederverwendet, beim Malen läuft also kein UI-Durchlauf pro Frame.
- **Prüfung ohne Fenster:** `imagine --ui-snapshot OUT.png [DATEI]` rendert Leinwand und Oberfläche offscreen (Layout-Kontrolle per Bild).

- **Nachgereicht:**
  - Dateidialoge (Öffnen mit „Zuletzt geöffnet“, Speichern unter, Exportieren unter) und Neu-Dialog (Vorgaben, weiß/transparent), Dokumente zur Laufzeit wechseln.
  - Rückfrage bei ungespeicherten Änderungen (Speichern / Verwerfen / Abbrechen), auch beim Beenden. Speichern ohne eigene Datei fragt nach dem Ort.
  - Einstellungen in `~/.config/imagine/settings.json` (Pinsel, Farbe, Werkzeug, Zauberstab, Wischfinger, zuletzt geöffnet, letzter Ordner).
  - Pinsel-HUD per Alt+Rechtsziehen.

- **Ebenen-Vorschaubilder:** GPU-verkleinert (Warp-Kernel, bis 8×8 Supersampling) in ein 64²-Textur-Array, nur bei geänderten Kacheln neu, höchstens 4 pro Frame und nie während eines Strichs.

**Phase 3 abgeschlossen** (2026-11-20): zuletzt Vorschaubilder für Gruppen (gestapelte Ebenen-Vorschauen) und Masken (auf der GPU verkleinert, grau, rotes Kreuz wenn aus). (Frei belegbare Tastenkürzel: erledigt 2026-11-20 – Bearbeiten › Tastaturbefehle, Strg+Alt+Umschalt+K; gespeichert in den Einstellungen; Konflikte wie Photoshop.) (Docking/abtrennbare Panels: erledigt 2026-11-20 mit egui_dock, Layout wird gespeichert.)

## Stand Phase 4 (20.11.2026) – Nicht-destruktiv, Filter, Farbe (erster Teil)

- **Einstellungsebenen:** Helligkeit/Kontrast, Tonwertkorrektur, Gradationskurven (monoton-kubisch), Belichtung (linear), Dynamik, Farbton/Sättigung (inkl. Färben), Schwarzweiß, Umkehren, Tontrennung, Schwellenwert.
  - Kanalweise Arten werden zu einer 256er-Tabelle gebacken, die übrigen rechnet der Flatten-Shader direkt. Eine CPU-Referenz prüft jede Art auf der GPU.
  - Jede Einstellungsebene hat eine Maske (aus der Auswahl), die live bemalt werden kann. Deckkraft mischt.
  - Füllmethoden wirken wie bei einer Ebene, die das eingestellte Bild enthält und auf das Darunterliegende beschnitten ist (z. B. Kurven im Modus Luminanz).
  - Eigenschaften-Panel mit Kurven-Editor, Dateiformat v3.
- **Filter (GPU):** Gaußscher Weichzeichner (separabel; ab 10 px Radius auf einer verkleinerten Kopie, deckungsgleich mit der analytischen Sprungantwort), Unscharf maskieren, Hochpass, Rauschen. Sie wirken durch die Auswahl, als ein Undo-Schritt, mit Live-Vorschau des sichtbaren Bereichs.
- **Farbmanagement beim Import:** Eingebettete ICC-Profile (z. B. Display P3, Adobe RGB) werden mit moxcms nach linearem sRGB im erweiterten Bereich umgerechnet. Farben außerhalb von sRGB bleiben in f16 erhalten, statt abgeschnitten zu werden.
- **Histogramm** in Tonwertkorrektur (mit ziehbaren Reglern für Tiefen, Gamma, Lichter) und Kurven. Es zeigt das Bild unterhalb der Einstellungsebene: Dieser Teil wird stapelweise in temporäre Kacheln geflattet, eine Mip-Stufe gelesen und gezählt; zwischengespeichert, bis sich darunter etwas ändert.
- **Smart-Filter:** Filter können nicht-destruktiv an einer Pixelebene hängen (Option im Filterdialog). Sie stehen unter der Ebene in der Ebenenliste, lassen sich ein- und ausschalten, per Klick bearbeiten (mit Live-Vorschau), entfernen oder in die Pixel übernehmen. Die Engine rechnet die Filterkette einmal pro Änderung über die ganze Ebene und speichert das Ergebnis zwischen. Composite, Histogramm, Pipette und Export sehen das gefilterte Bild; Dateiformat v4.
- **Verflüssigen** (Strg+Umschalt+X): Schieben, Rekonstruieren, Strudel, Zusammenziehen, Aufblähen. Ein Verschiebungsfeld (alle 8 px ein Stützpunkt, bei Riesenbildern gröber) wird auf der CPU bemalt; die GPU tastet immer die unveränderte Ebene bikubisch an der verschobenen Stelle ab, also kein Schärfeverlust durch wiederholtes Neuabtasten. Die Vorschau rechnet nur berührte, sichtbare Kacheln. Strg+Z nimmt innerhalb der Sitzung Striche zurück; Übernehmen ist ein Undo-Schritt.
- **Export mit Farbprofil:** sRGB, Display P3, Adobe RGB oder Rec. 2020, 8 oder 16 Bit (PNG/TIFF), JPEG-Qualität. Weite Farbräume betten ihr ICC-Profil ein (PNG, JPEG, WebP, TIFF), die Umrechnung geht vom linearen Arbeitsfarbraum aus, also bleiben P3-Farben aus dem Import erhalten. Die Einstellungen merkt sich das Programm.
- **Agenten:** `adjustment_add`, `adjustment_set`, `filter_apply`, `histogram`; `export` mit `color_space`, `bit_depth`, `quality`; `filter_apply` mit `smart`, `smart_filter_set`, `liquify`.

| Messung (100 MP, RX 6700 XT) | Ergebnis |
|---|---|
| Gauß r = 2 / 10 / 40 / 200 | 52 / 114 / 64 / 29 ms |
| Unscharf maskieren r = 2 | 51 ms |
| Rauschen | 17 ms |
| Kurven-Einstellungsebene (kompletter Neuaufbau) | 42 ms |
| 24-MP-PNG mit Display-P3-Profil laden | 133 ms (ohne Profil 67 ms) |
| Verflüssigen 100 MP: Tupfer r = 300 / Übernehmen | 0,09 ms / 21 ms |
| Histogramm unter einer Ebene | 36 ms, danach aus dem Cache 0,08 ms |
| 24-MP-TIFF exportieren: sRGB / Display P3 / P3 16 Bit | 65 / 112 / 118 ms |

**Noch offen in Phase 4:**
- Monitorprofil: zurückgestellt (niri kann kein Wayland-Farbmanagement, der Nutzer hat kein Profil)

## Stand Phase 5 (20.11.2026) – Retusche (begonnen)

- **Kopierstempel** (`S`): Alt+Klick setzt die Quelle, ein Kreuz zeigt beim Malen, woher kopiert wird. Ausgerichtet (Abstand bleibt) oder nicht; Quelle ist die Ebene oder das ganze Bild (dann auch auf eine leere Ebene). Läuft über die Arbeitskopie des Wischfingers, jeder Tupfer kopiert pixelgenau mit ganzzahligem Versatz (kein Weichzeichnen durch Neuabtasten). Agent: `paint_stroke` mit `clone_offset`, `clone_all_layers`.
- **Reparaturpinsel** (`J`): malt wie der Stempel; beim Loslassen wird der Bereich (Vereinigung der Pinseltupfer) per Poisson-Angleich in die Umgebung eingepasst – Struktur von der Quelle, Farbe/Helligkeit vom Rand. Löser auf der CPU: kaskadisches Multigrid mit rot-schwarzem SOR, parallel über Zeilen (4 MP Bereich: 97 ms, 0,25 MP: 10 ms). Agent: `paint_stroke` mit `heal: true`.
- **Bereichsreparatur** (`J`, nochmal `J` wechselt zum Reparaturpinsel): über den Fleck malen, beim Loslassen sucht die Engine selbst eine Quelle – 64 Kandidaten-Versätze (16 Richtungen × 4 Abstände) werden über die Gradienten im Ring um die Stelle verglichen – und heilt von dort. Agent: `paint_stroke` mit `spot_heal: true`.
- **Inhaltsbasiert füllen** (Umschalt+Rücktaste; Strg+Umschalt+Rücktaste lernt aus allen Ebenen): füllt die Auswahl mit Struktur aus der Umgebung (Wexler/PatchMatch, mehrstufig). Gröbste Stufe: „Zwiebelschalen“-Füllen von außen nach innen; pro Stufe PatchMatch (Weitergabe + Zufallssuche, parallel) und gewichtete Abstimmung (Ähnlichkeit × Randnähe 1,3^−d); im ersten Durchgang jeder Stufe vergleichen Randausschnitte nur bekannte Pixel. Rechnet im Hintergrund, Ergebnis als ein Undo-Schritt (verworfen, falls die Ebene inzwischen geändert wurde). Agent: `content_aware_fill`.
  - Große Stufen (ab 4000 Zielpixeln) laufen auf der GPU (Compute-Shader `cafill.wgsl`: Bewerten, PatchMatch mit Sprungweiten 8/4/2/1 und lokaler Zufallssuche, Histogramm für σ², Gewichte, Abstimmung als Gather), die kleinen und der Zwiebelschalen-Start auf der CPU. Alles auf dem Worker-Thread.
  - Zeiten: 0,07-MP-Loch 0,1 s (CPU 0,25 s), 1-MP-Loch 0,77 s inkl. Lesen (CPU 3,6 s) – im Budget (< 1 s).
  - Schwäche: streng periodische Muster (Ziegel) in Bildecken können versetzt weiterlaufen; frei im Bild und bei Fotostrukturen gut.
- Pinsel-Glättung wechselt jetzt mit Umschalt+S (vorher S), Pinsel-Dynamik mit Umschalt+J (vorher J).

Phase 5 ist bis auf eine progressive Vorschau beim inhaltsbasierten Füllen fertig.

## Stand Phase 6 (20.11.2026) – Text & Vektor (erster Teil)

- **Text- und Formebenen** (`Content::Vector`): Die Beschreibung bleibt erhalten (Text mit Schrift, Größe, Schnitt, Farbe, Ausrichtung, Zeilenabstand, Laufweite, optional Absatzbreite; Formen: Rechteck mit Eckenradius, Ellipse, Polygon, SVG-Pfad, mit Fläche und Kontur), plus eine Transformation.
  - Neues Crate `imagine-vector`: Layout mit parley (Systemschriften über fontique), Rastern mit vello_cpu in parallelen 4×4-Kachel-Blöcken direkt in lineare f16-Kacheln – nur dort, wo Inhalt ist.
  - Die Engine rastert geänderte Vektorebenen und setzt die Kacheln ins Render-Dokument ein (wie Smart-Filter, die auch auf Textebenen wirken). Strg+T und Verschieben verändern nur die Transformation: Text bleibt scharf. Ebene › Rastern (Strg+Alt+R) macht Pixel daraus; Malen auf einer Textebene wird mit Hinweis abgelehnt. Dateiformat v5.
- **Werkzeuge:** Text (`T`: Klick legt Text an bzw. wählt vorhandenen zum Bearbeiten), Form (Umschalt+`U`: Rechteck/Ellipse aufziehen, Umschalt quadratisch, Alt von der Mitte; Fläche in Pinselfarbe). Eigenschaften-Panel mit Textfeld, Schriftliste mit Suche, Größe, Fett/Kursiv, Ausrichtung, Farbe, Zeilenabstand, Laufweite, Absatz; bei Formen Ecken, Fläche, Kontur. Tippen und Regler ergeben je einen Undo-Schritt.
- **Agenten:** `text_add`, `shape_add`, `vector_set` (JSON-Merge in die Beschreibung, auch `transform`), `layer_rasterize`; `document_info` zeigt Vektorebenen mit Beschreibung und Grenzen.

- **Drehen auffindbar gemacht:** Transformieren-Panel mit Winkelfeld, 90°/180°-Knöpfen und Spiegeln; Menüs Bild › drehen/spiegeln (exakt, Arbeitsfläche tauscht Breite/Höhe) und Ebene › drehen/spiegeln (um die Inhaltsmitte, 90°-Schritte pixelgenau). Behoben: Arbeitsfläche/Zuschneiden um ganze Kacheln verschob Vektorebenen nicht mit.

- **Ebenenstile** (an Pixel-, Text- und Formebenen; Panel „Ebenenstile“, „fx“ in der Ebenenliste): Kontur (außen/mittig/innen), Schlagschatten, Schatten nach innen, Schein nach außen/innen, Farbüberlagerung, Abgeflachte Kante & Relief (Kante innen, Relief, Kissen; Lichtwinkel und -höhe, Tiefe, Weichheit).
  - Berechnung auf der CPU aus dem Alpha der Ebene (nach Smart-Filtern bzw. Rastern): exakte euklidische Distanzfelder (Felzenszwalb–Huttenlocher), dreifacher Box-Blur ≈ Gauß, beleuchtetes Höhenfeld. Zwischengespeichert, bis sich Pixel oder Stil ändern; über 24 MP wird übersprungen.
  - Effekte außerhalb der Form (Schatten: Multiplizieren, Schein: Negativ multiplizieren, äußere Kontur) werden im Render-Dokument als eigene Ebenen direkt unter die Ebene gesetzt – mit deren Deckkraft, Maske und Sichtbarkeit – und wirken so wirklich auf das Darunterliegende; Effekte innen werden in die Ebene eingerechnet. Dateiformat v6. Agent: `layer_effects_set`.

- **Schriftwahl:** Fenster mit Suche, Filter nach Art (Sans, Serif, Mono, Schreibschrift, Display, Symbole – aus OS/2-Familienklasse, PANOSE und Festbreite, Name als Rückfall), Favoriten (Stern) und zuletzt verwendet (beide in den Einstellungen gespeichert); Schriften ohne lateinische Buchstaben ausgeblendet (zuschaltbar). Jede Familie mit dem eigenen Text als Vorschau; Pfeiltasten/Klick zeigen die Schrift live im Bild (Engine-Vorschau, Dokument unverändert), Enter übernimmt, Esc stellt zurück. Familien mit Schnitten als einzelne Systemfamilien (Plex: „… Medm“, „… SmBld“) werden zu einer Familie mit Schnitt-Auswahl zusammengefasst (937 → 294).
- Konturkanten der Ebenenstile subpixelgenau (Abstand zu Kantenpunkten aus Deckung und Gradient), keine Pixeltreppen mehr.

- **Text direkt im Bild bearbeiten:** Cursor, Markieren mit Maus (Klick, Doppel-/Dreifachklick, Ziehen, Umschalt+Klick) und Tastatur (Pfeile, Wortweise mit Strg, Pos1/Ende, Strg+A), Tippen, Rück-/Entf-Taste (mit Strg wortweise), Enter = neue Zeile, Esc/Strg+Enter beendet; leer gelassene Textebenen verschwinden. Arbeitet mit parleys Selection auf demselben Layout wie das Rendern, auch auf gedrehtem Text; Markierung und Cursor zeichnet die Oberfläche über die Leinwand. Eine Sitzung = ein Undo-Schritt.

- **Zwischenablage** (Wayland-Data-Device über SCTK, funktioniert in jedem Compositor): Text beim Bearbeiten und in Textfeldern; Pixel der Ebene bzw. aller Ebenen in der Auswahl als PNG (weiche Kanten → Alpha, auf den Inhalt beschnitten); Einfügen als neue Ebene (aus imagine an der alten Stelle, von außen mittig) oder als Textebene. Lesen und Schreiben auf eigenen Threads. Offener Frei-Transformieren-Rahmen wird beim Ändern anderer Dinge übernommen (wie Photoshop).

- **Zeichenstift** (`P`): Bézier-Pfade zeichnen (Ecke/Kurve, Schließen am Startpunkt → gefüllte Form, offen → Linie) und Punkte/Griffe bearbeiten (Alt: Ecke↔Kurve, Griffe einzeln), auch auf transformierten Pfaden. Neue Geometrie `Bezier` mit Knoten; Overlay über der Leinwand. Palettenfarbe wechselt jetzt mit Umschalt+P.

- **Ebenenstile erweitert** (2026-11-20): Füllmethode je Effekt (Standard wie Photoshop), Verlaufsüberlagerung (zwei Farben, linear/radial, Winkel, Skalierung, umkehren), „Globalen Lichteinfall verwenden“ für Schatten und Abgeflachte Kante (ein Winkel für alle Ebenen, ein Rückgängig-Schritt). Überlagerungen behalten weiche Kanten.

- **OpenType-Features** (Ligaturen, Kapitälchen, Mediäval-/Tabellenziffern, Brüche, Ordinalzeichen, Schwungschrift, Stilvarianten, Hoch-/Tiefstellung, Kerning); was die Schrift nicht hat, ist ausgegraut; PSD-Import übernimmt Photoshops Schalter.
- **Text auf Pfad:** Eigenschaften › Pfad wählt eine Pfadebene (Zeichenstift), Pfadanfang und Spiegeln; jede Glyphe wird auf der Kurve platziert und gedreht, offene Pfade schneiden überstehenden Text ab, geschlossene laufen herum. Bearbeitet wird im Textfeld der Eigenschaften (kein Cursor auf der Kurve).

- **Ebenenstile in PSD** (2026-11-20): Import und Export von Photoshops `lfx2`/`lmfx` – Schlagschatten, Schatten nach innen, Schein außen/innen (mit „Bereich“), Abgeflachte Kante & Relief, Farb- und Verlaufsüberlagerung, Kontur (Farbe); Füllmethoden (auch die ausgeschriebenen Namen neuer Versionen), Globaler Lichteinfall (Ressourcen 1037/1049), mehrere Effekte gleicher Art (CC-Listen), ausgeschaltete Effekte. PSD-Export schreibt die Ebenen ungerendert plus Stil (wie Photoshop), keine „– Effekt“-Ebenen mehr; Rundreise pixelgleich. Nicht übernommen (Warnung): Glanz, Musterüberlagerung, Kontur mit Verlauf/Muster, Abgeflachte Kante außen, Verläufe mit mehr als zwei Farben. Kantenprofil der Abgeflachten Kante jetzt rund (steil an der Kante) – dünne Schrift zeigt das Relief wie in Photoshop.

- **Form- und Füllebenen aus PSD** (2026-11-20): Volltonfarbe-Füllebenen (`SoCo`/`vscg`) mit Vektormaske (`vmsk`/`vsms`) und Kontur (`vstk`) werden editierbare Formebenen; ein Pfad bleibt ein Bézier-Pfad (Zeichenstift), mehrere werden mit Photoshops Pfadoperationen (Vereinigen, Subtrahieren, Schnittmenge, Ausschließen, auch Formgruppen) über i_overlay zu einer Kontur verrechnet. Füllebenen ohne Pfad: Rechteck über die Fläche. Konturen jetzt innen/mittig/außen mit Gehrungs-/Rund-/Abgeflachten Ecken und Linienenden (auch im Formen-Panel und für Agenten: `stroke_align`). Verlaufs- und Musterfüllungen siehe unten.
- **Fläche (Fill)** (2026-11-20): Deckkraft nur der Ebenenpixel, der Ebenenstil bleibt voll sichtbar (Ebenen-Panel neben Deckkraft, Agent `layer_set fill_opacity`, PSD `iOpa` in beide Richtungen, Dateiformat v7). Überlagerungen werden „in der Form“ verrechnet und wirken auch bei 0 % Fläche.

- **Ebenenmaske vor Ebenenstil** (2026-11-20): wie in Photoshop werden die Effekte aus dem maskierten Inhalt berechnet – eine Kontur läuft an der Maskenkante entlang, statt von der Maske abgeschnitten zu werden.

- **Formebenen im PSD-Export** (2026-11-20): Formebenen werden als Photoshop-Formen geschrieben (Volltonfüllung `SoCo`, Vektormaske `vsms`, Kontur `vstk` mit Ausrichtung/Ecken/Enden, Konturbreite mit der Transformation skaliert) – in Photoshop weiter editierbar; Rundreise pixelgleich. Ellipsen-Pfade sind jetzt geschlossen (Kontur innen/außen wirkte bei Ellipsen bisher mittig).

- **Editierbarer Text im PSD-Export** (2026-11-20): Textebenen werden als Photoshop-Text (`TySh`) geschrieben – Punkt- und Absatztext, Schrift (PostScript-Name der tatsächlich verwendeten Schrift), Größe, Farbe, Ausrichtung je Absatz, Zeilenabstand, Laufweite, OpenType-Schalter, Transformation. Die EngineData basiert auf Photoshops eigener (Schreiber erzeugt sie byte-gleich neu). Rundreise pixelgleich (auch gedreht); Text auf Pfad bleibt Pixel. Beim Import sitzt Text jetzt auf Photoshops Grundlinie (Abweichung zu Photoshops Rendering 0,036 → 0,0095 RMSE).

- **Wide-Gamut AVIF/JPEG XL** (2026-11-21): JPEG XL in allen Export-Farbräumen (Display P3, Adobe RGB, Rec. 2020; als Farbcodierung signalisiert, 8/16 Bit, verlustfrei bitgenau), AVIF in Display P3 und Rec. 2020 (10 Bit, CICP-Codes im `nclx`-Kasten; Adobe RGB hat keinen Code). Beim Öffnen werden AVIF-`nclx`-Angaben ausgewertet; Profile wie „…_P3_Srgb“ werden nicht mehr fälschlich als sRGB behandelt.

- **Musterüberlagerung** (2026-11-21): Ebenenstil mit Muster, Deckkraft, Skalierung, Versatz (am Dokument ausgerichtet, Füllmethode wie alle Effekte). Musterbibliothek: sechs eingebaute Muster plus PNGs in `$XDG_DATA_HOME/imagine/patterns`; Bearbeiten › Muster festlegen (Strg+Alt+Umschalt+D, sichtbare Pixel der Auswahl) und Muster importieren (Strg+Alt+Umschalt+Y: Bilder, GIMP-.pat). Das Muster steckt im Effekt, Dokumente bleiben vollständig. PSD: Musterblock (`Patt`/`Pat2`/`Pat3`) lesen und schreiben, `patternFill` im Ebenenstil – Photoshops „Wasser“-Muster kommt an, Rundreise pixelgleich. Agent: `pattern_list`, `pattern_import`, `pattern_define`, Muster per Name in `layer_effects_set`. Übersetzungen mit Kontext (`msgctxt`, `tr_ctx`) für mehrdeutige Wörter („Fill“ = Füllen / Fläche).

- **Verlaufs- und Musterfüllungen** (2026-11-21): Formebenen füllen wahlweise mit Farbe, Verlauf oder Muster (Formen-Panel: „Füllung“). Verläufe mit beliebig vielen Farb- und Deckkraftunterbrechungen, Mittelpunkten und Glättung; Stile Linear, Radial, Winkel, Reflektiert, Raute; Winkel, Skalierung, Umkehren. Neuer Verlaufseditor (Balken mit Unterbrechungen: klicken wählt, ziehen verschiebt, neben den Balken klicken fügt hinzu, wegziehen oder Papierkorb löscht), auch in der Verlaufsüberlagerung (die jetzt ebenfalls mehrere Farben und alle Stile kann). Geometrie und Glättung an Photoshops gespeicherten Pixeln vermessen (Länge = Mitte bis Kante entlang des Winkels, Winkelverlauf im Uhrzeigersinn, 100 % Glättung = halb Smoothstep): Linear/Radial 0,003–0,011 RMSE. PSD: Verlaufs- (`GdFl`) und Musterfüllebenen (`PtFl`, auch in `vscg`) werden editierbare Formebenen, auch Füllebenen mit leerer Vektormaske; Export schreibt sie zurück (Muster in den Musterblock), Rundreise pixelgleich. Nur Rauschverläufe bleiben Pixel. Agent: `shape_add gradient/gradient_style/gradient_angle/pattern/pattern_scale`, `vector_set {"paint": …}`, `gradient_overlay` mit `gradient`. Dateiformat v8.

- **Glanz (Satin)** (2026-11-21): letzter fehlender Photoshop-Ebenenstil. Form weichgezeichnet, zwei um ±Abstand entlang des Winkels verschobene Kopien, Betrag der Differenz, Umkehren, auf die Form begrenzt (Standard Multiplizieren). An Photoshops eigener Darstellung vermessen (`fxcmp`: Effekte auf Photoshops gespeicherten Ebenenpixeln gegen das Composite): Weichzeichnen vor dem Vergleich, Stärke 0,8 × Größe – RMSE 0,0049 (vorher fehlend). Innere Effekte jetzt in Photoshops fester Stapelreihenfolge (Muster, Verlauf, Farbe, Glanz, Schein innen, Schatten innen). PSD lesen/schreiben (`ChFX`/`satinMulti`, lineare Kontur), Rundreise pixelgleich; Agent `satin` in `layer_effects_set`.

- **Text auf Pfad direkt bearbeiten** (2026-11-21): Klicks werden auf die Kurve projiziert (Bogenlänge + Abstand → Koordinaten der flachen Zeile), Cursor und Auswahl laufen gebogen entlang der Kurve; Tippen, Markieren, Doppel-/Dreifachklick wie bei normalem Text.

### Phase 7 – Interop (begonnen 2026-11-20)

- **PSD/PSB-Import** (eigener Leser, `imagine-io/psd.rs`): Ebenen mit Namen (Unicode), Deckkraft (inkl. Flächendeckkraft), Füllmethoden, Sichtbarkeit, Gruppen (Abschnittstrenner), Ebenenmasken; 8/16/32 Bit; RGB, Graustufen, CMYK (naiv); roh/RLE/ZIP/ZIP mit Vorhersage; eingebettetes ICC-Profil; zusammengefügtes Bild als Rückfall. Ebenen werden kachelweise parallel gestreamt (619-MB-PSD mit 4 Ebenen: 1,4 s bis zum Export). Hinweise für Schnittmasken, Text-/Smart-/Einstellungs-/Füllebenen (nur Pixel) und fremde Füllmethoden. Getestet mit Dateien aus GIMP 3 und ImageMagick.

- **PSD/PSB-Export** (Exportieren unter … `.psd`/`.psb`, auch für Agenten): Ebenen, Gruppen (isoliert = „Normal“), Masken, Deckkraft, Füllmethoden, Sichtbarkeit, Unicode-Namen; Text, Formen und Smartfilter werden wie angezeigt gerastert, Ebenenstile bleiben als Photoshop-Ebenenstil editierbar; 8/16 Bit, RLE; eingebettetes ICC-Profil des gewählten Farbraums; zusammengefügtes Bild; PSB automatisch über 30 000 px. Einstellungsebenen fehlen noch als Ebene (nur im zusammengefügten Bild). Geprüft: Rundreise pixelgleich, GIMP 3 und ImageMagick lesen die Dateien.

- **Schnittmasken** (Strg+Alt+G, Ebene-Menü, Agenten `layer_set clip`): eine Ebene zeigt sich nur, wo die Basisebene darunter Pixel hat; Basis plus geschnittene Ebenen werden als eigene Gruppe berechnet (geschnittene Ebenen „atop“), Deckkraft und Füllmethode der Basis gelten für die Gruppe, ausgeblendete Basis blendet alle aus. Auch Gruppen, Text, Formen und Einstellungsebenen können Basis oder geschnitten sein; Effektebenen erben das. PSD-Import und -Export übernehmen Schnittmasken, `.imagine` speichert sie.

- **Gruppen „Hindurchwirken“** (Photoshops Standard, jetzt auch für neue Gruppen): die Ebenen der Gruppe wirken direkt auf alles darunter (Füllmethoden, Einstellungsebenen), Deckkraft und Maske der Gruppe blenden das Ergebnis über. Im Füllmethoden-Menü einer Gruppe ganz oben; eine andere Füllmethode macht die Gruppe isoliert. Ältere `.imagine`-Dateien bleiben isoliert (unverändertes Aussehen). PSD-Import/-Export übernehmen es; Agenten: `layer_set pass_through`.

- **PSD: Einstellungsebenen und Text.** Import und Export von Einstellungsebenen als echte Photoshop-Einstellungsebenen (Tonwertkorrektur, Gradationskurven, Helligkeit/Kontrast modern und „Früheren Wert“, Farbton/Sättigung inkl. Färben, Belichtung, Dynamik, Schwarzweiß, Umkehren, Tontrennung, Schwellenwert); Deskriptoren lesen/schreiben (`psd_data.rs`). Textebenen werden editierbar importiert (Text, Schrift aus PostScript-Name, Größe, Farbe, Ausrichtung, Zeilenabstand, Laufweite, Absatzbreite, Transformation; Lage an Photoshops Pixeln ausgerichtet); fehlende Arial/Helvetica/Times/Courier/Calibri/Cambria/Georgia werden durch metrisch gleiche freie Schriften ersetzt, sonst bleiben die Pixel. Text-Export bleibt Pixel (Photoshops EngineData lässt sich hier ohne Photoshop nicht prüfen).
- **Einstellungen rechnen wie Photoshop** (gemessen an Photoshops eigenen Renderings, `scripts/psd-reference.sh`): Kurven als natürlicher kubischer Spline, Kanal vor Gesamt bei Tonwert/Kurven, Helligkeit/Kontrast nach psd-tools (modern) bzw. Gerade (Legacy), Farbton/Sättigung in Photoshops Reihenfolge, Schwellenwert/Tontrennung in 8-Bit-Schritten. Abweichung (RMSE) jetzt 0,001–0,004, modern Helligkeit/Kontrast 0,013. Farbton/Sättigung für einzelne Farbbereiche fehlt noch (Hinweis beim Import).
- PSD-Masken mit Vektormaske (Kanal −3) werden gelesen.

- **Farbton/Sättigung mit Farbbereichen** (Rottöne … Magentatöne, wie Photoshop: Trapez-Gewichtung auf dem Farbton, Helligkeit in HSV, Sättigungen über Photoshops gemessene Kombinationstabelle von psd-tools), CPU und GPU identisch; PSD liest und schreibt die Bereiche. Abweichung zu Photoshop jetzt 0,002–0,007 (vorher bis 0,085).

- **AVIF und JPEG XL:** Öffnen (AVIF über dav1d, JPEG XL über jxl-oxide, auch HDR/ICC) und Exportieren (AVIF über ravif; JPEG XL über libjxl, 8/16 Bit, Qualität 100 = verlustfrei – geprüft bitgleich mit dem PNG-Export). Export vorerst nur sRGB.

- **Eigene Pinselspitzen und ABR-Import:** Bearbeiten › Pinsel importieren … (Strg+Alt+Umschalt+B, auch im Spitzen-Menü): Photoshop-ABR (v1/v2 und v6–10, Namen aus dem `desc`-Abschnitt; geprüft mit einer echten 6.2-Datei aus Kritas Testdaten) oder ein beliebiges Bild (Transparenz bzw. dunkel = Farbe). Spitzen liegen als graue PNGs in `~/.local/share/imagine/brushes`, bis zu 59 im Pinsel; Umschalt+B schaltet durch alle.

**Noch offen in Phase 7:** editierbarer Text im PSD-Export, AVIF/JPEG XL in weiten Farbräumen, Pinsel-Dynamik aus ABR.

### Phase 8 – Erweiterbarkeit & Kür (begonnen 2026-11-21)

- **Kamera-RAW** (2026-11-21): CR2/CR3, NEF, ARW, RAF (auch X-Trans), ORF, RW2, PEF, DNG … über `rawler` (Dekodieren, Demosaicing PPG/X-Trans bilinear, Zuschnitt); eigener Farbschritt in `imagine-io/raw.rs`: Weißabgleich (Wie Aufnahme, Automatisch = Grauwelt, Tageslicht, Temperatur/Tönung über die Planck-Kurve), Kameramatrix nach linearem sRGB ohne Gamut-Beschneidung (szenenlinear, Farben außerhalb sRGB bleiben), ausgefressene Lichter neutral, Belichtung, Ausrichtung. Beim Öffnen erscheint „RAW entwickeln“ mit Vorschau (1400 px, Regler live); Kommandozeile/`--export` entwickeln mit „Wie Aufnahme“. 24 MP in 0,3–0,4 s. Geprüft gegen LibRaw (`dcraw_emu -w -o 1 -4`) mit Canon 6D, Nikon D750, Sony A7 III, Fuji X-T2, iPhone-DNG: deckungsgleich bis auf die Canon, bei der rawler die Kamera-Automatik statt „Wie Aufnahme“ (1,1,1) liest. Agent: `document_open raw: {white_balance, exposure}`, `raw_info`. Werkzeug: `cargo run --example rawdev`.
- **RAW-Ebenen** (2026-11-21): eine geöffnete oder platzierte RAW-Datei bleibt eine RAW-Ebene – die Originaldatei steckt im Dokument (`.imagine` speichert sie einmal, inkrementelle Speicherungen verwenden sie weiter; Dateiformat v9), Weißabgleich/Belichtung bleiben im Eigenschaften-Panel änderbar (24 MP neu entwickeln ≈ 80 ms, das demosaizierte Bild der zuletzt entwickelten Datei bleibt im Speicher). Verschieben, Strg+T, Zuschneiden, Arbeitsfläche ändern wirken verlustfrei über eine eigene Transformation (bilinear mit geglätteten Kanten, starkes Verkleinern über vorab gemittelte Kopie). Rastern macht Pixel daraus, PSD-Export schreibt Pixel. Ebenen-Miniaturen jetzt auch für Text-, Form- und RAW-Ebenen. Crate `imagine-raw` (rawler + Farbschritt). Agent: `raw_set`, `place_image` mit RAW.
- **Lichter-Rekonstruktion und Rauschreduzierung** (2026-11-21): ausgefressene Kanäle werden aus den übrigen plus der Farbe am Rand der Fläche geschätzt („inpaint opposed“-artig), ganz ausgefressen bleibt neutral; Werte über 1 bleiben erhalten, Belichtung herunter holt Himmel mit Farbe und Verlauf zurück (Standard; „Beschneiden“ wie bisher). Wavelet-Rauschreduzierung (à trous, B3-Spline, weiche Schwelle) auf Wurzel-Werten (Schrotrauschen gleichmäßig), Rauschpegel aus der feinsten Ebene gemessen; Luminanz und Farbe getrennt (Farbe ab der zweiten Ebene auf halber Größe), Standard Farbe 25. 24 MP: 0,2 s (nur Farbe) bis 0,5 s. Regler wirken beim Loslassen; Weißabgleich/Belichtung bleiben schnell (entrauschtes Bild im Cache). Agent: `raw_set highlights/noise`.
- **Objektivkorrektur** (2026-11-21): Profile aus der lensfun-Datenbank des Systems (`/usr/share/lensfun`, Aktualisierungen von `lensfun-update-data` haben Vorrang; nichts wird mitgeliefert). Kamera und Objektiv werden über normalisierte Namen gefunden (alle Testdateien außer iPhone-DNG: richtiges Profil), Verzeichnung (ptlens/poly3/poly5) und Farbsäume (TCA linear/poly3) nach Brennweite, Vignettierung (pa) nach Brennweite und Blende interpoliert; Crop-Faktor- und Seitenverhältnis-Unterschiede zum Kalibriersensor werden umgerechnet. Automatischer Zoom ohne leere Ränder. Im Entwickeln-Schritt (Abtasten je Farbkanal), praktisch ohne Mehrkosten. Standard an; Schalter in Dialog/Eigenschaften, Agent `raw_set lens`, `raw_info lens_profile`.
- **Tonwert und Farbe** (2026-11-21): Kontrast, Lichter, Tiefen, Dynamik, Sättigung (−100…100, live). Kontrast/Tiefen in Blendenstufen um Mittelgrau (monoton), Lichter negativ = weiche Schulter auf dem größten Kanal (rekonstruierte Lichter über 1 werden sichtbar, Farben bleiben im Bereich), Dynamik wirkt vor allem auf blasse Farben. Rekonstruktions-Auswahl heißt jetzt „Ausgefressene Lichter“. Agent: `raw_set tone`.
- **X-Trans und DNG-Korrekturen** (2026-11-21): eigener Ablauf nach rawlers Skalierung – Bayer weiter PPG (bitgleich zu vorher), X-Trans jetzt Markesteijn (1 Durchgang; gegen LibRaw: RMSE 0,0131 → 0,0123, Schärfe +25 %, 3 Durchgänge brachten nichts bei doppelter Zeit; 0,65 s für 24 MP). DNG-Opcodes von Handys: GainMap (Objektivabschattung, bis 3,4× in den Ecken beim Pixel 4a) auf das Mosaik vor dem Demosaicing, WarpRectilinear (Verzeichnung/Farbsäume je Ebene, tangential) und FixVignetteRadial im Entwickeln-Schritt – Vorrang vor lensfun, Profilname „in der DNG enthalten“.

- **Filter-Plugins in WGSL** (2026-11-21, statt WASM: Pixel-Plugins gehören auf die GPU): `.wgsl`-Datei mit Kopfzeilen (`@name`, `@param` Zahl/bool/Farbe, `@reach`, `@canvas`) und `fn process(p) -> vec4<f32>`; imagine setzt ein Präludium davor (Quellpixel über Kachelgrenzen, `sample`, sRGB, Zufall, eine Funktion je Parameter) und übersetzt eine eigene Pipeline (Cache nach Quelltext). Prüfung vorab mit naga, Fehler mit Zeile der Plugin-Datei. Läuft wie eingebaute Filter: Vorschau, Smart-Filter (Shader steckt im Dokument), Undo, Auswahl; 24 MP < 20 ms. Ordner `~/.local/share/imagine/filters`, Menü Bild › Filter › Plugins (+ neu laden), vier Beispiele (Vignette, Pixelate, Chromatic Aberration, Clouds). Agenten: `filter_plugin_list`, `filter_plugin_install` (Agenten schreiben eigene Filter), `filter_apply {kind: plugin}`. Doku: `docs/filter-plugins.md`.
- **Aktionen, Skripte, Stapelverarbeitung** (2026-11-21): Datei › Aktionen – Aufnahme übersetzt UI-Aktionen in Agenten-Befehle (Filter, Korrekturen, Stile, RAW, Ebenen- und Auswahlbefehle, Marquee/Lasso, Verschieben, Bildgröße …; ohne Ebenen-IDs, wirkt auf die aktive Ebene), gespeichert als `NAME.jsonl` in `~/.local/share/imagine/actions`; Abspielen schrittweise mit Abbruch beim ersten Fehler. Ausführbare Dateien im Ordner sind Skripte (beliebige Sprache, gestartet mit `IMAGINE_SOCKET`). `imagine batch AKTION DATEIEN… [--out] [--format]`. Agenten: `action_list`, `action_run`. Doku: `docs/aktionen.md`.

- **Lokale KI: Auswahl** (2026-11-21): Crate `imagine-ai` mit ONNX Runtime aus dem System (zur Laufzeit geladen, Arch: `onnxruntime-cpu`; fehlt sie, sagen die Funktionen das). Modelle werden nach Rückfrage einmalig von huggingface.co nach `~/.local/share/imagine/models` geladen: SAM 2.1 small (Apache-2.0, 184 MB) und BiRefNet lite (MIT, 224 MB). Objektauswahl-Werkzeug (Umschalt+W; Klick wählt, Umschalt/Alt kombinieren): Bild einmal analysieren 0,8–1,7 s, jeder weitere Klick ~0,1 s (Einbettung im Cache). Auswahl › Motiv (Strg+Alt+S, ~5 s), Ebene › Hintergrund entfernen (Strg+Alt+B, Ebenenmaske). Rechnet auf der CPU (Ryzen 9700X); läuft im Hintergrund. Agenten: `select_subject`, `select_object`, `remove_background`, `ai_model_download`.
- **Lokale KI: Entfernen** (2026-11-21): LaMa (Apache-2.0, 208 MB, ONNX-Export von Carve, feste 512²). Entfernen-Werkzeug (J schaltet Bereichsreparatur → Entfernen → Reparaturpinsel): über das Störende malen (rosa Vorschau), beim Loslassen füllt LaMa es auf. Mit Auswahl: Bearbeiten › Entfernen (KI), Umschalt+Alt+Rücktaste (Auswahl +4 px). Läuft über die Inhaltsbasierte-Füllung-Maschinerie (`FillJob::run_with`, ein Rückgängig-Schritt, Ebene-geändert-Schutz); Kontext mind. 512 px je Seite bzw. ¾ der Lochgröße, größere Bereiche werden für das Modell verkleinert, nur das Loch übernimmt das Ergebnis. ~1–1,5 s auf der CPU. „Alle Ebenen aufnehmen“ wie bei der Bereichsreparatur. Nebenbei: SAM mit mehreren Punkten nimmt jetzt den bestbewerteten Vorschlag (vorher Index 0 – unbrauchbar bei diesem Export). Agent: `remove_area {all_layers, expand}`.
- **Lokale KI: Hochskalieren** (2026-11-21): Bild › Hochskalieren (KI) 2×/4× mit Real-ESRGAN General x4v3 (BSD-3, 5 MB; Qualcomm-AI-Hub-Export, Gewichte byte-gleich mit deren Release, HF-Spiegel). Feste 128er-Kacheln mit 12 px Überlappung, leere Kacheln übersprungen; 2× = 4× + Box-Verkleinerung. Alle Pixelebenen (auch außerhalb der Arbeitsfläche, transparente Ränder per Farb-„Bleed“), der Rest wie Bildgröße – ein Rückgängig-Schritt (`UpscaleJob`/`apply_upscale`). ~10 s je MP Ausgangsbild auf der CPU (x4plus wäre ~7× langsamer bei kaum sichtbarem Gewinn). Agent: `image_upscale {factor}`.
  Weiter: Entfernen-Werkzeug (LaMa), Hochskalieren (Real-ESRGAN), KI-Entrauschen; generatives Füllen über OpenAI-API (Anbieter austauschbar).

## Nächster konkreter Schritt

Phase 6 weiter: Zeichenstift, Text direkt im Bild bearbeiten.

## Nächster konkreter Schritt (Phase 0, erledigt)

**Phase 0 starten:** Cargo-Workspace anlegen, wgpu + winit-Fenster, Tablet-Events über `tablet-v2` auslesen und mit Druck auf eine sparse Tile-Canvas malen, ein 100-MP-Bild öffnen und mit 165 Hz schwenken/zoomen/drehen. Wenn das sitzt, steht der Stack.
