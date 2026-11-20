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

Tastenkürzel folgen Photoshop. Die Ebenenliste erscheint bis zum UI (Phase 3) im Terminal, die aktive Ebene im Fenstertitel.

| Aktion | Eingabe |
|---|---|
| Werkzeuge | `B` Pinsel, `E` Radierer, `M` Auswahlrechteck/-ellipse, `L` Lasso, `G` Verlauf linear/radial, `V` Verschieben |
| Malen | linke Maustaste / Stift; Radierer-Ende des Stifts; `[` `]` Größe, Ziffern Deckkraft, `H` Härte, `S` Glättung, `C` Farbe, `X` Schwarz↔Weiß |
| Ansicht | Leertaste+Ziehen schwenken, Mausrad zoomen, `R`+Ziehen drehen, Strg+`0` einpassen, Strg+`1` 100 %, `Q` Maskierungsmodus-Ansicht |
| Auswahl | Aufziehen mit Umschalt/Alt/beidem = hinzufügen/abziehen/Schnittmenge; Strg+`A` / Strg+`D` / Strg+Umschalt+`I`; Umschalt+`F6` weiche Kante |
| Füllen & Löschen | Alt+Rücktaste füllt Auswahl (bzw. Ebene), `Entf` löscht ausgewählte Pixel |
| Ebenen | Strg+Umschalt+`N` neu, Strg+`J` duplizieren, `Entf` (ohne Auswahl) löschen, Alt+`[` `]` auswählen, Strg+`[` `]` verschieben |
| Ebenen-Eigenschaften | Umschalt+Ziffer Deckkraft, Umschalt+`+`/`-` Füllmethode, Strg+`,` ein/aus, Strg+`G` gruppieren, Strg+Umschalt+`G` aufheben, `F7` Blend-Raum |
| Ebenenmasken | Strg+Umschalt+`M` neu (aus Auswahl), `\` Maske bemalen ↔ Pixel, Strg+Alt+`M` ein/aus, Strg+Alt+Umschalt+`M` anwenden |
| Verlauf & Datei | Strg+`Z` / Strg+Umschalt+`Z` / Strg+`Y`, Strg+`S` speichern (im Hintergrund), Strg+Umschalt+`E` PNG exportieren |
| Sonstiges | `F12` Benchmark, `F1` Hilfe, Strg+`Q` beenden |

## Aufbau

- `crates/imagine-core` – Dokumentmodell, Füllmethoden (CPU-Referenz), Verlauf, Kachel-Mathematik, View, Pinselstrich → Dabs (ohne GPU)
- `crates/imagine-gpu` – wgpu-Engine: Kachel-Pools, Strich-Puffer, Composite-Cache, WGSL-Shader
- `crates/imagine-io` – natives `.imagine`-Format (append-only), Export
- `crates/imagine-app` – Wayland (smithay-client-toolkit), Tablet-Protokoll, Eingabe, Loader, Headless-CLI, Agenten-Schnittstelle (`src/agent/`: Befehle, MCP, Socket)
