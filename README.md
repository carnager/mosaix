# imagine

GPU-first Bildeditor für Linux/Wayland – siehe [PLAN.md](PLAN.md).

Aktueller Stand: **Phase-0-Spike** (Stack-Validierung).

## Bauen & Starten

```sh
cargo run --release -- bild.jpg              # Bild öffnen (PNG, JPEG, WebP, TIFF, EXR)
cargo run --release -- --new 30000x30000     # leeres Dokument
cargo run --release -- --demo 10000x10000    # 100-MP-Testbild
cargo run --release -- --demo 10000x10000 --bench   # Benchmark, beendet sich danach
cargo run --release --features profile -- …  # mit Tracy-Profiling
cargo test --workspace                       # inkl. Headless-GPU-Tests (Vulkan)
```

Voraussetzungen: Wayland-Compositor, Vulkan-Treiber.

## Bedienung

| Aktion | Eingabe |
|---|---|
| Malen | linke Maustaste / Stift (Druck = Größe & Deckkraft) |
| Radieren | `E` umschalten, oder Radierer-Ende des Stifts |
| Schwenken | Leertaste + Ziehen, mittlere Maustaste, Stifttaste 1 |
| Zoomen | Mausrad (zum Cursor), Strg + Leertaste + Ziehen |
| Drehen | `R` halten + Ziehen, Umschalt + Mausrad, Stifttaste 2; Umschalt+`R` zurücksetzen |
| Ansicht | `0` einpassen, `1` = 100 % |
| Pinsel | `[` / `]` oder Strg + Mausrad Größe, `H` Härte, `S` Glättung, `C` Farbe, `B` Pinsel |
| Sonstiges | `F12` Benchmark, `F1` Hilfe, Strg+`Q` beenden |

## Aufbau

- `crates/imagine-core` – Kachel-Mathematik, View-Transformation, Farbkonvertierung, Pinselstrich → Dabs (ohne GPU)
- `crates/imagine-gpu` – wgpu-Device, sparse Kachel-Canvas, WGSL-Shader (Anzeige, Pinsel, Mips)
- `crates/imagine-app` – Wayland (smithay-client-toolkit), Tablet-Protokoll, Eingabe, Loader
