# Filter-Plugins (WGSL)

Eigene Filter laufen auf der GPU, genauso schnell wie die eingebauten: eine
`.wgsl`-Datei in `~/.local/share/imagine/filters/` (bzw. `$XDG_DATA_HOME/imagine/filters/`).
Sie erscheinen unter *Bild › Filter › Plugins* (*Plugins neu laden* liest den Ordner
erneut), mit Vorschau, als Smart-Filter, rückgängig machbar und für Agenten
(`filter_apply {"filter": {"kind": "plugin", "name": …}}`). Ein Smart-Filter speichert
den Shader im Dokument – die Datei funktioniert auch ohne installiertes Plugin.

## Aufbau

```wgsl
// @name Vignette
// @description Darkens the edges of the image
// @param amount "Amount" -100 100 -50 %
// @param radius "Radius" 1 3000 800 px log
// @param tint "Tint" color #000000
// @param inverted "Invert" bool false
// @reach 0
fn process(p: vec2<i32>) -> vec4<f32> {
    let c = src(p);
    …
}
```

- `@name` (Pflicht), `@description`
- `@param id "Beschriftung" min max standard [einheit] [log]` – Zahl, Regler (log = logarithmisch)
- `@param id "Beschriftung" bool [true]` – Schalter
- `@param id "Beschriftung" color [#rrggbb]` – Farbe
- `@reach px|param` – wie weit der Filter um ein Pixel herum liest (z. B. Weichzeichner); bestimmt, welche Kacheln außerhalb der Ebene mitgerechnet werden
- `@canvas` – malt die ganze Arbeitsfläche, auch wo die Ebene leer ist (Generatoren)

Höchstens 16 Zahlen an Parametern (eine Farbe zählt 3). Jeder Parameter ist
eine Funktion seines Namens: `amount()` → `f32`, `inverted()` → `bool`,
`tint()` → `vec3<f32>` (linear).

`process` bekommt die Dokumentposition des Pixels und gibt **vormultipliziertes
lineares RGBA** zurück. Hilfsfunktionen:

| Funktion | |
|---|---|
| `src(p: vec2<i32>) -> vec4<f32>` | Quellpixel (vormultipliziert, linear; außerhalb transparent) |
| `sample(p: vec2<f32>) -> vec4<f32>` | bilinear, Pixelmitten bei +0,5 |
| `unpremultiply(c)`, `premultiply(c)` | gerade ↔ vormultiplizierte Deckkraft |
| `to_srgb(v)`, `to_linear(v)` | sRGB-Kurve (`vec3<f32>`) |
| `luminance(rgb) -> f32` | Rec.-709-Luminanz (linear) |
| `canvas() -> vec2<f32>` | Dokumentgröße in px |
| `rand(p: vec2<i32>, seed: u32) -> f32` | 0…1, pro Pixel stabil |

Fehler (Kopfzeilen oder WGSL) nennen die Zeile der Plugin-Datei; kaputte Plugins
stehen ausgegraut im Menü (Fehler im Tooltip). Beispiele: `crates/imagine-app/plugins/`
(Vignette, Pixelate, Chromatic Aberration, Clouds) – eine Datei gleichen Namens im
Ordner ersetzt das eingebaute Beispiel.

Agenten: `filter_plugin_list` (Plugins, Parameter, Format), `filter_plugin_install`
(prüft und speichert eigenen WGSL-Code; `check_only` nur prüfen).
