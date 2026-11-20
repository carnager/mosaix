#!/bin/sh
# Compares imagine's rendering of Photoshop files with Photoshop's own
# renders (test files of psd-tools, MIT). Downloads them on first use.
#   scripts/psd-reference.sh [DIR]      (default: target/psd-reference)
# Prints the RMSE per file (0 = identical; < 0.01 is visually the same).
set -e
D=${1:-target/psd-reference}
B=https://raw.githubusercontent.com/psd-tools/psd-tools/main/tests/psd_files/adjustments
FILES="brightnesscontrast_rgb brightnesscontrast_legacy_rgb levels_rgb curves_rgb exposure_rgb
huesaturation_rgb huesaturation_colorize_rgb huesaturation_lightness_rgb huesaturation_saturation_rgb
invert_rgb posterize_rgb threshold_rgb"
mkdir -p "$D"
cargo build -q --release
for f in $FILES; do
  [ -s "$D/$f.psd" ] || curl -sf -o "$D/$f.psd" "$B/$f.psd"
  [ -s "$D/$f.png" ] || curl -sf -o "$D/$f.png" "$B/$f.png"
  target/release/imagine "$D/$f.psd" --export "$D/$f-imagine.png" >/dev/null 2>&1
  printf '%-34s %s\n' "$f" "$(magick compare -metric RMSE "$D/$f-imagine.png" "$D/$f.png" null: 2>&1 | grep -o '(.*)')"
done
