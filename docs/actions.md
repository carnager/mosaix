# Actions, scripts and batch processing

Everything runs on the same commands as the agent interface (`docs/agents.md`).
Folder: `~/.local/share/mosaix/actions/` (or `$XDG_DATA_HOME/mosaix/actions/`).

## Actions (macros)

*File › Actions › Start Recording*, then work as usual, *Stop Recording …*
and give it a name. Recorded are:

- Filters (also smart filters and plugins), adjustment layers and their changes, layer styles, RAW development
- Opacity, fill, blend mode, pass through, deleting a layer
- New layer, duplicate, group/ungroup, merge, add/apply mask, rasterize
- Selection: rectangle, ellipse, lasso (with add/subtract/intersect), all, deselect, inverse, feather
- Fill (Alt+Backspace), deleting the selection, moving, canvas size, image size, undo/redo

Not recorded: brush strokes and tool changes. Steps contain no layer IDs – they act
on whichever layer is active, so they work on any image.

An action is a text file `NAME.jsonl`, one command per line (`#` = comment),
written by hand or by an agent:

```jsonl
# Web look
{"cmd": "filter_apply", "filter": {"kind": "plugin", "name": "Vignette", "values": {"amount": -60}}}
{"cmd": "adjustment_add", "adjustment": {"kind": "brightness_contrast", "brightness": 5, "contrast": 20}}
{"cmd": "image_resize", "width": 1600}
```

Playing: click it in the menu (each step is one undo step; it stops at the first
error with a message). Agents: `action_list`, `action_run`.

## Scripts

Every **executable** file in the folder also shows up under *File › Actions* and
is started with `MOSAIX_SOCKET` (this window) and `MOSAIX` (the program path);
its output goes to the log. Any language, e.g.:

```sh
#!/bin/sh
# Make the active layer half transparent
echo '{"cmd": "layer_set", "opacity": 0.5}' | "$MOSAIX" run --attach -
```

For logic (loops over layers and so on): `document_info` returns the layer tree as JSON;
in Python e.g. `subprocess.run([mosaix, "run", "--attach", "-"], input=…)`, or talk
directly to the Unix socket (JSON lines, one answer per line).

## Batch processing

```sh
mosaix batch "Web Look" ~/Pictures/*.jpg --out ~/web --format webp
```

Opens each file (also RAW, PSD), plays the action and exports. Without `--out`
everything goes to `mosaix-batch/` next to the inputs; without `--format` in the same
format (PNG if the input format cannot be exported, e.g. RAW).
