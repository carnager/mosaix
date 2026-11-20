#!/usr/bin/env python3
"""i18n helper for imagine.

  scripts/i18n-check.py merge   – merge crates/imagine-core/i18n/parts/*.po into de.po
  scripts/i18n-check.py check   – list tr("…") / tr!("…") texts without a German entry,
                                  catalog entries no longer used, and string literals
                                  that still look German.
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT = os.path.join(ROOT, "crates/imagine-core/i18n/de.po")
HEADER = '''# German translations for imagine.
# The source language is English; msgid is the English text as written in the code.
msgid ""
msgstr ""
"Content-Type: text/plain; charset=UTF-8\\n"
"Language: de\\n"
'''


def unquote(s):
    s = s.strip()[1:-1]
    return bytes(s, "utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")


def quote(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t") + '"'


def parse(path):
    # Entries with a context are keyed "ctx\x04text" (as gettext does).
    out, cur, which = {}, {"id": None, "str": None, "ctx": None}, None

    def flush():
        if cur["id"] and cur["str"]:
            key = cur["id"] if cur["ctx"] is None else cur["ctx"] + "\x04" + cur["id"]
            out.setdefault(key, cur["str"])
        cur["id"] = cur["str"] = cur["ctx"] = None

    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("msgctxt "):
            flush()
            cur["ctx"], which = unquote(line[8:]), "ctx"
        elif line.startswith("msgid "):
            if which != "ctx":
                flush()
            cur["id"], which = unquote(line[6:]), "id"
        elif line.startswith("msgstr "):
            cur["str"], which = unquote(line[7:]), "str"
        elif line.startswith('"') and which:
            cur[which] = (cur[which] or "") + unquote(line)
    flush()
    return out


def write(entries):
    with open(CAT, "w", encoding="utf-8") as f:
        f.write(HEADER)
        for k in sorted(entries, key=str.lower):
            if "\x04" in k:
                ctx, text = k.split("\x04", 1)
                f.write(f"\nmsgctxt {quote(ctx)}\nmsgid {quote(text)}\nmsgstr {quote(entries[k])}\n")
            else:
                f.write(f"\nmsgid {quote(k)}\nmsgstr {quote(entries[k])}\n")


def sources():
    for crate in glob.glob(os.path.join(ROOT, "crates/*/src")):
        for dirpath, _, files in os.walk(crate):
            for fn in files:
                if fn.endswith(".rs") and fn != "i18n.rs":
                    yield os.path.join(dirpath, fn)


LIT = r'"((?:[^"\\]|\\.)*)"'


def used():
    keys = {}
    for p in sources():
        s = open(p, encoding="utf-8").read()
        for m in re.finditer(r"\btr!?\(\s*" + LIT, s):
            k = bytes(m.group(1), "utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")
            keys.setdefault(k, os.path.relpath(p, ROOT))
        # tr_ctx("context", "text"): keyed like msgctxt entries.
        for m in re.finditer(r"\btr_ctx\(\s*" + LIT + r"\s*,\s*" + LIT, s):
            dec = lambda t: bytes(t, "utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")
            keys.setdefault(dec(m.group(1)) + "\x04" + dec(m.group(2)), os.path.relpath(p, ROOT))
    return keys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "merge":
        entries = parse(CAT)
        conflicts = 0
        for part in sorted(glob.glob(os.path.join(os.path.dirname(CAT), "parts/*.po"))):
            for k, v in parse(part).items():
                if k in entries and entries[k] != v:
                    conflicts += 1
                    print(f"conflict {k!r}: {entries[k]!r} vs {v!r} ({os.path.basename(part)})")
                entries.setdefault(k, v)
        write(entries)
        print(f"{len(entries)} entries, {conflicts} conflicts")
        return
    cat = parse(CAT)
    keys = used()
    missing = [(k, f) for k, f in keys.items() if k not in cat]
    unused = [k for k in cat if k not in keys]
    german = []
    for p in sources():
        for n, line in enumerate(open(p, encoding="utf-8"), 1):
            code = line.split("//")[0]
            for m in re.finditer(LIT, code):
                if re.search(r"[äöüÄÖÜß]", m.group(1)):
                    german.append(f"{os.path.relpath(p, ROOT)}:{n}: {m.group(1)[:70]}")
    for k, f in missing:
        print(f"missing  {f}: {k!r}")
    for k in unused:
        print(f"unused   {k!r}")
    for g in german:
        print(f"german?  {g}")
    print(f"{len(keys)} used, {len(cat)} in catalog, {len(missing)} missing, {len(unused)} unused, {len(german)} German-looking literals")


main()
