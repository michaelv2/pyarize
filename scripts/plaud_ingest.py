"""Download one Plaud recording into ~/recordings/plaud/audio/ and file it in folders.csv.

Detects the format from the bytes (MP3, Ogg Opus, or Plaud raw Opus, which is wrapped
losslessly into Ogg with the raw original kept in raw/). Verifies Content-Length, a clean
ffmpeg decode and the API's duration before the file gets its final name, and keeps nothing
from a download that fails them, so it can be retried. Never prints the presigned URL.

usage: python3 -I scripts/plaud_ingest.py <file_id> <folder> [max_drift_s=3]
"""
import csv
import fcntl
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from plaud_raw2ogg import wrap

BASE = "https://platform.plaud.ai/developer/api/open/third-party"
ROOT = Path.home() / "recordings"
AUDIO, RAW, CSV = ROOT / "plaud" / "audio", ROOT / "plaud" / "raw", ROOT / "folders.csv"
FOLDERS = {"unfiled", "interview", "1:1 meeting", "group meeting", "family", "discard"}
fid, folder = sys.argv[1], sys.argv[2]
max_drift = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
if folder not in FOLDERS:
    sys.exit(f"unknown folder {folder!r}; expected one of {sorted(FOLDERS)}")
if fid in {r["id"] for r in csv.DictReader(open(CSV))}:
    sys.exit(f"{fid} is already in folders.csv")

tok = json.loads((Path.home() / ".plaud/tokens.json").read_text())
req = urllib.request.Request(f"{BASE}/files/{fid}", headers={
    "Authorization": f"Bearer {tok['access_token']}", "Accept": "application/json", "User-Agent": "pyarize/0.1"})
d = json.load(urllib.request.urlopen(req, timeout=30))

def write_private(path, data):
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as fh:
        fh.write(data)

with urllib.request.urlopen(urllib.request.Request(d["presigned_url"], headers={"User-Agent": "pyarize/0.1"}), timeout=300) as r:
    expected = int(r.headers.get("Content-Length") or -1)
    body = r.read()
if len(body) != expected:
    sys.exit(f"size {len(body)} != Content-Length {expected}")

if body[:3] == b"ID3" or (body[0] == 0xFF and body[1] & 0xE0 == 0xE0):
    kind, ext = "mp3", "mp3"
elif body[:4] == b"OggS":
    kind, ext = "ogg", "ogg"
elif body[0] == 0xB8 and len(body) % 80 == 0:
    kind, ext = "raw-opus", "ogg"
else:
    sys.exit(f"unrecognised format, first bytes {body[:8].hex()}")

local = datetime.fromisoformat(d["start_at"]).replace(tzinfo=timezone.utc).astimezone()
stem = f"{local:%Y-%m-%d_%H%M}_{fid}"
out, tmp = AUDIO / f"{stem}.{ext}", AUDIO / f"_{stem}.{ext}"
if out.exists():
    sys.exit(f"{out} already exists")
raw = None
if kind == "raw-opus":
    raw, body = body, wrap(body)
write_private(tmp, body)

def refuse(msg):
    tmp.unlink()
    sys.exit(msg)

decode = subprocess.run(["ffmpeg", "-v", "error", "-i", str(tmp), "-f", "null", "-"], capture_output=True, text=True)
errors = decode.stderr.strip().splitlines()
if decode.returncode or errors:
    refuse(f"decode errors: {len(errors)} lines (exit {decode.returncode})")
probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(tmp)],
                       capture_output=True, text=True)
drift = float(probe.stdout.strip() or "nan") - d["duration"] / 1000
if probe.returncode or not abs(drift) <= max_drift:
    refuse(f"duration check failed: drift {drift:+.1f}s {probe.stderr.strip()[:200]}")
# Kept only now, so a refused download leaves nothing behind to block its retry.
if raw is not None:
    write_private(RAW / f"{stem}.opus", raw)
tmp.replace(out)

processed = "yes" if (d.get("source_list") or d.get("note_list")) else "no"
# Header-aware, so columns added later (the review page's `title`) stay aligned. The lock is shared
# with earshot's review page, which rewrites folders.csv when a title or folder changes.
with open(ROOT / ".folders.lock", "a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    header = next(csv.reader(open(CSV)))
    with open(CSV, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=header, restval="").writerow({
            "id": fid, "start_local": f"{local:%Y-%m-%d %H:%M}", "duration_min": f"{d['duration'] / 60000:.1f}",
            "folder": folder, "plaud_transcribed": processed, "name": d["name"]})
print(f"ok {out.name}  {kind}  {len(body) / 1e6:.1f} MB  drift {drift:+.1f}s  plaud_transcribed={processed}  folder={folder}")
