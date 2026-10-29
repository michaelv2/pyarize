"""One-time download of pyannote community-1 at a pinned commit; verify and record file hashes.

Uses only huggingface_hub (pyannote is never imported). Checks every LFS file's sha256 against the
Hub's metadata and writes a manifest to ~/.pyarize/community-1.manifest.json.
Needs HF_TOKEN with the model's gate accepted; later runs are offline and need no token.

usage: HF_HUB_DISABLE_TELEMETRY=1 .venv/bin/python -I scripts/fetch_community1.py
"""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "pyannote/speaker-diarization-community-1"
api = HfApi()
info = api.model_info(REPO, files_metadata=True)
sha = info.sha
path = Path(snapshot_download(REPO, revision=sha))

files, bad = [], []
for s in info.siblings:
    p = path / s.rfilename
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    expected = s.lfs.sha256 if s.lfs else None
    if expected and digest != expected:
        bad.append(s.rfilename)
    files.append({"file": s.rfilename, "bytes": p.stat().st_size, "sha256": digest, "lfs": bool(s.lfs)})
    print(f"  {s.rfilename:40} {p.stat().st_size:>11,}  {digest[:16]}  {'LFS ok' if expected and digest == expected else ('LFS MISMATCH' if expected else 'git file')}")

out = Path.home() / ".pyarize"
out.mkdir(mode=0o700, exist_ok=True)
(out / "community-1.manifest.json").write_text(json.dumps({"repo": REPO, "revision": sha, "path": str(path), "files": files}, indent=1))
print(f"revision {sha}\nlocal {path}\nmismatches: {bad or 'none'}")
