# Plaud pipeline scripts

Working scripts for the revived Plaud → local transcription → diarization pipeline (2026-10).
They predate the SPEC v4 rewrite and the `pyarize/` package. Each one prints metrics only, never
transcript text or presigned URLs. Run them from the repo root.

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1. Ingest | `python3 -I scripts/plaud_ingest.py <file_id> <folder>` | `~/recordings/plaud/audio/<YYYY-MM-DD_HHMM>_<id>.{mp3,ogg}` and a row in `~/recordings/folders.csv` |
| 2. Transcribe | `<whisper-python> -I scripts/plaud_transcribe.py <audio>` | `~/recordings/plaud/transcripts/<stem>.{json,txt}` |
| 3. Diarize | `unshare -Urn .venv/bin/python -I scripts/plaud_diarize.py <audio> <transcript.json> [N]` | `<stem>.rttm`, `<stem>.diarized.{json,txt}` next to the transcript |

- **Ingest:** find new IDs by comparing the API listing with `folders.csv`. The API exposes no
  folders, so new recordings arrive `unfiled`, and earshot's brief.py files them under a suggested
  category (`interview`, `1:1 meeting`, `group meeting`, `family`; plus `discard`). The format is
  detected from the bytes, because Plaud serves MP3, Ogg Opus, or bare Opus packets.
  `plaud_raw2ogg.py` wraps bare packets into Ogg losslessly and keeps the original in
  `~/recordings/plaud/raw/`. The duration check allows 3 s of drift by default. Pass a larger
  value only after confirming the file decodes cleanly. The first post-backfill file came back
  9.4 s short.
- **Transcribe:** faster-whisper large-v3 on CUDA, from a separate Python environment with
  faster-whisper installed (`<whisper-python>`). The JSON keeps word
  timestamps for step 3. A 56-minute recording took about 6 minutes (RTF 0.10).
- **Diarize:** pyannote `speaker-diarization-community-1`. Omit `N` to let pyannote estimate the
  speaker count. Took 38 s for 56 minutes (RTF 0.011), with 1.6 GiB peak VRAM.
- **Smoothing.** Within a sentence, a run of another speaker's words that is 3 words or fewer, or
  under 1 s, takes its neighbour's speaker. Without it, a listener's "mm-hmm" over the speaker
  flips the speaker's own words to the listener, about 150 times an hour. Relabelled words keep
  `raw_speaker`. To redo only this step from an existing `.rttm`, with no audio or GPU, run
  `.venv/bin/python -I scripts/plaud_diarize.py --realign <transcript.json>`.

## Diarization environment

One-time setup:

```bash
uv venv .venv --python 3.12
uv pip sync --no-build requirements-diarize.txt
HF_HUB_DISABLE_TELEMETRY=1 .venv/bin/python -I scripts/fetch_community1.py   # needs HF_TOKEN once
```

`requirements-diarize.txt` is the exact set that passed the dependency audit on 2026-10-26. Any
change to it needs a new audit before install.

The audit passed **with conditions**, and `plaud_diarize.py` enforces them:

- **Telemetry is on by default in pyannote.audio 4.x.** It sends usage traces to
  `otel.pyannote.ai`. The script sets `PYANNOTE_METRICS_ENABLED=0` and `OTEL_SDK_DISABLED=true`
  before any import, and calls `set_telemetry_metrics(False)`.
- **The Hub is offline** (`HF_HUB_OFFLINE=1`). The model loads from the snapshot pinned in
  `~/.pyarize/community-1.manifest.json`.
- **Model weights are pickles.** pyannote unpickles them in full, so the script re-hashes them
  against the manifest and refuses to load on a mismatch.
- **No network at all.** `unshare -Urn` runs the process in a network namespace with only a
  downed loopback, so nothing can leave even if an opt-out is missed. The GPU still works.
- **Never use `precision-2` or a pyannoteAI API key.** Those send audio to pyannoteAI's servers.

Smoke tests need real voices. pyannote hears two espeak-ng voices as one speaker.
