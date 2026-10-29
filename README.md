# pyarize

Offline diarized transcription for private recordings: audio goes in, and you get a transcript
with word timestamps and a speaker label on every word. It runs on your own GPU with no network
access, so meetings and interviews never leave the machine.

**Status:** working scripts (milestone M0), used every day by
[earshot](https://github.com/michaelv2/earshot) to process Plaud recordings. The job service in
[SPEC.md](SPEC.md) isn't built yet: a Unix-socket API inside a network-isolated systemd unit,
plus a benchmark to choose models. English only.

## What's here

| Script | What it does |
|---|---|
| `scripts/plaud_ingest.py` | Downloads one Plaud recording and checks it: size against `Content-Length`, a full decode, and length against the API's figure. Plaud-specific, so it's moving to earshot |
| `scripts/plaud_raw2ogg.py` | Losslessly wraps Plaud's bare Opus packets in an Ogg container |
| `scripts/plaud_transcribe.py` | Transcribes with faster-whisper `large-v3` and keeps word timestamps |
| `scripts/plaud_diarize.py` | Labels speakers with pyannote `speaker-diarization-community-1`, matches the labels to words, and smooths out one-word label flicker |
| `scripts/fetch_community1.py` | One-time download of the pinned pyannote model, recording each file's hash |
| `requirements-diarize.txt` | Exact versions for the diarization environment, which passed a dependency audit |

[`scripts/README.md`](scripts/README.md) has the full commands and file layout.

## How it keeps audio private

- **No network at all.** Diarization runs under `unshare -Urn`, a network namespace containing
  only a loopback interface that is switched off. The GPU still works, and nothing can leave
  the machine even if an opt-out is missed.
- **Telemetry off.** pyannote.audio 4.x sends usage telemetry by default. The scripts disable it
  before any import, and load the model with the Hugging Face Hub offline.
- **Pinned, hash-checked weights.** pyannote's checkpoints are Python pickles, so the script
  re-hashes them against the saved manifest before every load and refuses to load on a mismatch.
- **Never the cloud tier.** pyannoteAI's `precision-2` model and API keys send audio to their
  servers, so the scripts never use them.

## Setup

Requirements: Linux or WSL2, an NVIDIA GPU with about 6 GB of free VRAM, `ffmpeg`, and
[`uv`](https://github.com/astral-sh/uv).

```bash
# 1. Diarization environment, from the audited pins
uv venv .venv --python 3.12
uv pip sync --no-build requirements-diarize.txt

# 2. Accept the model's terms on Hugging Face (pyannote/speaker-diarization-community-1),
#    then download it once with your token:
HF_TOKEN=... HF_HUB_DISABLE_TELEMETRY=1 .venv/bin/python -I scripts/fetch_community1.py

# 3. A separate environment with faster-whisper for transcription
uv venv ~/venvs/whisper && uv pip install --python ~/venvs/whisper faster-whisper
```

Any change to `requirements-diarize.txt` deserves a fresh dependency review before it's
installed.

## Usage

```bash
~/venvs/whisper/bin/python -I scripts/plaud_transcribe.py meeting.ogg
unshare -Urn .venv/bin/python -I scripts/plaud_diarize.py meeting.ogg ~/recordings/plaud/transcripts/meeting.json
```

The scripts print metrics only, never transcript text, and write their outputs (`.json`,
`.txt`, `.rttm`, `.diarized.json`, `.diarized.txt`) with mode 0600.

Measured on an RTX 4080 (real-time factor is processing time divided by audio length):

| Step | Speed |
|---|---|
| Transcription (large-v3, fp16) | RTF 0.06 to 0.17, i.e. 4 to 10 minutes per hour of audio, depending on how much is speech |
| Speaker labelling | RTF about 0.011, under a minute per hour, with 1.6 GiB of VRAM |

## License

MIT, see [LICENSE](LICENSE). The pyannote model and the Python dependencies have their own
licenses.
