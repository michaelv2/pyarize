# Pyarize — SPEC

**Status:** v4.1 (2026-10-27), a local diarized-transcription service. English only.
**Owner:** michaelv2
**Consumers:** `earshot` (Plaud ingestion, separate repo), `yt-queue` (migration under review, §9),
and possibly `memoize`.

> **What changed from v3.** v3 (2026-08-09) scoped pyarize to diarizing yt-queue transcripts, and
> recorded the Plaud pipeline as dropped. On 2026-10-18 Plaud ingestion was revived for work
> meetings and candidate interviews. v4 adds transcription and makes pyarize consumer-agnostic.
> Source ingestion moved to its own repo, `earshot`, along with the Plaud findings from v3's
> Appendix A. Appendix A here is now pyarize's own decision log.

---

## 1. Summary

Pyarize turns an audio file into a transcript with word timestamps and speaker labels. It runs
entirely on this machine, with no network access, so `private` audio such as meetings and
interviews is safe to send it.

```
[any audio ffmpeg can decode]
        │
        └──> pyarize: decode → transcribe → diarize → align ──> segments + words + speakers
```

Pyarize knows nothing about where audio comes from. Callers deal with their source's quirks,
such as Plaud's raw Opus packets, and hand over audio that ffmpeg can decode.

### Why a separate service

- **One audited environment.** The torch / pyannote / ASR dependency tree lives in one venv that
  has passed the dependency audit, instead of being pinned into every caller.
- **One sandbox.** The service runs with no network and holds no credentials. Its callers need
  both network and credentials (Plaud, Slack, YouTube), so splitting the process is what makes
  the no-egress guarantee possible (§4).
- **One GPU owner.** Speech workloads on this box share one lease and one idle-unload policy.
- **Evidence-based backends.** Transcription and diarization backends can be swapped after the
  benchmark (§8) without changing any caller.

---

## 2. Goals and non-goals

### Goals

- **G1:** One job API. Audio goes in. Out come segments with word timestamps and a speaker label
  on every word.
- **G2:** Consumer-agnostic. The service has no source-specific code and no caller-specific
  options.
- **G3:** Private by construction. No network egress at runtime, no credentials, models pinned
  and hash-checked, and transcript text never in logs.
- **G4:** Diarization never sinks a job. If diarization fails, the job still returns the
  unlabelled transcript, plus a warning.
- **G5:** Backends chosen by reproducible benchmark, not by leaderboard reputation.
- **G6:** Coexists with the other GPU workloads on this box without starving them.

### Non-goals

- **Source ingestion.** Plaud sync, and later voice-memo pickup, belong to `earshot`.
- **Languages other than English.** Transcription is fixed to English (decided 2026-10-27).
- **Speaker names or roles.** Labels are `SPEAKER_00`, `SPEAKER_01`, and so on, per job.
  Consumers add roles, either through a UI that tags speakers or an LLM that infers them from
  the conversation. There are no voiceprints and no identity across recordings.
- **Summaries or other LLM post-processing.** Consumers do that.
- **Realtime or streaming.** All callers are batch.
- **Serving other machines.** The API is a local Unix socket only (§4).
- **Cloud backends for caller audio.** Never. A cloud diarizer may appear only in the benchmark
  harness, on public corpora (§8.2).

---

## 3. Constraints

| Resource | Value | Implication |
|---|---|---|
| GPU | RTX 4080, 16 GB, shared with the Windows host. About 7.8 GB was in use on 2026-10-26 with no WSL process holding it. | The working budget is about 8 GB, not 16 |
| CPU | Ryzen 9 5900XT, 16 cores / 32 threads | CPU diarization is viable. CPU large-v3 transcription is too slow to be the default |
| Host | Windows + WSL2, systemd 255 (user manager running), cgroup v2 | §4's sandbox is built on a systemd user unit |
| GPU neighbours | A local Ollama used on demand, yt-queue's in-process faster-whisper, and a TTS server | VRAM contention is real. Large LLMs run on a separate server, so only these smaller workloads share this GPU |

**Measured on 2026-10-26** (M0, one 56-minute Plaud meeting):

| Stage | Backend | Speed (RTF) | Peak VRAM |
|---|---|---|---|
| Transcribe | faster-whisper large-v3, fp16, beam 5, VAD, word timestamps | 0.100 (5.6 min) | not measured |
| Diarize | pyannote community-1, exclusive mode | 0.011 (38 s) | 1.6 GiB |

**GPU arbitration.** A single advisory lock (`~/.pyarize/gpu.lock`, via `flock`) guards inference.
Models unload after an idle timeout (`PYARIZE_IDLE_UNLOAD_S`, default 300 s). Under contention,
jobs wait in the queue rather than falling back to CPU, because they're asynchronous. CPU
fallback is opt-in per job (`device: "cpu"`).

---

## 4. Security model

All of these were verified on this host on 2026-10-26 (Appendix A).

- **No network.** The service runs as a systemd **user** unit with `PrivateNetwork=yes`, which
  leaves the process only a downed loopback interface. `IPAddressDeny=` is **silently ignored**
  in user units (tested: traffic still went through), so it must not be relied on.
- **Unix-socket API.** Because `PrivateNetwork` also hides localhost, the API is served on a
  Unix socket, `$XDG_RUNTIME_DIR/pyarize/pyarize.sock`, mode 0600. A process on the host can
  connect to a service inside the namespace this way, and that service still can't reach the
  network. Only processes running as this user can call it.
- **No credentials at runtime.** `pyarize fetch-models` runs once, outside the sandbox, with
  `HF_TOKEN`. It downloads the models at pinned revisions and writes
  `~/.pyarize/models.manifest.json` with each file's sha256. The service runs with
  `HF_HUB_OFFLINE=1` and never sees a token.
- **Pinned, hash-checked weights.** pyannote's weights are pickles, loaded with
  `weights_only=False`, so loading them can run arbitrary code. Every load re-hashes the files
  against the manifest and refuses on any mismatch. Changing a model revision is a deliberate
  `fetch-models` run.
- **Telemetry forced off.** pyannote.audio 4.x sends usage traces to `otel.pyannote.ai` by
  default. `config.py` sets these before any import, and also calls
  `set_telemetry_metrics(False)`:
  `PYANNOTE_METRICS_ENABLED=0`, `OTEL_SDK_DISABLED=true`, `OTEL_TRACES_EXPORTER=none`,
  `OTEL_METRICS_EXPORTER=none`, `HF_HUB_DISABLE_TELEMETRY=1` and `DO_NOT_TRACK=1`. The network
  sandbox is the backstop if an opt-out is ever missed.
- **No cloud pipelines.** pyannote's `precision-2` and any pyannoteAI API key are never
  configured. That client is installed only because pyannote.audio requires it.
- **Model provenance policy.** Weights must come from publishers who can be held accountable,
  under licences that permit this use. Chinese-origin speech models are excluded, consistent with
  the 2026-10-26 decision after Cohere's Qwen3.8 report. That covers Qwen-ASR, SenseVoice/FunASR,
  Canary-Qwen (which uses a Qwen decoder) and MOSS. An ASR model has no tools, but a
  mis-transcription is an integrity risk, and a blanket rule is easier to keep than case-by-case
  calls.
- **Dependency policy.**
  - There is one lockfile. Install with `--no-build` and `--exclude-newer` set two weeks back.
  - Any change needs a security-auditor pass first. The 2026-10-26 audit passed with conditions,
    and this section is those conditions.
  - After any change, re-check `.pth` files and the telemetry code paths.
- **Data handling.**
  - Transcript text and recording titles never go into logs.
  - Job results are written 0600 under `~/.pyarize/jobs/`. They're deleted when the caller
    deletes the job, or after `PYARIZE_JOB_TTL_H` (default 24 h).
  - Audio is read where it is, either from a path or uploaded over the socket. Pyarize never
    keeps a copy.

---

## 5. Architecture

```
pyarize/
├── pyproject.toml, uv.lock    # audited pins (succeeds requirements-diarize.txt)
├── pyarize/
│   ├── config.py              # PYARIZE_* settings; telemetry and offline env set before imports
│   ├── audio.py               # ffmpeg decode → 16 kHz mono float32, in memory
│   ├── models.py              # fetch-models, manifest, verify-before-load
│   ├── transcribe/
│   │   ├── base.py            # Transcriber protocol → segments with words
│   │   ├── faster_whisper.py  # large-v3 (M0 default), large-v3-turbo
│   │   └── …                  # bench candidates (§8.1) that pass the gates
│   ├── diarize/
│   │   ├── base.py            # Diarizer protocol → list[Turn(start, end, speaker)]
│   │   ├── pyannote_backend.py
│   │   └── sortformer.py
│   ├── align.py               # word → speaker (max overlap, nearest turn in gaps); utterance grouping
│   ├── jobs.py                # SQLite job store, single GPU worker, restart recovery
│   ├── gpu.py                 # lease + idle unload
│   ├── service.py             # FastAPI on a Unix socket
│   ├── client.py              # stdlib-only client (http.client over AF_UNIX) for callers to vendor
│   └── cli.py                 # transcribe | fetch-models | serve | bench
├── systemd/pyarize.service    # user unit: PrivateNetwork=yes, socket under $XDG_RUNTIME_DIR
├── bench/                     # §8
├── scripts/                   # M0 scripts; superseded by the package at M1
└── tests/
```

- **`pyarize transcribe <file>`** runs the same pipeline as the service, in-process, inside
  `unshare -Urn`. It's for one-off use and debugging.
- **`client.py` uses only the standard library**, so callers need no new dependencies, and
  therefore no new audits, to use pyarize.
- **The M0 scripts.** `scripts/plaud_transcribe.py` and `scripts/plaud_diarize.py` become the
  first two backends plus `align.py`. `scripts/plaud_ingest.py` and `scripts/plaud_raw2ogg.py`
  move to `earshot`.

---

## 6. API

The API is HTTP over the Unix socket. Jobs are asynchronous, because a 60-minute recording
takes about 7 minutes.

| Method | Route | Purpose |
|---|---|---|
| POST | `/jobs` | Submit a job. Returns `202 {"job_id": …}` |
| GET | `/jobs/{id}?wait=60` | Status: `queued`, `running` (with phase), `done` or `failed`. `wait` long-polls |
| GET | `/jobs/{id}/result` | Result JSON |
| DELETE | `/jobs/{id}` | Delete the job and its artifacts |
| GET | `/health` | Backends, device, free VRAM, lease state, which models are loaded, queue depth |
| GET | `/backends` | Available backends, with their pinned model revisions |

**Request**

```jsonc
{
  "audio_path": "/home/you/recordings/plaud/audio/2026-10-26_1002_f_….ogg", // or multipart upload
  "segments": null,          // optional: caller's own transcript (with words) → diarize-only job
  "transcribe": {"backend": null, "language": "en", "word_timestamps": true},
  "diarize": {"enabled": true, "backend": null,
              "num_speakers": null, "min_speakers": null, "max_speakers": null},
  "device": null             // null = GPU via lease; "cpu" opts into CPU
}
```

- **Diarize-only jobs.** Callers that already have a transcript, like yt-queue today, send
  `segments` and get labels back.
- **Single-speaker audio**, such as voice memos, is sent with `diarize.enabled: false`.
- **Language.** `en` is the only accepted value, and the default. Fixing it skips Whisper's
  language detection, which can misfire on a noisy opening. Any other value is rejected.

**Result**

```jsonc
{
  "audio": {"duration_s": 3372.8, "sha256": "…"},
  "language": "en",
  "transcriber": {"backend": "faster_whisper_large_v3", "revision": "…"},
  "diarizer": {"backend": "pyannote_community1", "revision": "3533c8cf8e36"},  // null if disabled
  "num_speakers": 2,
  "segments": [{"start": 0.0, "end": 4.2, "speaker": "SPEAKER_00", "text": "…",
                "words": [{"start": 0.0, "end": 0.3, "word": " Good", "p": 0.99, "speaker": "SPEAKER_00"}]}],
  "turns": [{"start": 0.0, "end": 4.2, "speaker": "SPEAKER_00"}],   // RTTM-equivalent
  "quality": {"median_avg_logprob": -0.12, "low_confidence_segments": 0, "words_in_turn": 0.99},
  "timing": {"transcribe_rtf": 0.100, "diarize_rtf": 0.011, "queued_s": 0},
  "warnings": []             // e.g. "diarization_failed: …"; the transcript is still returned
}
```

- **How segments are formed.** With diarization on, `segments` are speaker utterances: runs of
  words from one speaker, split at pauses over 2 s. With diarization off, they are the
  transcriber's own segments.
- **Smoothing.** Within one sentence of a transcriber segment, a run of another speaker's words
  that is ≤ 3 words or < 1 s long takes its neighbour's speaker. Relabelled words keep
  `raw_speaker`. Whole short sentences, such as "Yeah.", keep their own label. The thresholds are
  tuned in M4 against cpWER on AMI.
- **Overlapping speech.** Exclusive diarization gives each word to exactly one speaker, so in
  overlaps the quieter speaker's words get attributed to the louder one.

---

## 7. Configuration

| Variable | Default | Notes |
|---|---|---|
| `PYARIZE_HOME` | `~/.pyarize` | Manifest, lock, job store, results |
| `PYARIZE_SOCKET` | `$XDG_RUNTIME_DIR/pyarize/pyarize.sock` | Created with mode 0600 |
| `PYARIZE_TRANSCRIBER` | `faster_whisper_large_v3` | Revisited after the bake-off (§8) |
| `PYARIZE_DIARIZER` | `pyannote_community1` | `pyannote_community1`, `pyannote_31` or `sortformer` |
| `PYARIZE_DEVICE` | `cuda` | Per-job `"cpu"` override |
| `PYARIZE_IDLE_UNLOAD_S` | `300` | Release VRAM when idle |
| `PYARIZE_JOB_TTL_H` | `24` | Delete results the caller never fetched |
| `HF_TOKEN` | — | Only for `fetch-models`; never in the service environment |

v3's `PYARIZE_HOST`/`PORT` and `PYARIZE_ALLOW_CLOUD` are gone.

---

## 8. Benchmark

### 8.1 Transcription

Every candidate must:
- produce word timestamps, natively or through forced alignment
- run offline in under 6 GB of VRAM
- have a licence that permits this use
- pass the provenance policy (§4)

The candidate list was checked against the Open ASR Leaderboard on 2026-10-26, and is re-checked
at the start of M4.

| # | Transcriber | Notes |
|---|---|---|
| T1 | faster-whisper large-v3 | Baseline, already measured (§3) |
| T2 | faster-whisper large-v3-turbo | Same family, expected to be several times faster |
| T3 | NVIDIA Parakeet TDT 0.6B v2 (English-only) | Fast TDT decoder. Brings NeMo's dependency tree |
| T4 | IBM Granite Speech 4.1 2B | Top of the leaderboard as of 2026-10. LLM decoder, so slower |
| T5 | Cohere Transcribe 03-2026 (2B, Apache 2.0) | Near the top. Needs a check on word timestamps |

T4 and T5 likely need forced alignment for word timestamps. That cost counts toward their speed.

**Data**

| Corpus | Licence | Role |
|---|---|---|
| AMI (test), single distant mic | CC-BY-4.0 | Primary for meetings. One far-field mic matches a Plaud on a table |
| AMI (test), headset mics | CC-BY-4.0 | Upper bound. Separates mic quality from model quality |
| Earnings-22 | CC-BY-SA-4.0 | Accented, jargon-heavy business speech |
| TED-LIUM 3 (test) | CC-BY-NC-ND-3.0, private benchmarking only | Talks, the yt-queue domain |

**Metrics**
- WER, after Whisper's text normalizer
- insertions in non-speech regions, to catch Whisper hallucinations
- RTF, cold and warm
- peak VRAM
- word-timestamp error against AMI's forced alignments

### 8.2 Diarization

| # | Diarizer | Notes |
|---|---|---|
| S1 | pyannote 3.1 | Baseline |
| S2 | pyannote community-1 | M0 default |
| S3 | NeMo Streaming Sortformer v2.1 | Very fast, but **degrades hard past 4 speakers** |
| C1 | pyannoteAI Precision-2 (API) | Optional quality ceiling. **Public corpora only**, run from the bench harness outside the service |

DiariZen is **excluded**. Its weights are CC BY-NC-4.0, and the licence question isn't worth the
few points it might buy.

**Data**
- **VoxConverse dev (CC-BY-4.0):** the yt-queue domain.
- **AMI test (CC-BY-4.0):** meetings. Now co-primary, because Plaud meetings are a main consumer.
- **Preprocessed copies:** both are on the Hub under `diarizers-community`.

**Metrics**
- DER at a 0 s collar with overlap included (the honest number), and at a 0.25 s collar without
  overlap (comparable to published figures)
- JER
- speaker-count error
- RTF
- peak VRAM

### 8.3 End-to-end and spot checks

- **cpWER on AMI** (via `meeteval`) for the best transcriber × diarizer pairs. This is the
  number closest to what a reader of the output sees.
- **Archive spot checks, no annotation.**
  - 10 yt-queue items, where the speaker count is known by inspection.
  - The Plaud archive, using speaker counts the owner supplies. That audio is private, so it's
    processed only by the service, and only aggregate metrics are recorded.

### 8.4 Protocol and gates

**Protocol**
- Cache raw outputs, so re-scoring needs no re-inference.
- Run three timing passes after warm-up and report the median. Measure cold start separately.
- Record pinned model revisions per run.
- Report per-file distributions with bootstrap confidence intervals, not just aggregates.

| Gate | Threshold |
|---|---|
| Default transcriber | Lowest WER on AMI distant-mic among candidates with RTF ≤ 0.15. It only replaces T1 if it beats it by more than T1's confidence interval |
| DER on VoxConverse dev (0 s collar, overlap) | ≤ 15% |
| Speaker count | Exact on ≥ 80% of files, and within ±1 on ≥ 95% |
| End-to-end RTF | ≤ 0.15 on the 4080, so a 60-minute recording finishes in ≤ 9 minutes. M0 managed 0.11 |
| Peak VRAM | ≤ 6 GB with both models loaded |
| Caller resilience | `earshot` and yt-queue both complete normally with pyarize stopped (queued or degraded, not failed) |

---

## 9. yt-queue migration: tradeoffs

Today yt-queue transcribes in-process. It runs faster-whisper with the `base` model by default,
with VAD and an artifact filter, and one worker slot per entry in `whisper_devices`. In its job
pipeline, transcription (Phase 2) is **fatal**, and summarize, score and archive are non-fatal.

| | Migrate transcription to pyarize | Keep it in yt-queue |
|---|---|---|
| Quality | Benchmark-chosen model plus diarization in one call | `base` unless reconfigured; diarization would be a second call |
| GPU | One owner and one lease, so no double-loaded Whisper | Two processes compete for VRAM with no coordination |
| Dependencies | yt-queue drops faster-whisper and CTranslate2 | yt-queue keeps its own speech stack and its own audits |
| Availability | **pyarize becomes a hard dependency of Phase 2.** Downtime means jobs wait | Self-contained |
| Throughput | One serial GPU worker. Bulk YouTube backfills would queue behind meetings | Parallel slots per device |
| Deployment | Same host only (Unix socket) | Can run anywhere |

**Recommendation.** First, integrate yt-queue as a **diarize-only** caller, sending its own
`segments`. That keeps v3's non-fatal design and is low-risk. Decide on full migration after
M4's numbers. If yt-queue migrates, keep its in-process `base` transcriber as a fallback for when
pyarize is unreachable, and add a `priority` field to jobs, so that interactive work isn't stuck
behind bulk backfills.

---

## 10. Milestones

| # | Deliverable | Exit criterion |
|---|---|---|
| **M0** ✅ | Working scripts (2026-10-26) | One Plaud meeting transcribed and diarized, offline, in a sandbox, with an audited env and pinned weights |
| **M1** | Package and CLI | `pyarize transcribe <file>` runs the M0 pipeline from the package. `fetch-models` writes the manifest. The combined faster-whisper + pyannote env passes the audit, which includes proving that CTranslate2's CUDA 12 libraries coexist with torch's CUDA 13 |
| **M2** | Service | Job API over the Unix socket. User unit with `PrivateNetwork=yes`. GPU lease and idle unload. Jobs survive a restart. Egress verified blocked under the unit |
| **M3** | First consumer | `earshot` sends Plaud recordings through the service, unattended |
| **M4** | Benchmark and bake-off | REPORT.md ranks T1–T5 and S1–S3 (+C1). Defaults set on evidence |
| **M5** | yt-queue, diarize-only | Labels flow into `segments_json`. Unlabelled fallback verified with the service stopped. Full-migration decision recorded |
| **M6** | Hardening | Tests, retention TTL, contention behaviour, dependency re-audit cadence |

The service is built before the benchmark this time. M0 already gives a sound default, and the
backend protocols mean the bake-off can swap models without touching callers.

---

## 11. Open questions

1. **yt-queue full migration** (§9). Decide after M4.
2. **memoize's integration.** Should memoize call pyarize directly, for example on Signal voice
   notes (memoize has no audio handling today)? Or should `earshot` deliver distilled notes?
   Raw transcripts are a poor memoize input (see `earshot` Appendix A). This doesn't change
   pyarize's API.
3. **Overlapping speech.** Is it acceptable to attribute overlaps exclusively, or should
   overlapped words carry both speakers?
4. **VRAM policy.** Keep both models resident (about 5 GB) or load them per stage? This depends
   on the M2 contention tests.

---

## Appendix A — Decision log

**2026-10-27: word-level smoothing.** Exclusive diarization gives a listener brief turns over
the speaker (likely a quiet "mm-hmm") that Whisper doesn't transcribe, so max-overlap assignment
flipped the speaker's own words to the listener mid-sentence. Measured on two 56-minute Plaud
recordings, re-aligned from the same RTTM:

| | Interview before → after | Meeting before → after |
|---|---|---|
| Utterances | 378 → 176 | 266 → 87 |
| Speaker switches inside one sentence | 211 → 18 | 169 → 4 |
| Short A-B-A islands | 161 → 35 | 117 → 14 |
| Islands that are their own Whisper segment | 14 → 35 of 35 | 5 → 8 of 14 |

**2026-10-27: English only.** The owner chose English-only transcription. `transcribe.language`
is fixed to `en` (§6), and English-only checkpoints such as Parakeet TDT 0.6B v2 (T3) are eligible
for the bake-off. Voice memos are deferred in `earshot`; the API keeps `diarize.enabled: false`
for any single-speaker caller.

**2026-10-26: v4 rescope (diarized transcription, consumer-agnostic).**

- **Ingestion goes in its own repo (`earshot`).**
  - The ingester needs network and credentials (Plaud, S3, Slack, the DS418), and pyarize must
    have neither.
  - Plaud's API churns (the ID prefix changed from `of_` to `f_`, it serves three formats, and
    cloud copies come back short), and that shouldn't touch an audited ML environment.
  - Integrations live with their callers. yt-queue and memoize integrate on their own side.
- **Speaker roles are a consumer concern.** Labels only, per job.
- **Sandbox, tested on this host:**

  | Mechanism | Outbound request | Usable for serving |
  |---|---|---|
  | No restriction (control) | 200 | — |
  | User unit with `IPAddressDeny=any` and `IPAddressAllow=localhost` | **200 (ignored)** | — |
  | User unit with `PrivateNetwork=yes` | blocked | Unix socket: host connected, egress still blocked |
  | `unshare -Urn` | blocked | GPU (`/dev/dxg`) reachable; used for M0 |

- **Dependency audit of pyannote.audio 4.0.7, 113 packages: PASS WITH CONDITIONS.**
  - All packages are legitimate. `httpx2` and `httpcore2` are the Pydantic-stewarded HTTPX fork,
    pulled in by huggingface-hub 2.2.0.
  - The conditions are §4's telemetry, offline, pinning and no-cloud rules.
  - The only `.pth` files that run at startup are the two standard ones.
  - The audit flagged torchaudio 2.11 against torch 2.14.1 as a possible ABI risk. In practice
    it imports, its functional ops run, and pyannote loads it during diarization.
  - setuptools 84.0.0 has GHSA-xmj3-9gf7-h52w, which affects Windows paths only.
- **Model pinned:** `pyannote/speaker-diarization-community-1` @ `3533c8cf8e36`. Its two `.bin`
  files are pickles; its `.npz` files are plain float arrays, loaded with pickling off.
- **Smoke-test lesson.** pyannote hears two espeak-ng voices as one speaker, and forcing
  `num_speakers=2` splits them badly. Synthetic voices can't test diarization quality.

**2026-08-09: v3 rescope to diarization-only.** The Plaud pipeline was dropped in favour of iOS
dictation into Signal. That was reversed on 2026-10-18 for work meetings and interviews; Signal
still covers personal memos. The v3 Plaud, WSL and memoize findings now live in `earshot`'s SPEC.
