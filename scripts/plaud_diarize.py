"""Diarize a Plaud recording with pyannote community-1 and attach speakers to its faster-whisper words.

Reads <stem>.json (from plaud_transcribe.py) and writes, next to it:
  <stem>.rttm            raw diarization turns
  <stem>.diarized.json   segments regrouped by speaker, words labelled
  <stem>.diarized.txt    "[HH:MM:SS] SPEAKER_00: ..." lines
Audio is decoded with the system ffmpeg and passed in memory, so torchcodec never opens the file.
Telemetry is forced off and the Hub is offline; the model loads from the pinned local snapshot in
~/.pyarize/community-1.manifest.json after its pickled weights are re-hashed. Run it under
`unshare -Urn` so it has no network at all. Prints metrics only, never transcript text.

Each word gets the speaker whose turn overlaps it most. Then, within each sentence of a Whisper
segment, a run of another speaker's words that is <= 3 words or < 1 s long takes its neighbour's
speaker. Exclusive diarization gives brief turns to a listener's "mm-hmm" over the speaker, which
Whisper doesn't transcribe, so without this the speaker's own words flip mid-sentence. Relabelled
words keep their original label in "raw_speaker".

--realign skips pyannote and re-runs only the word alignment from an existing <stem>.rttm
(no audio, model or GPU needed).

usage: unshare -Urn .venv/bin/python -I scripts/plaud_diarize.py <audio_path> <transcript.json> [num_speakers]
       .venv/bin/python -I scripts/plaud_diarize.py --realign <transcript.json>
"""
import os

# Before any pyannote / huggingface_hub / opentelemetry import: these are read at import time.
os.environ.update(PYANNOTE_METRICS_ENABLED="0", OTEL_SDK_DISABLED="true", OTEL_TRACES_EXPORTER="none",
                  OTEL_METRICS_EXPORTER="none", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")

import bisect
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

MAX_RUN_WORDS, MAX_RUN_S = 3, 1.0  # a run of another speaker inside a sentence this short is relabelled
SENTENCE_END = (".", "?", "!")


def hms(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


def write_private(path, text):
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as fh:
        fh.write(text)


def diarize(audio, num_speakers):
    """Run pyannote community-1 on the audio; return sorted (start, end, speaker) turns."""
    import numpy as np
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.telemetry import set_telemetry_metrics

    set_telemetry_metrics(False)
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(audio), "-map", "0:a:0", "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    waveform = torch.from_numpy(np.frombuffer(pcm, dtype=np.float32).copy()).unsqueeze(0)
    duration = waveform.shape[1] / 16000

    manifest = json.loads((Path.home() / ".pyarize" / "community-1.manifest.json").read_text())
    for f in manifest["files"]:
        if f["file"].endswith((".bin", ".npz", "config.yaml")):
            if hashlib.sha256((Path(manifest["path"]) / f["file"]).read_bytes()).hexdigest() != f["sha256"]:
                sys.exit(f"hash mismatch on {f['file']}; refusing to load")

    t0 = time.time()
    pipeline = Pipeline.from_pretrained(manifest["path"])
    pipeline.to(torch.device("cuda"))
    t_load = time.time() - t0
    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    output = pipeline({"waveform": waveform, "sample_rate": 16000}, **kwargs)
    t_run = time.time() - t0 - t_load
    print(f"model load {t_load:.0f}s, diarize {t_run:.0f}s, RTF {t_run / duration:.3f}, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    # 4.x returns both; the exclusive variant has one speaker at a time, which is what word alignment needs.
    annotation = getattr(output, "exclusive_speaker_diarization", None) or getattr(output, "speaker_diarization", output)
    return sorted((seg.start, seg.end, spk) for seg, _, spk in annotation.itertracks(yield_label=True)), manifest


def read_rttm(path):
    turns = []
    for line in path.read_text().splitlines():
        f = line.split()
        turns.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    return sorted(turns)


def assign(turns, words):
    """Label each word with the speaker it overlaps most (nearest turn if in a gap); return how many overlapped."""
    starts = [t[0] for t in turns]
    in_turn = 0
    for w in words:
        ws, we = w["start"], w["end"]
        i = bisect.bisect_right(starts, we)
        best, best_ov = None, 0.0
        for s, e, spk in turns[max(0, i - 4):i]:
            ov = min(e, we) - max(s, ws)
            if ov > best_ov:
                best, best_ov = spk, ov
        if best is None and turns:
            mid = (ws + we) / 2
            best = min(turns[max(0, i - 2):i + 2] or turns, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)))[2]
        w["speaker"] = best
        in_turn += best_ov > 0
    return in_turn


def smooth(segments):
    """Within each sentence of each Whisper segment, give short speaker runs their neighbour's speaker."""
    for seg in segments:
        sentences, cur = [], []
        for w in seg["words"]:
            cur.append(w)
            if w["word"].strip().rstrip("\"'”’)").endswith(SENTENCE_END):
                sentences.append(cur)
                cur = []
        if cur:
            sentences.append(cur)
        for sent in sentences:
            while True:
                runs = []
                for w in sent:
                    if runs and runs[-1][-1]["speaker"] == w["speaker"]:
                        runs[-1].append(w)
                    else:
                        runs.append([w])
                dur = lambda r: r[-1]["end"] - r[0]["start"]
                short = [i for i, r in enumerate(runs) if len(r) <= MAX_RUN_WORDS or dur(r) < MAX_RUN_S]
                if len(runs) < 2 or not short:
                    break
                i = min(short, key=lambda i: dur(runs[i]))
                j = max((j for j in (i - 1, i + 1) if 0 <= j < len(runs)), key=lambda j: dur(runs[j]))
                for w in runs[i]:
                    w.setdefault("raw_speaker", w["speaker"])
                    w["speaker"] = runs[j][0]["speaker"]
    relabelled = 0
    for seg in segments:
        for w in seg["words"]:
            if "raw_speaker" in w:
                if w["raw_speaker"] == w["speaker"]:
                    del w["raw_speaker"]
                else:
                    relabelled += 1
    return relabelled


realign = sys.argv[1] == "--realign"
args = sys.argv[2:] if realign else sys.argv[1:]
src = Path(args[0] if realign else args[1])
stem = src.with_suffix("")
rttm_out, json_out, txt_out = (stem.with_name(stem.name + ext) for ext in (".rttm", ".diarized.json", ".diarized.txt"))
for p in (json_out, txt_out) if realign else (rttm_out, json_out, txt_out):
    if p.exists():
        sys.exit(f"{p} already exists")
transcript = json.loads(src.read_text())

if realign:
    turns = read_rttm(rttm_out)
    manifest = json.loads((Path.home() / ".pyarize" / "community-1.manifest.json").read_text())
else:
    turns, manifest = diarize(Path(args[0]), int(args[2]) if len(args) > 2 else None)
    write_private(rttm_out, "".join(f"SPEAKER {stem.name} 1 {s:.3f} {e - s:.3f} <NA> <NA> {spk} <NA> <NA>\n" for s, e, spk in turns))

words = [w for seg in transcript["segments"] for w in seg["words"]]
in_turn = assign(turns, words)
relabelled = smooth(transcript["segments"])

utterances = []
for w in words:
    if utterances and utterances[-1]["speaker"] == w["speaker"] and w["start"] - utterances[-1]["end"] < 2.0:
        u = utterances[-1]
        u["end"], u["text"] = w["end"], u["text"] + w["word"]
        u["words"].append(w)
    else:
        utterances.append({"speaker": w["speaker"], "start": w["start"], "end": w["end"], "text": w["word"], "words": [w]})
for u in utterances:
    u["text"] = u["text"].strip()

write_private(json_out, json.dumps({**{k: v for k, v in transcript.items() if k != "segments"},
                                    "diarizer": f"{manifest['repo']}@{manifest['revision'][:12]}",
                                    "smoothing": {"max_run_words": MAX_RUN_WORDS, "max_run_s": MAX_RUN_S,
                                                  "relabelled_words": relabelled},
                                    "utterances": utterances}, indent=1))
write_private(txt_out, "".join(f"[{hms(u['start'])}] {u['speaker']}: {u['text']}\n" for u in utterances))

talk = defaultdict(float)
for s, e, spk in turns:
    talk[spk] += e - s
total = sum(talk.values())
print(f"{len(talk)} speakers, {len(turns)} turns, {len(utterances)} utterances; "
      f"{in_turn / max(len(words), 1):.0%} of {len(words)} words fall inside a speaker turn; "
      f"{relabelled} words relabelled by smoothing")
for spk, secs in sorted(talk.items(), key=lambda x: -x[1]):
    print(f"  {spk}: {secs / 60:5.1f} min ({secs / total:.0%})")
print("wrote " + "\n      ".join(str(p) for p in ((json_out, txt_out) if realign else (rttm_out, json_out, txt_out))))
