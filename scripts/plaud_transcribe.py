"""Transcribe a Plaud recording locally with faster-whisper; write JSON (segments + words) and TXT.

Output goes to <out_dir>/<stem>.{json,txt} (0600), default ~/recordings/plaud/transcripts/.
Prints quality metrics only, never transcript text. Needs a Python environment with faster-whisper
and CUDA; pyarize's diarization venv doesn't include it (see README).

usage: <whisper-python> -I scripts/plaud_transcribe.py <audio_path> [model=large-v3] [out_dir]
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel

audio = Path(sys.argv[1])
name = sys.argv[2] if len(sys.argv) > 2 else "large-v3"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path.home() / "recordings" / "plaud" / "transcripts"
OUT.mkdir(mode=0o700, exist_ok=True)
json_out, txt_out = OUT / f"{audio.stem}.json", OUT / f"{audio.stem}.txt"
for p in (json_out, txt_out):
    if p.exists():
        sys.exit(f"{p} already exists")

t0 = time.time()
model = WhisperModel(name, device="cuda", compute_type="float16")
t_load = time.time() - t0
segs_iter, info = model.transcribe(str(audio), beam_size=5, vad_filter=True, word_timestamps=True)
segs = [s for s in segs_iter if s.text.strip()]
t_run = time.time() - t0 - t_load

def hms(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"

def write_private(path, text):
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as fh:
        fh.write(text)

write_private(json_out, json.dumps({
    "audio": audio.name, "model": f"faster-whisper {name}", "language": info.language,
    "language_probability": round(info.language_probability, 3), "duration": round(info.duration, 2),
    "segments": [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                  "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3),
                  "compression_ratio": round(s.compression_ratio, 2),
                  "words": [{"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word, "p": round(w.probability, 3)}
                            for w in s.words or []]} for s in segs]}, indent=1))
write_private(txt_out, "".join(f"[{hms(s.start)}] {s.text.strip()}\n" for s in segs))

if not segs:  # e.g. a few seconds of pocket noise; the empty transcript still marks it as done
    print(f"model load {t_load:.0f}s, transcribe {t_run:.0f}s; no speech detected in {hms(info.duration)} of audio")
    print(f"wrote {json_out}\n      {txt_out}")
    sys.exit(0)

words = [w for s in segs for w in s.words or []]
speech = sum(s.end - s.start for s in segs)
repeats = sum(1 for a, b in zip(segs, segs[1:]) if a.text.strip() == b.text.strip())
print(f"model load {t_load:.0f}s, transcribe {t_run:.0f}s, RTF {t_run / info.duration:.3f}")
print(f"language {info.language} (p={info.language_probability:.2f}), audio {hms(info.duration)}, "
      f"{len(segs)} segments, {len(words)} words, speech covered {speech / info.duration:.0%}")
print(f"median avg_logprob {statistics.median(s.avg_logprob for s in segs):.2f}; "
      f"low-confidence segments (avg_logprob < -1): {sum(s.avg_logprob < -1 for s in segs)}; "
      f"repetitive (compression > 2.4): {sum(s.compression_ratio > 2.4 for s in segs)}; "
      f"identical consecutive: {repeats}; median word p {statistics.median(w.probability for w in words) if words else float('nan'):.2f}")
print(f"first segment {hms(segs[0].start)}, last segment ends {hms(segs[-1].end)}")
print(f"wrote {json_out}\n      {txt_out}")
