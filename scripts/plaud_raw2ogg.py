"""Losslessly wrap Plaud's raw Opus recordings (fixed 80-byte packets, no container) in Ogg.

Plaud stores some recordings as bare 20 ms CELT-wideband Opus packets (TOC 0xb8, 16 kHz mono,
32 kbps CBR). The packets are copied untouched; only Ogg pages and Opus headers are added.
"""
import struct
import sys
from pathlib import Path

PACKET = 80          # bytes per packet (32 kbps x 20 ms)
SAMPLES = 960        # 20 ms at Opus's fixed 48 kHz granule rate
PER_PAGE = 50        # packets per Ogg page (1 s of audio)
SERIAL = 0x504C4144  # "PLAD"

def _crc_table():
    t = []
    for i in range(256):
        r = i << 24
        for _ in range(8):
            r = ((r << 1) ^ 0x04C11DB7) if r & 0x80000000 else (r << 1)
        t.append(r & 0xFFFFFFFF)
    return t

CRC = _crc_table()

def ogg_crc(data):
    c = 0
    for b in data:
        c = ((c << 8) & 0xFFFFFFFF) ^ CRC[((c >> 24) ^ b) & 0xFF]
    return c

def page(packets, granule, seq, flags):
    lacing = b""
    for p in packets:
        lacing += b"\xff" * (len(p) // 255) + bytes([len(p) % 255])
    header = struct.pack("<4sBBqIIIB", b"OggS", 0, flags, granule, SERIAL, seq, 0, len(lacing)) + lacing
    body = b"".join(packets)
    crc = ogg_crc(header + body)
    return header[:22] + struct.pack("<I", crc) + header[26:] + body

def wrap(raw: bytes) -> bytes:
    if len(raw) % PACKET or any(raw[i] != 0xB8 for i in range(0, len(raw), PACKET)):
        raise ValueError("not a Plaud raw Opus stream (expected 80-byte packets with TOC 0xb8)")
    head = struct.pack("<8sBBHIhB", b"OpusHead", 1, 1, 0, 16000, 0, 0)  # v1, mono, pre-skip 0, 16 kHz input
    tags = b"OpusTags" + struct.pack("<I", 6) + b"pyarize" [:6] + struct.pack("<I", 0)
    out = [page([head], 0, 0, 0x02), page([tags], 0, 1, 0)]
    packets = [raw[i:i + PACKET] for i in range(0, len(raw), PACKET)]
    seq = 2
    for i in range(0, len(packets), PER_PAGE):
        chunk = packets[i:i + PER_PAGE]
        last = i + PER_PAGE >= len(packets)
        out.append(page(chunk, (i + len(chunk)) * SAMPLES, seq, 0x04 if last else 0))
        seq += 1
    return b"".join(out)

if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.write_bytes(wrap(src.read_bytes()))
    print(f"{src.name} -> {dst.name}: {len(src.read_bytes()) // PACKET} packets")
