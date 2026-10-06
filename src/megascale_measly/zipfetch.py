"""Pull a single member out of a remote zip over HTTP range requests.

The mega-scale archive on Zenodo is 1 GB but the table we need is one 210 MB
member inside it, so we read the zip's central directory, locate that member,
and inflate only its bytes.
"""

from __future__ import annotations

import struct
import subprocess
import zlib
from pathlib import Path


def _curl(url: str, byte_range: str | None = None, stream: bool = False):
    cmd = ["curl", "-sSL", "--fail"]
    if byte_range:
        cmd += ["-r", byte_range]
    cmd.append(url)
    if stream:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE)
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def _content_length(url: str) -> int:
    head = subprocess.run(["curl", "-sSLI", "--fail", url],
                          capture_output=True, check=True).stdout.decode()
    lengths = [line.split(":", 1)[1].strip() for line in head.splitlines()
               if line.lower().startswith("content-length")]
    return int(lengths[-1])


def _find_member(url: str, member: str) -> tuple[int, int, int, int]:
    """Return (local_header_offset, compressed_size, uncompressed_size, method)."""
    total = _content_length(url)
    tail = _curl(url, f"{max(0, total - 1_000_000)}-{total - 1}")
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("no end-of-central-directory record found")
    cd_size, cd_offset = struct.unpack("<II", tail[eocd + 12:eocd + 20])
    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
        eocd64 = tail.rfind(b"PK\x06\x06")
        cd_size, cd_offset = struct.unpack("<QQ", tail[eocd64 + 40:eocd64 + 56])

    cd = _curl(url, f"{cd_offset}-{cd_offset + cd_size - 1}")
    pos = 0
    while pos + 46 <= len(cd) and cd[pos:pos + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[pos + 10:pos + 12])
        csize, usize = struct.unpack("<II", cd[pos + 20:pos + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[pos + 28:pos + 34])
        offset, = struct.unpack("<I", cd[pos + 42:pos + 46])
        name = cd[pos + 46:pos + 46 + nlen].decode("utf-8", "replace")
        if name == member:
            return offset, csize, usize, method
        pos += 46 + nlen + elen + clen
    raise KeyError(f"{member!r} not found in archive")


def extract_member(url: str, member: str, dest: Path) -> Path:
    """Download and inflate one zip member to `dest`. No-op if it already exists."""
    dest = Path(dest)
    if dest.exists():
        return dest

    offset, csize, _, method = _find_member(url, member)
    header = _curl(url, f"{offset}-{offset + 29}")
    nlen, elen = struct.unpack("<HH", header[26:30])
    data_start = offset + 30 + nlen + elen

    decompressor = zlib.decompressobj(-15) if method == 8 else None
    partial = dest.with_suffix(dest.suffix + ".partial")
    proc = _curl(url, f"{data_start}-{data_start + csize - 1}", stream=True)
    with open(partial, "wb") as out:
        while chunk := proc.stdout.read(1 << 20):
            out.write(decompressor.decompress(chunk) if decompressor else chunk)
        if decompressor:
            out.write(decompressor.flush())
    if proc.wait() != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError("download failed")
    partial.rename(dest)
    return dest
