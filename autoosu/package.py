"""Metadata reading, audio conversion and .osz packaging."""
from __future__ import annotations

import re
import json
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .audio_io import find_ffmpeg, prepare_for_osu, transcode_mp3  # noqa: F401  (re-exported)
from .beatmap import Beatmap

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize(name: str) -> str:
    name = _ILLEGAL.sub("", name).strip().rstrip(".")
    return name or "untitled"


COVER_MIME = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def extract_cover(path: Path) -> Optional[Tuple[bytes, str]]:
    """Embedded cover art of an audio file as (bytes, extension), or None."""
    try:
        import mutagen
        from mutagen.flac import Picture

        f = mutagen.File(str(path))
        if f is None:
            return None
        pics = []
        if hasattr(f, "pictures") and f.pictures:                       # FLAC
            pics = [(p.data, p.mime) for p in f.pictures]
        elif f.tags is not None:
            tags = f.tags
            if hasattr(tags, "getall"):                                   # ID3 (mp3, aiff, wav)
                pics = [(a.data, a.mime) for a in tags.getall("APIC")]
            elif "covr" in tags:                                          # MP4 / m4a
                from mutagen.mp4 import MP4Cover

                for c in tags["covr"]:
                    pics.append((bytes(c), "image/png" if getattr(c, "imageformat", None) == MP4Cover.FORMAT_PNG else "image/jpeg"))
            elif "metadata_block_picture" in tags:                        # ogg vorbis / opus
                import base64

                for b64 in tags["metadata_block_picture"]:
                    p = Picture(base64.b64decode(b64))
                    pics.append((p.data, p.mime))
            elif "WM/Picture" in tags:                                    # wma
                for v in tags["WM/Picture"]:
                    raw = bytes(v.value) if hasattr(v, "value") else bytes(v)
                    pics.append((raw, "image/jpeg"))
        for data, mime in pics:
            if data and len(data) > 1000:
                ext = COVER_MIME.get((mime or "").lower())
                if ext is None:
                    ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
                return data, ext
    except Exception:
        return None
    return None


def prepare_background(data: bytes, ext: str, workdir: Path, max_side: int = 1920) -> Optional[Path]:
    """Write the cover as bg.jpg / bg.png for osu!, downscaling very large images."""
    try:
        from io import BytesIO

        from PIL import Image

        img = Image.open(BytesIO(data))
        img.load()
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
            ext = ".jpg" if img.mode in ("RGB", "L") else ".png"
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / f"bg{ext}"
        if ext == ".jpg":
            img.convert("RGB").save(out, quality=90)
        else:
            img.save(out)
        return out
    except Exception:
        return None


def read_metadata(path: Path) -> Tuple[str, str]:
    """Return (title, artist) from tags, falling back to an 'Artist - Title' file name."""
    title = artist = ""
    try:
        import mutagen

        tags = mutagen.File(str(path), easy=True)
        if tags:
            title = (tags.get("title") or [""])[0]
            artist = (tags.get("artist") or [""])[0]
    except Exception:
        pass
    stem = path.stem
    if not title:
        if " - " in stem:
            a, t = stem.split(" - ", 1)
            artist = artist or a.strip()
            title = t.strip()
        else:
            title = stem
    return sanitize(title), sanitize(artist or "Unknown Artist")


encode_mp3 = transcode_mp3


def prepare_audio(src: Path, workdir: Path) -> Path:
    """Copy (mp3 / ogg-vorbis) or transcode (anything else) the song so osu! can play it."""
    return prepare_for_osu(src, workdir)


def write_osz(beatmaps: List[Beatmap], audio: Path, out_dir: Path, extra_files: Sequence[Path] = (),
              manifest: Optional[dict] = None) -> Path:
    from .provenance import MANIFEST_NAME

    out_dir.mkdir(parents=True, exist_ok=True)
    first = beatmaps[0]
    osz = out_dir / sanitize(f"{first.artist} - {first.title} ({first.creator}).osz")
    names = [audio.name, *(Path(p).name for p in extra_files), *(sanitize(b.osu_filename()) for b in beatmaps)]
    if manifest is not None:
        names.append(MANIFEST_NAME)
    if len({n.casefold() for n in names}) != len(names):
        raise ValueError("Duplicate filenames in the output archive; choose distinct difficulties and assets")
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".osz.tmp", delete=False) as f:
            temp = Path(f.name)
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(audio, audio.name)
            for extra in extra_files:
                zf.write(extra, Path(extra).name)
            for bm in beatmaps:
                zf.writestr(sanitize(bm.osu_filename()), bm.to_osu().encode("utf-8"))
            if manifest is not None:
                zf.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
        temp.replace(osz)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return osz
