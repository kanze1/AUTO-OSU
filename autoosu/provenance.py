"""Generation declarations and local content records; neither is an authenticity certificate."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import __version__

SCHEMA = "autoosu.provenance/1"
CONTENT_SCHEMA = "autoosu-map/1"
MANIFEST_NAME = "autoosu-provenance.json"
MAX_MAP_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
HEX = re.compile(r"^[0-9a-f]{64}$")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def record_dir() -> Path:
    return Path(os.environ.get("AUTOOSU_RECORDS") or Path.home() / ".autoosu" / "provenance")


def _number(value: str) -> str:
    try:
        if len(value) > 64:
            raise ValueError("Numeric field is too long")
        n = Decimal(value.strip())
        if not n.is_finite() or abs(n) > Decimal("1e15") or n.as_tuple().exponent < -20:
            raise ValueError("Numeric field is outside the supported range")
        if n == 0:
            return "0"
        rendered = format(n, "f")
        return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered
    except InvalidOperation as exc:
        raise ValueError("Invalid numeric field") from exc


def _fields(lines: list[str]) -> dict[str, str]:
    fields = {}
    for line in lines:
        if ":" not in line:
            raise ValueError("Invalid section field")
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if key in fields:
            raise ValueError(f"Duplicate field: {key}")
        fields[key] = value
    return fields


def parse_sections(raw: bytes) -> tuple[int, dict[str, list[str]]]:
    if len(raw) > MAX_MAP_BYTES:
        raise ValueError("Beatmap exceeds the 16 MiB check limit")
    text = raw.decode("utf-8-sig")
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("//")]
    if not lines or not re.fullmatch(r"osu file format v\d+", lines[0]):
        raise ValueError("Missing osu! file format header")
    version = int(lines[0].rsplit("v", 1)[1])
    sections, current = {}, None
    for line in lines[1:]:
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            if current in sections:
                raise ValueError(f"Duplicate section: {current}")
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return version, sections


def _sample(value: str) -> list[str]:
    parts = value.split(":", 4)
    parts += ["0"] * max(0, 4 - len(parts))
    if len(parts) == 4:
        parts.append("")
    return [_number(p or "0") for p in parts[:4]] + [parts[4].strip()]


def fingerprint(raw: bytes) -> dict:
    """Hash an explicit standard-map field scope, not all media or all editor state.

    Includes format, mode, stacking, sample set, six difficulty fields, complete timing
    and hit-object fields, and breaks. Ignores metadata, colours, other events, audio
    filename, editor settings and formatting. Numeric spelling is normalized; ordering
    is retained. Version/field changes can conservatively prevent a record match.
    """
    version, sections = parse_sections(raw)
    if not 3 <= version <= 14:
        raise ValueError("Unsupported osu! file format version")
    general = _fields(sections.get("General", []))
    if _number(general.get("Mode", "0")) != "0":
        raise ValueError("Content matching currently supports osu!standard only")
    diff = _fields(sections.get("Difficulty", []))
    defaults = dict(HPDrainRate="5", CircleSize="5", OverallDifficulty="5",
                    ApproachRate=diff.get("OverallDifficulty", "5"), SliderMultiplier="1.4", SliderTickRate="1")
    if not sections.get("TimingPoints") or not sections.get("HitObjects"):
        raise ValueError("Beatmap needs timing points and hit objects")
    timing = []
    for line in sections["TimingPoints"]:
        parts = line.split(",")
        if not 2 <= len(parts) <= 8:
            raise ValueError("Invalid timing point")
        defaults_tp = ["0", "500", "4", "0", "0", "100", "1" if Decimal(_number(parts[1])) > 0 else "0", "0"]
        parts += defaults_tp[len(parts):]
        timing.append([_number(p.strip()) for p in parts])
    objects = []
    for line in sections["HitObjects"]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            raise ValueError("Invalid hit object")
        head = [_number(p) for p in parts[:5]]
        kind = int(head[3]) & (1 | 2 | 8 | 128)
        if kind == 1 and len(parts) in (5, 6):
            obj = head + [_sample(parts[5] if len(parts) > 5 else "")]
        elif kind == 8 and len(parts) in (6, 7):
            obj = head + [_number(parts[5]), _sample(parts[6] if len(parts) > 6 else "")]
        elif kind == 2 and 8 <= len(parts) <= 11:
            curve = parts[5].split("|")
            if curve[0] not in ("B", "C", "L", "P"):
                raise ValueError("Unsupported slider curve")
            anchors = [[_number(v) for v in p.split(":")] for p in curve[1:]]
            if not anchors or any(len(p) != 2 for p in anchors):
                raise ValueError("Invalid slider anchors")
            repeats = int(_number(parts[6]))
            if not 1 <= repeats <= 10000:
                raise ValueError("Invalid slider repeats")
            sounds = parts[8].split("|") if len(parts) > 8 and parts[8] else ["0"] * (repeats + 1)
            sets = parts[9].split("|") if len(parts) > 9 and parts[9] else ["0:0"] * (repeats + 1)
            obj = head + [[curve[0], anchors], str(repeats), _number(parts[7]),
                          [_number(p) for p in sounds], [[_number(v) for v in p.split(":")] for p in sets],
                          _sample(parts[10] if len(parts) > 10 else "")]
        else:
            raise ValueError("Unsupported or malformed standard hit object")
        objects.append(obj)
    breaks = []
    for line in sections.get("Events", []):
        parts = [p.strip() for p in line.split(",")]
        if parts[0] in ("2", "Break"):
            if len(parts) != 3:
                raise ValueError("Invalid break event")
            breaks.append([_number(p) for p in parts[1:]])
    content = dict(schema=CONTENT_SCHEMA, format=version, mode=0,
                   stack_leniency=_number(general.get("StackLeniency", "0.7")),
                   sample_set=general.get("SampleSet", "Normal"),
                   difficulty={k: _number(diff.get(k, v)) for k, v in defaults.items()},
                   timing=timing, hit_objects=objects, breaks=breaks)
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return dict(raw_sha256=sha256(raw), content_sha256=sha256(encoded), content_schema=CONTENT_SCHEMA)


def model_identity(kind: str, path: str | Path | None, cache: dict) -> dict:
    if path is None:
        return dict(engine="rules")
    from .models import MODELS

    source = Path(path).resolve()
    before = source.stat()
    key = ("provenance", kind, str(source), before.st_size, before.st_mtime_ns)
    if key not in cache:
        digest = file_hash(source)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Model changed while recording its identity; retry generation")
        official = digest == MODELS[kind].sha256
        cache[key] = dict(engine="model", identity="v0" if official else "custom-model", sha256=digest,
                          bytes=before.st_size)
    return dict(cache[key])


def engine_info(rhythm: dict, coord: dict) -> dict:
    count = sum(m["engine"] == "model" for m in (rhythm, coord))
    return dict(kind=("rules", "mixed", "model")[count], rhythm=rhythm, coord=coord)


def source_tags(engine: dict) -> str:
    return "autoosu kanzei " + ("rule-generated" if engine["kind"] == "rules" else "ai-generated") + " autoosu-" + engine["kind"]


def build_manifest(maps: list[dict], engine: dict, settings: dict, input_audio: Path, packaged_audio: Path) -> dict:
    return dict(schema=SCHEMA, generator="AUTO-OSU", app_version=__version__,
                generation_id=str(uuid.uuid4()), created_utc=datetime.now(timezone.utc).isoformat(),
                evidence="self-declaration; not a signature or proof of model execution",
                engine=engine, settings=settings, input_audio_sha256=file_hash(input_audio),
                packaged_audio_sha256=file_hash(packaged_audio), maps=maps)


def valid_manifest(value) -> bool:
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("generator") != "AUTO-OSU":
        return False
    if not isinstance(value.get("maps"), list) or not 1 <= len(value["maps"]) <= 1000:
        return False
    try:
        uuid.UUID(value["generation_id"])
    except (ValueError, KeyError, TypeError, AttributeError):
        return False
    for m in value["maps"]:
        if not isinstance(m, dict) or not isinstance(m.get("filename"), str) or m.get("content_schema") != CONTENT_SCHEMA:
            return False
        if any(not isinstance(m.get(k), str) or not HEX.fullmatch(m[k]) for k in ("raw_sha256", "content_sha256")):
            return False
    return True


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        tmp.replace(path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def store_record(manifest: dict, directory: Path | None = None) -> None:
    if not valid_manifest(manifest):
        raise ValueError("Invalid generated manifest")
    root = Path(directory) if directory is not None else record_dir()
    for entry in manifest["maps"]:
        digest = entry["content_sha256"]
        atomic_json(root / digest[:2] / digest / (manifest["generation_id"] + ".json"), manifest)


def local_matches(digests: dict, directory: Path) -> tuple[list[dict], list[str]]:
    digest = digests["content_sha256"]
    root = directory / digest[:2] / digest
    results, warnings = [], []
    if not root.exists():
        return results, warnings
    for i, path in enumerate(sorted(root.glob("*.json"))):
        if i >= 500:
            warnings.append("Local record lookup stopped after 500 records for this content")
            break
        try:
            if path.stat().st_size > MAX_MANIFEST_BYTES:
                raise ValueError("Local record exceeds the size limit")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not valid_manifest(value):
                raise ValueError("Invalid local record")
            matching = [m for m in value["maps"] if m["content_sha256"] == digest]
            if matching:
                results.append(dict(generation_id=value["generation_id"],
                                    match="raw" if any(m["raw_sha256"] == digests["raw_sha256"] for m in matching) else "content",
                                    engine=value.get("engine"), created_utc=value.get("created_utc")))
        except (OSError, ValueError, UnicodeError) as exc:
            warnings.append(f"Unreadable local record {path.name}: {exc}")
    return results, warnings
