"""Read-only source evidence for .osu, .osz and folders. Does not run a classifier."""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

from .provenance import (MANIFEST_NAME, MAX_MANIFEST_BYTES, MAX_MAP_BYTES, fingerprint,
                         local_matches, parse_sections, record_dir, valid_manifest)

MAX_ARCHIVE_ENTRIES = 4096
MAX_TOTAL_MAP_BYTES = 128 * 1024 * 1024


def _check_map(raw: bytes, name: str, records: Path, manifest=None) -> dict:
    result = dict(file=name, status="inconclusive", tags=[], manifest="absent", local_records=[], warnings=[])
    try:
        _, sections = parse_sections(raw)
        tags = []
        for line in sections.get("Metadata", []):
            key, _, value = line.partition(":")
            if key.strip() == "Tags":
                tags.extend(value.lower().split())
        result["tags"] = [t for t in tags if t in ("autoosu", "auto-osu", "ai-generated", "rule-generated",
                                                   "autoosu-model", "autoosu-mixed", "autoosu-rules")]
        if {"autoosu", "auto-osu"}.intersection(tags):
            result["status"] = "declaration"
        digests = fingerprint(raw)
        result.update(digests)
        if manifest is not None:
            matches = [m for m in manifest["maps"] if m["content_sha256"] == digests["content_sha256"]]
            if matches:
                result["manifest"] = "raw-match" if any(m["raw_sha256"] == digests["raw_sha256"] for m in matches) else "content-match"
                result["status"] = "declaration"
                result["declared_engine"] = manifest.get("engine")
                result["declared_generation_id"] = manifest["generation_id"]
            else:
                result["manifest"] = "no-matching-content"
                result["warnings"].append("Archive manifest does not describe this map's checked content")
        result["local_records"], warnings = local_matches(digests, records)
        result["warnings"].extend(warnings)
        if result["local_records"]:
            result["status"] = "local_record_match"
    except (OSError, ValueError, UnicodeError, ArithmeticError) as exc:
        result["warnings"].append(f"Content comparison unavailable: {exc}")
    return result


def _read_file(path: Path, records: Path) -> list[dict]:
    try:
        if path.suffix.lower() == ".osu":
            with path.open("rb") as stream:
                return [_check_map(stream.read(MAX_MAP_BYTES + 1), str(path), records)]
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_ENTRIES:
                raise ValueError("Archive exceeds the 4096-entry check limit")
            names = [e.filename.replace("\\", "/").casefold() for e in entries]
            if len(names) != len(set(names)):
                raise ValueError("Archive contains ambiguous duplicate entry names")
            maps = [e for e in entries if e.filename.lower().endswith(".osu") and not e.is_dir()]
            if not maps:
                raise ValueError("No .osu beatmaps found in archive")
            if sum(e.file_size for e in maps) > MAX_TOTAL_MAP_BYTES:
                raise ValueError("Archive exceeds the 128 MiB beatmap check limit")
            manifest, warnings = None, []
            candidates = [e for e in entries if e.filename.casefold() == MANIFEST_NAME.casefold()]
            if candidates:
                try:
                    entry = candidates[0]
                    if entry.file_size > MAX_MANIFEST_BYTES:
                        raise ValueError("Manifest exceeds the 2 MiB check limit")
                    with archive.open(entry) as f:
                        candidate = json.loads(f.read(MAX_MANIFEST_BYTES + 1).decode("utf-8"))
                    if not valid_manifest(candidate):
                        raise ValueError("Invalid or unsupported manifest")
                    manifest = candidate
                except (ValueError, OSError, UnicodeError, RuntimeError, zipfile.BadZipFile) as exc:
                    warnings.append(f"Manifest ignored: {exc}")
            results = []
            for entry in maps:
                name = f"{path} :: {entry.filename}"
                try:
                    if entry.file_size > MAX_MAP_BYTES:
                        raise ValueError("Beatmap exceeds the 16 MiB check limit")
                    with archive.open(entry) as stream:
                        result = _check_map(stream.read(MAX_MAP_BYTES + 1), name, records, manifest)
                except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
                    result = dict(file=name, status="error", warnings=[str(exc)])
                result["warnings"].extend(warnings)
                results.append(result)
            return results
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        return [dict(file=str(path), status="error", warnings=[str(exc)])]


def check_source(path: str | Path, *, recursive: bool = False, records_dir: Path | None = None) -> dict:
    source = Path(path).expanduser().resolve()
    records = Path(records_dir).expanduser().resolve() if records_dir is not None else record_dir().expanduser().resolve()
    if source.is_dir():
        files = []
        def raise_error(error):
            raise error
        for root, dirs, names in os.walk(source, followlinks=False, onerror=raise_error):
            base = Path(root)
            dirs[:] = sorted(d for d in dirs if not (base / d).is_symlink() and not d.startswith(".")) if recursive else []
            files.extend(base / n for n in sorted(names) if Path(n).suffix.lower() in (".osu", ".osz")
                         and not (base / n).is_symlink())
        if not files:
            raise ValueError("No .osu or .osz files found")
    elif source.suffix.lower() in (".osu", ".osz"):
        files = [source]
    else:
        raise ValueError("Choose an .osu file, an .osz archive, or a folder")
    results = [item for file in files for item in _read_file(file, records)]
    return dict(schema="autoosu.source-report/1", records_scope=str(records),
                statistical_detection="unavailable: independent evaluation gate not met",
                interpretation="Declarations can be copied. A local match is limited to this store and the checked fields. No match does not imply human authorship.",
                results=results)


def format_report(report: dict, lang: str = "en") -> str:
    from .i18n import STRINGS

    def message(key, **values):
        return STRINGS[key].get(lang, STRINGS[key]["en"]).format(**values)
    lines = [message("source.explanation"), message("source.scope", path=report["records_scope"]), ""]
    for item in report["results"]:
        lines.append(item["file"])
        lines.append("  " + message("source.status." + item["status"]))
        if item.get("tags"):
            lines.append("  Tags: " + " ".join(item["tags"]))
        if item.get("declared_engine"):
            engine = item["declared_engine"]
            if isinstance(engine, dict) and engine.get("kind") in ("model", "mixed", "rules"):
                lines.append("  " + message("source.declared_engine") + ": " + message("source.engine." + engine["kind"]))
                for part in ("rhythm", "coord"):
                    model = engine.get(part)
                    if isinstance(model, dict) and model.get("identity") in ("v0", "custom-model"):
                        lines.append("  " + message("source.layer." + part) + ": " + model["identity"])
        for record in item.get("local_records", []):
            lines.append("  " + message("source.match." + record["match"]) + " · " + record["generation_id"])
        for warning in item.get("warnings", []):
            lines.append("  " + warning)
        lines.append("")
    return "\n".join(lines)
