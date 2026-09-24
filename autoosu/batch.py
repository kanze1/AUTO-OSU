"""Sequential folder inference with isolated outputs, model reuse and a durable report."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

from .audio_io import SUPPORTED_EXTS
from .generate import GenerateResult, generate


def scan_inputs(source: Path, recursive: bool = False, exclude_dirs: Sequence[Path] = ()) -> list[Path]:
    source = Path(source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input does not exist: {source}")
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_EXTS:
            raise ValueError(f"Unsupported audio/video format: {source.suffix or source.name}")
        return [source]
    excluded = [Path(p).resolve() for p in exclude_dirs]

    def ignored(path: Path) -> bool:
        return path.is_symlink() or any(path.resolve().is_relative_to(p) for p in excluded)

    files = []
    def scan_error(error):
        raise error
    for root, dirs, names in os.walk(source, onerror=scan_error, followlinks=False):
        base = Path(root)
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and not ignored(base / d)) if recursive else []
        for name in names:
            path = base / name
            if not name.startswith('.') and path.suffix.lower() in SUPPORTED_EXTS and not ignored(path):
                files.append(path)
    return sorted(files, key=lambda p: str(p.relative_to(source)).casefold())


def output_for(source: Path, root: Path, destination: Path) -> Path:
    """Different input paths/extensions remain distinct, even if their song tags match."""
    identity = source.relative_to(root).as_posix()
    suffix = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:12]
    stem = re.sub(r'[^\w .-]', '_', source.stem).strip(' .')[:72] or 'song'
    return destination / f"{stem}-{suffix}"


@dataclass
class BatchItem:
    source: str
    output_dir: str
    status: str = "pending"
    osz: Optional[str] = None
    error: Optional[str] = None
    elapsed_s: float = 0.0
    device: str = ""
    generation_id: Optional[str] = None
    provenance_recorded: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class BatchResult:
    report: Path
    items: list[BatchItem]
    elapsed_s: float = 0.0
    cancelled: bool = False

    @property
    def succeeded(self) -> int:
        return sum(item.status == "done" for item in self.items)

    @property
    def failed(self) -> int:
        return sum(item.status == "error" for item in self.items)

    def save(self) -> None:
        data = {"elapsed_s": self.elapsed_s, "cancelled": self.cancelled,
                "succeeded": self.succeeded, "failed": self.failed,
                "items": [asdict(item) for item in self.items]}
        temporary = self.report.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.report)


def generate_batch(source: Path, difficulties: list[str], out_dir: Path, *, recursive: bool = False,
                   preview: bool = False, debug_plot: bool = False,
                   cancel: Optional[threading.Event] = None,
                   progress: Optional[Callable[[float, str], None]] = None,
                   on_item: Optional[Callable[[int, int, BatchItem], None]] = None,
                   on_result: Optional[Callable[[GenerateResult], None]] = None,
                   log=print, **kwargs) -> BatchResult:
    source, out_dir = Path(source).resolve(), Path(out_dir).resolve()
    if not source.is_dir():
        raise ValueError(f"Batch input must be a folder: {source}")
    if source == out_dir:
        raise ValueError("Choose an output folder different from the input folder")
    files = scan_inputs(source, recursive, exclude_dirs=[out_dir] if out_dir.is_relative_to(source) else [])
    if not files:
        raise ValueError("No supported audio or video files found in this folder")
    if not difficulties:
        raise ValueError("Choose at least one difficulty")
    if kwargs.get("rhythm_model") or kwargs.get("coord_model"):
        from .devices import resolve_device
        kwargs["device"] = resolve_device(kwargs.get("device"))
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"batch-report-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    result = BatchResult(report, [BatchItem(str(p), str(output_for(p, source, out_dir))) for p in files])
    result.save()
    started = time.perf_counter()
    models = {}
    notify = on_item or (lambda i, n, item: None)
    update = progress or (lambda f, message: None)
    try:
        for i, item in enumerate(result.items):
            if cancel is not None and cancel.is_set():
                result.cancelled = True
                for pending in result.items[i:]:
                    pending.status = "cancelled"
                break
            item.status = "running"
            result.save()
            notify(i, len(files), item)
            song, destination = Path(item.source), Path(item.output_dir)
            log(f"[{i + 1}/{len(files)}] {song.name}")
            t0 = time.perf_counter()
            try:
                res = generate(song, difficulties, destination, model_cache=models, log=log,
                               progress=lambda f, m, idx=i: update((idx + f) / len(files), m), **kwargs)
                item.osz, item.device, item.status = str(res.osz), res.device, "done"
                item.generation_id = (getattr(res, "provenance", None) or {}).get("generation_id")
                item.provenance_recorded = getattr(res, "provenance_recorded", False)
                item.warnings.extend(getattr(res, "warnings", []))
                # An optional preview failure must not hide a successfully written beatmap.
                try:
                    if preview:
                        from .preview import render_preview
                        for diff in res.diffs:
                            path = destination / f"{song.stem} [{diff.preset.name}]_preview.mp3"
                            render_preview(res.audio_file, diff.beatmap, path, click_shift_ms=res.osu_shift_ms)
                    if debug_plot:
                        from .debug import plot_debug
                        plot_debug(res, destination / f"{song.stem}_debug.png")
                except Exception as exc:
                    item.warnings.append(f"Optional preview/debug output: {exc}")
                    log(item.warnings[-1])
                if on_result:
                    on_result(res)
                del res
            except Exception as exc:
                item.status, item.error = "error", f"{type(exc).__name__}: {exc}"
                log(f"Failed: {song.name}: {item.error}")
            item.elapsed_s = time.perf_counter() - t0
            result.elapsed_s = time.perf_counter() - started
            result.save()
            notify(i, len(files), item)
            update((i + 1) / len(files), "batch item done")
    finally:
        models.clear()
        result.elapsed_s = time.perf_counter() - started
        result.save()
    return result
