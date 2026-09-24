"""JSON-lines inference process for the app-managed GPU environment."""
from __future__ import annotations

import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path


def emit(event, **data):
    print(json.dumps(dict(event=event, **data), ensure_ascii=False), flush=True)


def summary(result):
    return dict(osz=str(result.osz), elapsed_s=result.elapsed_s, device=result.device, bpm=result.timing.bpm,
                generation_id=(result.provenance or {}).get("generation_id"),
                provenance_recorded=result.provenance_recorded, warnings=result.warnings,
                evaluation_path=str(result.evaluation_path) if result.evaluation_path else None,
                diffs=[dict(name=d.preset.name, **d.summary()) for d in result.diffs])


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    request = json.loads(Path(args[0]).read_text(encoding='utf-8'))
    song, out = Path(request['source']), Path(request['out_dir'])
    log = lambda text: emit('log', text=text)
    progress = lambda fraction, message: emit('progress', fraction=fraction, message=message)
    try:
        if song.is_dir():
            from .batch import generate_batch
            class Cancellation:
                def is_set(self):
                    return Path(request['cancel_file']).exists()
            result = generate_batch(
                song, request['difficulties'], out, recursive=request.get('recursive', False),
                preview=request.get('preview', False), cancel=Cancellation(), log=log, progress=progress,
                on_item=lambda i, n, item: emit('batch_item', index=i, total=n, item=asdict(item)), **request['kwargs'])
            emit('batch_done', report=str(result.report), items=[asdict(item) for item in result.items],
                 elapsed_s=result.elapsed_s, cancelled=result.cancelled)
        else:
            from .generate import generate
            result = generate(song, request['difficulties'], out, log=log, progress=progress, **request['kwargs'])
            if request.get('preview'):
                from .preview import render_preview
                for diff in result.diffs:
                    render_preview(result.audio_file, diff.beatmap, out/f'{song.stem} [{diff.preset.name}]_preview.mp3',
                                   click_shift_ms=result.osu_shift_ms)
            emit('single_done', result=summary(result))
        return 0
    except Exception as exc:
        emit('log', text=traceback.format_exc())
        emit('error', text=f'{type(exc).__name__}: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
