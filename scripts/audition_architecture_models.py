"""Frozen full-song A/B/C audition of v0 and two attribute-trained checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from slider import Beatmap as IndependentBeatmap
from slider.beatmap import Slider as IndependentSlider

from autoosu.beatmap import Slider
from autoosu.generate import generate, OSU_TIMING_SHIFT_MS
from autoosu.preview import render_preview
from autoosu.provenance import atomic_json
from autoosu.source_check import check_source
from scripts.audition_skill_models import GROUPS, digest, run_lengths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--v0", type=Path, required=True)
    ap.add_argument("--best", type=Path, required=True)
    ap.add_argument("--continued", type=Path, required=True)
    ap.add_argument("--coord", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", type=Path, help="JSON list of named song groups, presets, star conditions and controls")
    args = ap.parse_args()
    cases = (json.loads(args.cases.read_text(encoding="utf-8")) if args.cases else
             [dict(name=f"song-{i}", group=group, preset="Insane", star_rating=6.5,
                   controls=dict(highlight_mode="off")) for i, group in enumerate(GROUPS, 1)])
    args.out.mkdir(parents=True, exist_ok=False)
    modes = {"A-v0": args.v0, "B-v1-best": args.best, "C-v1-continued": args.continued}
    model_records = {}
    for mode, path in modes.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model_records[mode] = dict(path=str(path), sha256=digest(path), step=checkpoint["step"])
        del checkpoint
    settings = dict(seed=240924, decode_steps=12, coord_steps=100,
                    temperature=.9, cfg_scale=1., device="cuda")
    recipe = dict(models=model_records, coord_sha256=digest(args.coord), settings=settings,
                  cases=cases, timing="reference; no reference rhythm supplied",
                  groups=[case["group"] for case in cases], corpus_sha256=digest(args.corpus / "corpus.json"),
                  source_sha256={str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "autoosu").rglob("*.py")},
                  script_sha256=digest(__file__), packages={n:version(n) for n in ("torch", "slider", "rosu-pp-py")},
                  scope="Development full-song audition, not independent test acceptance or osu! client playtesting")
    atomic_json(args.out / "recipe.json", recipe)
    songs = {s["group"]:s for s in json.loads((args.corpus / "corpus.json").read_text(encoding="utf-8"))["songs"]}
    rows, cache = [], {}
    packs = args.out / "maps"
    packs.mkdir()
    for number, case in enumerate(cases, 1):
        group = case["group"]
        song = songs[group]
        audio, reference = args.corpus / song["audio"], args.corpus / song["reference"]
        if digest(audio) != song["audio_sha256"] or digest(reference) != song["reference_sha256"]:
            raise ValueError("Frozen audition source changed")
        original = IndependentBeatmap.parse(reference.read_text(encoding="utf-8-sig"))
        for mode, checkpoint in modes.items():
            print(f"START {number} {mode} {original.title}", flush=True)
            result = generate(audio, [case["preset"]], args.out / "generated" / f"{number:02d}-{mode}",
                rhythm_model=str(checkpoint), coord_model=str(args.coord),
                bpm=song["bpm"], offset_ms=song["offset_ms"] + OSU_TIMING_SHIFT_MS,
                title=f"{original.title} [{case['name']} {mode}]", artist=original.artist, creator=f"AUTO-OSU {mode}",
                star_rating=case["star_rating"], controls=case["controls"], records_dir=args.out / "records", model_cache=cache,
                log=lambda *_: None, **settings)
            diff = result.diffs[0]
            with zipfile.ZipFile(result.osz) as archive:
                text = archive.read(next(n for n in archive.namelist() if n.endswith(".osu"))).decode("utf-8-sig")
            parsed = IndependentBeatmap.parse(text)
            imported = parsed.hit_objects(stacking=False)
            if len(imported) != len(diff.beatmap.hit_objects):
                raise ValueError("Independent parser object count mismatch")
            curve_failures = []
            for own, other in zip(diff.beatmap.hit_objects, imported):
                if isinstance(own, Slider):
                    expected_sounds = own.edge_sounds or [own.hitsound] + [0] * own.repeats
                    if not isinstance(other, IndependentSlider) or other.edge_sounds != expected_sounds or other.repeat != own.repeats or other.hitsound != own.body_hitsound:
                        raise ValueError("Independent parser slider attributes mismatch")
                    points = np.array([tuple(other.curve(float(t))) for t in np.linspace(0, 1, 1001)])
                    if not np.isfinite(points).all() or np.any(points < [-.01, -.01]) or np.any(points > [512.01, 384.01]):
                        curve_failures.append(own.time)
            source = check_source(result.osz, records_dir=args.out / "records")
            if any(r["manifest"] != "raw-match" or not any(m["match"] == "raw" for m in r["local_records"])
                   for r in source["results"]):
                raise ValueError("Provenance record mismatch")
            if diff.measurement["status"] != "ok":
                raise ValueError("Star measurement failed")
            target = packs / f"{number:02d}-{mode}-{diff.measurement['stars']:.2f}star.osz"
            shutil.copy2(result.osz, target)
            preview = render_preview(result.audio_file, diff.beatmap,
                args.out / "rhythm-previews" / f"{number:02d}-{mode}.mp3", click_shift_ms=result.osu_shift_ms)
            sliders = [o for o in diff.beatmap.hit_objects if isinstance(o, Slider)]
            row = dict(song=number, case=case, group=group, title=original.title, artist=original.artist, bpm=song["bpm"],
                duration_s=song["duration_s"], mode=mode, stars=diff.measurement["stars"], summary=diff.summary(),
                runs=run_lengths(diff.beatmap, result.timing.beat_length),
                attributes=dict(new_combos=sum(o.new_combo for o in diff.beatmap.hit_objects),
                    hitsounds=dict(Counter(o.hitsound for o in diff.beatmap.hit_objects)),
                    repeat_sliders=sum(o.repeats > 1 for o in sliders),
                    nonzero_tail_sounds=sum(o.edge_sounds[-1] != 0 for o in imported if isinstance(o, IndependentSlider)),
                    curve_types=dict(Counter(o.curve_type for o in sliders))),
                diagnostics=diff.diagnostics, curves_outside_or_nonfinite_ms=curve_failures,
                independent_parser_objects=len(imported), elapsed_s=result.elapsed_s, warnings=result.warnings,
                file=target.relative_to(args.out).as_posix(), sha256=digest(target),
                preview=preview.relative_to(args.out).as_posix())
            rows.append(row)
            atomic_json(args.out / "results.json", rows)
            print(json.dumps(row, ensure_ascii=True), flush=True)
    lines = ["# AUTO-OSU 新旧模型对照", "", f"{len(cases)} 组完整开发歌曲，每组三版，共 {len(rows)} 张谱面。",
        "A：已发布 v0（40,000 步）；B：新版节奏 F1 最佳（10,000 步）；C：续训最佳（21,000 步）。",
        "三版均用同一 coord v0、相同种子和参考 timing。每组共享输入条件；目标星级控制器可按实测结果调整条件和间距。实际星级并不保证相同。",
        "将 maps 中的 .osz 拖入 osu!，按歌名后的 case 名称及 A/B/C 对比。",
        "rhythm-previews 只用合成点击声辅助比较节奏与新连击，不播放真实 osu! 音效样本；音效映射请在 osu! 中听。",
        "没有使用技能标签控制；歌曲覆盖不同速度和节奏结构，不代表模型已学会按标签生成特定谱型。",
        "已做独立解析和曲线检查；尚未进行 osu! 客户端试玩。生成警告和逐张数据见 results.json。", "",
        *[f"- {case['name']}：{case['preset']}，初始星级条件 {case['star_rating']}，控制参数 `{json.dumps(case['controls'], ensure_ascii=False)}`。" for case in cases], "",
        "| 歌曲 | BPM | 版本 | 实测星级 | 物件 | 滑条 | 最长连续圆圈 | 新连击 | 折返滑条 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['song']:02d} {r['title']} | {r['bpm']:.0f} | {r['mode']} | {r['stars']:.2f} | "
            f"{r['summary']['objects']} | {r['summary']['sliders']} | {r['runs']['longest_run']} | "
            f"{r['attributes']['new_combos']} | {r['attributes']['repeat_sliders']} |")
    (args.out / "试听说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with zipfile.ZipFile(args.out / "AUTO-OSU-v1-ABC-comparison.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("试听说明.md", "results.json", "recipe.json"):
            archive.write(args.out / name, name)
        for folder in (packs, args.out / "rhythm-previews"):
            for path in sorted(folder.iterdir()):
                archive.write(path, path.relative_to(args.out).as_posix())
    print(f"COMPLETE {len(rows)} maps", flush=True)


if __name__ == "__main__":
    main()
