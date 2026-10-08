"""Generate actual same-song skill auditions and component ablations from frozen weights."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autoosu.beatmap import Circle
from autoosu.generate import generate, OSU_TIMING_SHIFT_MS
from autoosu.preferences import pattern_metrics
from autoosu.provenance import atomic_json
from autoosu.preview import render_preview
from autoosu.source_check import check_source


GROUPS = ["1b03caaee2c6f160d693ff9a6727f49267e3f5777d51e428e36e7ddea29b2505",
          "193214fecc8770b97767a43f00a8e63e5d491e9a3208afb8466c507beda867d2",
          "10483e0c557b4fb9ddb63680a8dc09b7178afdae8c77fc4e6c482384fe821dd9",
          "02f892e29b0847cf4be34326e328a738c6716741e4c4f14a0d50e64bd6b28b45"]
MODES = {"v0": "旧版对照", "unknown": "新版无标签", "jumps": "跳跃", "bursts": "短串",
         "streams": "连打", "slider_tech": "滑条技巧"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_lengths(beatmap, beat_ms):
    runs, run, previous = [], 0, None
    for obj in beatmap.hit_objects:
        if not isinstance(obj, Circle):
            if run:
                runs.append(run)
            run, previous = 0, None
            continue
        if previous is not None and abs(obj.time - previous.time - beat_ms / 4) <= 3:
            run += 1
        else:
            if run:
                runs.append(run)
            run = 1
        previous = obj
    if run:
        runs.append(run)
    return dict(short_run_objects=sum(n for n in runs if 3 <= n <= 7),
                long_run_objects=sum(n for n in runs if n >= 8),
                sustained_run_objects=sum(n for n in runs if n >= 16), longest_run=max(runs, default=0))


def main():
    from slider import Beatmap as IndependentBeatmap
    from slider.beatmap import Slider as IndependentSlider
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rhythm", type=Path, required=True)
    ap.add_argument("--coord", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=ROOT / "out/difficulty-evaluation")
    args = ap.parse_args()
    corpus = json.loads((args.corpus / "corpus.json").read_text(encoding="utf-8"))
    songs = {song["group"]: song for song in corpus["songs"]}
    missing = set(GROUPS) - set(songs)
    if missing:
        raise ValueError(f"Missing audition songs: {missing}")
    args.out.mkdir(parents=True, exist_ok=False)
    settings = dict(seed=240924, decode_steps=12, coord_steps=100, temperature=.9,
                    cfg_scale=1., device="cuda", star_rating=6.5)
    recipe = dict(settings=settings, groups=GROUPS, modes=MODES, timing="reference",
                  controls=dict(highlight_mode="off"), models={
                      "rhythm": digest(args.rhythm), "coord": digest(args.coord),
                      "rhythm_v0": digest(ROOT / "models/rhythm_v0.pt"),
                      "coord_v0": digest(ROOT / "models/coord_v0.pt")},
                  packages={name: version(name) for name in ("torch", "slider", "rosu-pp-py")},
                  source_sha256={str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "autoosu").rglob("*.py")},
                  audition_script_sha256=digest(__file__),
                  corpus_sha256=digest(args.corpus / "corpus.json"),
                  evidence_scope="development audition; not independent of v0 pretraining")
    atomic_json(args.out / "recipe.json", recipe)
    rows, cache = [], {}
    packs = args.out / "试听谱面"
    packs.mkdir()
    for number, group in enumerate(GROUPS, 1):
        song = songs[group]
        audio = args.corpus / song["audio"]
        if digest(audio) != song["audio_sha256"]:
            raise ValueError("Frozen audio changed")
        reference = args.corpus / song["reference"]
        if digest(reference) != song["reference_sha256"]:
            raise ValueError("Frozen reference changed")
        original = IndependentBeatmap.parse(reference.read_text(encoding="utf-8-sig"))
        modes = list(MODES)
        if number == 1:
            modes += ["jumps_rhythm_only", "jumps_coord_only", "streams_rhythm_only", "streams_coord_only"]
        for mode in modes:
            rhythm = ROOT / "models/rhythm_v0.pt" if mode == "v0" else args.rhythm
            coord = ROOT / "models/coord_v0.pt" if mode == "v0" else args.coord
            skills = {}
            if mode not in ("v0", "unknown"):
                label = mode.split("_rhythm_only")[0].split("_coord_only")[0]
                skills = {part: {label: 1} for part in ("rhythm", "coord")
                          if not (mode.endswith("_rhythm_only") and part == "coord")
                          and not (mode.endswith("_coord_only") and part == "rhythm")}
            result = generate(audio, ["Insane"], args.out / "generated" / group / mode,
                rhythm_model=str(rhythm), coord_model=str(coord), model_skills=skills,
                bpm=song["bpm"], offset_ms=song["offset_ms"] + OSU_TIMING_SHIFT_MS,
                title=f"{original.title} [M1 {mode}]", artist=original.artist,
                creator=f"AUTO-OSU M1 {mode}",
                controls=dict(highlight_mode="off"), records_dir=args.out / "records", model_cache=cache,
                log=lambda *_: None, **settings)
            diff = result.diffs[0]
            with zipfile.ZipFile(result.osz) as archive:
                content = archive.read(next(n for n in archive.namelist() if n.endswith(".osu")))
                parsed = IndependentBeatmap.parse(content.decode("utf-8-sig"))
                parsed_objects = len(parsed.hit_objects(stacking=False))
                if parsed_objects != len(diff.beatmap.hit_objects):
                    raise ValueError("Independent parser object count differs")
                sliders = [obj for obj in parsed.hit_objects(stacking=False) if isinstance(obj, IndependentSlider)]
                outside = []
                for slider in sliders:
                    points = np.array([tuple(slider.curve(float(t))) for t in np.linspace(0, 1, 1001)])
                    if not np.isfinite(points).all() or np.any(points < [-.01, -.01]) or np.any(points > [512.01, 384.01]):
                        outside.append(slider.time.total_seconds() * 1000)
            if diff.measurement["status"] != "ok":
                raise ValueError("Actual star measurement failed")
            source = check_source(result.osz, records_dir=args.out / "records")
            if len(source["results"]) != 1 or any(item["manifest"] != "raw-match"
                    or not any(record["match"] == "raw" for record in item["local_records"])
                    for item in source["results"]):
                raise ValueError("Exported map does not match its manifest and local generation record")
            target = packs / f"{number:02d}-{mode}-{diff.measurement['stars']:.2f}star.osz"
            shutil.copy2(result.osz, target)
            preview = None
            if number == 1 and mode in ("jumps", "bursts", "streams", "slider_tech"):
                preview = render_preview(result.audio_file, diff.beatmap,
                    args.out / "点击声试听" / f"01-{mode}.mp3", click_shift_ms=result.osu_shift_ms)
            rows.append(dict(song=number, title=original.title, artist=original.artist,
                group=group, mode=mode, stars=diff.measurement["stars"],
                summary=diff.summary(), patterns=pattern_metrics(diff.beatmap, result.timing.beat_length),
                runs=run_lengths(diff.beatmap, result.timing.beat_length), diagnostics=diff.diagnostics,
                independently_parsed_objects=parsed_objects, source=source,
                curves=dict(parser="slider", samples_per_curve=1001, stacking=False,
                            sliders=len(sliders), outside_time_ms=outside),
                file=target.relative_to(args.out).as_posix(), sha256=digest(target),
                preview=preview.relative_to(args.out).as_posix() if preview else None,
                elapsed_s=result.elapsed_s, warnings=result.warnings))
            atomic_json(args.out / "results.json", rows)
            print(json.dumps({k: rows[-1][k] for k in ("song", "mode", "stars", "patterns", "elapsed_s")}), flush=True)
    instructions = ["# M1 标签模型试听包", "", "四首熟悉的开发歌曲；每首含旧版、新版无标签及四种技能条件。",
        "谱面名称中的星级为实际测量值。相同星级输入不保证输出实际星级相同，请结合下面的对照表判断。",
        "第一首另有仅节奏 / 仅坐标标签消融。参考 timing；highlight 关闭以分离标签效果。",
        "点击声试听文件可先听节奏；位置、滑条和手感需要将对应 .osz 导入 osu! 查看。",
        "这些是实验输出；请检查技能偏好、音乐性、恢复段、滑条和手感。尚未做客户端试玩或编辑器另存。", "",
        "| 歌曲 | 版本 | 实测星级 | 物件 | 最长四分之一拍连续圆圈 |", "|---|---|---:|---:|---:|"]
    for row in rows:
        instructions.append(f"| {row['song']}: {row['title']} | {MODES.get(row['mode'], row['mode'])} | {row['stars']:.2f} | "
                            f"{row['summary']['objects']} | {row['runs']['longest_run']} |")
    (args.out / "试听说明.md").write_text("\n".join(instructions) + "\n", encoding="utf-8")
    with zipfile.ZipFile(args.out / "AUTO-OSU-M1-试听包.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.out / "试听说明.md", "试听说明.md")
        for path in sorted(packs.glob("*.osz")):
            archive.write(path, f"谱面/{path.name}")
        for path in sorted((args.out / "点击声试听").glob("*")):
            archive.write(path, f"点击声试听/{path.name}")
    print(f"COMPLETE {len(rows)} maps", flush=True)


if __name__ == "__main__":
    main()
