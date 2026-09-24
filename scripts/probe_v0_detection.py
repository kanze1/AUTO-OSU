"""Offline feasibility screen, never an application detector.

Prepare a local corpus, generate with an explicitly frozen checkout, then evaluate.
Raw songs, maps, features and fitted classifiers stay under ignored out/.
Historical reference maps are not individually verified human ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def split(group):
    n = int(hashlib.sha256(("autoosu-d1-v1:" + group).encode()).hexdigest()[:8], 16) % 5
    return "train" if n < 3 else "calibration" if n == 3 else "test"


def prepare(args):
    if (args.out / "corpus.json").exists():
        raise ValueError("Corpus already exists; reuse it or choose a fresh --out directory")
    references, candidates = {}, {}
    for shard in sorted(args.shards.glob("*.tar")):
        with tarfile.open(shard) as archive:
            for member in archive:
                if not member.name.endswith(".json") or member.size > 32 * 1024 * 1024:
                    continue
                data = json.load(archive.extractfile(member))
                group = data["audio_hash"]
                maps = [b for b in data["beatmaps"] if b.get("mode") == 0
                        and b.get("approved") in (1, 2, 4)
                        and 3 <= float(b.get("difficultyrating") or 0) <= 7
                        and "1900" < str(b.get("last_update") or "")[:4] <= "2020"]
                if not maps:
                    continue
                b = min(maps, key=lambda m: (abs(float(m["difficultyrating"]) - 5), int(m["beatmap_id"])))
                references[group] = dict(group=group, split=split(group), label="historical_reference",
                                         beatmap_id=b["beatmap_id"], creator_id=b["creator_id"],
                                         content=b["content"], shard=shard.name, member=member.name)
                if 60 <= float(data.get("audio_length", 0)) <= 150:
                    candidates[group] = dict(group=group, split=split(group), shard=shard.name,
                                             member=member.name.replace(".json", ".opus"))
    # Ten songs per partition: balanced positive support, no song crosses partitions.
    chosen = []
    for partition, count in (("train", args.songs // 2), ("calibration", args.songs // 4),
                             ("test", args.songs - args.songs // 2 - args.songs // 4)):
        eligible = sorted((c for c in candidates.values() if c["split"] == partition),
                          key=lambda c: hashlib.sha256(("sample:" + c["group"]).encode()).hexdigest())
        if len(eligible) < count:
            raise ValueError("Not enough eligible independent songs")
        chosen.extend(eligible[:count])
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    # Keep every reference: one map per unique audio, not one row per difficulty.
    for group, item in sorted(references.items()):
        item = item.copy()
        text = item.pop("content")
        path = args.out / "reference" / f"{group}.osu"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
        rows.append(dict(item, path=str(path.relative_to(args.out)), sha256=hashlib.sha256(text.encode()).hexdigest()))
    for item in chosen:
        path = args.out / "audio" / f"{item['group']}.opus"
        path.parent.mkdir(exist_ok=True)
        with tarfile.open(args.shards / item["shard"]) as archive:
            path.write_bytes(archive.extractfile(item["member"]).read())
        item["path"] = str(path.relative_to(args.out))
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(args.out / "corpus.json", dict(schema=1, songs=chosen, references=rows,
                                       reference_label="historical, individually unverified; not release ground truth"))
    print(json.dumps(dict(songs=len(chosen), references=len(rows))), flush=True)


def run(args):
    # The caller supplies a git archive of the baseline code, excluding uncommitted edits.
    sys.path.insert(0, str(args.frozen.resolve()))
    from autoosu.generate import generate
    from autoosu.models import sha256_of

    corpus = json.loads((args.out / "corpus.json").read_text(encoding="utf-8"))
    results_path = args.out / "generated.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    completed = {(r["group"], r["label"]) for r in results}
    settings = dict(seed=240924, decode_steps=12, coord_steps=100, temperature=0.9, cfg_scale=1.0,
                    star_rating=5.5, device="cuda")
    recipe = dict(code_commit=args.commit, settings=settings,
                  models={k: sha256_of(args.models / f"{k}_v0.pt") for k in ("rhythm", "coord")})
    recipe_path = args.out / "recipe.json"
    if results and (not recipe_path.exists() or json.loads(recipe_path.read_text()) != recipe):
        raise ValueError("Existing output uses a different recipe; choose a fresh --out directory")
    save(recipe_path, recipe)
    cache = {}
    for i, song in enumerate(corpus["songs"]):
        for label, rhythm, coord in (("both_models", True, True), ("rhythm_only", True, False),
                                     ("coord_only", False, True), ("rules", False, False)):
            if (song["group"], label) in completed:
                continue
            start = time.perf_counter()
            result = generate(args.out / song["path"], ["Insane"], args.out / "generated" / song["group"] / label,
                              rhythm_model=str(args.models / "rhythm_v0.pt") if rhythm else None,
                              coord_model=str(args.models / "coord_v0.pt") if coord else None,
                              title="D1 frozen probe", artist=song["group"][:12], model_cache=cache,
                              log=lambda *_: None, **settings)
            with zipfile.ZipFile(result.osz) as z:
                name = next(n for n in z.namelist() if n.endswith(".osu"))
                raw = z.read(name)
            path = result.osz.parent / "map.osu"
            path.write_bytes(raw)
            results.append(dict(group=song["group"], split=song["split"], label=label,
                                path=str(path.relative_to(args.out)), sha256=hashlib.sha256(raw).hexdigest(),
                                elapsed_s=round(time.perf_counter() - start, 3)))
            save(results_path, results)
            print(f"{i + 1}/{len(corpus['songs'])} {label}: {results[-1]['elapsed_s']:.1f}s", flush=True)


def features(text, edit=False, partial=False):
    """Content features only: no title, tags, author, filename, preset fields or whitespace."""
    sections, section = {}, ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line
            sections[section] = []
        elif line and not line.startswith("//"):
            sections.setdefault(section, []).append(line)
    timing = [l.split(",") for l in sections.get("[TimingPoints]", [])]
    reds = [(float(p[0]), float(p[1])) for p in timing if float(p[1]) > 0]
    objects = [l.split(",") for l in sections.get("[HitObjects]", [])]
    if partial:
        objects = objects[len(objects) // 4:3 * len(objects) // 4]
    if len(objects) < 30 or not reds:
        return None
    times = np.array([float(p[2]) for p in objects])
    xy = np.array([[float(p[0]), float(p[1])] for p in objects])
    if edit:
        # Synthetic edit stress, not an actual osu! editor round-trip.
        xy[::10] += np.array([7, -5])
        times[::10] += 3
    types = np.array([int(p[3]) for p in objects])
    dt = np.diff(times)
    xy_diff = np.diff(xy, axis=0)
    dist = np.linalg.norm(xy_diff, axis=1)
    red_times, bls = np.array(reds).T
    indices = np.maximum(0, np.searchsorted(red_times, times, side="right") - 1)
    beats = (times - red_times[indices]) / bls[indices]
    frac = np.abs(beats * 4 - np.round(beats * 4))
    intervals = dt / bls[indices[:-1]]
    angles = np.sum(xy_diff[1:] * xy_diff[:-1], axis=1) / np.maximum(dist[1:] * dist[:-1], 1)
    sliding = [p for p in objects if int(p[3]) & 2]
    lengths = [float(p[7]) for p in sliding]
    bins = np.histogram(times, bins=max(2, int((times[-1] - times[0]) / 8000)))[0]
    def quantiles(values):
        return np.quantile(values if len(values) else [0], [.1, .5, .9]).tolist()
    return [float(np.mean(types & 2 != 0)), float(np.mean(types & 8 != 0)), float(np.mean(types & 4 != 0)),
            float(np.mean(frac < .015)), float(np.mean(dt <= 0)), float(np.mean(dist < 5)),
            float(np.std(bins) / max(np.mean(bins), 1)),
            *quantiles(intervals), *quantiles(dist), *quantiles(angles), *quantiles(lengths),
            sum(p[5].startswith("L|") for p in sliding) / max(len(sliding), 1),
            sum(int(p[6]) > 1 for p in sliding) / max(len(sliding), 1)]


def evaluate(args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    corpus = json.loads((args.out / "corpus.json").read_text())
    generated = json.loads((args.out / "generated.json").read_text())
    rows, abstentions = [], []
    for row in corpus["references"] + generated:
        text = (args.out / row["path"]).read_text(encoding="utf-8-sig")
        try:
            f = features(text)
        except (ValueError, IndexError, ZeroDivisionError) as exc:
            f = None
            abstentions.append(dict(label=row["label"], split=row["split"], path=row["path"], reason=str(exc)))
        else:
            if f is None:
                abstentions.append(dict(label=row["label"], split=row["split"], path=row["path"], reason="too few objects or missing timing"))
        if f is not None:
            rows.append(dict(row, features=f))
            if row["label"] == "both_models" and row["split"] == "test":
                for kind in ("edit", "partial"):
                    f2 = features(text, edit=kind == "edit", partial=kind == "partial")
                    if f2 is not None:
                        rows.append(dict(row, label="both_models_" + kind, features=f2))
    # Independent generator files are external specificity probes, never training data.
    if args.other:
        for path in sorted(args.other.glob("*.osz")):
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if name.endswith(".osu"):
                        f = features(z.read(name).decode("utf-8-sig"))
                        if f is not None:
                            rows.append(dict(group="external-unverified-song-overlap", split="test",
                                             label="other_generator", path=path.name + "/" + name, features=f))
    # Classifier is trained on dual-model outputs only; hybrid coverage is reported separately.
    training = [r for r in rows if r["split"] == "train" and r["label"] in ("both_models", "historical_reference")]
    negatives = [r for r in rows if r["split"] == "calibration" and r["label"] == "historical_reference"]
    report = dict(schema=1, recipe=json.loads((args.out / "recipe.json").read_text()),
                  independent_human_ground_truth=False, actual_editor_roundtrip=False,
                  feature_count=len(training[0]["features"]), abstentions=abstentions, models={})
    for name, clf in (("logistic", make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=24))),
                      ("forest", RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=24, n_jobs=4))):
        clf.fit([r["features"] for r in training], [r["label"] == "both_models" for r in training])
        threshold = float(np.nextafter(clf.predict_proba([r["features"] for r in negatives])[:, 1].max(), np.inf))
        metrics = {}
        for label in sorted({r["label"] for r in rows}):
            items = [r for r in rows if r["split"] == "test" and r["label"] == label]
            if not items:
                continue
            predictions = clf.predict_proba([r["features"] for r in items])[:, 1] >= threshold
            flagged = int(predictions.sum())
            groups = len({r["group"] for r in items})
            metrics[label] = dict(n=len(items), audio_groups=groups,
                                  flagged=flagged, rate=flagged / len(items), not_flagged=len(items) - flagged,
                                  zero_event_one_sided_95_upper_by_audio_group=(1 - .05 ** (1 / groups)) if flagged == 0 else None,
                                  flagged_examples=[r.get("beatmap_id", r["path"]) for r, pred in zip(items, predictions) if pred][:20])
        report["models"][name] = dict(threshold=threshold, calibration_negatives=len(negatives),
                                      training_rows=len(training), tests=metrics)
    report["decision"] = "Do not ship: exploratory corpus lacks verified human provenance and independent low-FPR/other-generator coverage. Unflagged means inconclusive."
    report["limitations"] = ["One preset, seed and automatic timing path; only current v0 code snapshot.",
                             "Reference songs may have been in generator training; this is not a model-generalization study.",
                             "Synthetic coordinate/time edits and cropping do not reproduce editor resaving or mixed authorship.",
                             "Audio-hash grouping does not deduplicate different recordings of the same song; mapper overlap remains.",
                             "Other-generator sample has too few independent songs for attribution claims."]
    save(args.out / "report.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "evaluate"))
    parser.add_argument("--out", type=Path, default=ROOT / "out/d1-v0-probe")
    parser.add_argument("--shards", type=Path, default=ROOT / "data/hf/compressed")
    parser.add_argument("--songs", type=int, default=40)
    parser.add_argument("--frozen", type=Path, default=ROOT / "out/d1-v0-probe/frozen")
    parser.add_argument("--commit", default="8f508921534bef39c4f7beb54201a09fb7db8aca")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--other", type=Path, default=ROOT / "out/baselines")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
