import io
import json
import tarfile

import pytest

from scripts.audit_training_data import audit, cross_split_groups, tag_candidates


def test_audio_split_conflict_across_different_sets():
    rows = [{"source_audio_hash": "same", "legacy_split": "train"},
            {"source_audio_hash": "same", "legacy_split": "validation"},
            {"source_audio_hash": "other", "legacy_split": "train"},
            {"source_audio_hash": None, "legacy_split": "validation"}]
    result = cross_split_groups(rows, "source_audio_hash")
    assert result == {"groups": 2, "mixed_groups": 1, "maps_in_mixed_groups": 2,
                      "validation_maps_in_mixed_groups": 1}


def test_literal_candidates_do_not_match_substrings():
    assert tag_candidates("jumping mainstream dreamstream") == []
    assert tag_candidates("JUMPS skillset/streams") == ["jumps", "skillset/streams"]


def make_corpus(tmp_path, *, source_id=1, tags="jumps anime"):
    shards = tmp_path / "shards"
    shards.mkdir()
    data = json.dumps({"audio_hash": "declared-audio", "beatmaps": [
        {"beatmap_id": source_id, "tags": tags, "content": "osu file format v14"}]}).encode()
    with tarfile.open(shards / "data-000000.tar", "w") as tf:
        member = tarfile.TarInfo("000001.json")
        member.size = len(data)
        tf.addfile(member, io.BytesIO(data))
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({"beatmap_id": 1, "beatmapset_id": 1,
                                "track": "data-000000_000001", "sr": 6.5,
                                "n_objects": 100}) + "\n", encoding="utf-8")
    return shards, index


@pytest.mark.parametrize("tags", [None, "", "jumps anime"])
def test_source_tags_never_become_reviewed_labels(tmp_path, tags):
    shards, index = make_corpus(tmp_path, tags=tags)
    report = audit(shards, index, tmp_path / "out")
    row = json.loads((tmp_path / "out/manifest.jsonl").read_text())
    assert report["matched_maps"] == 1
    assert not report["training_ready"]
    assert report["reviewed_skill_label_maps"] == 0
    assert set(row["skill_labels"].values()) == {None}
    assert row["source_tags"] == tags
    assert row["label_status"] == "unreviewed"


def test_missing_source_cannot_report_complete_inventory(tmp_path):
    shards, index = make_corpus(tmp_path, source_id=2)
    report = audit(shards, index, tmp_path / "out")
    assert report["status"] == "incomplete_source_join"
    assert report["missing_source_records"] == 1


def test_existing_review_directory_is_preserved(tmp_path):
    shards, index = make_corpus(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    review = out / "review_queue.jsonl"
    review.write_text("existing human review", encoding="utf-8")
    with pytest.raises(FileExistsError):
        audit(shards, index, out)
    assert review.read_text() == "existing human review"


def test_duplicate_index_keys_are_rejected(tmp_path):
    shards, index = make_corpus(tmp_path)
    index.write_text(index.read_text() * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        audit(shards, index, tmp_path / "out")
    assert not (tmp_path / "out").exists()
