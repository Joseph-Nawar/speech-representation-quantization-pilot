from __future__ import annotations

from collections import Counter
import io
import json
from pathlib import Path
import zlib

import numpy as np
import pytest
import soundfile as sf
import zipfile

from sqpilot.p2_data import (
    audit_audio_payloads,
    build_official_member_plan,
    make_split_manifest,
    materialize_official,
    model_rows,
    plan_local_upper_bound,
    preflight_resource_limits,
    select_max_complete_grid,
    verify_manifest_hash,
    verify_candidate_metadata,
    verify_historical_micro_test_disjoint,
)


def test_max_complete_grid_uses_locked_size_priority_then_lexicographic_ids():
    cells = {(f"a{i}", f"p{j}") for i in range(4) for j in range(3)}
    cells |= {(f"b{i}", f"q{j}") for i in range(3) for j in range(4)}
    result = select_max_complete_grid(
        cells,
        min_speakers=3,
        max_speakers=4,
        min_prompts=3,
        max_prompts=4,
        max_cells=12,
    )
    assert result["speakers"] == ["a0", "a1", "a2", "a3"]
    assert result["prompts"] == ["p0", "p1", "p2"]
    assert len(result["cells"]) == 12


def test_max_complete_grid_fails_closed_on_duplicate_or_insufficient_cells():
    with pytest.raises(ValueError, match="duplicate candidate cell"):
        select_max_complete_grid(
            [("s1", "p1"), ("s1", "p1")],
            min_speakers=1,
            max_speakers=2,
            min_prompts=1,
            max_prompts=2,
            max_cells=4,
        )
    with pytest.raises(ValueError, match="No complete grid"):
        select_max_complete_grid(
            [("s1", "p1")],
            min_speakers=2,
            max_speakers=2,
            min_prompts=2,
            max_prompts=2,
            max_cells=4,
        )


def test_official_member_plan_is_mic1_only_and_rejects_text_or_path_drift():
    speakers, prompts = ["p225", "p226"], ["002", "003"]
    candidate = {"speakers": speakers, "prompts": prompts}
    entries = []
    texts = {}
    offset = 0
    for speaker in speakers:
        for prompt in prompts:
            texts[f"{speaker}_{prompt}"] = "canonical sentence"
            path = f"wav48_silence_trimmed/{speaker}/{speaker}_{prompt}_mic1.flac"
            entries.append([path, 17, 128, 100, offset])
            offset += 256
    entries.append(["zz-sentinel", 0, 0, 0, offset])
    plan = build_official_member_plan(candidate, entries, texts, max_cells=4, max_range_bytes=2048)
    assert len(plan["rows"]) == 4
    assert all(row["mic"] == "mic1" and row["path"].endswith("_mic1.flac") for row in plan["rows"])
    assert all(row["transcript_sha256"] for row in plan["rows"])
    bad_texts = dict(texts, p226_003="different sentence")
    with pytest.raises(ValueError, match="canonical prompt transcript"):
        build_official_member_plan(candidate, entries, bad_texts, max_cells=4, max_range_bytes=2048)
    bad_entries = [row.copy() for row in entries]
    bad_entries[0][0] = bad_entries[0][0].replace("mic1", "mic2")
    with pytest.raises(ValueError, match="mic1"):
        build_official_member_plan(candidate, bad_entries, texts, max_cells=4, max_range_bytes=2048)


def test_official_member_plan_preserves_p0_registered_speaker_prefix_variants():
    speaker, prompt = "s5", "002"
    path = f"wav48_silence_trimmed/{speaker}/{speaker}_{prompt}_mic1.flac"
    entries = [[path, 17, 128, 100, 0], ["zz-sentinel", 0, 1, 1, 256]]
    plan = build_official_member_plan(
        {"speakers": [speaker], "prompts": [prompt]}, entries,
        {f"{speaker}_{prompt}": "same canonical sentence"}, max_cells=1, max_range_bytes=512,
    )
    assert plan["rows"][0]["id"] == "s5_002_mic1"


def test_candidate_metadata_cross_audits_pinned_mirror_to_official_source():
    candidate = {"speakers": ["p225", "p226"], "prompts": ["002", "003"]}
    texts = {f"{s}_{p}": "  SAME canonical sentence  " for s in candidate["speakers"]
             for p in candidate["prompts"]}
    official = [[f"wav48_silence_trimmed/{s}/{s}_{p}_mic1.flac", 23, 128, 90, i * 200]
                for i, (s, p) in enumerate((s, p) for s in candidate["speakers"] for p in candidate["prompts"])]
    mirror = [{"id": f"{s}_{p}", "speaker_id": s, "mic_id": "mic1", "path": f"{s}/{s}_{p}_mic1.flac",
               "text_original": "same canonical sentence", "duration_s": 1.4}
              for s in candidate["speakers"] for p in candidate["prompts"]]
    result = verify_candidate_metadata(candidate, official, texts, mirror)
    assert result["candidate_cells_cross_audited"] == 4
    assert result["mirror_mic1_candidate_rows"] == 4
    bad_mirror = [dict(row) for row in mirror]
    bad_mirror[0]["path"] = bad_mirror[0]["path"].replace("mic1", "mic2")
    with pytest.raises(ValueError, match="mirror path or microphone"):
        verify_candidate_metadata(candidate, official, texts, bad_mirror)


def test_mirror_candidate_metadata_duplicate_ids_fail_closed():
    candidate = {"speakers": ["p225"], "prompts": ["002"]}
    texts = {"p225_002": "sentence"}
    official = [["wav48_silence_trimmed/p225/p225_002_mic1.flac", 23, 128, 90, 0]]
    row = {"id": "p225_002", "speaker_id": "p225", "mic_id": "mic1", "path": "p225/p225_002_mic1.flac",
           "text_original": "sentence", "duration_s": 1.4}
    with pytest.raises(ValueError, match="duplicate mirror recording"):
        verify_candidate_metadata(candidate, official, texts, [row, dict(row)])


def _flac(values: np.ndarray, rate: int = 48000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, values, rate, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def test_audio_audit_logs_objective_rejects_and_deduplicates_by_audio_hash():
    valid = _flac(np.sin(np.arange(48000, dtype=np.float32) * 0.02) * 0.4)
    silent = _flac(np.zeros(48000, dtype=np.float32))
    rows = [
        {"id": "p225_002_mic1", "speaker": "p225", "prompt": "002", "part": "", "mic": "mic1",
         "path": "p225_002_mic1.flac"},
        {"id": "p225_003_mic1", "speaker": "p225", "prompt": "003", "part": "", "mic": "mic1",
         "path": "p225_003_mic1.flac"},
        {"id": "p226_002_mic1", "speaker": "p226", "prompt": "002", "part": "", "mic": "mic1",
         "path": "p226_002_mic1.flac"},
    ]
    accepted, rejected = audit_audio_payloads(
        rows,
        {rows[0]["id"]: valid, rows[1]["id"]: valid, rows[2]["id"]: silent},
    )
    assert [row["id"] for row in accepted] == ["p225_002_mic1"]
    assert len(rejected) == 2
    assert rejected[0]["reason"] == "duplicate_audio_content"
    assert rejected[1]["reason"].startswith("invalid_audio:")
    assert accepted[0]["sample_rate_hz"] == 16000
    assert accepted[0]["duration_seconds"] == pytest.approx(1.0)
    assert accepted[0]["decoded_pcm16k_sha256"]


def test_audio_audit_checks_official_size_and_crc_before_decode():
    payload = _flac(np.sin(np.arange(48000, dtype=np.float32) * 0.02) * 0.4)
    row = {"id": "p225_002_mic1", "speaker": "p225", "prompt": "002", "mic": "mic1",
           "path": "p225_002_mic1.flac", "size": len(payload) + 1, "crc32": zlib.crc32(payload)}
    accepted, rejected = audit_audio_payloads([row], {row["id"]: payload})
    assert not accepted
    assert rejected[0]["reason"] == "official_crc_or_size_mismatch"


def test_audio_file_metadata_duration_limit_runs_before_full_decode(tmp_path, monkeypatch):
    from sqpilot.p2_data import audit_audio_files

    payload = _flac(np.sin(np.arange(480000 + 4800, dtype=np.float32) * 0.02) * 0.4)
    row = {"id": "p225_002_mic1", "speaker": "p225", "prompt": "002", "mic": "mic1",
           "path": "p225_002_mic1.flac", "size": len(payload), "crc32": zlib.crc32(payload)}
    (tmp_path / f"{row['id']}.flac").write_bytes(payload)
    monkeypatch.setattr("sqpilot.p2_data.preprocess", lambda _payload: pytest.fail("decoder ran before duration cap"))
    accepted, rejected, _ = audit_audio_files([row], tmp_path)
    assert not accepted
    assert rejected[0]["reason"] == "duration_out_of_range"


def test_full_split_meets_original_support_and_model_rows_never_include_test():
    speakers = [f"p{i:03d}" for i in range(40)]
    prompts = [f"{i:03d}" for i in range(16)]
    rows = [
        {
            "id": f"{speaker}_{prompt}_mic1",
            "speaker": speaker,
            "prompt": prompt,
            "mic": "mic1",
            "audio_sha256": f"{i:064x}",
            "decoded_pcm16k_sha256": f"{i+1:064x}",
        }
        for i, (speaker, prompt) in enumerate((s, p) for s in speakers for p in prompts)
    ]
    manifest = make_split_manifest(rows, seed=20261122)
    assert Counter(row["partition"] for row in manifest["rows"]) == {
        "train": 384,
        "dev": 128,
        "test": 128,
    }
    selected = model_rows(rows, manifest)
    assert selected and all(row["partition"] in {"train", "dev"} for row in selected)
    assert not any(row["partition"] == "test" for row in selected)
    assert manifest["support_minima"] == {
        "train_per_speaker": 8,
        "dev_per_speaker": 3,
        "test_per_speaker": 3,
        "train_per_prompt": 12,
        "dev_per_prompt": 5,
        "test_per_prompt": 5,
    }


def test_split_rejects_cross_partition_audio_duplicates_and_bad_manifest_hash():
    speakers, prompts = [f"p{i:03d}" for i in range(40)], [f"{i:03d}" for i in range(16)]
    rows = [
        {"id": f"{s}_{p}_mic1", "speaker": s, "prompt": p, "mic": "mic1",
         "audio_sha256": f"{i+1:064x}", "decoded_pcm16k_sha256": f"{i+1000:064x}"}
        for i, (s, p) in enumerate((s, p) for s in speakers for p in prompts)
    ]
    rows[1]["decoded_pcm16k_sha256"] = rows[0]["decoded_pcm16k_sha256"]
    with pytest.raises(ValueError, match="duplicate audio content across cells"):
        make_split_manifest(rows, seed=20261122)
    document = {"rows": [{"id": "one"}]}
    digest = __import__("hashlib").sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert verify_manifest_hash(document, digest)
    with pytest.raises(ValueError, match="[Mm]anifest SHA256 mismatch"):
        verify_manifest_hash(document, "0" * 64)


def test_split_rejects_duplicate_source_ids_and_preflight_bounds_worst_case_pcm():
    speakers, prompts = [f"p{i:03d}" for i in range(40)], [f"{i:03d}" for i in range(16)]
    rows = [
        {"id": f"{s}_{p}_mic1", "speaker": s, "prompt": p, "mic": "mic1",
         "audio_sha256": f"{i+1:064x}", "decoded_pcm16k_sha256": f"{i+1000:064x}"}
        for i, (s, p) in enumerate((s, p) for s in speakers for p in prompts)
    ]
    rows[-1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError, match="duplicate clip ID"):
        make_split_manifest(rows, seed=20261122)
    plan = {"candidate_cells": 100, "planned_flac_bytes": 12_345}
    assert plan_local_upper_bound(plan, max_duration_seconds=10, sample_rate_hz=16_000, bytes_per_sample=4) == (
        12_345 + 100 * 10 * 16_000 * 4
    )


def test_historical_p1_micro_ids_must_remain_outside_the_new_sealed_test_partition():
    micro = {"rows": [{"id": "clip-a", "part": "train"}, {"id": "clip-b", "part": "dev"}]}
    split = {"rows": [{"id": "clip-a", "partition": "train"}, {"id": "clip-b", "partition": "dev"},
                      {"id": "clip-c", "partition": "test"}]}
    assert verify_historical_micro_test_disjoint(micro, split)["overlap_count"] == 0
    split["rows"][0]["partition"] = "test"
    with pytest.raises(ValueError, match="historical P1 model access overlaps sealed P2 test"):
        verify_historical_micro_test_disjoint(micro, split)


def _one_member_zip_plan():
    buffer = io.BytesIO()
    path = "wav48_silence_trimmed/p225/p225_002_mic1.flac"
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(path, b"bounded official audio payload")
        archive.writestr("zz-sentinel", b"x")
        first = archive.getinfo(path)
        second = archive.getinfo("zz-sentinel")
    data = buffer.getvalue()
    row = {
        "id": "p225_002_mic1",
        "speaker": "p225",
        "prompt": "002",
        "mic": "mic1",
        "path": path,
        "crc32": first.CRC,
        "size": first.file_size,
        "compressed": first.compress_size,
        "offset": first.header_offset,
        "span": second.header_offset - first.header_offset,
        "transcript_sha256": "a" * 64,
    }
    plan = {"rows": [row], "candidate_cells": 1, "planned_range_bytes": row["span"]}
    return plan, data


def test_official_download_is_range_bounded_resumable_and_manifest_keyed(tmp_path, monkeypatch):
    plan, archive_bytes = _one_member_zip_plan()

    class FakeRemote:
        calls = 0

        def __init__(self, url, total, etag):
            assert url.startswith("https://datashare.ed.ac.uk/")
            self.bytes = 0

        def pull(self, offset, length):
            type(self).calls += 1
            payload = archive_bytes[offset:offset + length]
            assert len(payload) == length
            self.bytes += len(payload)
            return payload

    monkeypatch.setattr("sqpilot.p2_data.Remote", FakeRemote)
    audio_dir, receipt_dir = tmp_path / "audio", tmp_path / "receipts"
    first = materialize_official(plan, audio_dir, receipt_dir, max_payload_bytes=4096)
    assert first["downloaded_cells"] == 1
    assert first["cached_cells"] == 0
    assert first["actual_range_bytes"] == plan["planned_range_bytes"]
    assert (audio_dir / "p225_002_mic1.flac").read_bytes() == b"bounded official audio payload"
    calls_after_first = FakeRemote.calls

    second = materialize_official(plan, audio_dir, receipt_dir, max_payload_bytes=4096)
    assert second["downloaded_cells"] == 0
    assert second["cached_cells"] == 1
    assert second["actual_range_bytes"] == plan["planned_range_bytes"]
    assert second["actual_range_bytes_this_call"] == 0
    assert FakeRemote.calls == calls_after_first

    changed = json.loads(json.dumps(plan))
    changed["rows"][0]["transcript_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="changed plan manifest"):
        materialize_official(changed, audio_dir, receipt_dir, max_payload_bytes=4096)


def test_official_download_cap_fails_before_network_request(tmp_path, monkeypatch):
    plan, _ = _one_member_zip_plan()

    class NoNetwork:
        def __init__(self, *args, **kwargs):
            pytest.fail("network request started before bound check")

    monkeypatch.setattr("sqpilot.p2_data.Remote", NoNetwork)
    with pytest.raises(ValueError, match="range payload exceeds cap"):
        materialize_official(
            plan,
            tmp_path / "audio",
            tmp_path / "receipts",
            max_payload_bytes=plan["planned_range_bytes"] - 1,
        )


def test_p2_resource_limits_fail_closed_before_audio_download():
    limits = {
        "P2_max_audio_cells": 1134,
        "P2_max_audio_payload_bytes": 2 * 1024**3,
        "P2_max_local_decoded_bytes": 4 * 1024**3,
        "P2_min_free_disk_bytes": 5 * 1024**3,
    }
    assert preflight_resource_limits(
        limits, cells=1134, range_bytes=1024, local_bytes=2048, free_disk_bytes=6 * 1024**3
    )
    with pytest.raises(ValueError, match="RESOURCE_OVERRIDE_REQUIRED.*cells"):
        preflight_resource_limits(limits, cells=1135, range_bytes=1024, local_bytes=2048, free_disk_bytes=6 * 1024**3)
    with pytest.raises(ValueError, match="RESOURCE_OVERRIDE_REQUIRED.*range bytes"):
        preflight_resource_limits(
            limits, cells=100, range_bytes=2 * 1024**3 + 1, local_bytes=2048, free_disk_bytes=6 * 1024**3
        )
    with pytest.raises(ValueError, match="RESOURCE_OVERRIDE_REQUIRED.*local decoded"):
        preflight_resource_limits(
            limits, cells=100, range_bytes=1024, local_bytes=4 * 1024**3 + 1, free_disk_bytes=6 * 1024**3
        )
    with pytest.raises(ValueError, match="RESOURCE_OVERRIDE_REQUIRED.*free disk"):
        preflight_resource_limits(
            limits, cells=100, range_bytes=1024, local_bytes=2048, free_disk_bytes=5 * 1024**3 - 1
        )
