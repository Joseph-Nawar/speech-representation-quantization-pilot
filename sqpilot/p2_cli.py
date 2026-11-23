"""Bounded, sealed-test-safe P2 execution for protocol v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch
import yaml

from tools.p0_source_ranges import norm
from sqpilot.data import preprocess
from sqpilot.model import (
    LFS_SHA256,
    SOURCE_HASH,
    check_close,
    check_model_files,
    feature,
    load_model,
    masked_mean,
    padded_frames,
    processor,
)
from sqpilot.p2_data import (
    audit_audio_files,
    build_official_member_plan,
    canonical_json,
    make_split_manifest,
    materialize_official,
    model_rows,
    plan_local_upper_bound,
    preflight_resource_limits,
    verify_candidate_metadata,
    verify_historical_micro_test_disjoint,
    write_json_atomic,
)
from sqpilot.quant import stable_hash
from sqpilot.p2_engine import (
    P2_C_GRID,
    _atomic_npz,
    array_sha,
    cached_features,
    evaluation_metrics,
    fit_frozen_probe,
    g1_gate,
    g2_gate,
    head_array_predictions,
    head_digest,
)
from sqpilot.stats import BOOTSTRAP_SPEC_SHA256
from sqpilot.quant import apply_from_master, cost_book, discover, fingerprint, master_weights
from tools.p0_model import write_json


STUDY_ID = "speech-quant-diagnostic-20261122-v3"
PROTOCOL_VERSION = 3
CONFIG = Path("configs/pilot.yaml")
P0_P1_STATUS = Path("P0_P1_STATUS.json")
P0_V1_STATUS = Path("reports/P0_STATUS_V1.json")
P2_Q002_RESULT = Path("runs/P2_Q002/q002-synthetic.json")
RUN_DIR = Path("runs") / STUDY_ID
DATA_DIR = Path("data") / STUDY_ID
AMENDED_HISTORIC_PATHS = {"configs/pilot.yaml", "docs/01_protocol.md", "docs/04_analysis_and_gates.md"}


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def read_json(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_csv_atomic(path: Path, columns: list[str], rows: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite changed manifest {path}")
        return digest_bytes(payload)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return digest_bytes(payload)


def _verify_hash_inventory(inventory: dict, *, amended_paths: set[str] | None = None) -> dict:
    amended_paths = amended_paths or set()
    checked, amended = [], []
    for relative_path, expected in inventory.items():
        path = Path(relative_path)
        if not path.is_file():
            raise ValueError(f"historical evidence is missing: {relative_path}")
        actual = file_sha(path)
        if actual == expected:
            checked.append(relative_path)
        elif relative_path in amended_paths:
            amended.append({"path": relative_path, "historical_sha256": expected, "current_sha256": actual,
                            "disposition": "protocol-v3 amendment"})
        else:
            raise ValueError(f"historical hash mismatch: {relative_path}: expected {expected}, got {actual}")
    unexpected_amendments = {row["path"] for row in amended} - amended_paths
    if unexpected_amendments:
        raise ValueError(f"unrecognized historical amendments: {sorted(unexpected_amendments)}")
    return {"verified_count": len(checked), "verified_paths": checked, "amended_paths": amended}


def verify_prior_evidence() -> dict:
    p0p1 = read_json(P0_P1_STATUS)
    previous = read_json(P0_V1_STATUS)
    if p0p1.get("status") != "PASS" or p0p1.get("P0", {}).get("status") != "PASS" or p0p1.get("P1", {}).get("status") != "PASS":
        raise ValueError("P0/P1 source status is not a recorded pass")
    if p0p1.get("study_id") != "speech-quant-diagnostic-20261122-v2" or p0p1.get("protocol_version") != 2:
        raise ValueError("P0/P1 status does not identify the frozen Q-001 v2 study")
    versioned = _verify_hash_inventory(p0p1["sha256"]["versioned"], amended_paths=AMENDED_HISTORIC_PATHS)
    local = _verify_hash_inventory(p0p1["sha256"]["local_evidence"])
    original_sources = previous["sha256"]["ignored_local_evidence"]
    source = _verify_hash_inventory(original_sources)
    for report in ("reports/P0_PREFLIGHT.md", "reports/P0_PREFLIGHT_Q001.md", "reports/P1_SMOKE.md", "reports/P0_STATUS_V1.json"):
        if report not in p0p1["sha256"]["versioned"] or file_sha(Path(report)) != p0p1["sha256"]["versioned"][report]:
            raise ValueError(f"prior scientific report changed: {report}")

    q002 = read_json(P2_Q002_RESULT)
    if (q002.get("study_id"), q002.get("protocol_version"), q002.get("Q002")) != (
        STUDY_ID, PROTOCOL_VERSION, "PREOUTCOME_SYNTHETIC_DIAGNOSTIC_RECORDED"
    ):
        raise ValueError("Q-002 synthetic result is absent or mismatched")
    if q002.get("test_audio_model_forwards") != 0 or q002.get("test_prediction_reads") != 0:
        raise ValueError("Q-002 synthetic stage has nonzero test/model access counters")
    result_without_self_hash = {key: value for key, value in q002.items()
                                if key not in {"wall_seconds", "scientific_results_sha256"}}
    q002_scientific_sha = digest_bytes(canonical_json(result_without_self_hash))
    if q002.get("scientific_results_sha256") != q002_scientific_sha:
        raise ValueError("Q-002 synthetic result checksum is invalid")
    return {
        "P0_P1_status_sha256": file_sha(P0_P1_STATUS),
        "P0_v1_status_sha256": file_sha(P0_V1_STATUS),
        "P0_P1_versioned": versioned,
        "P0_P1_local_evidence": local,
        "P0_source_evidence": source,
        "P0_P1_reports_preserved": True,
        "Q002_result_sha256": file_sha(P2_Q002_RESULT),
        "Q002_scientific_results_sha256": q002_scientific_sha,
    }


def _load_mirror_metadata() -> tuple[list[dict], dict[str, str]]:
    files = sorted(Path("cache/p0-source").glob("larry-*.json"))
    if len(files) != 22:
        raise ValueError(f"pinned mirror metadata shard count changed: expected 22, got {len(files)}")
    rows = []
    hashes = {}
    for path in files:
        payload = path.read_bytes()
        hashes[str(path)] = digest_bytes(payload)
        shard = json.loads(payload)
        if not isinstance(shard, list):
            raise ValueError(f"pinned mirror shard is not a list: {path}")
        rows.extend(shard)
    if len(rows) != 88_156:
        raise ValueError(f"pinned mirror metadata row count changed: expected 88156, got {len(rows)}")
    return rows, hashes


def _candidate_and_source_plan(config: dict, p0p1: dict) -> tuple[dict, dict, dict]:
    candidate_path = Path("cache/p0-source/candidate-grid-audit.json")
    official_index_path = Path("cache/p0-source/official-index.json")
    official_texts_path = Path("cache/p0-source/official-texts.json")
    candidate = read_json(candidate_path)
    if (len(candidate.get("speakers", [])), len(candidate.get("prompts", []))) != (63, 18):
        raise ValueError("P2 audio authorization is limited to the verified 63 x 18 candidate grid")
    expected_candidate_sha = p0p1["candidate_grid"]["sha256"]
    if file_sha(candidate_path) != expected_candidate_sha:
        raise ValueError("P0 candidate-grid source manifest changed")
    official = p0p1["source"]["official"]
    if official.get("license") != "CC BY 4.0" or official.get("release") != "VCTK 0.92 September 2019":
        raise ValueError("official VCTK release or license evidence changed")
    if official.get("archive_bytes") != 11_747_302_977 or official.get("archive_published_md5_etag") != "8a6ba2946b36fcbef0212cad601f4bfa":
        raise ValueError("official archive size or published ETag/MD5 differs from P0 evidence")

    official_index = json.loads(official_index_path.read_bytes())
    official_texts = json.loads(official_texts_path.read_bytes())
    mirror_rows, mirror_hashes = _load_mirror_metadata()
    cross_audit = verify_candidate_metadata(candidate, official_index, official_texts, mirror_rows)
    plan = build_official_member_plan(
        candidate,
        official_index,
        official_texts,
        max_cells=int(config["resources"]["P2_max_audio_cells"]),
        max_range_bytes=int(config["resources"]["P2_max_audio_payload_bytes"]),
    )
    source = {
        "official": {
            "release": official["release"],
            "license": official["license"],
            "doi": official["doi"],
            "item_id": official["item_id"],
            "bitstream_id": official["bitstream_id"],
            "archive_bytes": official["archive_bytes"],
            "etag_md5": official["archive_published_md5_etag"],
            "audio_download_source": "official VCTK ZIP; bounded HTTP Range requests",
        },
        "mirror": {
            "repository": "thelarryproject/vctk",
            "revision": p0p1["source"]["thelarryproject/vctk"]["revision"],
            "producer_revision": p0p1["source"]["thelarryproject/vctk"]["producer_revision"],
            "audio_parity": "UNVERIFIED; no mirror audio is used",
            "metadata_shards_sha256": mirror_hashes,
        },
        "candidate_grid_sha256": file_sha(candidate_path),
        "official_index_sha256": file_sha(official_index_path),
        "official_texts_sha256": file_sha(official_texts_path),
        "cross_audit": cross_audit,
        "plan_sha256": digest_bytes(canonical_json(plan)),
        "candidate_cells": plan["candidate_cells"],
        "planned_range_bytes": plan["planned_range_bytes"],
        "planned_flac_bytes": plan["planned_flac_bytes"],
    }
    return plan, source, {"official_index": official_index, "official_texts": official_texts}


def validate_locked_config(config: dict) -> bool:
    """Reject any P2 config drift in source, split, model, probes, quantizer, stats, or gates."""
    if (config.get("study_id"), config.get("protocol_version"), config.get("amendment"),
            config.get("parent_study_id")) != (STUDY_ID, PROTOCOL_VERSION, "Q-002",
                                               "speech-quant-diagnostic-20261122-v2"):
        raise ValueError("P2 requires the predeclared v3 study/version and v2 parent")
    data = config["data"]
    if (data.get("corpus"), data.get("microphone"), data.get("min_speakers"), data.get("max_speakers"),
            data.get("min_prompts"), data.get("max_prompts"), data.get("duration_seconds"),
            data.get("sample_rate_hz"), data.get("split"), data.get("split_seed"), data.get("official_url"),
            data.get("candidate_hf_mirrors")) != (
        "vctk_0.92", "mic1", 40, 64, 16, 24, [1.0, 10.0], 16000,
        {"train": 0.6, "dev": 0.2, "test": 0.2}, 20261122,
        "https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928/full",
        ["thelarryproject/vctk", "kth-tmh/vctk"],
    ):
        raise ValueError("registered G0 source, grid, duration, resampling, or split setting changed")
    support = {key: data.get(key) for key in (
        "train_min_per_speaker", "dev_min_per_speaker", "test_min_per_speaker",
        "train_min_per_prompt", "dev_min_per_prompt", "test_min_per_prompt",
    )}
    if support != {
        "train_min_per_speaker": 8, "dev_min_per_speaker": 3, "test_min_per_speaker": 3,
        "train_min_per_prompt": 12, "dev_min_per_prompt": 5, "test_min_per_prompt": 5,
    }:
        raise ValueError("registered split-support minimums changed")
    model = config["model"]
    if (model.get("model_id"), model.get("revision"), model.get("processor_revision"),
            model.get("expected_transformer_blocks"), model.get("pooling"), model.get("freeze_encoder")) != (
        "microsoft/wavlm-base-plus", "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "4c66d4806a428f2e922ccfa1a962776e232d487b", 12,
        "mask_aware_mean_of_last_four_hidden_layers", True,
    ):
        raise ValueError("registered WavLM model/processor revision, pooling, or freeze state changed")
    probes = config["probes"]
    if (probes.get("kind"), probes.get("C_grid"), probes.get("macro_f1_min"),
            probes.get("min_above_majority_macro_f1"), probes.get("heads_train_only_on_full_precision")) != (
        "standard_scaler_multinomial_logistic_regression", [0.01, 0.1, 1.0, 10.0], 0.55, 0.20, True,
    ):
        raise ValueError("registered frozen-probe architecture, C grid, or G1 thresholds changed")
    quant = config["quantization"]
    expected_eligible = [
        "attention.q_proj.weight", "attention.k_proj.weight", "attention.v_proj.weight",
        "attention.out_proj.weight", "feed_forward.intermediate_dense.weight", "feed_forward.output_dense.weight",
    ]
    if (quant.get("kind"), quant.get("eligible"), quant.get("excluded_block_linear"), quant.get("group_size"),
            quant.get("bits"), quant.get("scale_dtype"), quant.get("primary_budget_fraction"),
            quant.get("budgets_fraction_between_uniform4_and_uniform8"),
            quant.get("budget_cost_tolerance_percent"), quant.get("bias_layernorm_conv_excluded")) != (
        "symmetric_groupwise_weight_only_fake_quant", expected_eligible, "attention.gru_rel_pos_linear.weight",
        64, [4, 8], "float16", 0.50, [0.25, 0.50, 0.75], 0.5, True,
    ):
        raise ValueError("Q-001 projection coverage, groupwise quantizer, bit widths, or budget changed")
    analysis = config["analysis"]
    if (analysis.get("bootstrap_kind"), analysis.get("bootstrap_factor_distribution"),
            analysis.get("bootstrap_cell_weight"), analysis.get("bootstrap_factor_normalization"),
            analysis.get("bootstrap_repetitions"), analysis.get("bootstrap_seed"),
            analysis.get("confidence_level"), analysis.get("bootstrap_interval"),
            analysis.get("bootstrap_streams"), analysis.get("bootstrap_paired_conditions_share_weights"),
            analysis.get("bootstrap_spec_sha256")) != (
        "positive_weight_crossed_factor_multiplier", {"name": "gamma", "shape": 1.0, "scale": 1.0},
        "speaker_weight_times_prompt_weight", "mean_one_per_replicate", 2000, 20261124, 0.95,
        "percentile", ["dev", "test"], True, BOOTSTRAP_SPEC_SHA256,
    ):
        raise ValueError("adopted Q-002 statistical specification changed")
    expected_gates = {
        "G2_min_uniform4_drop_any_task": 0.03,
        "G2_max_uniform8_drop_each_task": 0.05,
        "G3_max_spearman": 0.35,
        "G3_max_top4_overlap": 2,
        "G3_min_layers_with_cross_task_drop_difference": 3,
        "G3_min_cross_task_drop_difference": 0.03,
        "G3_max_spearman_bootstrap_upper_CI": 0.70,
        "G4_min_dev_headroom_gain": 0.02,
        "G4_test_CI_lower_must_exceed": 0.0,
        "G5_manual_novelty_inspection": True,
    }
    if config.get("gates") != expected_gates:
        raise ValueError("registered G2 threshold or G3/G4/G5 gate changed")
    resources = config["resources"]
    if (resources.get("P2_max_audio_cells"), resources.get("P2_max_audio_payload_bytes"),
            resources.get("P2_max_local_decoded_bytes"), resources.get("P2_min_free_disk_bytes"),
            resources.get("prohibit_full_corpus_download_without_owner_approval"),
            resources.get("stage_stop_after")) != (
        1134, 2 * 1024**3, 4 * 1024**3, 5 * 1024**3, True, "P2",
    ):
        raise ValueError("approved P2 scope or resource ceiling changed")
    return True


def _sampled_os_state() -> dict:
    usage = shutil.disk_usage(".")
    return {
        "disk_total_bytes": usage.total,
        "disk_used_bytes": usage.total - usage.free,
        "disk_free_bytes": usage.free,
        "host_os": platform.platform(),
        "host_arch": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def _write_or_verify_sealed_partition(rows: list[dict], split_manifest: dict, staging: Path) -> dict:
    by_id = {row["id"]: row for row in rows}
    split_by_id = {row["id"]: row["partition"] for row in split_manifest["rows"]}
    if len(split_by_id) != len(split_manifest["rows"]):
        raise ValueError("duplicate IDs in split manifest")
    train_dir = DATA_DIR / "train_dev_audio"
    test_dir = DATA_DIR / "sealed_test_audio"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    # A prior interrupted G0 move is recoverable only through this G0-only path.
    test_dir.chmod(0o700)
    test_files = []
    train_files = []
    for clip_id, partition in sorted(split_by_id.items()):
        if clip_id not in by_id:
            raise ValueError(f"split manifest references missing G0 row {clip_id}")
        source = staging / f"{clip_id}.flac"
        destination_dir = train_dir if partition in {"train", "dev"} else test_dir
        destination = destination_dir / f"{clip_id}.flac"
        record = by_id[clip_id]
        expected = record["audio_sha256"]
        if source.is_file():
            if file_sha(source) != expected:
                raise ValueError(f"G0 staging audio changed before partitioning: {clip_id}")
            if destination.exists() and file_sha(destination) != expected:
                raise ValueError(f"partitioned audio changed during G0 resume: {clip_id}")
            if not destination.exists():
                os.replace(source, destination)
        elif not destination.is_file() or file_sha(destination) != expected:
            raise ValueError(f"G0 resume cannot locate verified audio: {clip_id}")
        if partition == "test":
            destination.chmod(0)
            test_files.append(destination)
        else:
            train_files.append(destination)
    test_dir.chmod(0)
    return {
        "model_accessible_audio_directory": str(train_dir),
        "sealed_test_audio_directory": str(test_dir),
        "train_dev_audio_files": len(train_files),
        "sealed_test_audio_files": len(test_files),
        "sealed_test_files_mode": "000",
        "test_audio_model_forwards": 0,
        "test_prediction_reads": 0,
    }


def run_g0(*, out: Path) -> dict:
    started = time.monotonic()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        previous = read_json(out)
        body = {key: value for key, value in previous.items() if key != "stage_sha256"}
        if previous.get("stage_sha256") != digest_bytes(canonical_json(body)):
            raise ValueError("existing G0 stage artifact failed its embedded SHA256")
        print(f"G0 replay verified: {out}", flush=True)
        return previous

    result = {
        "stage": "G0",
        "study_id": STUDY_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": "BLOCKED_INVALID",
        "test_audio_model_forwards": 0,
        "test_prediction_reads": 0,
        "test_model_feature_rows": 0,
        "test_outcome_metric_rows": 0,
    }
    try:
        commit = _code_commit()
        config = yaml.safe_load(CONFIG.read_bytes())
        validate_locked_config(config)
        p0p1 = read_json(P0_P1_STATUS)
        historical = verify_prior_evidence()
        plan, source, _ = _candidate_and_source_plan(config, p0p1)
        result.update({"code_commit": commit, "source": source,
                       "candidate_cells": plan["candidate_cells"],
                       "planned_range_bytes": plan["planned_range_bytes"]})
        local_upper_bound = plan_local_upper_bound(
            plan,
            max_duration_seconds=float(config["data"]["duration_seconds"][1]),
            sample_rate_hz=int(config["data"]["sample_rate_hz"]),
            bytes_per_sample=4,
        )
        before = _sampled_os_state()
        limits = preflight_resource_limits(
            config["resources"],
            cells=plan["candidate_cells"],
            range_bytes=plan["planned_range_bytes"],
            local_bytes=local_upper_bound,
            free_disk_bytes=before["disk_free_bytes"],
        )
        result.update({"resource_preflight": limits, "resources": {"before_download": before,
                       "model_forwards": 0, "test_audio_model_forwards": 0, "test_prediction_reads": 0}})
        print(
            f"G0 source preflight: {plan['candidate_cells']} cells, "
            f"{plan['planned_range_bytes']} range bytes, "
            f"{local_upper_bound} upper-bound local bytes; beginning bounded official ranges",
            flush=True,
        )
        staging = DATA_DIR / "g0_staging_audio"
        receipt_dir = RUN_DIR / "g0" / "download_receipts"
        transfer = materialize_official(
            plan,
            staging,
            receipt_dir,
            max_payload_bytes=int(config["resources"]["P2_max_audio_payload_bytes"]),
            batch_size=16,
        )
        accepted, rejected, audit_resources = audit_audio_files(
            plan["rows"], staging,
            max_local_bytes=int(config["resources"]["P2_max_local_decoded_bytes"]),
        )
        split_manifest = make_split_manifest(accepted, seed=int(config["data"]["split_seed"]))
        prior_p1_isolation = verify_historical_micro_test_disjoint(
            read_json(Path("runs/P0_Q001/micro-plan.json")), split_manifest,
        )
        accepted_by_id = {row["id"]: row for row in accepted}
        selected_ids = {row["id"] for row in split_manifest["rows"]}
        final_rows = [accepted_by_id[clip_id] for clip_id in sorted(selected_ids)]
        if len(split_manifest["speakers"]) < int(config["data"]["min_speakers"]):
            raise ValueError("G0_FAIL: complete eligible grid has fewer than 40 speakers")
        if len(split_manifest["prompts"]) < int(config["data"]["min_prompts"]):
            raise ValueError("G0_FAIL: complete eligible grid has fewer than 16 shared prompts")
        if len(final_rows) != int(split_manifest["cells"]):
            raise ValueError("G0_FAIL: split manifest does not cover the complete selected grid")

        final_audio_state = _write_or_verify_sealed_partition(accepted, split_manifest, staging)
        after = _sampled_os_state()
        if after["disk_free_bytes"] < int(config["resources"]["P2_min_free_disk_bytes"]):
            raise ValueError("RESOURCE_OVERRIDE_REQUIRED: disk safety margin fell below 5 GiB after G0")
        if audit_resources["combined_local_audio_bytes"] > int(config["resources"]["P2_max_local_decoded_bytes"]):
            raise ValueError("RESOURCE_OVERRIDE_REQUIRED: decoded/extracted bytes exceeded P2 cap")

        manifest_dir = Path("manifests/v3")
        grid_rows = [
            {"id": row["id"], "speaker": row["speaker"], "prompt": row["prompt"], "mic": row["mic"],
             "transcript_sha256": row["transcript_sha256"], "audio_sha256": row["audio_sha256"],
             "decoded_pcm16k_sha256": row["decoded_pcm16k_sha256"],
             "duration_seconds": f"{row['duration_seconds']:.9f}"}
            for row in final_rows
        ]
        grid_sha = write_csv_atomic(
            manifest_dir / "grid.csv",
            ["id", "speaker", "prompt", "mic", "transcript_sha256", "audio_sha256",
             "decoded_pcm16k_sha256", "duration_seconds"],
            grid_rows,
        )
        split_rows = [
            {"id": row["id"], "speaker": row["speaker"], "prompt": row["prompt"],
             "partition": row["partition"], "audio_sha256": row["audio_sha256"],
             "decoded_pcm16k_sha256": row["decoded_pcm16k_sha256"]}
            for row in split_manifest["rows"]
        ]
        split_csv_sha = write_csv_atomic(
            manifest_dir / "splits.csv",
            ["id", "speaker", "prompt", "partition", "audio_sha256", "decoded_pcm16k_sha256"],
            split_rows,
        )
        label_maps = {
            "study_id": STUDY_ID,
            "speaker_labels": split_manifest["speakers"],
            "prompt_labels": split_manifest["prompts"],
        }
        label_sha = write_json_atomic(manifest_dir / "label_maps.json", label_maps)
        split_path = RUN_DIR / "g0" / "split_manifest.json"
        split_sha = write_json_atomic(split_path, split_manifest)
        source_path = RUN_DIR / "g0" / "source_plan.json"
        source_plan_sha = write_json_atomic(source_path, {"source": source, "plan": plan,
                                                           "P0_evidence": historical})
        audio_path = RUN_DIR / "g0" / "audio_integrity.json"
        stable_transfer = {
            key: transfer[key] for key in (
                "plan_sha256", "planned_range_bytes", "reserved_range_bytes_total", "actual_range_bytes",
                "compressed_audio_payload_bytes", "extracted_flac_bytes", "total_verified_cells", "max_payload_bytes"
            )
        }
        audio_sha = write_json_atomic(audio_path, {
            "accepted": accepted,
            "rejected": rejected,
            "audit_resources": audit_resources,
            "transfer": stable_transfer,
            "source_audio_sha256_unique": audit_resources["audio_sha256_unique"],
            "decoded_pcm_sha256_unique": audit_resources["decoded_pcm_sha256_unique"],
        })
        ids_path = RUN_DIR / "g0" / "ids_and_hashes.json"
        ids_sha = write_json_atomic(ids_path, {
            "candidate_ids": sorted(row["id"] for row in plan["rows"]),
            "candidate_ids_sha256": digest_bytes(canonical_json(sorted(row["id"] for row in plan["rows"]))),
            "selected_grid_ids": [row["id"] for row in split_rows],
            "selected_grid_ids_sha256": digest_bytes(canonical_json([row["id"] for row in split_rows])),
            "split_manifest_sha256": split_manifest["sha256"],
            "train_dev_model_ids": sorted(row["id"] for row in split_rows if row["partition"] in {"train", "dev"}),
            "test_ids_for_integrity_only": sorted(row["id"] for row in split_rows if row["partition"] == "test"),
        })

        result.update({
            "status": "PASS",
            "G0": "PASS",
            "code_commit": commit,
            "verdict_if_stopped_here": "P2_CONTINUE_G1",
            "source": source,
            "resource_preflight": limits,
            "transfer": stable_transfer,
            "audio_integrity": {
                "candidate_files_checked": audit_resources["candidate_files_checked"],
                "accepted_candidate_cells": len(accepted),
                "rejected_candidate_cells": len(rejected),
                "reject_counts_by_reason": {reason: sum(row["reason"].startswith(reason) for row in rejected)
                                            for reason in sorted({row["reason"].split(":")[0] for row in rejected})},
                "final_speakers": len(split_manifest["speakers"]),
                "final_prompts": len(split_manifest["prompts"]),
                "final_cells": split_manifest["cells"],
                "grid_selection_priority": ["unique_cells", "speakers", "prompts", "lexicographic IDs"],
                "grid_speakers": split_manifest["speakers"],
                "grid_prompts": split_manifest["prompts"],
                "split_sha256": split_manifest["sha256"],
                "support_minima": split_manifest["support_minima"],
                "support_observed": split_manifest["support_observed"],
                "split_counts": {part: sum(row["partition"] == part for row in split_rows)
                                 for part in ("train", "dev", "test")},
                "exact_cell_partition": True,
                "cross_partition_content_duplicates": 0,
                "near_duplicate_microphone_channels": "excluded structurally: mic1 only; mic2 never materialized",
            },
            "test_isolation": final_audio_state,
            "historical_P1_test_overlap": prior_p1_isolation,
            "artifacts": {
                str(manifest_dir / "grid.csv"): grid_sha,
                str(manifest_dir / "splits.csv"): split_csv_sha,
                str(manifest_dir / "label_maps.json"): label_sha,
                str(split_path): split_sha,
                str(source_path): source_plan_sha,
                str(audio_path): audio_sha,
                str(ids_path): ids_sha,
            },
            "resources": {
                "before_download": before,
                "after_g0": after,
                "disk_bytes_consumed": max(0, after["disk_used_bytes"] - before["disk_used_bytes"]),
                "audio_transfer_range_bytes": transfer["actual_range_bytes"],
                "audio_payload_bytes": transfer["compressed_audio_payload_bytes"],
                "extracted_flac_bytes": audit_resources["extracted_flac_bytes"],
                "decoded_pcm_bytes": audit_resources["decoded_pcm_bytes"],
                "combined_local_audio_bytes": audit_resources["combined_local_audio_bytes"],
                "peak_rss_bytes": after["peak_rss_bytes"],
                "wall_seconds": round(time.monotonic() - started, 3),
                "model_forwards": 0,
                "test_audio_model_forwards": 0,
                "test_prediction_reads": 0,
            },
        })
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        if "RESOURCE_OVERRIDE_REQUIRED" in str(error):
            result.update(status="RESOURCE_OVERRIDE_REQUIRED", G0="NOT_PASSED", blocker=message)
        elif str(error).startswith("G0_FAIL:") or "No complete grid meets" in str(error) or "partition fails predeclared" in str(error):
            result.update(status="NO_GO_CURRENT_DESIGN", G0="FAIL", blocker=message)
        else:
            result.update(status="BLOCKED_INVALID", G0="BLOCKED", blocker=message)
        result["resources"] = {
            "wall_seconds": round(time.monotonic() - started, 3),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "model_forwards": 0,
            "test_audio_model_forwards": 0,
            "test_prediction_reads": 0,
        }
    body = dict(result)
    body["stage_sha256"] = digest_bytes(canonical_json(body))
    write_json_atomic(out, body)
    print(f"G0 {result['status']}; artifact={out}", flush=True)
    return body


def _verify_stage(path: Path, stage_name: str) -> dict:
    stage = read_json(path)
    body = {key: value for key, value in stage.items() if key != "stage_sha256"}
    if stage.get("stage") != stage_name or stage.get("stage_sha256") != digest_bytes(canonical_json(body)):
        raise ValueError(f"{stage_name} artifact is absent, mismatched, or failed SHA256 verification")
    return stage


def _code_commit() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "sqpilot", "tests", "configs/pilot.yaml", "docs/01_protocol.md",
         "docs/04_analysis_and_gates.md", "docs/amendments/Q001_SCOPE.md", "docs/amendments/Q002_BOOTSTRAP.md",
         "tools"],
        text=True,
    ).strip()
    if status:
        raise ValueError("P2 implementation or protocol files changed after the frozen code commit")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if len(commit) != 40:
        raise ValueError("cannot identify frozen P2 code commit")
    return commit


def _assert_code_unchanged_since(reference_commit: str, current_commit: str) -> None:
    completed = subprocess.run(
        ["git", "diff", "--quiet", reference_commit, current_commit, "--", "sqpilot", "tests",
         "configs/pilot.yaml", "docs/01_protocol.md", "docs/04_analysis_and_gates.md",
         "docs/amendments/Q001_SCOPE.md", "docs/amendments/Q002_BOOTSTRAP.md", "tools"],
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("P2 code or protocol changed after its G0 frozen commit")


def _g0_inputs() -> tuple[dict, dict, dict, list[dict], dict]:
    g0 = _verify_stage(RUN_DIR / "g0" / "g0.json", "G0")
    if g0.get("G0") != "PASS" or g0.get("status") != "PASS":
        raise ValueError("G0 did not pass; G1 is forbidden")
    for relative_path, expected_sha in g0.get("artifacts", {}).items():
        path = Path(relative_path)
        if not path.is_file() or file_sha(path) != expected_sha:
            raise ValueError(f"G0 artifact changed after its integrity freeze: {relative_path}")
    integrity = read_json(RUN_DIR / "g0" / "audio_integrity.json")
    split = read_json(RUN_DIR / "g0" / "split_manifest.json")
    if split.get("sha256") != g0["audio_integrity"]["split_sha256"]:
        raise ValueError("G0 split hash differs from the status artifact")
    from sqpilot.p2_data import verify_manifest_hash
    verify_manifest_hash(split, split["sha256"])
    model_input_rows = model_rows(integrity["accepted"], split)
    if not model_input_rows or any(row["partition"] not in {"train", "dev"} for row in model_input_rows):
        raise ValueError("sealed test rows reached the P2 modeling input")
    test_ids = {row["id"] for row in split["rows"] if row["partition"] == "test"}
    model_ids = {row["id"] for row in model_input_rows}
    if test_ids & model_ids:
        raise ValueError("sealed test IDs overlap the P2 modeling input")
    audio_dir = DATA_DIR / "train_dev_audio"
    expected_ids = {row["id"] for row in model_input_rows}
    actual_ids = {path.stem for path in audio_dir.glob("*.flac")}
    if actual_ids != expected_ids:
        raise ValueError("model-accessible audio directory has extra/missing clip IDs")
    test_dir = DATA_DIR / "sealed_test_audio"
    if test_dir.stat().st_mode & 0o777 != 0:
        raise ValueError("sealed test audio directory permissions are not mode 000")
    for clip_id in test_ids:
        if (audio_dir / f"{clip_id}.flac").exists():
            raise ValueError("sealed test audio is present in the model-accessible directory")
    g0_ids = {row["id"]: row for row in integrity["accepted"]}
    for row in model_input_rows:
        local = audio_dir / f"{row['id']}.flac"
        if not local.is_file() or file_sha(local) != row["audio_sha256"]:
            raise ValueError(f"train/dev audio failed frozen SHA256 replay: {row['id']}")
        if row["id"] not in g0_ids:
            raise ValueError("train/dev model input is not in the G0 integrity records")
    return g0, integrity, split, model_input_rows, {"audio_dir": audio_dir, "test_ids": sorted(test_ids),
            "model_ids": sorted(model_ids), "split": split}


def _load_frozen_model(config: dict, historic: dict, monitor) -> tuple[torch.nn.Module, object, tuple, list[dict], str, str]:
    torch.set_num_threads(4)
    torch.manual_seed(int(config["data"]["split_seed"]))
    checkpoint = check_model_files(config, historic)
    if not torch.backends.mps.is_available():
        # P1 separately validated CPU/FP16. Preserve FP16 precision and report the actual backend.
        device = "cpu"
    else:
        device = "mps"
    dtype = torch.float16
    model = load_model(checkpoint, dtype, device)
    extractor = processor()
    expected = read_json(Path("runs/P0/model.json"))["tensor_shapes"]
    selection, inventory = discover(model, config, expected)
    if len(selection) != 12 or sum(len(group) for group in selection) != 72:
        raise ValueError("runtime model inventory does not contain the exact 72 eligible Q-001 matrices")
    monitor.sample()
    return model, extractor, selection, inventory, device, str(dtype).removeprefix("torch.")


def _read_verified_signal(row: dict, audio_dir: Path, monitor) -> np.ndarray:
    path = audio_dir / f"{row['id']}.flac"
    payload = path.read_bytes()
    monitor.audio_reads += 1
    if digest_bytes(payload) != row["audio_sha256"]:
        raise ValueError(f"audio changed after G0: {row['id']}")
    signal, duration = preprocess(payload)
    pcm_sha = digest_bytes(np.ascontiguousarray(signal, dtype=np.float32).tobytes())
    if pcm_sha != row["decoded_pcm16k_sha256"] or abs(duration - row["duration_seconds"]) > 1e-9:
        raise ValueError(f"decoded waveform changed after G0: {row['id']}")
    monitor.sample()
    return signal


class _StageMonitor:
    def __init__(self):
        self.start = time.monotonic()
        self.model_forwards = 0
        self.audio_reads = 0
        self.mps_memory_bytes = 0
        self.minimum_disk_free_bytes = shutil.disk_usage(".").free

    def sample(self):
        self.minimum_disk_free_bytes = min(self.minimum_disk_free_bytes, shutil.disk_usage(".").free)
        if torch.backends.mps.is_available():
            try:
                self.mps_memory_bytes = max(self.mps_memory_bytes, int(torch.mps.driver_allocated_memory()))
            except (RuntimeError, AttributeError):
                pass

    def feature(self, model, extractor, signal, **kwargs):
        self.model_forwards += 1
        value = feature(model, extractor, signal, **kwargs)
        self.sample()
        return value

    def result(self) -> dict:
        self.sample()
        return {
            "wall_seconds": round(time.monotonic() - self.start, 3),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "sampled_peak_mps_driver_bytes": self.mps_memory_bytes,
            "mps_memory_is_reported_separately_from_rss": True,
            "minimum_disk_free_bytes": self.minimum_disk_free_bytes,
            "model_forwards": self.model_forwards,
            "train_dev_audio_reads": self.audio_reads,
            "test_audio_model_forwards": 0,
            "test_prediction_reads": 0,
        }


def _fp_cache_key(g0: dict, model_id: str, device: str, precision: str, commit: str) -> str:
    return stable_hash({
        "study_id": STUDY_ID,
        "stage": "fp_reference_pooled_features_train_dev",
        "g0_stage_sha256": g0["stage_sha256"],
        "split_sha256": g0["audio_integrity"]["split_sha256"],
        "model_revision": model_id,
        "processor_revision": model_id,
        "checkpoint_sha256": LFS_SHA256,
        "wavlm_source_sha256": SOURCE_HASH,
        "pooling": "attention-mask-aware mean of last four hidden states",
        "precision": precision,
        "device": device,
        "code_commit": commit,
    })


def _feature_matrix(model, extractor, rows, audio_dir, monitor, *, label):
    outputs = []
    for index, row in enumerate(rows, start=1):
        signal = _read_verified_signal(row, audio_dir, monitor)
        outputs.append(monitor.feature(model, extractor, signal))
        if index % 50 == 0 or index == len(rows):
            print(f"{label} pooled features: {index}/{len(rows)}", flush=True)
    result = np.asarray(np.stack(outputs), dtype=np.float32)
    if result.shape != (len(rows), 768) or not np.isfinite(result).all():
        raise ValueError("P2 pooled WavLM feature matrix changed shape or contains non-finite values")
    return result


def _prediction_artifact(path: Path, arrays: dict[str, np.ndarray]) -> str:
    return _atomic_npz(path, arrays)


def run_g1(*, out: Path) -> dict:
    if out.exists():
        stage = _verify_stage(out, "G1")
        print(f"G1 replay verified: {out}", flush=True)
        return stage
    monitor = _StageMonitor()
    result = {"stage": "G1", "study_id": STUDY_ID, "protocol_version": PROTOCOL_VERSION,
              "status": "BLOCKED_INVALID", "G1": "BLOCKED", "test_audio_model_forwards": 0,
              "test_prediction_reads": 0}
    try:
        commit = _code_commit()
        config = yaml.safe_load(CONFIG.read_bytes())
        validate_locked_config(config)
        g0, integrity, split, rows, isolation = _g0_inputs()
        _assert_code_unchanged_since(g0["code_commit"], commit)
        historical = read_json(P0_V1_STATUS)["sha256"]["ignored_local_evidence"]
        model, extractor, selection, inventory, device, precision = _load_frozen_model(config, historical, monitor)
        if precision != "float16":
            raise ValueError("G1 reference precision must remain FP16 under the validated P1 protocol")
        baseline_fingerprint = fingerprint(model)
        first_signal = _read_verified_signal(rows[0], isolation["audio_dir"], monitor)
        baseline = monitor.feature(model, extractor, first_signal)
        repeat = monitor.feature(model, extractor, first_signal)
        repeat_max = check_close(baseline, repeat, rtol=1e-4, atol=2e-5, check="FP16 repeat")
        padded = monitor.feature(model, extractor, first_signal, pad=640)
        sample_padding_max = check_close(baseline, padded, rtol=0.01, atol=0.008, check="FP16 sample padding")
        monitor.model_forwards += 1
        frame_padded = padded_frames(model, extractor, first_signal, 5)
        monitor.sample()
        frame_padding_max = check_close(baseline, frame_padded, rtol=0.01, atol=0.008,
                                        check="FP16 transformer padding")
        if fingerprint(model) != baseline_fingerprint:
            raise ValueError("FP model weights changed during G1 pooling checks")
        result["padding_checks"] = {"repeat_max_abs": repeat_max,
                                    "sample_padding_max_abs": sample_padding_max,
                                    "transformer_padding_max_abs": frame_padding_max,
                                    "reference_weight_fingerprint_sha256": stable_hash(baseline_fingerprint)}

        key = _fp_cache_key(g0, config["model"]["revision"], device, precision, commit)
        feature_path = RUN_DIR / "g1" / "features_train_dev_fp16.npz"
        clip_ids = [row["id"] for row in rows]
        features, cache_hit = cached_features(
            feature_path, key, clip_ids,
            lambda: _feature_matrix(model, extractor, rows, isolation["audio_dir"], monitor, label="G1 FP16"),
        )
        replay, replay_hit = cached_features(feature_path, key, clip_ids,
                                             lambda: (_ for _ in ()).throw(AssertionError("cache replay computed features")))
        if not replay_hit or not np.array_equal(features, replay):
            raise ValueError("FP train/dev feature cache replay failed")
        result["feature_cache"] = {"path": str(feature_path), "key_sha256": key,
                                   "features_sha256": array_sha(features), "file_sha256": file_sha(feature_path),
                                   "cache_hit_before_run": cache_hit, "replay_hit": replay_hit,
                                   "ordered_clip_ids_sha256": digest_bytes(canonical_json(clip_ids))}

        label_maps = read_json(Path("manifests/v3/label_maps.json"))
        labels_by_task = {
            "speaker": label_maps["speaker_labels"],
            "prompt": label_maps["prompt_labels"],
        }
        heads = {}
        dev_indices = [index for index, row in enumerate(rows) if row["partition"] == "dev"]
        if any(row["partition"] == "test" for row in rows) or any(rows[index]["id"] in isolation["test_ids"] for index in dev_indices):
            raise ValueError("sealed test rows reached G1 prediction or C selection")
        prediction_arrays = {"ids": np.asarray([rows[index]["id"] for index in dev_indices], dtype=np.str_)}
        for task in ("speaker", "prompt"):
            head_path = RUN_DIR / "g1" / "heads" / f"{task}.npz"
            head_result = fit_frozen_probe(features, rows, task=task, labels=labels_by_task[task], out=head_path)
            heads[task] = head_result
            result.setdefault("probe_training", {})[task] = head_result["metadata"]
            prediction_arrays[f"{task}_true"] = np.asarray([rows[index][task] for index in dev_indices], dtype=np.str_)
            prediction_arrays[f"{task}_pred"] = np.asarray(head_result["prediction"], dtype=np.str_)
        prediction_path = RUN_DIR / "g1" / "dev_predictions.npz"
        prediction_sha = _prediction_artifact(prediction_path, prediction_arrays)

        metrics = {task: heads[task]["metadata"]["development_metrics"] for task in ("speaker", "prompt")}
        gate = g1_gate(metrics, config["probes"])
        result.update({
            "status": "PASS" if gate["pass"] else "NO_GO_CURRENT_DESIGN",
            "G1": "PASS" if gate["pass"] else "FAIL",
            "gate": gate,
            "study_id": STUDY_ID,
            "protocol_version": PROTOCOL_VERSION,
            "code_commit": commit,
            "data_split_sha256": split["sha256"],
            "g0_stage_sha256": g0["stage_sha256"],
            "model_revision": config["model"]["revision"],
            "processor_revision": config["model"]["processor_revision"],
            "checkpoint_sha256": LFS_SHA256,
            "wavlm_source_sha256": SOURCE_HASH,
            "device": device,
            "reference_precision": precision,
            "feature_cache": {"path": str(feature_path), "key_sha256": key,
                              "features_sha256": array_sha(features), "file_sha256": file_sha(feature_path),
                              "cache_hit_before_run": cache_hit, "replay_hit": replay_hit,
                              "ordered_clip_ids_sha256": digest_bytes(canonical_json(clip_ids))},
            "padding_checks": {"repeat_max_abs": repeat_max, "sample_padding_max_abs": sample_padding_max,
                               "transformer_padding_max_abs": frame_padding_max,
                               "reference_weight_fingerprint_sha256": stable_hash(baseline_fingerprint)},
            "probe_training": {
                task: {"C_selection": heads[task]["metadata"]["candidate_development_macro_f1"],
                       "selected_C": heads[task]["metadata"]["selected_C"],
                       "head_sha256": heads[task]["metadata"]["head_sha256"],
                       "probe_file_sha256": heads[task]["metadata"]["probe_file_sha256"],
                       "train_class_support": heads[task]["metadata"]["train_class_support"],
                       "development_class_support": heads[task]["metadata"]["development_class_support"],
                       "development_prediction_sha256": heads[task]["metadata"]["development_prediction_sha256"],
                       "train_samples": heads[task]["metadata"]["train_samples"],
                       "development_samples": heads[task]["metadata"]["development_samples"],
                       "solver": heads[task]["metadata"]["solver"],
                       "max_iter": heads[task]["metadata"]["max_iter"],
                       "tolerance": heads[task]["metadata"]["tolerance"],
                       "metrics": heads[task]["metadata"]["development_metrics"]}
                for task in ("speaker", "prompt")
            },
            "dev_prediction_artifact": {"path": str(prediction_path), "sha256": prediction_sha,
                                        "samples": len(dev_indices)},
            "model_inventory": {"eligible_matrices": 72, "excluded_gate_matrices": 12,
                                "total_block_linears": 84, "inventory_sha256": stable_hash(inventory)},
            "test_isolation": {"test_ids_in_model_rows": 0, "test_audio_model_forwards": 0,
                               "test_prediction_reads": 0, "model_accessible_ids": len(isolation["model_ids"])},
            "resources": monitor.result(),
        })
    except Exception as error:
        result.update({"status": "BLOCKED_INVALID", "G1": "BLOCKED",
                       "blocker": f"{type(error).__name__}: {error}", "resources": monitor.result()})
    body = dict(result)
    body["stage_sha256"] = digest_bytes(canonical_json(body))
    write_json_atomic(out, body)
    print(f"G1 {result['status']}; artifact={out}", flush=True)
    return body


def _load_head(path: Path, expected_sha: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"frozen probe artifact is unreadable: {path}: {exc}") from exc
    if head_digest(arrays) != expected_sha:
        raise ValueError(f"frozen probe/scaler SHA256 changed: {path}")
    return arrays


def _quantized_dev_features(model, extractor, rows, audio_dir, monitor, *, label):
    outputs = []
    for index, row in enumerate(rows, start=1):
        if row["partition"] != "dev":
            raise ValueError("quantized model input must contain development rows only")
        signal = _read_verified_signal(row, audio_dir, monitor)
        outputs.append(monitor.feature(model, extractor, signal))
        if index % 50 == 0 or index == len(rows):
            print(f"{label} features: {index}/{len(rows)}", flush=True)
    values = np.asarray(np.stack(outputs), dtype=np.float32)
    if values.shape != (len(rows), 768) or not np.isfinite(values).all():
        raise ValueError("quantized dev feature matrix is invalid")
    return values


def run_g2(*, out: Path) -> dict:
    if out.exists():
        stage = _verify_stage(out, "G2")
        print(f"G2 replay verified: {out}", flush=True)
        return stage
    monitor = _StageMonitor()
    result = {"stage": "G2", "study_id": STUDY_ID, "protocol_version": PROTOCOL_VERSION,
              "status": "BLOCKED_INVALID", "G2": "BLOCKED", "test_audio_model_forwards": 0,
              "test_prediction_reads": 0}
    try:
        commit = _code_commit()
        config = yaml.safe_load(CONFIG.read_bytes())
        validate_locked_config(config)
        g0, integrity, split, all_model_rows, isolation = _g0_inputs()
        g1 = _verify_stage(RUN_DIR / "g1" / "g1.json", "G1")
        if g1.get("G1") != "PASS" or g1.get("status") != "PASS":
            raise ValueError("G1 did not pass; G2 is forbidden")
        if g1.get("data_split_sha256") != split["sha256"] or g1.get("g0_stage_sha256") != g0["stage_sha256"]:
            raise ValueError("G1/G0 provenance mismatch before quantized evaluation")
        if g1.get("code_commit") != commit:
            raise ValueError("P2 code commit changed after the frozen G1 probe evaluation")
        historical = read_json(P0_V1_STATUS)["sha256"]["ignored_local_evidence"]
        model, extractor, selection, inventory, device, precision = _load_frozen_model(config, historical, monitor)
        if device != g1["device"] or precision != g1["reference_precision"] or precision != "float16":
            raise ValueError("G2 backend or precision differs from the frozen G1 FP reference")
        dev_rows = [row for row in all_model_rows if row["partition"] == "dev"]
        train_dev_rows = all_model_rows
        if not dev_rows or set(row["id"] for row in dev_rows) & set(isolation["test_ids"]):
            raise ValueError("invalid dev-only G2 row selection")
        train_dev_ids = [row["id"] for row in train_dev_rows]
        fp_key = g1["feature_cache"]["key_sha256"]
        fp_cache_path = Path(g1["feature_cache"]["path"])
        fp_features, fp_cache_hit = cached_features(
            fp_cache_path, fp_key, train_dev_ids,
            lambda: (_ for _ in ()).throw(AssertionError("G2 must replay the frozen G1 FP feature cache")),
        )
        if not fp_cache_hit or array_sha(fp_features) != g1["feature_cache"]["features_sha256"]:
            raise ValueError("G1 FP feature cache failed G2 provenance/replay")
        dev_indices = [index for index, row in enumerate(train_dev_rows) if row["partition"] == "dev"]
        fp_dev_features = fp_features[dev_indices]
        predictions_path = Path(g1["dev_prediction_artifact"]["path"])
        with np.load(predictions_path, allow_pickle=False) as archive:
            frozen_predictions = {key: archive[key].copy() for key in archive.files}
        if file_sha(predictions_path) != g1["dev_prediction_artifact"]["sha256"]:
            raise ValueError("frozen G1 dev prediction artifact checksum changed")
        if frozen_predictions["ids"].tolist() != [row["id"] for row in dev_rows]:
            raise ValueError("frozen G1 dev prediction IDs changed order or support")
        result["frozen_dev_prediction_artifact_sha256"] = g1["dev_prediction_artifact"]["sha256"]
        if frozen_predictions["ids"].tolist() != [row["id"] for row in dev_rows]:
            raise ValueError("frozen G1 dev prediction IDs changed order or support")
        label_maps = read_json(Path("manifests/v3/label_maps.json"))
        expected_labels = {"speaker": label_maps["speaker_labels"], "prompt": label_maps["prompt_labels"]}
        labels = {}
        for task in ("speaker", "prompt"):
            probe_meta = g1["probe_training"][task]
            head_path = RUN_DIR / "g1" / "heads" / f"{task}.npz"
            arrays = _load_head(head_path, probe_meta["head_sha256"])
            if arrays["classes"].tolist() != expected_labels[task]:
                raise ValueError(f"frozen {task} head classes changed from the G0 label map")
            if frozen_predictions[f"{task}_true"].tolist() != [row[task] for row in dev_rows]:
                raise ValueError(f"frozen {task} true labels differ from the dev manifest")
            pred = head_array_predictions(fp_dev_features, arrays)
            if not np.array_equal(pred, frozen_predictions[f"{task}_pred"]):
                raise ValueError("FP dev predictions changed since G1 head freeze")
            expected_pred_sha = array_sha(np.asarray(pred, dtype=np.str_))
            if expected_pred_sha != probe_meta["development_prediction_sha256"]:
                raise ValueError("FP dev prediction hash changed")
            labels[task] = arrays

        # Q-001 cost reconciliation: fixed model parameters plus calculated packed matrices.
        flattened = tuple(name for block_names in selection for name in block_names)
        if len(flattened) != 72 or len(set(flattened)) != 72:
            raise ValueError("Q-001 eligible matrix coverage differs from 72 unique weights")
        baseline_fingerprint = fingerprint(model)
        immutable_master = master_weights(model)
        probe_numeric_bytes = sum(
            int(value.nbytes) for arrays in labels.values() for name, value in arrays.items() if name != "classes"
        )
        model_cost = cost_book(model, selection, torch.float16)
        total_cost = cost_book(model, selection, torch.float16, probe_numeric_bytes)
        p0_costs = read_json(Path("runs/P0_Q001/p0.json"))["model_budget_estimates"]["fp16"]
        cost_summary = {
            "model_only": {
                "B4_bytes": model_cost.total((4,) * 12),
                "B025_bytes": int(model_cost.budget(0.25)),
                "B050_bytes": int(model_cost.budget(0.50)),
                "B075_bytes": int(model_cost.budget(0.75)),
                "B8_bytes": model_cost.total((8,) * 12),
                "fixed_model_parameter_bytes": model_cost.fixed,
                "per_block_uniform4_uniform8_bytes": [
                    {"block": block + 1, "uniform4": costs[0], "uniform8": costs[1]}
                    for block, costs in enumerate(model_cost.blocks)
                ],
            },
            "model_plus_frozen_probe_numeric_arrays": {
                "B4_bytes": total_cost.total((4,) * 12),
                "B050_bytes": int(total_cost.budget(0.50)),
                "B8_bytes": total_cost.total((8,) * 12),
                "fixed_model_and_probe_bytes": total_cost.fixed,
                "probe_numeric_bytes": probe_numeric_bytes,
            },
            "comparison_to_P0_Q001": {
                "all4_exact": model_cost.total((4,) * 12) == p0_costs["all4_bytes"],
                "all8_exact": model_cost.total((8,) * 12) == p0_costs["all8_bytes"],
                "B050_exact": int(model_cost.budget(0.50)) == p0_costs["fractions"]["0.5"]["ceiling_bytes"],
                "reported_primary_B050_bytes": p0_costs["fractions"]["0.5"]["ceiling_bytes"],
            },
            "budgets_monotonic": model_cost.total((4,) * 12) < model_cost.budget(0.25) < model_cost.budget(0.50)
                < model_cost.budget(0.75) < model_cost.total((8,) * 12),
            "accounting_label": "calculated packed bytes; no serialized packed artifact or inference acceleration",
            "layout": {"group_size": 64, "scale_dtype": "float16", "scale_bytes_per_group": 2,
                       "packing": "ceil(group_weights * bits / 8), partial group padded per row",
                       "unused_signed_code": {"4": -8, "8": -128}},
        }
        if not all(cost_summary["comparison_to_P0_Q001"][key] for key in ("all4_exact", "all8_exact", "B050_exact")):
            raise ValueError("calculated packed cost does not reconcile exactly with frozen Q-001 accounting")
        if not cost_summary["budgets_monotonic"]:
            raise ValueError("P2 packed byte budgets are not monotonically ordered")

        fp_metrics = {}
        for task in ("speaker", "prompt"):
            y_true = [row[task] for row in dev_rows]
            pred = frozen_predictions[f"{task}_pred"].tolist()
            train_y = [row[task] for row in train_dev_rows if row["partition"] == "train"]
            fp_metrics[task] = evaluation_metrics(y_true, pred, labels[task]["classes"].tolist(), train_y)
        result["conditions"] = {"FP": {"metrics": fp_metrics}}
        condition_results = {"FP": {"metrics": fp_metrics, "feature_sha256": array_sha(fp_dev_features),
                                     "prediction_source": "G1 frozen train/dev FP cache and prediction artifact"}}
        condition_prediction_arrays = {"ids": np.asarray([row["id"] for row in dev_rows], dtype=np.str_)}
        for task in ("speaker", "prompt"):
            condition_prediction_arrays[f"{task}_true"] = np.asarray([row[task] for row in dev_rows], dtype=np.str_)
            condition_prediction_arrays[f"{task}_FP_pred"] = frozen_predictions[f"{task}_pred"]

        for bits, condition in ((4, "uniform4"), (8, "uniform8")):
            condition_path = RUN_DIR / "g2" / f"features_{condition}_dev.npz"
            condition_weight_hash = None
            with apply_from_master(model, immutable_master, flattened, bits) as weight_stats:
                intervention_fingerprint = fingerprint(model)
                changed = {name for name in intervention_fingerprint
                           if intervention_fingerprint[name] != baseline_fingerprint[name]}
                if changed != set(flattened):
                    raise ValueError(f"{condition} changed parameters outside the 72 Q-001 matrices")
                condition_weight_hash = stable_hash(intervention_fingerprint)
                feature_key = stable_hash({
                    "study_id": STUDY_ID,
                    "stage": condition,
                    "g0_sha256": g0["stage_sha256"],
                    "split_sha256": split["sha256"],
                    "dev_ids": [row["id"] for row in dev_rows],
                    "fp_head_hashes": {task: g1["probe_training"][task]["head_sha256"] for task in labels},
                    "quantizer": config["quantization"],
                    "bits": bits,
                    "weight_fingerprint_sha256": condition_weight_hash,
                    "model_revision": config["model"]["revision"],
                    "checkpoint_sha256": LFS_SHA256,
                    "device": device,
                    "precision": precision,
                    "code_commit": commit,
                })
                values, cache_hit = cached_features(
                    condition_path, feature_key, [row["id"] for row in dev_rows],
                    lambda: _quantized_dev_features(model, extractor, dev_rows, isolation["audio_dir"], monitor,
                                                    label=f"G2 {condition}"),
                )
                cache_replay, replay_hit = cached_features(
                    condition_path, feature_key, [row["id"] for row in dev_rows],
                    lambda: (_ for _ in ()).throw(AssertionError("quantized cache replay computed features")),
                )
                if not replay_hit or not np.array_equal(values, cache_replay):
                    raise ValueError(f"{condition} cached features failed replay")
                repeated = _read_verified_signal(dev_rows[0], isolation["audio_dir"], monitor)
                repeated_feature = monitor.feature(model, extractor, repeated)
                repeat_max = check_close(values[0], repeated_feature, rtol=1e-4, atol=2e-5,
                                         check=f"{condition} same-config replay")
                predictions = {}
                metrics = {}
                for task in ("speaker", "prompt"):
                    predictions[task] = head_array_predictions(values, labels[task]).astype(np.str_)
                    if not np.array_equal(head_array_predictions(cache_replay, labels[task]), predictions[task]):
                        raise ValueError(f"{condition} cached and uncached predictions differ for {task}")
                    metrics[task] = evaluation_metrics(
                        [row[task] for row in dev_rows], predictions[task].tolist(),
                        labels[task]["classes"].tolist(),
                        [row[task] for row in train_dev_rows if row["partition"] == "train"],
                    )
                condition_prediction_arrays[f"speaker_{condition}_pred"] = predictions["speaker"]
                condition_prediction_arrays[f"prompt_{condition}_pred"] = predictions["prompt"]
                condition_results[condition] = {
                    "metrics": metrics,
                    "feature_sha256": array_sha(values),
                    "feature_cache_key_sha256": feature_key,
                    "feature_cache_file_sha256": file_sha(condition_path),
                    "cache_hit_before_run": cache_hit,
                    "cache_replay_hit": replay_hit,
                    "same_config_repeat_max_abs": repeat_max,
                    "changed_weight_count": len(changed),
                    "weight_fingerprint_sha256": condition_weight_hash,
                    "weight_stats": weight_stats,
                    "probe_sha256_unchanged": {task: head_digest(labels[task]) == g1["probe_training"][task]["head_sha256"]
                                               for task in labels},
                }
                result["conditions"] = dict(condition_results)
            if fingerprint(model) != baseline_fingerprint:
                raise ValueError(f"{condition} did not restore the complete FP master model")
            # Reapply the same condition from the immutable master and verify replay; then restore again.
            with apply_from_master(model, immutable_master, flattened, bits):
                repeat_signal = _read_verified_signal(dev_rows[0], isolation["audio_dir"], monitor)
                reapply = monitor.feature(model, extractor, repeat_signal)
                repeat_again = check_close(values[0], reapply, rtol=1e-4, atol=2e-5,
                                           check=f"{condition} restored/reapplied condition")
            if fingerprint(model) != baseline_fingerprint:
                raise ValueError(f"{condition} repeated application did not restore the FP master model")
            condition_results[condition]["reapplied_repeat_max_abs"] = repeat_again
            print(f"G2 {condition}: restored immutable FP master", flush=True)

        restored = monitor.feature(model, extractor, _read_verified_signal(dev_rows[0], isolation["audio_dir"], monitor))
        restored_max = check_close(fp_dev_features[0], restored, rtol=1e-4, atol=2e-5,
                                   check="quantized-then-restored versus fresh FP")
        if fingerprint(model) != baseline_fingerprint:
            raise ValueError("final model state differs from the immutable FP master")
        heads_unchanged = {
            task: (head_digest(_load_head(RUN_DIR / "g1" / "heads" / f"{task}.npz",
                                          g1["probe_training"][task]["head_sha256"]))
                   == g1["probe_training"][task]["head_sha256"]
                   and file_sha(RUN_DIR / "g1" / "heads" / f"{task}.npz")
                   == g1["probe_training"][task]["probe_file_sha256"])
            for task in labels
        }
        if not all(heads_unchanged.values()) or not all(
            all(result["probe_sha256_unchanged"].values()) for result in
            (condition_results["uniform4"], condition_results["uniform8"])
        ):
            raise ValueError("frozen FP heads/scalers changed during quantization")
        g2_gate_result = g2_gate(
            {task: fp_metrics[task]["macro_f1"] for task in fp_metrics},
            {task: condition_results["uniform4"]["metrics"][task]["macro_f1"] for task in fp_metrics},
            {task: condition_results["uniform8"]["metrics"][task]["macro_f1"] for task in fp_metrics},
            config["gates"],
        )
        prediction_path = RUN_DIR / "g2" / "dev_predictions.npz"
        prediction_sha = _prediction_artifact(prediction_path, condition_prediction_arrays)
        result.update({
            "status": "PASS" if g2_gate_result["pass"] else "NO_GO_CURRENT_DESIGN",
            "G2": "PASS" if g2_gate_result["pass"] else "FAIL",
            "gate": g2_gate_result,
            "code_commit": commit,
            "g0_stage_sha256": g0["stage_sha256"],
            "g1_stage_sha256": g1["stage_sha256"],
            "data_split_sha256": split["sha256"],
            "model_revision": config["model"]["revision"],
            "processor_revision": config["model"]["processor_revision"],
            "checkpoint_sha256": LFS_SHA256,
            "device": device,
            "reference_precision": precision,
            "conditions": condition_results,
            "model_restoration": {"full_fingerprint_equals_master": True,
                                  "restored_fp_feature_max_abs": restored_max,
                                  "head_scaler_hashes_unchanged": heads_unchanged,
                                  "eligible_weights_modified_only": True},
            "costs": cost_summary,
            "dev_prediction_artifact": {"path": str(prediction_path), "sha256": prediction_sha,
                                        "samples": len(dev_rows)},
            "test_isolation": {"test_ids_in_model_rows": 0, "test_audio_model_forwards": 0,
                               "test_prediction_reads": 0, "dev_prediction_rows": len(dev_rows)},
            "resources": monitor.result(),
        })
    except Exception as error:
        result.update({"status": "BLOCKED_INVALID", "G2": "BLOCKED",
                       "blocker": f"{type(error).__name__}: {error}", "resources": monitor.result()})
    body = dict(result)
    body["stage_sha256"] = digest_bytes(canonical_json(body))
    write_json_atomic(out, body)
    print(f"G2 {result['status']}; artifact={out}", flush=True)
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    g0 = subparsers.add_parser("g0", help="verify official source, bounded audio, grid, and split")
    g0.add_argument("--out", type=Path, default=RUN_DIR / "g0" / "g0.json")
    g1 = subparsers.add_parser("g1", help="fit and evaluate frozen FP probes on train/dev only")
    g1.add_argument("--out", type=Path, default=RUN_DIR / "g1" / "g1.json")
    g2 = subparsers.add_parser("g2", help="evaluate only FP, uniform-4 and uniform-8 dev conditions")
    g2.add_argument("--out", type=Path, default=RUN_DIR / "g2" / "g2.json")
    args = parser.parse_args(argv)
    if args.stage == "g0":
        result = run_g0(out=args.out)
        return 0 if result["status"] == "PASS" else 2
    if args.stage == "g1":
        result = run_g1(out=args.out)
        return 0 if result["status"] == "PASS" else 2
    if args.stage == "g2":
        result = run_g2(out=args.out)
        return 0 if result["status"] == "PASS" else 2
    raise AssertionError(f"unreachable P2 stage {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())

