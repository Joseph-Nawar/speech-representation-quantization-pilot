"""Command line for the pre-outcome Q-002 synthetic statistical validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import yaml

from sqpilot.stats import BOOTSTRAP_SPEC_SHA256
from tools.q002_synthetic_mc import run_diagnostic


STUDY_ID = "speech-quant-diagnostic-20261122-v3"
PROTOCOL_VERSION = 3
CONFIG_PATH = Path("configs/pilot.yaml")
EXPECTED_SCENARIOS = (
    "balanced_no_interaction",
    "imbalanced_no_interaction",
    "balanced_interaction",
    "imbalanced_interaction",
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _scientific_hash(document: dict) -> str:
    body = {key: value for key, value in document.items() if key not in {"wall_seconds", "scientific_results_sha256"}}
    return hashlib.sha256(_canonical(body)).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _validate_config() -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_bytes())
    if (config.get("study_id"), config.get("protocol_version"), config.get("parent_study_id")) != (
        STUDY_ID,
        PROTOCOL_VERSION,
        "speech-quant-diagnostic-20261122-v2",
    ):
        raise ValueError("Q-002 requires the v3 study ID with v2 as parent")
    analysis = config["analysis"]
    if analysis.get("bootstrap_spec_sha256") != BOOTSTRAP_SPEC_SHA256:
        raise ValueError("Configured bootstrap specification hash does not match implementation")
    if analysis.get("bootstrap_repetitions") != 2000 or analysis.get("bootstrap_seed") != 20261124:
        raise ValueError("Configured pilot repetitions or bootstrap seed changed")
    return config


def _validate_result(result: dict) -> None:
    if result.get("diagnostic") != "synthetic_only_crossed_cluster_macro_f1_coverage":
        raise ValueError("Q-002 diagnostic did not identify the required synthetic-only procedure")
    scenarios = result.get("scenarios")
    if not isinstance(scenarios, list) or tuple(row.get("name") for row in scenarios) != EXPECTED_SCENARIOS:
        raise ValueError("Q-002 diagnostic must contain four validated scenarios in locked order")
    for row in scenarios:
        coverage = row.get("empirical_95pct_percentile_interval_coverage")
        interval = row.get("coverage_wilson_95pct_mc_interval")
        widths = (row.get("mean_interval_width"), row.get("median_interval_width"))
        if (
            not isinstance(coverage, (int, float))
            or not math.isfinite(coverage)
            or not 0 <= coverage <= 1
            or not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in interval)
            or interval[0] > interval[1]
            or not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in widths)
            or row.get("intervals_all_finite_positive_width") is not True
        ):
            raise ValueError("Q-002 diagnostic must contain four validated scenarios in locked order")


def _load_valid_existing(path: Path, implementation_sha256: str) -> dict:
    try:
        document = json.loads(path.read_bytes())
        valid = (
            document.get("study_id") == STUDY_ID
            and document.get("protocol_version") == PROTOCOL_VERSION
            and document.get("bootstrap_spec_sha256") == BOOTSTRAP_SPEC_SHA256
            and document.get("implementation_sha256") == implementation_sha256
            and document.get("scientific_results_sha256") == _scientific_hash(document)
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        valid = False
        document = {}
    if not valid:
        raise ValueError("Existing Q-002 artifact failed integrity check; preserving it without overwrite")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("validate",))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    _validate_config()
    implementation_sha256 = hashlib.sha256(
        Path("sqpilot/stats.py").read_bytes() + Path("tools/q002_synthetic_mc.py").read_bytes()
    ).hexdigest()

    if args.out.exists():
        _load_valid_existing(args.out, implementation_sha256)
        print(f"Q-002 synthetic artifact replay verified: {args.out}", flush=True)
        return 0

    result = run_diagnostic()
    _validate_result(result)
    result.update(
        {
            "study_id": STUDY_ID,
            "protocol_version": PROTOCOL_VERSION,
            "Q002": "PREOUTCOME_SYNTHETIC_DIAGNOSTIC_RECORDED",
            "bootstrap_spec_sha256": BOOTSTRAP_SPEC_SHA256,
            "implementation_sha256": implementation_sha256,
            "test_audio_model_forwards": 0,
            "test_prediction_reads": 0,
        }
    )
    result["scientific_results_sha256"] = _scientific_hash(result)
    _atomic_write(args.out, _canonical(result) + b"\n")
    print(
        f"Q-002 synthetic diagnostic recorded: {len(result['scenarios'])} scenarios; "
        f"sha256={result['scientific_results_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
