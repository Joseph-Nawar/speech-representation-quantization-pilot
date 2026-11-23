from __future__ import annotations

import json

import pytest

import sqpilot.q002_cli as cli


def _synthetic_result():
    return {
        "diagnostic": "synthetic_only_crossed_cluster_macro_f1_coverage",
        "scenarios": [
            {
                "name": name,
                "empirical_95pct_percentile_interval_coverage": 0.95,
                "coverage_wilson_95pct_mc_interval": [0.92, 0.98],
                "mean_interval_width": 0.1,
                "median_interval_width": 0.1,
                "mean_weighted_point_error": 0.0,
                "intervals_all_finite_positive_width": True,
            }
            for name in (
                "balanced_no_interaction",
                "imbalanced_no_interaction",
                "balanced_interaction",
                "imbalanced_interaction",
            )
        ],
        "wall_seconds": 0.5,
    }


def test_validate_command_writes_atomic_provenance_and_replays_without_overwrite(tmp_path, monkeypatch):
    output = tmp_path / "q002.json"
    calls = []

    def diagnostic():
        calls.append(1)
        return _synthetic_result()

    monkeypatch.setattr(cli, "run_diagnostic", diagnostic)
    assert cli.main(["validate", "--out", str(output)]) == 0
    first = json.loads(output.read_text())
    assert first["study_id"] == "speech-quant-diagnostic-20261122-v3"
    assert first["protocol_version"] == 3
    assert first["Q002"] == "PREOUTCOME_SYNTHETIC_DIAGNOSTIC_RECORDED"
    assert len(first["scientific_results_sha256"]) == 64
    assert cli.main(["validate", "--out", str(output)]) == 0
    assert len(calls) == 1
    assert json.loads(output.read_text()) == first


def test_validate_command_fails_closed_on_corrupt_existing_stage_artifact(tmp_path, monkeypatch):
    output = tmp_path / "q002.json"
    monkeypatch.setattr(cli, "run_diagnostic", _synthetic_result)
    assert cli.main(["validate", "--out", str(output)]) == 0
    output.write_text('{"corrupt":true}')
    with pytest.raises(ValueError, match="Existing Q-002 artifact failed integrity check"):
        cli.main(["validate", "--out", str(output)])


def test_validate_command_rejects_incomplete_result_before_writing(tmp_path, monkeypatch):
    output = tmp_path / "q002.json"
    monkeypatch.setattr(
        cli,
        "run_diagnostic",
        lambda: {"diagnostic": "synthetic_only_crossed_cluster_macro_f1_coverage", "scenarios": []},
    )
    with pytest.raises(ValueError, match="four validated scenarios"):
        cli.main(["validate", "--out", str(output)])
    assert not output.exists()
