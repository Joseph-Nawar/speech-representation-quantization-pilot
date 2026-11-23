from __future__ import annotations

from pathlib import Path

import yaml

from sqpilot.stats import BOOTSTRAP_SPEC, BOOTSTRAP_SPEC_SHA256


def test_config_study_lineage_and_bootstrap_spec_are_exact():
    config = yaml.safe_load(Path("configs/pilot.yaml").read_text())
    analysis = config["analysis"]
    assert config["study_id"] == "speech-quant-diagnostic-20261122-v3"
    assert config["protocol_version"] == 3
    assert config["parent_study_id"] == "speech-quant-diagnostic-20261122-v2"
    assert config["amendments"] == ["Q-001", "Q-002"]
    assert analysis["bootstrap_kind"] == BOOTSTRAP_SPEC["method"]
    assert analysis["bootstrap_repetitions"] == BOOTSTRAP_SPEC["repetitions"]
    assert analysis["bootstrap_seed"] == BOOTSTRAP_SPEC["root_seed"]
    assert analysis["confidence_level"] == BOOTSTRAP_SPEC["confidence_level"]
    assert analysis["bootstrap_spec_sha256"] == BOOTSTRAP_SPEC_SHA256


def test_q002_protocol_keeps_q001_model_quantizer_and_decision_thresholds():
    config = yaml.safe_load(Path("configs/pilot.yaml").read_text())
    assert config["model"]["revision"] == "4c66d4806a428f2e922ccfa1a962776e232d487b"
    assert config["model"]["processor_revision"] == config["model"]["revision"]
    assert config["quantization"]["bits"] == [4, 8]
    assert len(config["quantization"]["eligible"]) == 6
    assert config["quantization"]["primary_budget_fraction"] == 0.5
    assert config["probes"]["C_grid"] == [0.01, 0.1, 1.0, 10.0]
    assert config["probes"]["macro_f1_min"] == 0.55
    assert config["probes"]["min_above_majority_macro_f1"] == 0.20
    assert config["gates"]["G2_min_uniform4_drop_any_task"] == 0.03
    assert config["gates"]["G2_max_uniform8_drop_each_task"] == 0.05
