from collections import Counter

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from sqpilot.p2_engine import (
    cached_features,
    evaluation_metrics,
    fit_frozen_probe,
    g1_gate,
    g2_gate,
    head_array_predictions,
    head_digest,
)
from sqpilot.p2_cli import validate_locked_config
import yaml
from pathlib import Path


def _rows():
    rows = []
    for part, n in (("train", 8), ("dev", 4), ("test", 2)):
        for label in ("a", "b"):
            for index in range(n):
                rows.append({"id": f"{part}-{label}-{index}", "partition": part, "prompt": label,
                             "speaker": label})
    return rows


def test_metrics_use_fixed_labels_and_train_majority_baseline():
    train = ["a"] * 9 + ["b"] * 3
    dev_true = ["a", "a", "b", "b"]
    dev_pred = ["a", "a", "a", "a"]
    result = evaluation_metrics(dev_true, dev_pred, ["a", "b"], train)
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["accuracy"] == 0.5
    assert result["class_support"] == {"a": 2, "b": 2}
    assert result["majority_class"] == "a"
    assert result["majority_macro_f1"] == pytest.approx(1 / 3)


def test_registered_g1_g2_thresholds_and_signed_drops_are_literal():
    gate = {"macro_f1_min": 0.55, "min_above_majority_macro_f1": 0.20}
    assert g1_gate({"speaker": {"macro_f1": 0.75, "majority_macro_f1": 0.55},
                    "prompt": {"macro_f1": 0.56, "majority_macro_f1": 0.36}}, gate)["pass"]
    assert not g1_gate({"speaker": {"macro_f1": 0.749, "majority_macro_f1": 0.55},
                        "prompt": {"macro_f1": 0.56, "majority_macro_f1": 0.36}}, gate)["pass"]
    g2 = {"G2_min_uniform4_drop_any_task": 0.03, "G2_max_uniform8_drop_each_task": 0.05}
    result = g2_gate({"speaker": 0.8, "prompt": 0.7},
                     {"speaker": 0.77, "prompt": 0.71},
                     {"speaker": 0.75, "prompt": 0.75}, g2)
    assert result["pass"]
    assert result["drops"]["uniform8"]["prompt"] == pytest.approx(-0.05)
    assert not g2_gate({"speaker": 0.8, "prompt": 0.7},
                       {"speaker": 0.79, "prompt": 0.69},
                       {"speaker": 0.749, "prompt": 0.7}, g2)["pass"]


def test_current_protocol_config_is_exact_and_gate_drift_fails_closed():
    config = yaml.safe_load(Path("configs/pilot.yaml").read_bytes())
    assert validate_locked_config(config)
    changed = yaml.safe_load(Path("configs/pilot.yaml").read_bytes())
    changed["gates"]["G2_min_uniform4_drop_any_task"] = 0.02
    with pytest.raises(ValueError, match="registered G2 threshold or G3"):
        validate_locked_config(changed)
    wrong_stop = yaml.safe_load(Path("configs/pilot.yaml").read_bytes())
    wrong_stop["resources"]["stage_stop_after"] = "P1"
    with pytest.raises(ValueError, match="approved P2 scope or resource ceiling"):
        validate_locked_config(wrong_stop)


def test_probe_fits_scaler_and_head_on_train_only_then_selects_dev_c(tmp_path):
    rows = [row for row in _rows() if row["partition"] in {"train", "dev"}]
    features = np.asarray([
        [float(row["prompt"] == "a"), float(row["prompt"] == "b"), float(i % 3)]
        for i, row in enumerate(rows)
    ], dtype=np.float32)
    result = fit_frozen_probe(features, rows, task="prompt", labels=["a", "b"], out=tmp_path / "head.npz")
    train_indices = np.asarray([i for i, row in enumerate(rows) if row["partition"] == "train"])
    assert np.allclose(result["scaler"].mean_, features[train_indices].astype(np.float64).mean(axis=0))
    assert result["metadata"]["solver"] == "lbfgs"
    assert result["metadata"]["max_iter"] == 2000
    assert result["metadata"]["selected_C"] == 0.01
    assert set(result["metadata"]["candidate_development_macro_f1"].values()) == {1.0}
    assert result["metadata"]["head_sha256"] == head_digest(result["arrays"])
    assert (tmp_path / "head.npz").is_file()

    all_rows = _rows()
    all_features = np.zeros((len(all_rows), 3), dtype=np.float32)
    with pytest.raises(ValueError, match="test partition"):
        fit_frozen_probe(all_features, all_rows, task="prompt", labels=["a", "b"], out=tmp_path / "bad.npz")


def test_probe_rejects_test_rows_and_missing_train_or_dev_support(tmp_path):
    rows = _rows()
    features = np.zeros((len(rows), 3), dtype=np.float32)
    with pytest.raises(ValueError, match="test partition"):
        fit_frozen_probe(features, rows, task="prompt", labels=["a", "b"], out=tmp_path / "bad.npz")
    permitted = [row for row in rows if row["partition"] != "test"]
    with pytest.raises(ValueError, match="training classes"):
        fit_frozen_probe(np.zeros((len(permitted), 3), dtype=np.float32), permitted,
                         task="speaker", labels=["a", "b", "c"], out=tmp_path / "bad2.npz")


def test_cached_features_bind_ordered_ids_and_replay_predictions(tmp_path):
    rows = _rows()
    rows = [row for row in rows if row["partition"] in {"train", "dev"}]
    ids = [row["id"] for row in rows]
    values = np.arange(len(ids) * 3, dtype=np.float32).reshape(len(ids), 3)
    path = tmp_path / "features.npz"
    calls = Counter()

    def compute():
        calls["compute"] += 1
        return values.copy()

    uncached, hit = cached_features(path, "a" * 64, ids, compute)
    replay, hit_again = cached_features(path, "a" * 64, ids, compute)
    assert not hit and hit_again
    assert calls["compute"] == 1
    assert np.array_equal(uncached, replay)
    model = LogisticRegression().fit(values[:, :2], np.asarray([row["prompt"] for row in rows]))
    from_uncached = model.predict(uncached[:, :2])
    from_cached = model.predict(replay[:, :2])
    assert np.array_equal(from_uncached, from_cached)
    assert np.array_equal(head_array_predictions(uncached[:, :2], {
        "mean": np.zeros(2), "scale": np.ones(2), "coef": model.coef_,
        "intercept": model.intercept_, "classes": model.classes_,
    }), from_uncached)
    with pytest.raises(ValueError, match="stale or corrupt"):
        cached_features(path, "b" * 64, ids, compute)
    with pytest.raises(ValueError, match="clip ID order"):
        cached_features(path, "a" * 64, ids[::-1], compute)

