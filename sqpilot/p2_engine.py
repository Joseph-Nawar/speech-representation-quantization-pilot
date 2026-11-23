"""Dev-only feature cache and frozen probe mechanics for the bounded P2 stage."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


P2_C_GRID = (0.01, 0.1, 1.0, 10.0)
SOLVER = "lbfgs"
MAX_ITER = 2000
TOLERANCE = 1e-6
PROBE_SEED = 20261122


def array_sha(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = sha256(json.dumps({"dtype": str(contiguous.dtype), "shape": contiguous.shape},
                               separators=(",", ":")).encode())
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def head_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def evaluation_metrics(y_true, y_pred, labels, train_labels) -> dict:
    labels = list(labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("fixed evaluation labels must be non-empty and unique")
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("development true/predicted labels must be non-empty and paired")
    if set(y_true) - set(labels) or set(y_pred) - set(labels):
        raise ValueError("development labels contain values outside the frozen class set")
    support = Counter(y_true)
    if set(support) != set(labels):
        raise ValueError("every frozen class must have development true-label support")
    train_counts = Counter(train_labels)
    if not train_counts or set(train_counts) != set(labels):
        raise ValueError("every frozen class must have training support for the majority baseline")
    majority = min(labels, key=lambda label: (-train_counts[label], str(label)))
    majority_predictions = np.full(len(y_true), majority, dtype=object)
    fixed_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    majority_f1 = float(f1_score(y_true, majority_predictions, labels=labels,
                                 average="macro", zero_division=0))
    return {
        "macro_f1": fixed_f1,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "class_support": {str(label): int(support[label]) for label in labels},
        "majority_class": str(majority),
        "majority_macro_f1": majority_f1,
        "improvement_over_majority": fixed_f1 - majority_f1,
        "sample_count": len(y_true),
    }


def g1_gate(task_metrics: dict[str, dict], config: dict) -> dict:
    minimum = float(config["macro_f1_min"])
    margin = float(config["min_above_majority_macro_f1"])
    criteria = {}
    for task in ("speaker", "prompt"):
        value = task_metrics[task]
        score_margin = Decimal(str(value["macro_f1"])) - Decimal(str(value["majority_macro_f1"]))
        criteria[task] = {
            "macro_f1_at_least_minimum": value["macro_f1"] >= minimum,
            "improvement_at_least_margin": score_margin >= Decimal(str(margin)),
            "macro_f1": value["macro_f1"],
            "majority_macro_f1": value["majority_macro_f1"],
            "improvement_over_majority": value["macro_f1"] - value["majority_macro_f1"],
        }
    return {"pass": all(row["macro_f1_at_least_minimum"] and row["improvement_at_least_margin"]
                         for row in criteria.values()), "thresholds": {"macro_f1_min": minimum,
                         "min_above_majority_macro_f1": margin}, "criteria": criteria}


def g2_gate(fp_macro_f1: dict[str, float], all4_macro_f1: dict[str, float],
            all8_macro_f1: dict[str, float], config: dict) -> dict:
    if any(set(row) != {"speaker", "prompt"} for row in (fp_macro_f1, all4_macro_f1, all8_macro_f1)):
        raise ValueError("G2 requires exactly prompt and speaker task scores in all conditions")
    drops = {
        condition: {task: float(fp_macro_f1[task] - scored[task]) for task in ("speaker", "prompt")}
        for condition, scored in (("uniform4", all4_macro_f1), ("uniform8", all8_macro_f1))
    }
    min4 = float(config["G2_min_uniform4_drop_any_task"])
    max8 = float(config["G2_max_uniform8_drop_each_task"])
    meets4 = any(Decimal(str(fp_macro_f1[task])) - Decimal(str(all4_macro_f1[task])) >= Decimal(str(min4))
                 for task in ("speaker", "prompt"))
    meets8 = all(Decimal(str(fp_macro_f1[task])) - Decimal(str(all8_macro_f1[task])) <= Decimal(str(max8))
                 for task in ("speaker", "prompt"))
    criteria = {
        "uniform4_drop_at_least_minimum_on_any_task": meets4,
        "uniform8_drop_at_most_maximum_on_each_task": meets8,
    }
    return {"pass": all(criteria.values()), "thresholds": {"uniform4_min_drop_any_task": min4,
            "uniform8_max_drop_each_task": max8}, "criteria": criteria, "drops": drops}


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[name], arrays[name]) for name in arrays):
                raise ValueError(f"refusing to overwrite changed frozen probe artifact {path}")
        return sha256(path.read_bytes()).hexdigest()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return sha256(path.read_bytes()).hexdigest()


def head_array_predictions(features: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    mean = np.asarray(arrays["mean"], dtype=np.float64)
    scale = np.asarray(arrays["scale"], dtype=np.float64)
    coefficients = np.asarray(arrays["coef"], dtype=np.float64)
    intercept = np.asarray(arrays["intercept"], dtype=np.float64)
    classes = np.asarray(arrays["classes"])
    if values.ndim != 2 or values.shape[1] != mean.size or scale.shape != mean.shape:
        raise ValueError("head feature geometry changed")
    if intercept.shape not in {(len(classes),), (1,)}:
        raise ValueError("frozen logistic head geometry changed")
    if coefficients.shape == (1, mean.size) and len(classes) == 2:
        decision = ((values - mean) / scale) @ coefficients[0] + intercept[0]
        if not np.isfinite(decision).all():
            raise ValueError("frozen head produced non-finite decision scores")
        return classes[(decision > 0).astype(np.int64)]
    if coefficients.shape != (len(classes), mean.size) or intercept.shape != (len(classes),):
        raise ValueError("frozen logistic head geometry changed")
    scores = ((values - mean) / scale) @ coefficients.T + intercept
    if not np.isfinite(scores).all():
        raise ValueError("frozen head produced non-finite decision scores")
    return classes[np.argmax(scores, axis=1)]


def cached_features(
    path: Path,
    key: str,
    clip_ids: list[str],
    compute,
) -> tuple[np.ndarray, bool]:
    """Load or atomically create a finite feature matrix bound to exact ordered IDs."""
    if len(key) != 64 or any(char not in "0123456789abcdef" for char in key.lower()):
        raise ValueError("feature cache key must be a SHA256 digest")
    if not clip_ids or len(clip_ids) != len(set(clip_ids)):
        raise ValueError("feature cache clip IDs must be non-empty and unique")
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as archive:
                values = archive["values"]
                stored_key = str(archive["key"].item())
                stored_ids = archive["clip_ids"].tolist()
                stored_sha = str(archive["sha256"].item())
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"stale or corrupt feature cache {path}: {exc}") from exc
        if stored_ids != clip_ids:
            raise ValueError(f"feature cache clip ID order mismatch: {path}")
        if stored_key != key or stored_sha != array_sha(values):
            raise ValueError(f"stale or corrupt feature cache {path}")
        hit = True
    else:
        values = np.asarray(compute())
        if values.ndim != 2 or values.shape[0] != len(clip_ids):
            raise ValueError("computed feature cache has the wrong row count or rank")
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("computed features must be finite float32 values")
        arrays = {
            "values": np.ascontiguousarray(values),
            "key": np.asarray(key),
            "clip_ids": np.asarray(clip_ids, dtype=np.str_),
            "sha256": np.asarray(array_sha(values)),
        }
        _atomic_npz(path, arrays)
        hit = False
    if values.ndim != 2 or values.shape != (len(clip_ids), values.shape[1]) or values.shape[1] == 0:
        raise ValueError("cached feature matrix has invalid shape")
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("cached feature matrix must be finite float32 values")
    return np.ascontiguousarray(values), hit


def fit_frozen_probe(
    features: np.ndarray,
    rows: list[dict],
    *,
    task: str,
    labels: list[str],
    out: Path,
    c_grid: tuple[float, ...] = P2_C_GRID,
) -> dict:
    """Fit the predeclared train-only scaler/head and select C on dev macro-F1."""
    if len(rows) != len(features) or not rows:
        raise ValueError("feature matrix and P2 manifest rows differ")
    if any(row.get("partition") == "test" for row in rows):
        raise ValueError("sealed test partition is forbidden in probe fitting or C selection")
    if any(row.get("partition") not in {"train", "dev"} for row in rows):
        raise ValueError("probe input contains an unknown partition")
    if task not in {"speaker", "prompt"}:
        raise ValueError("probe task must be speaker or prompt")
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[0] != len(rows) or values.shape[1] == 0:
        raise ValueError("invalid pooled feature matrix shape")
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("pooled features must be finite float32 values")
    if tuple(c_grid) != P2_C_GRID:
        raise ValueError("P2 C grid is frozen at [0.01, 0.1, 1, 10]")
    train = np.asarray([row["partition"] == "train" for row in rows], dtype=bool)
    dev = np.asarray([row["partition"] == "dev" for row in rows], dtype=bool)
    if not train.any() or not dev.any():
        raise ValueError("both train and development partitions are required")
    y = np.asarray([str(row[task]) for row in rows], dtype=np.str_)
    labels = [str(label) for label in labels]
    train_labels, dev_labels = y[train], y[dev]
    if set(train_labels) != set(labels):
        raise ValueError("training classes differ from the frozen label map")
    if set(dev_labels) != set(labels):
        raise ValueError("development classes differ from the frozen label map")

    # Both the scaler and logistic heads see training data only.
    scaler = StandardScaler().fit(values[train].astype(np.float64))
    scaled_train = scaler.transform(values[train].astype(np.float64))
    scaled_dev = scaler.transform(values[dev].astype(np.float64))
    candidates = []
    for c_value in P2_C_GRID:
        head = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver=SOLVER,
            random_state=PROBE_SEED,
            max_iter=MAX_ITER,
            tol=TOLERANCE,
        )
        head.fit(scaled_train, train_labels)
        if int(head.n_iter_.max()) >= MAX_ITER:
            raise ValueError(f"frozen {task} logistic probe failed to converge for C={c_value}")
        predicted = head.predict(scaled_dev)
        score = float(f1_score(dev_labels, predicted, labels=labels, average="macro", zero_division=0))
        candidates.append((score, c_value, head, predicted))
    best_score, best_c, best_head, best_prediction = max(candidates, key=lambda item: (item[0], -item[1]))
    arrays = {
        "mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scale": np.asarray(scaler.scale_, dtype=np.float64),
        "variance": np.asarray(scaler.var_, dtype=np.float64),
        "coef": np.asarray(best_head.coef_, dtype=np.float64),
        "intercept": np.asarray(best_head.intercept_, dtype=np.float64),
        "classes": np.asarray(best_head.classes_, dtype=np.str_),
    }
    digest = head_digest(arrays)
    if not np.array_equal(head_array_predictions(values[dev], arrays), best_prediction):
        raise ValueError("saved frozen probe arrays do not replay sklearn predictions")
    file_digest = _atomic_npz(out, arrays)
    metrics = evaluation_metrics(dev_labels.tolist(), best_prediction.tolist(), labels, train_labels.tolist())
    if metrics["macro_f1"] != best_score:
        raise ValueError("selected probe metric changed during artifact construction")
    metadata = {
        "task": task,
        "selected_C": best_c,
        "candidate_development_macro_f1": {str(c_value): score for score, c_value, _, _ in candidates},
        "solver": SOLVER,
        "max_iter": MAX_ITER,
        "tolerance": TOLERANCE,
        "class_weight": "balanced",
        "random_state": PROBE_SEED,
        "scaler_fit_partition": "train_only",
        "head_fit_partition": "train_only",
        "feature_dimension": int(values.shape[1]),
        "train_samples": int(train.sum()),
        "development_samples": int(dev.sum()),
        "train_class_support": {label: int(np.sum(train_labels == label)) for label in labels},
        "development_class_support": {label: int(np.sum(dev_labels == label)) for label in labels},
        "head_sha256": digest,
        "probe_file_sha256": file_digest,
        "development_prediction_sha256": array_sha(np.asarray(best_prediction, dtype=np.str_)),
        "development_metrics": metrics,
    }
    return {"scaler": scaler, "head": best_head, "arrays": arrays, "prediction": best_prediction,
            "metadata": metadata}

