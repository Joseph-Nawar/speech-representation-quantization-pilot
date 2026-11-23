"""Positive-weight crossed-factor bootstrap for fixed-label macro-F1.

This is an approximate multiway multiplier bootstrap. It is not represented as
an exact finite-sample 95% coverage procedure.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
BOOTSTRAP_SPEC = {
    "method": "positive_weight_crossed_factor_multiplier",
    "factor_distribution": "Gamma(shape=1,scale=1)",
    "cell_weight": "speaker_weight*prompt_weight",
    "factor_normalization": "mean_one_per_replicate",
    "repetitions": 2000,
    "root_seed": 20261124,
    "confidence_level": 0.95,
    "interval": "percentile_linear",
    "streams": ["dev", "test"],
    "paired_conditions_share_weights": True,
    "fixed_label_macro_f1_zero_division": 0,
}
BOOTSTRAP_SPEC_SHA256 = hashlib.sha256(
    json.dumps(BOOTSTRAP_SPEC, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _label_indices(values: Sequence[Hashable], labels: Sequence[Hashable]) -> NDArray[np.int64]:
    if len(labels) == 0 or len(set(labels)) != len(labels):
        raise ValueError("labels must be a non-empty sequence of unique class values")
    mapping = {label: i for i, label in enumerate(labels)}
    try:
        return np.asarray([mapping[value] for value in values], dtype=np.int64)
    except (KeyError, TypeError) as exc:
        raise ValueError("observed true/predicted label is absent from fixed labels") from exc


def _validate_observations(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels: Sequence[Hashable],
    weights: Sequence[float] | np.ndarray,
) -> tuple[NDArray[np.int64], NDArray[np.int64], FloatArray]:
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and have equal length")
    true_i = _label_indices(y_true, labels)
    pred_i = _label_indices(y_pred, labels)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (len(y_true),):
        raise ValueError("weights must have one value per evaluation cell")
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("all cell weights must be finite and strictly positive")
    return true_i, pred_i, w


def weighted_macro_f1(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels: Sequence[Hashable],
    weights: Sequence[float] | np.ndarray,
) -> float:
    """Return fixed-label weighted macro-F1 with undefined class F1 set to 0."""
    true_i, pred_i, w = _validate_observations(y_true, y_pred, labels, weights)
    n_classes = len(labels)
    confusion = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(confusion, (true_i, pred_i), w)
    true_support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    denominator = true_support + predicted_support
    class_f1 = np.divide(
        2.0 * np.diag(confusion),
        denominator,
        out=np.zeros(n_classes, dtype=np.float64),
        where=denominator > 0,
    )
    score = float(class_f1.mean())
    if not np.isfinite(score):
        raise ValueError("weighted macro-F1 produced a non-finite value")
    return score


def crossed_gamma_weights(
    speakers: Sequence[Hashable],
    prompts: Sequence[Hashable],
    cell_speakers: Sequence[Hashable],
    cell_prompts: Sequence[Hashable],
    *,
    seed: int,
    repetitions: int,
) -> FloatArray:
    """Generate product Gamma(1,1) weights in the caller's fixed cell order."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if len(speakers) == 0 or len(prompts) == 0:
        raise ValueError("speaker and prompt factor levels must be non-empty")
    if len(set(speakers)) != len(speakers) or len(set(prompts)) != len(prompts):
        raise ValueError("speaker and prompt factor levels must be unique")
    if len(cell_speakers) == 0 or len(cell_speakers) != len(cell_prompts):
        raise ValueError("cell speaker/prompt arrays must be non-empty and equal length")
    if len(set(zip(cell_speakers, cell_prompts, strict=True))) != len(cell_speakers):
        raise ValueError("duplicate evaluation cell in crossed bootstrap input")
    speaker_index = {level: i for i, level in enumerate(speakers)}
    prompt_index = {level: i for i, level in enumerate(prompts)}
    try:
        s_idx = np.asarray([speaker_index[s] for s in cell_speakers], dtype=np.int64)
        p_idx = np.asarray([prompt_index[p] for p in cell_prompts], dtype=np.int64)
    except (KeyError, TypeError) as exc:
        raise ValueError("evaluation cell references an unknown speaker or prompt") from exc
    if set(cell_speakers) != set(speakers) or set(cell_prompts) != set(prompts):
        raise ValueError("each declared factor level must occur in the evaluation cells")

    rng = np.random.default_rng(seed)
    speaker_weights = rng.gamma(shape=1.0, scale=1.0, size=(repetitions, len(speakers)))
    prompt_weights = rng.gamma(shape=1.0, scale=1.0, size=(repetitions, len(prompts)))
    # Per-factor mean-one normalization is a global positive rescaling of all
    # cell weights in a replicate and therefore leaves F1 unchanged.
    speaker_weights /= speaker_weights.mean(axis=1, keepdims=True)
    prompt_weights /= prompt_weights.mean(axis=1, keepdims=True)
    cell_weights = speaker_weights[:, s_idx] * prompt_weights[:, p_idx]
    if not np.all(np.isfinite(cell_weights)) or np.any(cell_weights <= 0):
        raise ValueError("Gamma product weights must remain finite and positive")
    return np.asarray(cell_weights, dtype=np.float64)


def bootstrap_macro_f1(
    y_true: Sequence[Hashable],
    y_pred: Sequence[Hashable],
    labels: Sequence[Hashable],
    weights: np.ndarray,
) -> FloatArray:
    """Score each shared-weight replicate, retaining every fixed class label."""
    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(y_true):
        raise ValueError("weights must have shape (replicates, evaluation cells)")
    if matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)) or np.any(matrix <= 0):
        raise ValueError("every bootstrap cell weight must be finite and positive")
    # Every class supported in the observed evaluation set remains supported
    # under strictly positive weights in every replicate.
    for label in labels:
        observed = np.fromiter((value == label for value in y_true), dtype=bool, count=len(y_true))
        if not np.any(observed):
            raise ValueError(f"fixed class {label!r} has no observed true-label support")
        if np.any(matrix[:, observed].sum(axis=1) <= 0):
            raise ValueError(f"fixed class {label!r} lost true-label support in a replicate")
    return np.asarray(
        [weighted_macro_f1(y_true, y_pred, labels, row) for row in matrix],
        dtype=np.float64,
    )


def paired_bootstrap_delta(
    y_true: Sequence[Hashable],
    prediction_a: Sequence[Hashable],
    prediction_b: Sequence[Hashable],
    labels: Sequence[Hashable],
    shared_weights: np.ndarray,
) -> FloatArray:
    """Return paired A-minus-B weighted macro-F1 using identical cell weights."""
    if len(prediction_a) != len(prediction_b):
        raise ValueError("paired conditions must have the same number of evaluation cells")
    score_a = bootstrap_macro_f1(y_true, prediction_a, labels, shared_weights)
    score_b = bootstrap_macro_f1(y_true, prediction_b, labels, shared_weights)
    delta = score_a - score_b
    if not np.all(np.isfinite(delta)):
        raise ValueError("paired bootstrap delta contains non-finite values")
    return delta


def derive_substream_seed(
    *,
    root_seed: int,
    stream_name: str,
    study_id: str,
    split_hash: str,
    bootstrap_spec_hash: str,
) -> int:
    """Derive a stable, provenance-bound RNG seed for a named partition."""
    if root_seed < 0 or stream_name not in BOOTSTRAP_SPEC["streams"] or not study_id:
        raise ValueError("root seed, named dev/test stream, and study ID must be valid")
    for name, value in (("split_hash", split_hash), ("bootstrap_spec_hash", bootstrap_spec_hash)):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError(f"{name} must be a 64-character SHA256 hex digest")
    material = json.dumps(
        {
            "root_seed": int(root_seed),
            "stream_name": stream_name,
            "study_id": study_id,
            "split_hash": split_hash.lower(),
            "bootstrap_spec_hash": bootstrap_spec_hash.lower(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little", signed=False)


def bootstrap_cache_key(
    *, study_id: str, split_hash: str, bootstrap_spec_hash: str, stream_name: str
) -> str:
    """Hash mandatory study/split/spec/partition identity into cache keys."""
    seed = derive_substream_seed(
        root_seed=20261124,
        stream_name=stream_name,
        study_id=study_id,
        split_hash=split_hash,
        bootstrap_spec_hash=bootstrap_spec_hash,
    )
    material = json.dumps(
        {
            "study_id": study_id,
            "split_hash": split_hash.lower(),
            "bootstrap_spec_hash": bootstrap_spec_hash.lower(),
            "stream_name": stream_name,
            "derived_seed": seed,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(material).hexdigest()


def percentile_interval(scores: Sequence[float], *, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap scores must be a non-empty finite vector")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    tail = (1.0 - confidence) * 50.0
    lo, hi = np.percentile(values, [tail, 100.0 - tail], method="linear")
    return float(lo), float(hi)
