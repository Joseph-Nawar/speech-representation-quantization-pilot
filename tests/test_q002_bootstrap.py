from __future__ import annotations

import numpy as np
import pytest

from sqpilot.stats import (
    bootstrap_cache_key,
    bootstrap_macro_f1,
    crossed_gamma_weights,
    derive_substream_seed,
    paired_bootstrap_delta,
    percentile_interval,
    weighted_macro_f1,
)


def _perfect_crossed_labels(n_speakers: int, n_prompts: int):
    speakers = [f"s{i:02d}" for i in range(n_speakers)]
    prompts = [f"p{i:02d}" for i in range(n_prompts)]
    cell_speakers = [s for s in speakers for _ in prompts]
    cell_prompts = prompts * n_speakers
    return speakers, prompts, cell_speakers, cell_prompts


def _legacy_deletion_bootstrap_mean(n_speakers=63, n_prompts=18, repetitions=4000):
    """Reproduce multinomial factor resampling plus fixed-label zero filling."""
    rng = np.random.default_rng(20261124)
    speaker_seen = np.zeros((repetitions, n_speakers), dtype=bool)
    prompt_seen = np.zeros((repetitions, n_prompts), dtype=bool)
    for r in range(repetitions):
        speaker_seen[r, np.unique(rng.integers(n_speakers, size=n_speakers))] = True
        prompt_seen[r, np.unique(rng.integers(n_prompts, size=n_prompts))] = True
    # With a perfect classifier each represented class has F1=1; the legacy
    # fixed-label rule assigns F1=0 to every class omitted by resampling.
    return float(np.mean(np.concatenate([speaker_seen.mean(axis=1), prompt_seen.mean(axis=1)])))


def test_old_multinomial_rule_reproduces_deletion_induced_bias():
    legacy_mean = _legacy_deletion_bootstrap_mean()
    assert 0.58 < legacy_mean < 0.70


def test_perfect_63_by_18_classifier_is_one_in_every_weighted_replicate():
    speakers, prompts, cell_speakers, cell_prompts = _perfect_crossed_labels(63, 18)
    weights = crossed_gamma_weights(
        speakers,
        prompts,
        cell_speakers,
        cell_prompts,
        seed=20261124,
        repetitions=2000,
    )
    for truth, labels in ((cell_prompts, prompts), (cell_speakers, speakers)):
        scores = bootstrap_macro_f1(truth, truth, labels, weights)
        assert scores.shape == (2000,)
        assert np.array_equal(scores, np.ones(2000))


def test_gamma_weights_keep_every_original_true_class_positive():
    speakers, prompts, cell_speakers, cell_prompts = _perfect_crossed_labels(7, 5)
    weights = crossed_gamma_weights(
        speakers,
        prompts,
        cell_speakers,
        cell_prompts,
        seed=17,
        repetitions=128,
    )
    for labels in (speakers, prompts):
        support = np.stack([weights[:, np.asarray(cell_speakers if labels is speakers else cell_prompts) == label].sum(1)
                            for label in labels])
        assert np.all(support > 0)
    assert np.all(weights > 0)
    assert np.any(weights != weights[0, 0])


def test_paired_identical_conditions_have_exactly_zero_delta():
    speakers, prompts, cell_speakers, cell_prompts = _perfect_crossed_labels(9, 6)
    truth = np.array([int(p[1:]) % 3 for p in cell_prompts])
    prediction = truth.copy()
    weights = crossed_gamma_weights(
        speakers,
        prompts,
        cell_speakers,
        cell_prompts,
        seed=91,
        repetitions=256,
    )
    delta = paired_bootstrap_delta(
        truth, prediction, prediction.copy(), [0, 1, 2], weights
    )
    assert np.all(np.abs(delta) <= 1e-12)


def test_weighted_macro_f1_matches_independent_confusion_reference():
    y_true = ["a", "b", "a", "c", "b", "c", "a"]
    y_pred = ["a", "a", "b", "c", "c", "c", "c"]
    weights = np.array([0.5, 2.0, 1.5, 0.25, 2.0, 0.25, 1.5])
    labels = ["a", "b", "c", "never_predicted"]

    # Independent slow confusion-matrix reference; the supported but never
    # predicted label has precision/F1=0, as sklearn zero_division=0 does.
    confusion = {true: {pred: 0.0 for pred in labels} for true in labels}
    for true, pred, weight in zip(y_true, y_pred, weights, strict=True):
        confusion[true][pred] += float(weight)
    f1s = []
    for label in labels:
        tp = confusion[label][label]
        true_support = sum(confusion[label].values())
        predicted_support = sum(confusion[other][label] for other in labels)
        denominator = true_support + predicted_support
        f1s.append(2 * tp / denominator if denominator else 0.0)
    expected = sum(f1s) / len(labels)

    actual = weighted_macro_f1(y_true, y_pred, labels, weights)
    assert actual == pytest.approx(expected, abs=1e-14)


def test_named_streams_are_reproducible_distinct_and_paired_within_stream():
    common = dict(
        root_seed=20261124,
        study_id="speech-quant-diagnostic-20261122-v3",
        split_hash="a" * 64,
        bootstrap_spec_hash="b" * 64,
    )
    dev_seed = derive_substream_seed(stream_name="dev", **common)
    assert dev_seed == derive_substream_seed(stream_name="dev", **common)
    test_seed = derive_substream_seed(stream_name="test", **common)
    assert dev_seed != test_seed

    speakers, prompts, cell_speakers, cell_prompts = _perfect_crossed_labels(8, 6)
    dev_a = crossed_gamma_weights(speakers, prompts, cell_speakers, cell_prompts,
                                  seed=dev_seed, repetitions=64)
    dev_b = crossed_gamma_weights(speakers, prompts, cell_speakers, cell_prompts,
                                  seed=dev_seed, repetitions=64)
    test = crossed_gamma_weights(speakers, prompts, cell_speakers, cell_prompts,
                                 seed=test_seed, repetitions=64)
    assert np.array_equal(dev_a, dev_b)
    assert not np.array_equal(dev_a, test)


def test_bootstrap_cache_key_binds_study_split_spec_and_partition():
    base = dict(
        study_id="speech-quant-diagnostic-20261122-v3",
        split_hash="a" * 64,
        bootstrap_spec_hash="b" * 64,
        stream_name="dev",
    )
    key = bootstrap_cache_key(**base)
    assert key == bootstrap_cache_key(**base)
    assert key != bootstrap_cache_key(**{**base, "study_id": "other"})
    assert key != bootstrap_cache_key(**{**base, "split_hash": "c" * 64})
    assert key != bootstrap_cache_key(**{**base, "bootstrap_spec_hash": "d" * 64})
    assert key != bootstrap_cache_key(**{**base, "stream_name": "test"})


@pytest.mark.parametrize(
    "truth,prediction,labels,weights",
    [
        ([0, 1], [0, 2], [0, 1], [1.0, 1.0]),
        ([0, 1], [0, 1], [0, 1], [1.0, 0.0]),
        ([0, 1], [0, 1], [0, 1], [1.0, np.nan]),
    ],
)
def test_invalid_labels_or_nonpositive_nonfinite_weights_fail_closed(
    truth, prediction, labels, weights
):
    with pytest.raises(ValueError):
        weighted_macro_f1(truth, prediction, labels, np.asarray(weights))


def test_percentile_interval_rejects_nonfinite_scores_and_uses_fixed_confidence():
    assert percentile_interval(np.arange(1000), confidence=0.95) == pytest.approx((24.975, 974.025))
    with pytest.raises(ValueError):
        percentile_interval([0.0, float("nan")], confidence=0.95)
