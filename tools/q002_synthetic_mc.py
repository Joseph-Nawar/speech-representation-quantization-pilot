"""Bounded synthetic diagnostic for the Q-002 approximate crossed bootstrap.

No real audio, models, labels, predictions, or pilot outcomes are imported.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.special import expit

from sqpilot.stats import (
    bootstrap_macro_f1,
    crossed_gamma_weights,
    percentile_interval,
    weighted_macro_f1,
)


SCENARIOS = (
    {"name": "balanced_no_interaction", "prompt_class_counts": [4, 4, 4, 4], "interaction_sd": 0.0},
    {"name": "imbalanced_no_interaction", "prompt_class_counts": [8, 4, 2, 2], "interaction_sd": 0.0},
    {"name": "balanced_interaction", "prompt_class_counts": [4, 4, 4, 4], "interaction_sd": 0.8},
    {"name": "imbalanced_interaction", "prompt_class_counts": [8, 4, 2, 2], "interaction_sd": 0.8},
)
BASE_LOGITS = np.asarray([1.9, 1.35, 0.8, 0.25], dtype=np.float64)
SPEAKER_SD = 0.55
PROMPT_SD = 0.40


def _population_macro_f1(prompt_class_counts: list[int], interaction_sd: float) -> float:
    """Integrate the known DGP's expected confusion matrix by Gauss-Hermite."""
    counts = np.asarray(prompt_class_counts, dtype=np.float64)
    proportions = counts / counts.sum()
    nodes, gh_weights = np.polynomial.hermite.hermgauss(100)
    total_sd = math.sqrt(SPEAKER_SD**2 + PROMPT_SD**2 + interaction_sd**2)
    normal_draws = math.sqrt(2.0) * total_sd * nodes
    correct_probability = np.asarray(
        [
            np.dot(gh_weights, expit(logit + normal_draws)) / math.sqrt(math.pi)
            for logit in BASE_LOGITS
        ],
        dtype=np.float64,
    )
    confusion = np.zeros((len(BASE_LOGITS), len(BASE_LOGITS)), dtype=np.float64)
    for truth, prevalence in enumerate(proportions):
        confusion[truth, truth] = prevalence * correct_probability[truth]
        for prediction in range(len(BASE_LOGITS)):
            if prediction != truth:
                confusion[truth, prediction] = (
                    prevalence * (1.0 - correct_probability[truth]) / (len(BASE_LOGITS) - 1)
                )
    true_support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    return float((2 * np.diag(confusion) / (true_support + predicted_support)).mean())


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return [center - radius, center + radius]


def run_diagnostic(
    *, seed: int = 20261124, outer_repetitions: int = 250, bootstrap_repetitions: int = 500
) -> dict:
    if outer_repetitions < 100 or bootstrap_repetitions < 200:
        raise ValueError("synthetic diagnostic is bounded but requires >=100 outer and >=200 inner reps")
    started = time.monotonic()
    n_speakers, n_prompts = 40, 16
    speakers = [f"synthetic-speaker-{i:02d}" for i in range(n_speakers)]
    prompts = [f"synthetic-prompt-{i:02d}" for i in range(n_prompts)]
    cell_speakers = [speaker for speaker in speakers for _ in prompts]
    cell_prompts = prompts * n_speakers
    scenario_results = []

    for scenario_index, scenario in enumerate(SCENARIOS):
        class_counts = scenario["prompt_class_counts"]
        prompt_classes = np.concatenate(
            [np.full(count, label, dtype=np.int64) for label, count in enumerate(class_counts)]
        )
        truth = np.tile(prompt_classes, n_speakers)
        target = _population_macro_f1(class_counts, scenario["interaction_sd"])
        labels = list(range(len(BASE_LOGITS)))
        covered = 0
        widths: list[float] = []
        point_errors: list[float] = []

        for outer in range(outer_repetitions):
            dataset_seed = seed + scenario_index * 1_000_000 + outer * 2
            rng = np.random.default_rng(dataset_seed)
            speaker_effects = rng.normal(0, SPEAKER_SD, n_speakers)
            prompt_effects = rng.normal(0, PROMPT_SD, n_prompts)
            interactions = rng.normal(0, scenario["interaction_sd"], (n_speakers, n_prompts))
            linear = (
                BASE_LOGITS[truth]
                + np.repeat(speaker_effects, n_prompts)
                + np.tile(prompt_effects, n_speakers)
                + interactions.reshape(-1)
            )
            correct = rng.random(truth.size) < expit(linear)
            wrong_offsets = rng.integers(1, len(BASE_LOGITS), size=truth.size)
            predicted = np.where(correct, truth, (truth + wrong_offsets) % len(BASE_LOGITS))

            weights = crossed_gamma_weights(
                speakers,
                prompts,
                cell_speakers,
                cell_prompts,
                seed=dataset_seed + 1,
                repetitions=bootstrap_repetitions,
            )
            scores = bootstrap_macro_f1(truth, predicted, labels, weights)
            low, high = percentile_interval(scores, confidence=0.95)
            point = weighted_macro_f1(truth, predicted, labels, np.ones(truth.size))
            covered += int(low <= target <= high)
            widths.append(high - low)
            point_errors.append(point - target)

        successful = covered / outer_repetitions
        scenario_results.append(
            {
                **scenario,
                "outer_repetitions": outer_repetitions,
                "bootstrap_repetitions": bootstrap_repetitions,
                "population_macro_f1_from_known_dgp": target,
                "empirical_95pct_percentile_interval_coverage": successful,
                "coverage_wilson_95pct_mc_interval": _wilson_interval(covered, outer_repetitions),
                "median_interval_width": float(np.median(widths)),
                "mean_interval_width": float(np.mean(widths)),
                "mean_weighted_point_error": float(np.mean(point_errors)),
                "intervals_all_finite_positive_width": bool(
                    np.all(np.isfinite(widths)) and np.all(np.asarray(widths) > 0)
                ),
            }
        )

    return {
        "diagnostic": "synthetic_only_crossed_cluster_macro_f1_coverage",
        "dgp": {
            "speakers": n_speakers,
            "prompts": n_prompts,
            "fixed_macro_f1_labels": len(BASE_LOGITS),
            "speaker_effect_sd": SPEAKER_SD,
            "prompt_effect_sd": PROMPT_SD,
            "class_logits": BASE_LOGITS.tolist(),
            "misclassification": "uniform among other classes",
            "factor_weight_distribution": "independent Gamma(1,1) per speaker and prompt; product per cell",
            "factor_weights_normalized_to_mean_one": True,
            "target": "macro-F1 of the analytically integrated superpopulation confusion matrix for the fixed four labels",
        },
        "seed": seed,
        "outer_repetitions": outer_repetitions,
        "bootstrap_repetitions": bootstrap_repetitions,
        "scenarios": scenario_results,
        "interpretation_limits": [
            "Synthetic diagnostic only; no audio, model, labels, or pilot predictions were read.",
            "Coverage is assessed for this stated DGP, not proven for the pilot or as exact frequentist coverage.",
            "The four-label DGP does not reproduce the pilot's 18-prompt and 63-speaker label counts.",
            "Coverage Monte Carlo intervals quantify only finite outer-simulation error.",
            "No interval tuning or optional stopping is performed.",
        ],
        "wall_seconds": round(time.monotonic() - started, 3),
    }
