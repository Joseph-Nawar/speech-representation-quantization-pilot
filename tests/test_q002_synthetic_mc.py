from __future__ import annotations

import pytest

from tools.q002_synthetic_mc import run_diagnostic


def test_synthetic_monte_carlo_is_bounded_reproducible_and_reports_limits():
    first = run_diagnostic(seed=712, outer_repetitions=100, bootstrap_repetitions=200)
    second = run_diagnostic(seed=712, outer_repetitions=100, bootstrap_repetitions=200)
    assert first["diagnostic"] == "synthetic_only_crossed_cluster_macro_f1_coverage"
    assert len(first["scenarios"]) == 4
    assert [row["name"] for row in first["scenarios"]] == [
        "balanced_no_interaction",
        "imbalanced_no_interaction",
        "balanced_interaction",
        "imbalanced_interaction",
    ]
    for left, right in zip(first["scenarios"], second["scenarios"], strict=True):
        assert left["empirical_95pct_percentile_interval_coverage"] == right[
            "empirical_95pct_percentile_interval_coverage"
        ]
        assert left["median_interval_width"] == right["median_interval_width"]
        assert left["intervals_all_finite_positive_width"]
        assert 0 <= left["empirical_95pct_percentile_interval_coverage"] <= 1
    assert "no audio" in first["interpretation_limits"][0].lower()
    assert any("not proven for the pilot" in item.lower()
               for item in first["interpretation_limits"])
    assert first["dgp"]["speakers"] == 40
    assert first["dgp"]["prompts"] == 16
    assert first["dgp"]["fixed_macro_f1_labels"] == 4


def test_synthetic_monte_carlo_rejects_unbounded_tiny_runs():
    with pytest.raises(ValueError, match="requires >=100 outer"):
        run_diagnostic(seed=1, outer_repetitions=99, bootstrap_repetitions=200)
    with pytest.raises(ValueError, match="requires >=100 outer"):
        run_diagnostic(seed=1, outer_repetitions=100, bootstrap_repetitions=199)
