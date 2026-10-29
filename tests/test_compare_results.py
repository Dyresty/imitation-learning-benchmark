import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_results import mcnemar_exact_p, wilson_ci


def test_wilson_ci_matches_known_value():
    low, high = wilson_ci(29, 50)
    assert math.isclose(low, 0.44, abs_tol=0.01)
    assert math.isclose(high, 0.71, abs_tol=0.01)


def test_wilson_ci_bounds():
    low, high = wilson_ci(0, 50)
    assert low == 0.0
    assert 0.0 < high < 0.1
    low, high = wilson_ci(50, 50)
    assert high == 1.0


def test_wilson_ci_empty_sample_is_nan():
    low, high = wilson_ci(0, 0)
    assert math.isnan(low) and math.isnan(high)


def test_mcnemar_no_discordant_pairs():
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_symmetric_and_small_for_lopsided_pairs():
    assert mcnemar_exact_p(23, 1) == mcnemar_exact_p(1, 23)
    assert mcnemar_exact_p(23, 1) < 0.001


def test_mcnemar_balanced_pairs_not_significant():
    assert mcnemar_exact_p(5, 5) == 1.0
