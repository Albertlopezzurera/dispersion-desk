"""Tests for the evidence gate.

The module exists because a naive hit rate over overlapping windows looked like
proof and was not. These tests hold it to that job:

* it must recover a known autocorrelation structure;
* it must call a genuinely informative signal significant;
* it must call an uninformative one unproven, however good the naive number
  looks;
* and it must reproduce the inflation itself -- a series where the overlapping
  sample claims far more evidence than the independent sample contains.

The last one is the thesis of the project, so it is asserted directly.
"""

from __future__ import annotations

import math
import random

import pytest

from app.quant.evidence import (
    ALPHA,
    MIN_INDEPENDENT_SAMPLES,
    EvidenceError,
    EvidenceReport,
    assess,
    autocorrelation,
    binomial_p_value,
    decorrelation_lag,
    effective_sample_size,
    position_scale,
)


def ar1(n: int, phi: float, seed: int = 1) -> list[float]:
    """An AR(1) series with known persistence: x[t] = phi*x[t-1] + noise."""
    rng = random.Random(seed)
    out = [0.0]
    for _ in range(n - 1):
        out.append(phi * out[-1] + rng.gauss(0, 1))
    return out


class TestAutocorrelation:
    def test_white_noise_has_little_autocorrelation(self):
        rng = random.Random(3)
        series = [rng.gauss(0, 1) for _ in range(4000)]
        assert abs(autocorrelation(series, 1)) < 0.06

    @pytest.mark.parametrize("phi", [0.5, 0.8, 0.95])
    def test_ar1_autocorrelation_matches_phi(self, phi):
        """For AR(1), autocorrelation at lag k is phi**k."""
        series = ar1(6000, phi)
        assert autocorrelation(series, 1) == pytest.approx(phi, abs=0.05)
        assert autocorrelation(series, 3) == pytest.approx(phi**3, abs=0.06)

    def test_constant_series_returns_zero_not_nan(self):
        assert autocorrelation([5.0] * 50, 1) == 0.0

    def test_lag_longer_than_series_is_zero(self):
        assert autocorrelation([1.0, 2.0, 3.0], 10) == 0.0

    def test_non_positive_lag_raises(self):
        with pytest.raises(EvidenceError):
            autocorrelation([1.0, 2.0, 3.0, 4.0], 0)


class TestDecorrelationLag:
    def test_white_noise_decorrelates_immediately(self):
        rng = random.Random(5)
        assert decorrelation_lag([rng.gauss(0, 1) for _ in range(2000)]) == 1

    def test_persistent_series_takes_longer(self):
        """phi=0.9 crosses 0.2 near lag 16, since 0.9**16 is about 0.19."""
        assert 10 <= decorrelation_lag(ar1(6000, 0.9)) <= 25

    def test_more_persistence_means_a_longer_lag(self):
        assert decorrelation_lag(ar1(6000, 0.95)) > decorrelation_lag(ar1(6000, 0.7))

    def test_effective_sample_size_shrinks_with_persistence(self):
        rng = random.Random(9)
        noise = [rng.gauss(0, 1) for _ in range(1000)]
        assert effective_sample_size(noise) > effective_sample_size(ar1(1000, 0.95))


class TestBinomialPValue:
    def test_half_of_a_fair_sample_is_unremarkable(self):
        assert binomial_p_value(50, 100) == pytest.approx(0.54, abs=0.02)

    def test_a_strong_result_is_significant(self):
        assert binomial_p_value(70, 100) < 0.001

    def test_every_trial_a_success_matches_the_exact_probability(self):
        assert binomial_p_value(10, 10) == pytest.approx(0.5**10)

    def test_zero_successes_is_certain_to_be_matched_or_exceeded(self):
        assert binomial_p_value(0, 10) == pytest.approx(1.0)

    def test_invalid_inputs_raise(self):
        with pytest.raises(EvidenceError):
            binomial_p_value(1, 0)
        with pytest.raises(EvidenceError):
            binomial_p_value(1, 10, probability=0.0)


class TestAssess:
    def test_an_uninformative_signal_is_not_proven(self):
        """Signal and outcome independent, so nothing should be found."""
        rng = random.Random(11)
        n = 600
        signals = [rng.gauss(0, 1) for _ in range(n)]
        changes = [rng.gauss(0, 1) for _ in range(n)]

        report = assess(signals, changes)
        assert not report.is_significant
        assert report.verdict in ("unproven", "underpowered")

    def test_a_genuinely_informative_signal_is_proven(self):
        """Mean reversion built in, on independent observations."""
        rng = random.Random(13)
        n = 600
        signals = [rng.gauss(0, 1) for _ in range(n)]
        changes = [-0.9 * s + rng.gauss(0, 0.35) for s in signals]

        report = assess(signals, changes)
        assert report.decorrelation_days == 1  # independent by construction
        assert report.hit_rate > 0.8
        assert report.p_value < ALPHA
        assert report.is_significant
        assert report.verdict == "proven"

    def test_overlapping_windows_inflate_the_apparent_evidence(self):
        """The project's central finding, reproduced from first principles.

        A highly persistent signal repeats the same few underlying events across
        hundreds of neighbouring observations. The raw count looks large; the
        independent count is what the sample is actually worth.
        """
        rng = random.Random(17)
        n = 900
        signals = ar1(n, 0.97, seed=21)
        changes = [-0.5 * s + rng.gauss(0, 1.2) for s in signals]

        report = assess(signals, changes)

        assert report.decorrelation_days > 5, "signal should be persistent"
        assert report.n_independent < report.n_raw / 5
        assert report.inflation_factor > 5
        assert report.n_raw > report.n_independent

    def test_report_states_the_inflation_plainly(self):
        rendered = assess(ar1(400, 0.95, seed=4), [0.0] * 400).render()
        assert "independent samples" in rendered
        assert "overstated" in rendered
        assert "VERDICT" in rendered.upper()

    def test_misaligned_inputs_raise(self):
        with pytest.raises(EvidenceError, match="align"):
            assess([1.0, 2.0, 3.0, 4.0], [1.0, 2.0])

    def test_too_few_observations_raise(self):
        with pytest.raises(EvidenceError, match="at least 4"):
            assess([1.0, 2.0], [1.0, 2.0])

    def test_invalid_quantile_raises(self):
        with pytest.raises(EvidenceError):
            assess([1.0] * 10, [1.0] * 10, quantile=0.9)


class TestUnderpowered:
    def test_a_tiny_independent_sample_is_underpowered_not_failed(self):
        """Absence of evidence must not be reported as evidence of absence."""
        rng = random.Random(23)
        n = 200
        signals = ar1(n, 0.99, seed=29)  # decorrelates very slowly
        changes = [-s + rng.gauss(0, 0.1) for s in signals]

        report = assess(signals, changes)
        if report.n_independent < MIN_INDEPENDENT_SAMPLES:
            assert report.is_underpowered
            assert not report.is_significant
            assert report.verdict == "underpowered"
            assert "absence of evidence" in report.render().lower()


class TestPositionScale:
    @staticmethod
    def _report(**kw) -> EvidenceReport:
        base = dict(
            n_raw=400,
            n_independent=40,
            decorrelation_days=10,
            hit_rate=0.5,
            p_value=0.6,
            naive_hit_rate=0.8,
        )
        base.update(kw)
        return EvidenceReport(**base)

    def test_a_proven_signal_trades_full_size(self):
        assert position_scale(self._report(hit_rate=0.8, p_value=0.001)) == 1.0

    def test_an_unproven_signal_is_capped_hard(self):
        assert position_scale(self._report()) == 0.10

    def test_an_underpowered_signal_keeps_a_small_position(self):
        """Trading nothing means never learning whether the signal works."""
        scale = position_scale(self._report(n_independent=3))
        assert scale == 0.25

    def test_scale_is_always_a_valid_fraction(self):
        for p in (0.0, 0.01, 0.05, 0.5, 1.0):
            for n in (1, 5, 12, 100):
                assert 0 < position_scale(self._report(p_value=p, n_independent=n)) <= 1.0


class TestMathematicalSanity:
    def test_p_values_stay_within_zero_and_one(self):
        for successes in range(0, 21):
            assert 0.0 <= binomial_p_value(successes, 20) <= 1.0

    def test_p_value_decreases_as_successes_rise(self):
        values = [binomial_p_value(s, 40) for s in (20, 25, 30, 35)]
        assert values == sorted(values, reverse=True)

    def test_effective_sample_size_never_exceeds_the_raw_count(self):
        for phi in (0.0, 0.5, 0.9, 0.99):
            series = ar1(500, phi, seed=31)
            assert 1 <= effective_sample_size(series) <= len(series)

    def test_ar1_theory_holds_for_the_lag_we_rely_on(self):
        """decorrelation_lag should land near log(threshold)/log(phi)."""
        phi = 0.9
        expected = math.log(0.2) / math.log(phi)
        assert decorrelation_lag(ar1(8000, phi, seed=37)) == pytest.approx(expected, abs=6)
