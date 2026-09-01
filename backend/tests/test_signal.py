"""Tests for the signal engine: dispersion and realised correlation.

The central check is a *round trip through the theory*. Constructing an index
volatility from a known correlation and then asking the solver to recover that
correlation is the strongest available test, because the two directions use the
formula in opposite ways -- an algebra error would not cancel.

The second theme is refusal. This signal drives real orders, so every function
must return ``None`` rather than a plausible-looking number when its inputs
cannot support one. Those cases get as much attention as the successes.
"""

from __future__ import annotations

import math
import random

import pytest

from app.quant.dispersion import (
    DispersionError,
    basket_implied_vol,
    build_signal,
    compute_snapshot,
    implied_correlation,
)
from app.quant.realized import (
    RealizedError,
    average_pairwise_correlation,
    compute_realized_stats,
    correlation_risk_premium,
    log_returns,
    realized_volatility,
)

WEIGHTS = {"NVDA": 0.30, "MSFT": 0.25, "AAPL": 0.20, "AMZN": 0.15, "META": 0.10}
IVS = {"NVDA": 0.42, "MSFT": 0.22, "AAPL": 0.25, "AMZN": 0.31, "META": 0.35}


def index_iv_for(rho: float, ivs=IVS, weights=WEIGHTS) -> float:
    """Index volatility implied by the parts at a given average correlation."""
    weighted = {s: weights[s] * ivs[s] for s in weights}
    total = sum(weighted.values())
    sum_sq = sum(v * v for v in weighted.values())
    return math.sqrt(sum_sq + rho * (total * total - sum_sq))


class TestImpliedCorrelation:
    @pytest.mark.parametrize("rho", [0.0, 0.15, 0.40, 0.65, 0.90, 0.99, 1.0])
    def test_recovers_a_known_correlation(self, rho):
        recovered = implied_correlation(index_iv_for(rho), IVS, WEIGHTS)
        assert recovered is not None
        assert recovered == pytest.approx(rho, abs=1e-12)

    def test_recovers_negative_correlation(self):
        """Not typical for equities, but the algebra must still hold."""
        recovered = implied_correlation(index_iv_for(-0.10), IVS, WEIGHTS)
        assert recovered == pytest.approx(-0.10, abs=1e-12)

    def test_single_name_basket_is_undefined_not_zero(self):
        """With one name there are no pairs, so correlation has no meaning."""
        assert implied_correlation(0.30, {"NVDA": 0.30}, {"NVDA": 1.0}) is None

    def test_out_of_range_result_is_returned_not_clamped(self):
        """An impossible rho is information: it says the inputs disagree.

        Clamping would hide exactly the data-quality failure the risk engine's
        sanity gate exists to catch.
        """
        rho = implied_correlation(index_iv_for(1.0) * 1.35, IVS, WEIGHTS)
        assert rho is not None and rho > 1.0


class TestDispersionSnapshot:
    @pytest.mark.parametrize("rho", [0.05, 0.35, 0.70, 0.95])
    def test_ratio_is_bounded_by_one(self, rho):
        snap = compute_snapshot(index_iv_for(rho), IVS, WEIGHTS)
        assert 0 < snap.dispersion_ratio <= 1.0 + 1e-12

    def test_ratio_is_monotonic_in_correlation(self):
        ratios = [
            compute_snapshot(index_iv_for(r), IVS, WEIGHTS).dispersion_ratio
            for r in (0.1, 0.3, 0.5, 0.7, 0.9)
        ]
        assert ratios == sorted(ratios)

    def test_perfect_correlation_collapses_the_ratio_to_one(self):
        snap = compute_snapshot(index_iv_for(1.0), IVS, WEIGHTS)
        assert snap.dispersion_ratio == pytest.approx(1.0, abs=1e-12)
        assert snap.spread == pytest.approx(0.0, abs=1e-12)

    def test_spread_is_positive_when_correlation_is_imperfect(self):
        assert compute_snapshot(index_iv_for(0.4), IVS, WEIGHTS).spread > 0

    def test_basket_iv_is_the_weighted_average(self):
        expected = sum(WEIGHTS[s] * IVS[s] for s in WEIGHTS)
        assert basket_implied_vol(IVS, WEIGHTS) == pytest.approx(expected)


class TestDispersionRefusesBadInput:
    def test_missing_constituent_raises(self):
        with pytest.raises(DispersionError, match="missing implied volatility"):
            basket_implied_vol({"NVDA": 0.4}, WEIGHTS)

    def test_weights_must_sum_to_one(self):
        with pytest.raises(DispersionError, match="sum to 1.0"):
            basket_implied_vol(IVS, {"NVDA": 0.5, "MSFT": 0.2})

    @pytest.mark.parametrize("bad", [0.0, -0.2, float("nan"), float("inf")])
    def test_invalid_constituent_iv_raises(self, bad):
        with pytest.raises(DispersionError):
            basket_implied_vol({**IVS, "NVDA": bad}, WEIGHTS)

    @pytest.mark.parametrize("bad", [0.0, -0.1, float("nan")])
    def test_invalid_index_iv_raises(self, bad):
        with pytest.raises(DispersionError):
            compute_snapshot(bad, IVS, WEIGHTS)

    def test_empty_inputs_raise(self):
        with pytest.raises(DispersionError):
            basket_implied_vol({}, {})


class TestSignalGate:
    """Under-sampled history must produce no signal, not a confident one."""

    def setup_method(self):
        self.snap = compute_snapshot(index_iv_for(0.5), IVS, WEIGHTS)

    def test_too_little_history_is_neutral_with_no_z_score(self):
        sig = build_signal(self.snap, [0.7, 0.72, 0.69], z_entry=1.5, min_sample=20)
        assert sig.direction == "neutral"
        assert sig.z_score is None
        assert not sig.is_actionable

    def test_zero_variance_history_is_neutral(self):
        sig = build_signal(self.snap, [0.7] * 40, z_entry=1.5, min_sample=20)
        assert sig.direction == "neutral"
        assert sig.z_score is None

    def test_rich_index_vol_says_sell_index_vol(self):
        history = [0.60 + 0.001 * i for i in range(40)]  # centred below the snapshot
        sig = build_signal(self.snap, history, z_entry=1.0, min_sample=20)
        assert sig.z_score is not None and sig.z_score > 1.0
        assert sig.direction == "sell_index_vol"
        assert sig.is_actionable

    def test_cheap_index_vol_says_buy_index_vol(self):
        history = [0.95 + 0.001 * i for i in range(40)]  # centred above
        sig = build_signal(self.snap, history, z_entry=1.0, min_sample=20)
        assert sig.z_score is not None and sig.z_score < -1.0
        assert sig.direction == "buy_index_vol"

    def test_inside_the_band_is_neutral(self):
        history = [self.snap.dispersion_ratio + 0.01 * ((-1) ** i) for i in range(40)]
        sig = build_signal(self.snap, history, z_entry=3.0, min_sample=20)
        assert sig.direction == "neutral"

    def test_non_finite_history_entries_are_discarded(self):
        history = [0.7, float("nan"), float("inf"), -1.0] + [0.7 + 0.001 * i for i in range(40)]
        sig = build_signal(self.snap, history, z_entry=1.0, min_sample=20)
        assert sig.sample_size == 41  # the three junk values dropped, 0.7 kept

    def test_non_positive_entry_threshold_raises(self):
        with pytest.raises(DispersionError):
            build_signal(self.snap, [0.7] * 40, z_entry=0.0)


class TestRealizedStatistics:
    @staticmethod
    def _series(returns: list[float], start: float = 100.0) -> list[float]:
        prices = [start]
        for r in returns:
            prices.append(prices[-1] * math.exp(r))
        return prices

    def test_log_returns_invert_the_price_path(self):
        prices = self._series([0.01, -0.02, 0.005])
        assert log_returns(prices) == pytest.approx([0.01, -0.02, 0.005], abs=1e-12)

    def test_non_positive_price_is_an_error_not_a_zero_return(self):
        with pytest.raises(RealizedError, match="non-positive"):
            log_returns([100.0, 0.0, 100.0])

    def test_volatility_annualises_daily_deviation(self):
        rng = random.Random(11)
        daily = 0.012
        prices = self._series([rng.gauss(0, daily) for _ in range(2000)])
        vol = realized_volatility(prices)
        assert vol is not None
        assert vol == pytest.approx(daily * math.sqrt(252), rel=0.10)

    def test_too_short_a_series_returns_none(self):
        assert realized_volatility([100.0, 101.0, 102.0]) is None

    def test_flat_prices_have_no_definable_volatility(self):
        assert realized_volatility([100.0] * 60) is None

    def test_identical_series_are_perfectly_correlated(self):
        rng = random.Random(3)
        factor = [rng.gauss(0, 0.01) for _ in range(80)]
        closes = {s: self._series(factor) for s in WEIGHTS}
        stats = compute_realized_stats(closes, WEIGHTS)
        assert stats is not None
        assert stats.average_correlation == pytest.approx(1.0, abs=1e-9)
        assert stats.dispersion_ratio == pytest.approx(1.0, abs=1e-9)

    def test_independent_series_show_near_zero_correlation(self):
        rng = random.Random(5)
        closes = {s: self._series([rng.gauss(0, 0.015) for _ in range(400)]) for s in WEIGHTS}
        stats = compute_realized_stats(closes, WEIGHTS)
        assert stats is not None
        assert abs(stats.average_correlation) < 0.15
        assert stats.dispersion_ratio < 0.8  # low correlation damps index vol hard

    def test_a_partial_basket_is_refused(self):
        rng = random.Random(9)
        closes = {"NVDA": self._series([rng.gauss(0, 0.01) for _ in range(80)])}
        assert compute_realized_stats(closes, WEIGHTS) is None

    def test_a_member_with_too_little_data_refuses_the_whole_basket(self):
        rng = random.Random(13)
        closes = {s: self._series([rng.gauss(0, 0.01) for _ in range(80)]) for s in WEIGHTS}
        closes["META"] = [100.0, 101.0]
        assert compute_realized_stats(closes, WEIGHTS) is None

    def test_weights_must_sum_to_one(self):
        with pytest.raises(RealizedError, match="sum to 1.0"):
            compute_realized_stats({}, {"NVDA": 0.5})

    def test_correlation_needs_two_names(self):
        assert average_pairwise_correlation({"NVDA": [0.01] * 40}, {"NVDA": 1.0}) is None


class TestCorrelationRiskPremium:
    def test_premium_is_the_difference(self):
        assert correlation_risk_premium(0.62, 0.41) == pytest.approx(0.21)

    def test_negative_premium_when_implied_is_below_realised(self):
        assert correlation_risk_premium(0.30, 0.55) == pytest.approx(-0.25)

    @pytest.mark.parametrize(
        "implied,realised",
        [(None, 0.4), (0.6, None), (None, None), (float("nan"), 0.4), (0.6, float("inf"))],
    )
    def test_missing_or_invalid_inputs_never_read_as_zero_premium(self, implied, realised):
        """A missing estimate must not look like "no premium", which is tradable."""
        assert correlation_risk_premium(implied, realised) is None


class TestTheoryRoundTrip:
    """Implied and realised paths must agree when fed the same correlation.

    The two engines compute index volatility by different routes -- one inverts
    the option-implied relation, the other builds it up from return statistics.
    Agreement between them is what licenses subtracting one correlation from the
    other to form the premium the desk trades.
    """

    def test_both_engines_agree_on_a_common_correlation(self):
        rng = random.Random(21)
        target_rho = 0.6
        n = 600

        # Return series with a known common-factor correlation.
        factor = [rng.gauss(0, 1) for _ in range(n)]
        beta = math.sqrt(target_rho)
        closes: dict[str, list[float]] = {}
        for sym in WEIGHTS:
            daily = IVS[sym] / math.sqrt(252)
            prices = [100.0]
            for i in range(n):
                shock = beta * factor[i] + math.sqrt(1 - target_rho) * rng.gauss(0, 1)
                prices.append(prices[-1] * math.exp(daily * shock))
            closes[sym] = prices

        stats = compute_realized_stats(closes, WEIGHTS)
        assert stats is not None
        assert stats.average_correlation == pytest.approx(target_rho, abs=0.10)

        # The implied side, fed the same correlation, recovers it exactly.
        recovered = implied_correlation(index_iv_for(target_rho), IVS, WEIGHTS)
        assert recovered == pytest.approx(target_rho, abs=1e-12)

        # The premium between them is therefore near zero, as it must be when
        # implied and realised describe the same world.
        premium = correlation_risk_premium(recovered, stats.average_correlation)
        assert premium is not None
        assert abs(premium) < 0.10
