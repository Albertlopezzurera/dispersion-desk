"""Correctness tests for the Black-Scholes core.

These are not smoke tests.  Every downstream number the desk produces -- the
dispersion signal, position sizing, the greek P&L attribution -- inherits any
error made here, so the pricer is checked against properties that must hold
*identically*, not approximately:

* put-call parity (an arbitrage relation, exact in the model)
* textbook reference values
* analytic greeks vs. finite differences of the pricer itself
* round-trip price -> implied vol -> price
* refusal to invent a volatility from an un-invertible quote
"""

from __future__ import annotations

import math

import pytest

from app.quant.black_scholes import (
    BlackScholesError,
    bs_greeks,
    bs_price,
    implied_volatility,
    no_arbitrage_bounds,
)

# A spread of regimes: at-the-money, both wings, short and long dated, with and
# without a dividend yield.
CASES = [
    # (S,     K,     T,      r,     sigma, q)
    (100.0, 100.0, 1.00, 0.04, 0.20, 0.00),
    (100.0, 120.0, 0.25, 0.04, 0.35, 0.00),
    (100.0, 80.0, 0.50, 0.02, 0.15, 0.02),
    (450.0, 455.0, 0.0822, 0.045, 0.18, 0.013),  # ~30 DTE SPY-like
    (37.5, 40.0, 2.00, 0.05, 0.60, 0.00),
]


class TestPutCallParity:
    """C - P == S*e^{-qT} - K*e^{-rT}. Exact in the model, so tolerance is tiny."""

    @pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
    def test_parity_holds(self, S, K, T, r, sigma, q):
        call = bs_price(S, K, T, r, sigma, q, "call")
        put = bs_price(S, K, T, r, sigma, q, "put")
        expected = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert call - put == pytest.approx(expected, abs=1e-10)

    @pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
    def test_delta_parity(self, S, K, T, r, sigma, q):
        """delta_call - delta_put == e^{-qT}, differentiating parity in S."""
        dc = bs_greeks(S, K, T, r, sigma, q, "call").delta
        dp = bs_greeks(S, K, T, r, sigma, q, "put").delta
        assert dc - dp == pytest.approx(math.exp(-q * T), abs=1e-10)

    @pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
    def test_gamma_and_vega_are_type_independent(self, S, K, T, r, sigma, q):
        """Calls and puts on the same contract share gamma and vega."""
        gc = bs_greeks(S, K, T, r, sigma, q, "call")
        gp = bs_greeks(S, K, T, r, sigma, q, "put")
        assert gc.gamma == pytest.approx(gp.gamma, rel=1e-12)
        assert gc.vega == pytest.approx(gp.vega, rel=1e-12)


class TestReferenceValues:
    """Values cross-checked against the standard textbook parameterisation."""

    def test_atm_call_reference(self):
        # S=100, K=100, T=1, r=5%, sigma=20%, q=0  ->  10.4506 (Hull)
        assert bs_price(100, 100, 1.0, 0.05, 0.20, 0.0, "call") == pytest.approx(10.4506, abs=1e-4)

    def test_atm_put_reference(self):
        assert bs_price(100, 100, 1.0, 0.05, 0.20, 0.0, "put") == pytest.approx(5.5735, abs=1e-4)

    def test_atm_call_delta_is_slightly_above_half(self):
        """Drift pushes ATM call delta just over 0.5 when r > 0."""
        delta = bs_greeks(100, 100, 1.0, 0.05, 0.20, 0.0, "call").delta
        assert 0.5 < delta < 0.7

    def test_theta_is_negative_for_long_options(self):
        """Owning an option costs time value every day."""
        assert bs_greeks(100, 100, 0.5, 0.04, 0.25, 0.0, "call").theta < 0
        assert bs_greeks(100, 100, 0.5, 0.04, 0.25, 0.0, "put").theta < 0

    def test_vega_and_gamma_are_positive(self):
        g = bs_greeks(100, 100, 0.5, 0.04, 0.25, 0.0, "call")
        assert g.vega > 0
        assert g.gamma > 0


class TestGreeksAgainstFiniteDifferences:
    """The analytic greeks must equal numerical derivatives of bs_price.

    This is the strongest available check: it catches a wrong formula, a
    dropped discount factor, or a scaling mistake, because the two sides are
    computed by completely different routes.
    """

    @pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_delta(self, S, K, T, r, sigma, q, kind):
        h = S * 1e-5
        numeric = (
            bs_price(S + h, K, T, r, sigma, q, kind) - bs_price(S - h, K, T, r, sigma, q, kind)
        ) / (2 * h)
        assert bs_greeks(S, K, T, r, sigma, q, kind).delta == pytest.approx(numeric, abs=1e-6)

    @pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_gamma(self, S, K, T, r, sigma, q, kind):
        h = S * 1e-4
        numeric = (
            bs_price(S + h, K, T, r, sigma, q, kind)
            - 2 * bs_price(S, K, T, r, sigma, q, kind)
            + bs_price(S - h, K, T, r, sigma, q, kind)
        ) / (h * h)
        assert bs_greeks(S, K, T, r, sigma, q, kind).gamma == pytest.approx(numeric, abs=1e-5)

    @pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_vega_is_per_volatility_point(self, S, K, T, r, sigma, q, kind):
        """dPrice/dSigma scaled by 1/100, matching the desk convention."""
        h = 1e-6
        numeric = (
            bs_price(S, K, T, r, sigma + h, q, kind) - bs_price(S, K, T, r, sigma - h, q, kind)
        ) / (2 * h)
        assert bs_greeks(S, K, T, r, sigma, q, kind).vega == pytest.approx(numeric / 100.0, rel=1e-5)

    @pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_theta_is_per_calendar_day(self, S, K, T, r, sigma, q, kind):
        """Price change as one day of life is removed."""
        h = 1e-6
        d_price_d_t = (
            bs_price(S, K, T + h, r, sigma, q, kind) - bs_price(S, K, T - h, r, sigma, q, kind)
        ) / (2 * h)
        expected = -d_price_d_t / 365.0
        assert bs_greeks(S, K, T, r, sigma, q, kind).theta == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_rho_is_per_rate_point(self, S, K, T, r, sigma, q, kind):
        h = 1e-7
        numeric = (
            bs_price(S, K, T, r + h, sigma, q, kind) - bs_price(S, K, T, r - h, sigma, q, kind)
        ) / (2 * h)
        assert bs_greeks(S, K, T, r, sigma, q, kind).rho == pytest.approx(numeric / 100.0, rel=1e-4)


class TestImpliedVolatilityRoundTrip:
    @pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_recovers_the_input_volatility(self, S, K, T, r, sigma, q, kind):
        price = bs_price(S, K, T, r, sigma, q, kind)
        recovered = implied_volatility(price, S, K, T, r, q, kind)
        assert recovered is not None
        assert recovered == pytest.approx(sigma, abs=1e-6)

    @pytest.mark.parametrize("sigma", [0.02, 0.10, 0.45, 1.20, 3.00])
    def test_converges_across_wide_vol_regimes(self, sigma):
        price = bs_price(100, 100, 0.5, 0.03, sigma, 0.0, "call")
        assert implied_volatility(price, 100, 100, 0.5, 0.03, 0.0, "call") == pytest.approx(
            sigma, abs=1e-6
        )

    def test_deep_otm_short_dated_still_inverts(self):
        """The near-zero-vega wing is where Newton-Raphson would diverge."""
        S, K, T, r, sigma = 100.0, 145.0, 0.02, 0.04, 0.55
        price = bs_price(S, K, T, r, sigma, 0.0, "call")
        recovered = implied_volatility(price, S, K, T, r, 0.0, "call")
        assert recovered is not None
        assert recovered == pytest.approx(sigma, abs=1e-4)


class TestImpliedVolatilityRefusesBadQuotes:
    """The solver must return None rather than invent a number.

    A fabricated IV would silently corrupt the dispersion ratio, so these cases
    matter as much as the successful inversions.
    """

    def test_price_below_intrinsic_returns_none(self):
        # A call quoted below its own no-arbitrage floor.
        assert implied_volatility(0.5, 100, 80, 1.0, 0.04, 0.0, "call") is None

    def test_price_above_upper_bound_returns_none(self):
        # A call cannot be worth more than the discounted spot.
        assert implied_volatility(500.0, 100, 80, 1.0, 0.04, 0.0, "call") is None

    def test_zero_and_negative_prices_return_none(self):
        assert implied_volatility(0.0, 100, 100, 1.0, 0.04, 0.0, "call") is None
        assert implied_volatility(-1.0, 100, 100, 1.0, 0.04, 0.0, "call") is None

    def test_expired_contract_returns_none(self):
        assert implied_volatility(5.0, 100, 100, 0.0, 0.04, 0.0, "call") is None

    def test_bounds_bracket_the_fair_price(self):
        for S, K, T, r, sigma, q in CASES:
            for kind in ("call", "put"):
                lo, hi = no_arbitrage_bounds(S, K, T, r, q, kind)
                price = bs_price(S, K, T, r, sigma, q, kind)
                assert lo - 1e-9 <= price <= hi + 1e-9


class TestDegenerateInputs:
    def test_expired_option_is_worth_intrinsic(self):
        assert bs_price(110, 100, 0.0, 0.04, 0.20, 0.0, "call") == pytest.approx(10.0, abs=1e-9)
        assert bs_price(90, 100, 0.0, 0.04, 0.20, 0.0, "put") == pytest.approx(10.0, abs=1e-9)
        assert bs_price(90, 100, 0.0, 0.04, 0.20, 0.0, "call") == pytest.approx(0.0, abs=1e-9)

    def test_expired_option_has_step_delta_and_no_other_greeks(self):
        g = bs_greeks(110, 100, 0.0, 0.04, 0.20, 0.0, "call")
        assert g.delta == 1.0
        assert (g.gamma, g.vega, g.theta, g.rho) == (0.0, 0.0, 0.0, 0.0)

    def test_zero_volatility_collapses_to_discounted_intrinsic(self):
        price = bs_price(100, 90, 1.0, 0.05, 0.0, 0.0, "call")
        assert price == pytest.approx(100 - 90 * math.exp(-0.05), abs=1e-9)

    @pytest.mark.parametrize(
        "S,K,T",
        [(0.0, 100.0, 1.0), (-1.0, 100.0, 1.0), (100.0, 0.0, 1.0), (100.0, 100.0, -0.5)],
    )
    def test_invalid_inputs_raise(self, S, K, T):
        with pytest.raises(BlackScholesError):
            bs_price(S, K, T, 0.04, 0.2, 0.0, "call")


class TestMonotonicity:
    """Structural properties that must hold for any correct pricer."""

    def test_call_price_increases_with_volatility(self):
        prices = [bs_price(100, 100, 0.5, 0.04, s, 0.0, "call") for s in (0.10, 0.20, 0.40, 0.80)]
        assert prices == sorted(prices)

    def test_call_price_decreases_with_strike(self):
        prices = [bs_price(100, k, 0.5, 0.04, 0.25, 0.0, "call") for k in (80, 90, 100, 110, 120)]
        assert prices == sorted(prices, reverse=True)

    def test_call_delta_stays_within_unit_interval(self):
        for k in (60, 80, 100, 120, 160):
            d = bs_greeks(100, k, 0.5, 0.04, 0.25, 0.0, "call").delta
            assert 0.0 <= d <= 1.0

    def test_put_delta_stays_within_negative_unit_interval(self):
        for k in (60, 80, 100, 120, 160):
            d = bs_greeks(100, k, 0.5, 0.04, 0.25, 0.0, "put").delta
            assert -1.0 <= d <= 0.0
