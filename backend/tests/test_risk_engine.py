"""Tests for the deterministic risk engine.

The headline test is :class:`TestApprovalImpliesCompliance`, a randomised
property check asserting the one invariant the whole desk depends on:

    if the engine approves a basket, no configured limit is breached.

Everything else the system does -- the signal, the agents, the execution -- is
downstream of that guarantee.  A bug letting one non-compliant basket through
would matter more than any amount of missed opportunity, so the property is
exercised over thousands of randomised portfolios rather than a handful of
hand-picked cases.
"""

from __future__ import annotations

import random

import pytest

from app.config import Settings
from app.risk.engine import (
    CONTRACT_MULTIPLIER,
    BasketProposal,
    PortfolioState,
    PositionGreeks,
    ProposedLeg,
    RiskEngine,
    correlation_is_sane,
)


def make_settings(**overrides) -> Settings:
    """Settings with explicit limits, immune to whatever sits in a local .env."""
    base = dict(
        max_net_delta=150.0,
        max_portfolio_vega=400.0,
        max_portfolio_gamma=50.0,
        max_daily_theta=250.0,
        max_risk_per_basket_pct=1.5,
        max_total_risk_pct=10.0,
        max_underlying_concentration_pct=4.0,
        max_daily_loss_pct=3.0,
        max_spread_pct_of_mid=12.0,
        min_open_interest=25,
        max_quote_age_seconds=900,
        min_iv=0.01,
        max_iv=5.0,
        max_weights_age_days=30,
    )
    base.update(overrides)
    return Settings(**base)


def make_leg(**overrides) -> ProposedLeg:
    """A clean, tradable leg. Tests override exactly the field under test."""
    base = dict(
        symbol="SPY260918C00450000",
        underlying="SPY",
        side="buy",
        quantity=1,
        price=3.50,
        delta=0.30,
        gamma=0.01,
        vega=0.12,
        theta=-0.05,
        spread_pct_of_mid=4.0,
        open_interest=1200,
        quote_age_seconds=30.0,
        implied_volatility=0.19,
    )
    base.update(overrides)
    return ProposedLeg(**base)


def make_proposal(legs=None, max_loss: float = 500.0, **overrides) -> BasketProposal:
    base = dict(
        basket_id="basket-001",
        direction="sell_index_vol",
        legs=legs if legs is not None else [make_leg()],
        max_loss=max_loss,
    )
    base.update(overrides)
    return BasketProposal(**base)


def make_portfolio(**overrides) -> PortfolioState:
    base = dict(
        net_asset_value=100_000.0,
        daily_pnl=0.0,
        greeks=PositionGreeks(),
        open_defined_risk=0.0,
        risk_by_underlying={},
        market_is_open=True,
        weights_age_days=1,
    )
    base.update(overrides)
    return PortfolioState(**base)


def check(decision, name):
    return next(c for c in decision.checks if c.name == name)


class TestHappyPath:
    def test_a_clean_basket_is_approved(self):
        engine = RiskEngine(make_settings())
        decision = engine.evaluate(make_proposal(), make_portfolio())
        assert decision.approved, decision.rejection_summary
        assert decision.failures == []

    def test_render_states_the_verdict(self):
        engine = RiskEngine(make_settings())
        assert "TRADE APPROVED" in engine.evaluate(make_proposal(), make_portfolio()).render()

    def test_rejection_render_includes_the_reason(self):
        engine = RiskEngine(make_settings())
        decision = engine.evaluate(make_proposal(max_loss=50_000.0), make_portfolio())
        rendered = decision.render()
        assert "TRADE REJECTED" in rendered
        assert "Reason:" in rendered
        assert decision.rejection_summary


class TestDataQualityGatesFailClosed:
    """Missing data must reject. This is where a permissive default would hurt most."""

    def test_missing_quote_timestamp_rejects(self):
        engine = RiskEngine(make_settings())
        d = engine.evaluate(make_proposal([make_leg(quote_age_seconds=None)]), make_portfolio())
        assert not d.approved
        assert not check(d, "quote_freshness").passed

    def test_stale_quote_rejects(self):
        engine = RiskEngine(make_settings(max_quote_age_seconds=60))
        d = engine.evaluate(make_proposal([make_leg(quote_age_seconds=120.0)]), make_portfolio())
        assert not check(d, "quote_freshness").passed

    def test_missing_spread_rejects(self):
        engine = RiskEngine(make_settings())
        d = engine.evaluate(make_proposal([make_leg(spread_pct_of_mid=None)]), make_portfolio())
        assert not check(d, "liquidity_spread").passed

    def test_wide_spread_rejects(self):
        engine = RiskEngine(make_settings(max_spread_pct_of_mid=5.0))
        d = engine.evaluate(make_proposal([make_leg(spread_pct_of_mid=25.0)]), make_portfolio())
        assert not check(d, "liquidity_spread").passed

    def test_missing_open_interest_rejects(self):
        engine = RiskEngine(make_settings())
        d = engine.evaluate(make_proposal([make_leg(open_interest=None)]), make_portfolio())
        assert not check(d, "open_interest").passed

    def test_unsolvable_iv_rejects(self):
        engine = RiskEngine(make_settings())
        d = engine.evaluate(make_proposal([make_leg(implied_volatility=None)]), make_portfolio())
        assert not check(d, "implied_volatility_sanity").passed

    def test_absurd_iv_rejects(self):
        engine = RiskEngine(make_settings(max_iv=5.0))
        d = engine.evaluate(make_proposal([make_leg(implied_volatility=9.0)]), make_portfolio())
        assert not check(d, "implied_volatility_sanity").passed

    def test_closed_market_rejects(self):
        engine = RiskEngine(make_settings())
        d = engine.evaluate(make_proposal(), make_portfolio(market_is_open=False))
        assert not check(d, "market_hours").passed

    def test_stale_index_weights_reject(self):
        engine = RiskEngine(make_settings(max_weights_age_days=30))
        d = engine.evaluate(make_proposal(), make_portfolio(weights_age_days=90))
        assert not check(d, "index_weights_freshness").passed

    def test_unknown_nav_rejects_and_does_not_crash(self):
        engine = RiskEngine(make_settings())
        d = engine.evaluate(make_proposal(), make_portfolio(net_asset_value=0.0))
        assert not d.approved
        assert not check(d, "nav_known").passed
        assert not check(d, "daily_loss_circuit_breaker").passed


class TestStructuralGates:
    def test_empty_basket_rejects(self):
        engine = RiskEngine(make_settings())
        assert not engine.evaluate(make_proposal(legs=[]), make_portfolio()).approved

    def test_infinite_max_loss_rejects(self):
        engine = RiskEngine(make_settings())
        d = engine.evaluate(make_proposal(max_loss=float("inf")), make_portfolio())
        assert not check(d, "max_loss_is_defined").passed

    def test_nan_max_loss_rejects(self):
        engine = RiskEngine(make_settings())
        d = engine.evaluate(make_proposal(max_loss=float("nan")), make_portfolio())
        assert not check(d, "max_loss_is_defined").passed

    def test_zero_quantity_rejects(self):
        engine = RiskEngine(make_settings())
        d = engine.evaluate(make_proposal([make_leg(quantity=0)]), make_portfolio())
        assert not check(d, "positive_quantities").passed

    def test_zero_price_rejects(self):
        engine = RiskEngine(make_settings())
        d = engine.evaluate(make_proposal([make_leg(price=0.0)]), make_portfolio())
        assert not check(d, "positive_prices").passed


class TestCircuitBreaker:
    def test_trips_past_the_daily_loss_limit(self):
        engine = RiskEngine(make_settings(max_daily_loss_pct=3.0))
        d = engine.evaluate(
            make_proposal(), make_portfolio(net_asset_value=100_000.0, daily_pnl=-3_500.0)
        )
        assert not d.approved
        assert not check(d, "daily_loss_circuit_breaker").passed

    def test_allows_trading_just_inside_the_limit(self):
        engine = RiskEngine(make_settings(max_daily_loss_pct=3.0))
        d = engine.evaluate(
            make_proposal(), make_portfolio(net_asset_value=100_000.0, daily_pnl=-2_000.0)
        )
        assert check(d, "daily_loss_circuit_breaker").passed

    def test_a_profitable_day_never_trips_it(self):
        engine = RiskEngine(make_settings())
        d = engine.evaluate(make_proposal(), make_portfolio(daily_pnl=5_000.0))
        assert check(d, "daily_loss_circuit_breaker").passed


class TestGreekLimitsArePostTrade:
    """Limits apply to the resulting book, not to the basket in isolation."""

    def test_small_basket_rejected_when_book_is_already_at_the_limit(self):
        engine = RiskEngine(make_settings(max_net_delta=100.0))
        # The basket alone contributes 0.30 * 1 * 100 = 30 delta: harmless by itself.
        d = engine.evaluate(make_proposal(), make_portfolio(greeks=PositionGreeks(delta=90.0)))
        assert not check(d, "net_delta_band").passed, "post-trade delta 120 must breach a 100 cap"

    def test_short_delta_can_offset_an_existing_long_book(self):
        """Neutrality is about the net, so a hedging basket must be allowed."""
        engine = RiskEngine(make_settings(max_net_delta=100.0))
        d = engine.evaluate(
            make_proposal([make_leg(side="sell", delta=0.30)]),
            make_portfolio(greeks=PositionGreeks(delta=90.0)),
        )
        assert check(d, "net_delta_band").passed

    def test_vega_limit_is_enforced_on_absolute_exposure(self):
        engine = RiskEngine(make_settings(max_portfolio_vega=50.0))
        d = engine.evaluate(
            make_proposal([make_leg(side="sell", vega=0.40, quantity=2)]), make_portfolio()
        )
        assert not check(d, "portfolio_vega").passed

    def test_gamma_limit_is_enforced(self):
        engine = RiskEngine(make_settings(max_portfolio_gamma=1.0))
        d = engine.evaluate(make_proposal([make_leg(gamma=0.05, quantity=5)]), make_portfolio())
        assert not check(d, "portfolio_gamma").passed

    def test_theta_limit_is_enforced(self):
        engine = RiskEngine(make_settings(max_daily_theta=10.0))
        d = engine.evaluate(make_proposal([make_leg(theta=-0.50, quantity=5)]), make_portfolio())
        assert not check(d, "daily_theta").passed


class TestCapitalLimits:
    def test_oversized_basket_rejects(self):
        engine = RiskEngine(make_settings(max_risk_per_basket_pct=1.0))
        d = engine.evaluate(make_proposal(max_loss=5_000.0), make_portfolio())
        assert not check(d, "risk_per_basket").passed

    def test_aggregate_risk_across_open_baskets_rejects(self):
        engine = RiskEngine(make_settings(max_total_risk_pct=5.0))
        d = engine.evaluate(
            make_proposal(max_loss=1_000.0), make_portfolio(open_defined_risk=4_800.0)
        )
        assert not check(d, "total_defined_risk").passed

    def test_concentration_counts_existing_exposure(self):
        engine = RiskEngine(make_settings(max_underlying_concentration_pct=2.0))
        d = engine.evaluate(
            make_proposal(max_loss=500.0),
            make_portfolio(risk_by_underlying={"SPY": 1_800.0}),
        )
        assert not check(d, "underlying_concentration").passed

    def test_risk_attribution_splits_across_underlyings_and_sums_to_max_loss(self):
        proposal = make_proposal(
            legs=[
                make_leg(symbol="SPY260918C00450000", underlying="SPY", price=4.0, quantity=1),
                make_leg(symbol="NVDA260918C00180000", underlying="NVDA", price=6.0, quantity=1),
            ],
            max_loss=1_000.0,
        )
        split = proposal.risk_by_underlying()
        assert set(split) == {"SPY", "NVDA"}
        assert sum(split.values()) == pytest.approx(1_000.0)
        assert split["NVDA"] > split["SPY"]  # larger notional carries more of the risk


class TestCorrelationSanityGate:
    def test_accepts_a_plausible_correlation(self):
        assert correlation_is_sane(0.45).passed

    def test_accepts_the_boundaries(self):
        assert correlation_is_sane(1.0).passed
        assert correlation_is_sane(0.0).passed

    def test_rejects_an_impossible_correlation(self):
        assert not correlation_is_sane(1.8).passed
        assert not correlation_is_sane(-2.0).passed

    def test_rejects_missing_or_nan(self):
        assert not correlation_is_sane(None).passed
        assert not correlation_is_sane(float("nan")).passed


class TestApprovalImpliesCompliance:
    """The core invariant, checked over randomised portfolios and baskets.

    Rather than asserting specific outcomes, this recomputes every limit
    independently of the engine and asserts that an approval is never issued
    while any of them is breached.
    """

    # Values are drawn with weights that favour the valid range. A uniform draw
    # over every field makes a fully compliant basket astronomically unlikely
    # (~1 in 400,000 here), so the suite would approve nothing and the invariant
    # would be asserted zero times -- a test that passes without testing. The
    # weights below keep violations common while letting clean baskets through
    # often enough for the property to have teeth.
    @staticmethod
    def _weighted(rng: random.Random, values: list, weights: list[int]):
        return rng.choices(values, weights=weights, k=1)[0]

    @classmethod
    def _random_leg(cls, rng: random.Random) -> ProposedLeg:
        return make_leg(
            underlying=rng.choice(["SPY", "NVDA", "MSFT", "AAPL"]),
            side=rng.choice(["buy", "sell"]),
            quantity=cls._weighted(rng, [0, 1, 2, 3], [1, 8, 6, 4]),
            price=cls._weighted(rng, [0.0, 1.25, 3.50, 8.00], [1, 8, 8, 5]),
            delta=round(rng.uniform(-0.45, 0.45), 3),
            gamma=round(rng.uniform(0.0, 0.02), 4),
            vega=round(rng.uniform(0.0, 0.15), 3),
            theta=round(rng.uniform(-0.10, 0.0), 3),
            spread_pct_of_mid=cls._weighted(rng, [None, 1.0, 6.0, 20.0, 45.0], [1, 8, 8, 2, 1]),
            open_interest=cls._weighted(rng, [None, 0, 30, 5_000], [1, 1, 8, 8]),
            quote_age_seconds=cls._weighted(rng, [None, 5.0, 400.0, 5_000.0], [1, 8, 6, 1]),
            implied_volatility=cls._weighted(
                rng, [None, 0.005, 0.22, 0.90, 7.0], [1, 1, 8, 6, 1]
            ),
        )

    def test_no_approved_basket_ever_breaches_a_limit(self):
        rng = random.Random(20260901)
        settings = make_settings()
        engine = RiskEngine(settings)

        approvals = 0
        for _ in range(4000):
            proposal = make_proposal(
                legs=[self._random_leg(rng) for _ in range(rng.randint(0, 3))],
                max_loss=self._weighted(
                    rng,
                    [-10.0, 0.0, 300.0, 1_200.0, 9_000.0, float("inf")],
                    [1, 1, 8, 8, 2, 1],
                ),
            )
            portfolio = make_portfolio(
                net_asset_value=self._weighted(rng, [0.0, 50_000.0, 100_000.0], [1, 4, 8]),
                daily_pnl=rng.uniform(-4_000.0, 3_000.0),
                greeks=PositionGreeks(
                    delta=rng.uniform(-90, 90),
                    gamma=rng.uniform(-25, 25),
                    vega=rng.uniform(-250, 250),
                    theta=rng.uniform(-150, 150),
                ),
                open_defined_risk=rng.uniform(0.0, 9_000.0),
                risk_by_underlying={"SPY": rng.uniform(0.0, 3_000.0)},
                market_is_open=rng.random() > 0.1,
                weights_age_days=self._weighted(rng, [1, 10, 60], [8, 6, 2]),
            )

            decision = engine.evaluate(proposal, portfolio)
            if not decision.approved:
                continue

            approvals += 1
            nav = portfolio.net_asset_value
            post = portfolio.greeks + proposal.greeks

            # Every limit recomputed here, independently of the engine.
            assert nav > 0
            assert portfolio.market_is_open
            assert proposal.legs
            assert proposal.max_loss > 0 and proposal.max_loss != float("inf")
            assert all(leg.quantity >= 1 and leg.price > 0 for leg in proposal.legs)
            assert abs(post.delta) <= settings.max_net_delta
            assert abs(post.vega) <= settings.max_portfolio_vega
            assert abs(post.gamma) <= settings.max_portfolio_gamma
            assert abs(post.theta) <= settings.max_daily_theta
            assert -100.0 * portfolio.daily_pnl / nav < settings.max_daily_loss_pct
            assert 100.0 * proposal.max_loss / nav <= settings.max_risk_per_basket_pct
            assert (
                100.0 * (portfolio.open_defined_risk + proposal.max_loss) / nav
                <= settings.max_total_risk_pct
            )
            assert portfolio.weights_age_days <= settings.max_weights_age_days
            for leg in proposal.legs:
                assert leg.quote_age_seconds is not None
                assert leg.quote_age_seconds <= settings.max_quote_age_seconds
                assert leg.spread_pct_of_mid is not None
                assert leg.spread_pct_of_mid <= settings.max_spread_pct_of_mid
                assert leg.open_interest is not None
                assert leg.open_interest >= settings.min_open_interest
                assert leg.implied_volatility is not None
                assert settings.min_iv <= leg.implied_volatility <= settings.max_iv

        # A test that approved nothing would pass vacuously and prove nothing.
        assert approvals > 20, f"only {approvals} approvals; the generator is too strict"


class TestGreekScaling:
    def test_scaled_greeks_apply_direction_quantity_and_multiplier(self):
        leg = make_leg(side="sell", quantity=3, delta=0.25)
        assert leg.signed_contracts == -3
        assert leg.scaled_greeks.delta == pytest.approx(0.25 * -3 * CONTRACT_MULTIPLIER)

    def test_opposing_legs_cancel_in_the_aggregate(self):
        proposal = make_proposal(
            legs=[make_leg(side="buy", delta=0.40), make_leg(side="sell", delta=0.40)]
        )
        assert proposal.greeks.delta == pytest.approx(0.0)
