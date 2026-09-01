"""Tests for greek P&L attribution.

Two things are being proven here.

**Reconciliation.**  The components plus the residual must equal the realised
P&L exactly, in every case, including nonsensical ones.  An attribution that can
lose or invent money is worse than no attribution at all, because it would be
trusted.

**Discrimination.**  When only one risk factor moves, the decomposition must
attribute the P&L to *that* factor.  This is what makes the module usable as a
correctness test for the strategy: if a delta-neutral basket makes money on a
pure volatility move and the attribution says "vega", the desk is behaving as
designed; if it says "delta", something is broken.
"""

from __future__ import annotations

import pytest

from app.quant.attribution import (
    CONTRACT_MULTIPLIER,
    LegSnapshot,
    attribute_basket,
    attribute_leg,
    render_attribution,
)
from app.quant.black_scholes import bs_price

RATE = 0.04


def snapshot(
    *,
    spot=100.0,
    strike=100.0,
    t=0.25,
    iv=0.25,
    contracts=1,
    option_type="call",
    symbol="TEST260918C00100000",
    price=None,
) -> LegSnapshot:
    """A leg whose price is the model price, so the inputs stay self-consistent."""
    return LegSnapshot(
        symbol=symbol,
        option_type=option_type,
        signed_contracts=contracts,
        spot=spot,
        strike=strike,
        time_to_expiry=t,
        implied_volatility=iv,
        price=bs_price(spot, strike, t, RATE, iv, 0.0, option_type) if price is None else price,
        risk_free_rate=RATE,
    )


class TestReconciliation:
    """Components + residual == realised P&L, exactly. No exceptions."""

    @pytest.mark.parametrize(
        "exit_kwargs",
        [
            dict(spot=100.0, iv=0.25, t=0.25),  # nothing moved
            dict(spot=105.0, iv=0.25, t=0.25),  # spot only
            dict(spot=100.0, iv=0.31, t=0.25),  # vol only
            dict(spot=100.0, iv=0.25, t=0.20),  # time only
            dict(spot=87.0, iv=0.55, t=0.05),  # everything, violently
            dict(spot=140.0, iv=0.10, t=0.001),  # near expiry, deep ITM
        ],
    )
    @pytest.mark.parametrize("contracts", [1, -1, 7, -4])
    def test_components_sum_to_total(self, exit_kwargs, contracts):
        entry = snapshot(contracts=contracts)
        exit_ = snapshot(contracts=contracts, **exit_kwargs)
        a = attribute_leg(entry, exit_)
        assert a.explained + a.residual == pytest.approx(a.total, abs=1e-9)

    def test_reconciles_with_slippage(self):
        entry = snapshot()
        exit_ = snapshot(spot=103.0, iv=0.27, t=0.22)
        a = attribute_leg(entry, exit_, fill_price=entry.price + 0.08)
        assert a.explained + a.residual == pytest.approx(a.total, abs=1e-9)

    def test_total_matches_a_naive_pnl_calculation(self):
        entry = snapshot(contracts=3)
        exit_ = snapshot(contracts=3, spot=104.0, iv=0.26, t=0.20)
        expected = (exit_.price - entry.price) * 3 * CONTRACT_MULTIPLIER
        assert attribute_leg(entry, exit_).total == pytest.approx(expected, abs=1e-9)


class TestDiscrimination:
    """One factor moves; the attribution must name that factor."""

    def test_pure_volatility_move_attributes_to_vega(self):
        entry = snapshot()
        exit_ = snapshot(iv=0.28)  # +3 vol points, nothing else changed
        a = attribute_leg(entry, exit_)

        assert a.dominant_driver == "vega"
        assert a.delta_pnl == pytest.approx(0.0, abs=1e-12)
        assert a.gamma_pnl == pytest.approx(0.0, abs=1e-12)
        assert a.theta_pnl == pytest.approx(0.0, abs=1e-12)
        assert a.vega_pnl > 0  # long an option, vol rose
        assert a.is_reliable

    def test_short_vega_loses_when_volatility_rises(self):
        entry = snapshot(contracts=-2)
        a = attribute_leg(entry, snapshot(contracts=-2, iv=0.30))
        assert a.vega_pnl < 0
        assert a.total < 0

    def test_pure_time_decay_attributes_to_theta(self):
        entry = snapshot()
        exit_ = snapshot(t=0.25 - 5 / 365)  # five days pass, nothing else moves
        a = attribute_leg(entry, exit_)

        assert a.dominant_driver == "theta"
        assert a.theta_pnl < 0  # long option, time is a cost
        assert a.delta_pnl == pytest.approx(0.0, abs=1e-12)
        assert a.vega_pnl == pytest.approx(0.0, abs=1e-12)

    def test_short_option_earns_theta(self):
        entry = snapshot(contracts=-1)
        a = attribute_leg(entry, snapshot(contracts=-1, t=0.25 - 5 / 365))
        assert a.theta_pnl > 0

    def test_pure_spot_move_attributes_to_direction(self):
        entry = snapshot()
        a = attribute_leg(entry, snapshot(spot=102.0))
        assert a.dominant_driver in ("delta", "gamma")
        assert a.delta_pnl > 0
        assert a.vega_pnl == pytest.approx(0.0, abs=1e-12)
        assert a.theta_pnl == pytest.approx(0.0, abs=1e-12)

    def test_gamma_is_positive_for_a_long_option_in_either_direction(self):
        """Convexity pays the owner whichever way the underlying moves."""
        entry = snapshot()
        up = attribute_leg(entry, snapshot(spot=104.0))
        down = attribute_leg(entry, snapshot(spot=96.0))
        assert up.gamma_pnl > 0
        assert down.gamma_pnl > 0

    def test_residual_is_small_for_a_local_move(self):
        entry = snapshot()
        a = attribute_leg(entry, snapshot(spot=100.5, iv=0.255, t=0.25 - 1 / 365))
        assert a.is_reliable
        assert a.residual_fraction < 0.05

    def test_large_move_is_flagged_as_unreliable_not_silently_trusted(self):
        entry = snapshot(t=0.25)
        a = attribute_leg(entry, snapshot(spot=175.0, iv=1.20, t=0.01))
        assert not a.is_reliable
        assert "WARNING" in render_attribution(a)


class TestSlippage:
    def test_slippage_is_always_a_cost(self):
        entry = snapshot()
        paid_up = attribute_leg(entry, snapshot(), fill_price=entry.price + 0.10)
        paid_down = attribute_leg(entry, snapshot(), fill_price=entry.price - 0.10)
        assert paid_up.slippage < 0
        assert paid_down.slippage < 0

    def test_slippage_is_a_cost_for_short_positions_too(self):
        entry = snapshot(contracts=-3)
        a = attribute_leg(entry, snapshot(contracts=-3), fill_price=entry.price - 0.05)
        assert a.slippage < 0

    def test_no_fill_price_means_no_slippage(self):
        entry = snapshot()
        assert attribute_leg(entry, snapshot()).slippage == 0.0

    def test_slippage_scales_with_size(self):
        small_entry = snapshot(contracts=1)
        big_entry = snapshot(contracts=10)
        small = attribute_leg(
            small_entry, snapshot(contracts=1), fill_price=small_entry.price + 0.05
        )
        big = attribute_leg(big_entry, snapshot(contracts=10), fill_price=big_entry.price + 0.05)
        assert big.slippage == pytest.approx(small.slippage * 10)


class TestBasketAttribution:
    def test_the_strategy_thesis_a_neutral_basket_earns_from_vega(self):
        """A delta-neutral straddle held through a pure volatility move.

        This is the desk's claim in miniature: neutral in direction, paid in
        vega.  If this test ever attributed the P&L to delta, the strategy would
        not be doing what it says it does.
        """
        call_entry = snapshot(option_type="call", contracts=1, symbol="C")
        put_entry = snapshot(option_type="put", contracts=1, symbol="P")
        call_exit = snapshot(option_type="call", contracts=1, symbol="C", iv=0.30)
        put_exit = snapshot(option_type="put", contracts=1, symbol="P", iv=0.30)

        a = attribute_basket([(call_entry, call_exit, None), (put_entry, put_exit, None)])

        assert a.dominant_driver == "vega"
        assert a.vega_pnl > 0
        assert a.total > 0
        # A straddle is near delta-neutral, so direction contributes ~nothing.
        assert abs(a.delta_pnl) < abs(a.vega_pnl) * 0.01
        assert a.explained + a.residual == pytest.approx(a.total, abs=1e-9)

    def test_basket_totals_are_the_sum_of_their_legs(self):
        e1, x1 = snapshot(symbol="A", contracts=2), snapshot(symbol="A", contracts=2, iv=0.28)
        e2, x2 = snapshot(symbol="B", contracts=-1), snapshot(symbol="B", contracts=-1, iv=0.28)

        basket = attribute_basket([(e1, x1, None), (e2, x2, None)])
        legs = attribute_leg(e1, x1), attribute_leg(e2, x2)

        assert basket.total == pytest.approx(sum(a.total for a in legs))
        assert basket.vega_pnl == pytest.approx(sum(a.vega_pnl for a in legs))
        assert basket.symbol == "basket"

    def test_empty_basket_is_an_error(self):
        with pytest.raises(ValueError, match="empty basket"):
            attribute_basket([])


class TestInvalidInputs:
    def test_changing_position_size_is_rejected(self):
        with pytest.raises(ValueError, match="position size changed"):
            attribute_leg(snapshot(contracts=2), snapshot(contracts=3))

    def test_zero_size_is_rejected(self):
        with pytest.raises(ValueError, match="zero-size"):
            attribute_leg(snapshot(contracts=0), snapshot(contracts=0))


class TestRendering:
    def test_report_names_every_component(self):
        text = render_attribution(attribute_leg(snapshot(), snapshot(spot=103.0, iv=0.27)))
        for label in ("Delta", "Gamma", "Vega", "Theta", "Slippage", "Residual", "Dominant"):
            assert label in text
