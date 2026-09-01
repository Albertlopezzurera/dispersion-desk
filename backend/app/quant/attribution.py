"""Decomposing realised P&L into the risks that produced it.

Why this module is the desk's correctness test
----------------------------------------------
This is a volatility relative-value strategy.  It is delta-neutral by
construction, so its profits are supposed to come from **vega** -- being long
cheap volatility and short expensive volatility -- and to be eroded by
**theta**.  Direction is a risk the desk hedges away, not a bet it places.

That claim is falsifiable, and this module falsifies it.  Every closed position
is decomposed into

    P&L  =  delta effect + gamma effect + vega effect + theta effect
            + execution slippage + residual

If a run is profitable but the profit sits in ``delta``, the desk did not earn
that money the way it claims to: either the hedge is broken, the sizing is
wrong, or it got lucky on direction.  That is a bug report, not a result.  With
only a handful of trading sessions available, this decomposition says far more
about whether the system works than the P&L number does -- at that sample size
the P&L is mostly noise.

The arithmetic
--------------
A second-order Taylor expansion of the option price around the entry state:

    dP ~= delta*dS + 0.5*gamma*dS^2 + vega*d(sigma) + theta*dt

Greeks are taken at entry, in the conventions of ``black_scholes`` (vega per
volatility point, theta per calendar day), scaled by signed contracts and the
100-share multiplier.

``residual`` is what the expansion does not explain -- third-order terms, the
path taken between the two observations, cross-effects.  It is reported, never
hidden: a large residual means the move was too big for a local expansion and
the attribution should be read with care.  By construction the components plus
the residual reconcile to the realised P&L exactly, so the decomposition can
never quietly lose money.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.quant.black_scholes import Greeks, OptionType, bs_greeks

CONTRACT_MULTIPLIER = 100

# Above this share of gross P&L the Taylor expansion is not describing the move
# well, and the attribution is flagged rather than trusted.
RESIDUAL_WARNING_FRACTION = 0.25


@dataclass(frozen=True)
class LegSnapshot:
    """One leg observed at a point in time, with everything needed to price it."""

    symbol: str
    option_type: OptionType
    signed_contracts: int  # positive = long, negative = short
    spot: float
    strike: float
    time_to_expiry: float  # years
    implied_volatility: float
    price: float  # per share, observed mid
    risk_free_rate: float = 0.0

    @property
    def position_value(self) -> float:
        """Mark-to-market value of the position, in dollars."""
        return self.price * self.signed_contracts * CONTRACT_MULTIPLIER

    def greeks(self) -> Greeks:
        return bs_greeks(
            self.spot,
            self.strike,
            self.time_to_expiry,
            self.risk_free_rate,
            self.implied_volatility,
            0.0,
            self.option_type,
        )


@dataclass(frozen=True)
class Attribution:
    """Realised P&L split by the risk that produced it. All values in dollars."""

    symbol: str
    total: float
    delta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    theta_pnl: float
    slippage: float
    residual: float

    @property
    def explained(self) -> float:
        return self.delta_pnl + self.gamma_pnl + self.vega_pnl + self.theta_pnl + self.slippage

    @property
    def residual_fraction(self) -> float:
        """Residual as a share of gross attributed magnitude.

        Measured against the sum of absolute components rather than the net
        total, because a net near zero would make any residual look infinite.
        """
        gross = (
            abs(self.delta_pnl)
            + abs(self.gamma_pnl)
            + abs(self.vega_pnl)
            + abs(self.theta_pnl)
            + abs(self.slippage)
        )
        if gross <= 0:
            return 0.0
        return abs(self.residual) / gross

    @property
    def is_reliable(self) -> bool:
        return self.residual_fraction <= RESIDUAL_WARNING_FRACTION

    @property
    def dominant_driver(self) -> str:
        """The risk factor contributing most by magnitude.

        This answers "did we earn this the way we said we would?".
        """
        drivers = {
            "delta": abs(self.delta_pnl),
            "gamma": abs(self.gamma_pnl),
            "vega": abs(self.vega_pnl),
            "theta": abs(self.theta_pnl),
            "slippage": abs(self.slippage),
        }
        return max(drivers, key=lambda k: drivers[k])

    def __add__(self, other: "Attribution") -> "Attribution":
        return Attribution(
            symbol="portfolio",
            total=self.total + other.total,
            delta_pnl=self.delta_pnl + other.delta_pnl,
            gamma_pnl=self.gamma_pnl + other.gamma_pnl,
            vega_pnl=self.vega_pnl + other.vega_pnl,
            theta_pnl=self.theta_pnl + other.theta_pnl,
            slippage=self.slippage + other.slippage,
            residual=self.residual + other.residual,
        )


def attribute_leg(
    entry: LegSnapshot,
    exit_: LegSnapshot,
    fill_price: float | None = None,
) -> Attribution:
    """Decompose one leg's realised P&L.

    Args:
        entry: the leg as observed when the position was opened.
        exit_: the same leg at close, or at the current mark.
        fill_price: the price actually transacted at entry, per share.  When
            given, the gap between it and the observed mid is booked as
            ``slippage`` -- the cost of crossing the spread, usually the single
            largest cost an options desk pays.  When ``None``, slippage is zero
            and the entry mid is assumed to be the fill.

    Returns:
        An :class:`Attribution` whose components plus residual sum exactly to
        the realised P&L.
    """
    if entry.signed_contracts != exit_.signed_contracts:
        raise ValueError(
            f"{entry.symbol}: position size changed between snapshots "
            f"({entry.signed_contracts} -> {exit_.signed_contracts}); "
            "attribute each size separately"
        )
    if entry.signed_contracts == 0:
        raise ValueError(f"{entry.symbol}: cannot attribute a zero-size position")

    scale = entry.signed_contracts * CONTRACT_MULTIPLIER
    greeks = entry.greeks()

    d_spot = exit_.spot - entry.spot
    # Vega is quoted per volatility *point*, so the change must be too.
    d_vol_points = (exit_.implied_volatility - entry.implied_volatility) * 100.0
    # Theta is per calendar day; time to expiry shrinks as days pass.
    d_days = (entry.time_to_expiry - exit_.time_to_expiry) * 365.0

    delta_pnl = greeks.delta * d_spot * scale
    gamma_pnl = 0.5 * greeks.gamma * d_spot * d_spot * scale
    vega_pnl = greeks.vega * d_vol_points * scale
    theta_pnl = greeks.theta * d_days * scale

    # Slippage is the cost of the entry fill versus the mid the decision used.
    # Buying above the mid and selling below it both cost money, hence the
    # unconditional negative sign.
    if fill_price is None:
        slippage = 0.0
        effective_entry_value = entry.position_value
    else:
        slippage = -abs(fill_price - entry.price) * abs(scale)
        effective_entry_value = fill_price * scale

    total = exit_.position_value - effective_entry_value
    residual = total - (delta_pnl + gamma_pnl + vega_pnl + theta_pnl + slippage)

    return Attribution(
        symbol=entry.symbol,
        total=total,
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        vega_pnl=vega_pnl,
        theta_pnl=theta_pnl,
        slippage=slippage,
        residual=residual,
    )


def attribute_basket(legs: list[tuple[LegSnapshot, LegSnapshot, float | None]]) -> Attribution:
    """Aggregate attribution across every leg of a basket.

    Args:
        legs: ``(entry, exit, fill_price)`` triples, one per leg.
    """
    if not legs:
        raise ValueError("cannot attribute an empty basket")

    result = attribute_leg(*legs[0])
    for triple in legs[1:]:
        result = result + attribute_leg(*triple)

    return Attribution(
        symbol="basket",
        total=result.total,
        delta_pnl=result.delta_pnl,
        gamma_pnl=result.gamma_pnl,
        vega_pnl=result.vega_pnl,
        theta_pnl=result.theta_pnl,
        slippage=result.slippage,
        residual=result.residual,
    )


def render_attribution(a: Attribution) -> str:
    """Plain-text summary, written to the journal and shown in Trade Detail."""
    lines = [
        f"P&L ATTRIBUTION  --  {a.symbol}",
        f"  Total realised          {a.total:+12,.2f}",
        "",
        f"  Delta (direction)       {a.delta_pnl:+12,.2f}",
        f"  Gamma (convexity)       {a.gamma_pnl:+12,.2f}",
        f"  Vega  (volatility)      {a.vega_pnl:+12,.2f}",
        f"  Theta (time decay)      {a.theta_pnl:+12,.2f}",
        f"  Slippage (execution)    {a.slippage:+12,.2f}",
        f"  Residual (unexplained)  {a.residual:+12,.2f}",
        "",
        f"  Dominant driver: {a.dominant_driver}",
    ]
    if not a.is_reliable:
        lines.append(
            f"  WARNING: residual is {a.residual_fraction:.0%} of gross attribution; "
            "the move was too large for a local expansion to describe well."
        )
    return "\n".join(lines)
