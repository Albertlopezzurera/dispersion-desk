"""Turning a dispersion signal into defined-risk option structures, then orders.

The trade
---------
When implied correlation is rich, the desk sells index volatility and buys
single-name volatility.  Both sides must have a floor on their losses:

* **Index leg** -- a short iron condor on the index.  Selling a bare strangle
  would express the same view with unlimited loss, so the wings are bought.
  Four legs, one order, maximum loss = strike width minus the credit received.
* **Name legs** -- long strangles on the constituents whose volatility is
  cheapest relative to the index.  Buying options is inherently defined risk:
  the most that can be lost is the premium paid.

When implied correlation is cheap, both sides invert.

Legging risk, stated plainly
----------------------------
Alpaca caps a multi-leg order at four legs, so a basket cannot be filled
atomically: the condor is one order and each strangle is another.  Between the
first fill and the last, the desk is exposed to a partially-built position.  We
do not pretend otherwise -- the gap between the price each leg was quoted at and
the price it filled at is recorded and shows up as ``slippage`` in the
attribution, where it can be measured rather than assumed away.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date

from app.alpaca.client import AlpacaClient, OptionQuote, OrderLeg
from app.config import Settings
from app.quant.black_scholes import bs_greeks
from app.quant.surface import AtmVolatility, year_fraction
from app.risk.engine import CONTRACT_MULTIPLIER, BasketProposal, ProposedLeg

logger = logging.getLogger(__name__)

# How far out of the money the short strikes of the index condor sit, and how
# far beyond them the protective wings are bought. In units of the underlying's
# own expected move over the life of the contract, so the structure adapts to
# volatility instead of using a fixed dollar width.
SHORT_STRIKE_MOVES = 1.0
WING_WIDTH_MOVES = 0.5

# Long strangles on the names are placed just outside the money.
STRANGLE_MOVES = 0.5


class ExecutionError(RuntimeError):
    """A basket could not be constructed or submitted."""


@dataclass(frozen=True)
class Structure:
    """One submittable unit: at most four legs, one Alpaca order."""

    label: str
    underlying: str
    legs: list[ProposedLeg]
    max_loss: float
    net_price: float  # per share; positive = debit paid, negative = credit received

    def to_order_legs(self) -> list[OrderLeg]:
        return [
            OrderLeg(
                symbol=leg.symbol,
                side=leg.side,
                ratio_qty=leg.quantity,
                position_intent="buy_to_open" if leg.side == "buy" else "sell_to_open",
            )
            for leg in self.legs
        ]


def expected_move(spot: float, iv: float, t_years: float) -> float:
    """One standard deviation of the underlying's move over ``t_years``."""
    return spot * iv * (t_years**0.5)


def _nearest(
    quotes: list[OptionQuote],
    target_strike: float,
    kind: str,
    expiry: date,
    settings: Settings | None = None,
):
    """The closest *tradable* contract to a target strike.

    Liquidity is filtered here, not left to the risk gate. The distinction
    matters: the gate exists to refuse a bad trade, while this is about not
    proposing one in the first place. A structure built around a contract nobody
    quotes cannot be priced, cannot be filled, and wastes the whole cycle.

    In practice this is what moves the strikes. A one-sigma wing on a 50%-vol
    name lands on contracts with no open interest, so the builder walks inward
    until it finds strikes that actually trade. Relaxing the gates instead would
    have produced a trade on paper and nothing in the market.

    Falls back to the nearest quotable contract when nothing meets the liquidity
    bar, so the risk engine still sees the proposal and can reject it with a
    reason the operator can read.
    """
    candidates = [
        q for q in quotes if q.option_type == kind and q.expiration == expiry and q.mid is not None
    ]
    if not candidates:
        return None

    if settings is not None:
        liquid = [
            q
            for q in candidates
            if (q.open_interest or 0) >= settings.min_open_interest
            and (q.spread_pct_of_mid is not None)
            and q.spread_pct_of_mid <= settings.max_spread_pct_of_mid
        ]
        if liquid:
            candidates = liquid

    return min(candidates, key=lambda q: abs(q.strike - target_strike))


def _to_leg(quote: OptionQuote, side: str, qty: int, spot: float, t: float, r: float, iv: float):
    """Attach greeks, computed from the desk's own implied volatility."""
    g = bs_greeks(spot, quote.strike, t, r, iv, 0.0, quote.option_type)
    return ProposedLeg(
        symbol=quote.symbol,
        underlying=quote.underlying,
        side=side,
        quantity=qty,
        price=quote.mid or 0.0,
        delta=g.delta,
        gamma=g.gamma,
        vega=g.vega,
        theta=g.theta,
        spread_pct_of_mid=quote.spread_pct_of_mid,
        open_interest=quote.open_interest,
        quote_age_seconds=quote.age_seconds(),
        implied_volatility=iv,
    )


def build_index_condor(
    atm: AtmVolatility,
    chain: list[OptionQuote],
    today: date,
    risk_free_rate: float,
    short_side: bool,
    quantity: int = 1,
    size_scale: float = 1.0,
    settings: Settings | None = None,
) -> Structure | None:
    """Iron condor. ``short_side`` sells volatility, else buys it.

    ``size_scale`` narrows the wings. One contract is already the minimum
    tradable quantity, so the only remaining lever on a condor's worst case is
    how far the protective wing sits from the short strike: half the width is
    half the maximum loss. This is how the evidence gate actually shrinks a
    position rather than merely commenting on it.
    """
    t = year_fraction(today, atm.expiration)
    if t <= 0:
        return None

    move = expected_move(atm.spot, atm.implied_volatility, t)
    if move <= 0:
        return None

    put_short = _nearest(chain, atm.spot - SHORT_STRIKE_MOVES * move, "put", atm.expiration, settings)
    # Floor the wing so the strikes cannot collapse onto the short leg, which
    # would leave a structure with no protection at all.
    wing = max(0.15, WING_WIDTH_MOVES * max(0.0, min(1.0, size_scale)))
    put_wing = _nearest(
        chain, atm.spot - (SHORT_STRIKE_MOVES + wing) * move, "put", atm.expiration, settings
    )
    call_short = _nearest(chain, atm.spot + SHORT_STRIKE_MOVES * move, "call", atm.expiration, settings)
    call_wing = _nearest(
        chain, atm.spot + (SHORT_STRIKE_MOVES + wing) * move, "call", atm.expiration, settings
    )

    picked = [put_wing, put_short, call_short, call_wing]
    if any(q is None for q in picked):
        logger.warning("%s: incomplete condor; some strikes have no market", atm.underlying)
        return None
    if len({q.symbol for q in picked}) != 4:
        logger.warning("%s: condor strikes collapsed onto each other", atm.underlying)
        return None

    # Short condor: sell the inner strikes, buy the outer wings. Long: reverse.
    inner_side, outer_side = ("sell", "buy") if short_side else ("buy", "sell")
    legs = [
        _to_leg(put_wing, outer_side, quantity, atm.spot, t, risk_free_rate, atm.implied_volatility),
        _to_leg(put_short, inner_side, quantity, atm.spot, t, risk_free_rate, atm.implied_volatility),
        _to_leg(call_short, inner_side, quantity, atm.spot, t, risk_free_rate, atm.implied_volatility),
        _to_leg(call_wing, outer_side, quantity, atm.spot, t, risk_free_rate, atm.implied_volatility),
    ]

    # Net price: debits positive, credits negative.
    net = sum((leg.price if leg.side == "buy" else -leg.price) for leg in legs)

    put_width = put_short.strike - put_wing.strike
    call_width = call_wing.strike - call_short.strike
    widest = max(put_width, call_width)

    if short_side:
        # Credit received; worst case is the wider wing minus that credit.
        max_loss = (widest + net) * quantity * CONTRACT_MULTIPLIER
    else:
        # Debit paid; that is the whole risk.
        max_loss = net * quantity * CONTRACT_MULTIPLIER

    if max_loss <= 0:
        # A non-positive worst case means the quotes are inconsistent. The risk
        # engine would veto it anyway; refusing here keeps the reason clearer.
        logger.warning("%s: condor priced with non-positive max loss", atm.underlying)
        return None

    return Structure(
        label=f"{atm.underlying} {'short' if short_side else 'long'} iron condor",
        underlying=atm.underlying,
        legs=legs,
        max_loss=max_loss,
        net_price=net,
    )


def build_name_strangle(
    atm: AtmVolatility,
    chain: list[OptionQuote],
    today: date,
    risk_free_rate: float,
    long_side: bool,
    quantity: int = 1,
    size_scale: float = 1.0,
    settings: Settings | None = None,
) -> Structure | None:
    """One constituent leg of the dispersion basket.

    Long volatility is a bought strangle: the premium paid is the whole risk.
    Short volatility is an iron condor rather than a sold strangle, because the
    desk never opens a position whose loss is unbounded.
    """
    t = year_fraction(today, atm.expiration)
    if t <= 0:
        return None

    move = expected_move(atm.spot, atm.implied_volatility, t)
    if move <= 0:
        return None

    put = _nearest(chain, atm.spot - STRANGLE_MOVES * move, "put", atm.expiration, settings)
    call = _nearest(chain, atm.spot + STRANGLE_MOVES * move, "call", atm.expiration, settings)
    if put is None or call is None:
        return None

    if not long_side:
        # Selling a name's volatility with a naked strangle has unbounded loss,
        # which this desk does not trade. The same view expressed as an iron
        # condor is bounded by the wing width, so the short side of a dispersion
        # trade is built the same way on a constituent as on the index.
        #
        # Without this the `buy_index_vol` direction was unreachable: the signal
        # would fire, the builder would refuse every name, and the basket would
        # be abandoned. Found by running a live cycle, not by a unit test.
        return build_index_condor(
            atm, chain, today, risk_free_rate, True, quantity, size_scale, settings
        )

    legs = [
        _to_leg(put, "buy", quantity, atm.spot, t, risk_free_rate, atm.implied_volatility),
        _to_leg(call, "buy", quantity, atm.spot, t, risk_free_rate, atm.implied_volatility),
    ]
    net = sum((leg.price if leg.side == "buy" else -leg.price) for leg in legs)

    max_loss = net * quantity * CONTRACT_MULTIPLIER
    if max_loss <= 0:
        return None

    return Structure(
        label=f"{atm.underlying} long strangle",
        underlying=atm.underlying,
        legs=legs,
        max_loss=max_loss,
        net_price=net,
    )


def build_basket(
    direction: str,
    index_atm: AtmVolatility,
    index_chain: list[OptionQuote],
    name_atms: dict[str, AtmVolatility],
    name_chains: dict[str, list[OptionQuote]],
    today: date,
    settings: Settings,
    max_names: int = 2,
    size_scale: float = 1.0,
) -> tuple[BasketProposal, list[Structure]] | None:
    """Assemble the full dispersion basket.

    The names chosen are those whose implied volatility is *cheapest relative to
    the index*, since those are where the correlation view is most concentrated.
    Only ``max_names`` are taken: each extra name is another non-atomic order and
    another spread to cross.
    """
    if direction not in ("sell_index_vol", "buy_index_vol"):
        raise ExecutionError(f"unknown direction {direction!r}")

    selling_index = direction == "sell_index_vol"
    r = settings.risk_free_rate

    condor = build_index_condor(
        index_atm, index_chain, today, r, selling_index, 1, size_scale, settings
    )
    if condor is None:
        return None

    # Rank by the ratio of the name's vol to the index's: lowest first when we
    # want to own name volatility.
    ranked = sorted(
        name_atms.items(),
        key=lambda kv: kv[1].implied_volatility / index_atm.implied_volatility,
        reverse=not selling_index,
    )

    structures = [condor]
    for symbol, atm in ranked:
        if len(structures) > max_names:
            break
        chain = name_chains.get(symbol)
        if not chain:
            continue
        strangle = build_name_strangle(
            atm, chain, today, r, selling_index, 1, size_scale, settings
        )
        if strangle is not None:
            structures.append(strangle)

    if len(structures) < 2:
        logger.warning("basket has no tradable single-name leg; refusing to trade index alone")
        return None

    proposal = BasketProposal(
        basket_id=f"dsp-{uuid.uuid4().hex[:10]}",
        direction=direction,
        legs=[leg for s in structures for leg in s.legs],
        max_loss=sum(s.max_loss for s in structures),
        rationale=" + ".join(s.label for s in structures),
    )
    return proposal, structures


class Executor:
    """Submits approved structures. Never decides anything."""

    def __init__(self, client: AlpacaClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    async def submit(self, structures: list[Structure], quantity: int = 1) -> list[dict]:
        """Submit each structure as its own multi-leg order.

        Orders are placed at the structure's mid price.  The desk never sends
        market orders on options: the spread is the dominant cost, and paying it
        blindly would swamp the edge the strategy is trying to capture.

        If a later order fails, the earlier ones have already been sent.  That
        partial state is returned rather than hidden, so the caller can record
        exactly what reached the market.
        """
        self.settings.require_live_execution_allowed()

        submitted: list[dict] = []
        for structure in structures:
            # A credit structure is submitted as a negative limit price.
            limit = abs(structure.net_price)
            if limit <= 0:
                logger.warning("%s: zero net price, skipping", structure.label)
                continue
            try:
                order = await self.client.submit_mleg_order(
                    structure.to_order_legs(), qty=quantity, limit_price=limit
                )
                submitted.append({"structure": structure.label, "order": order})
                logger.info("submitted %s", structure.label)
            except Exception as exc:
                logger.error("failed to submit %s: %s", structure.label, exc)
                submitted.append({"structure": structure.label, "error": str(exc)})
                break  # Stop adding risk once a leg of the basket has failed.
        return submitted
