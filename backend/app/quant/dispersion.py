"""The dispersion signal: is index volatility rich or cheap versus its parts?

The economics
-------------
An index's volatility is not the average of its constituents' volatilities -- it
is that average *damped by imperfect correlation*.  For weights ``w`` and
constituent vols ``sigma_i``:

    sigma_index^2 = sum_i sum_j w_i w_j sigma_i sigma_j rho_ij

Since every ``rho_ij <= 1``, subadditivity gives ``sigma_index <= sum_i w_i sigma_i``
always.  The gap between the two sides is pure correlation.  That gap is what
this desk trades, and it is why the strategy is direction-neutral: nothing here
expresses a view on whether the market goes up or down.

Why a *ratio* rather than a difference
--------------------------------------
Alpaca's free ``indicative`` options feed publishes derived quotes, not true OPRA
quotes.  Any systematic multiplicative bias in that feed appears in both the
numerator and denominator of

    DR = sigma_index / sum_i w_i sigma_i

and therefore largely cancels.  A strategy that valued contracts in absolute
terms would inherit that bias in full.  This robustness is the main reason the
signal is expressed as a ratio, and it is the desk's central design bet.

``DR`` lives in ``(0, 1]``: it reaches 1 only when every pairwise correlation is
1.  High DR means index vol is expensive relative to the parts (sell index vol,
buy constituent vol); low DR means the reverse.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

# A basket whose weights sum this far from 1.0 is a caller bug, not rounding.
_WEIGHT_SUM_TOLERANCE = 1e-6


class DispersionError(ValueError):
    """Raised when a dispersion metric cannot be computed from the inputs."""


@dataclass(frozen=True)
class DispersionSnapshot:
    """One observation of the index-versus-basket volatility relationship."""

    index_iv: float
    basket_iv: float  # weighted average of constituent IVs
    dispersion_ratio: float
    implied_correlation: float | None
    constituent_ivs: dict[str, float]
    weights: dict[str, float]

    @property
    def spread(self) -> float:
        """Volatility points by which the basket exceeds the index.

        Never negative in theory; a negative reading means the data violates
        subadditivity and the feed should not be trusted.
        """
        return self.basket_iv - self.index_iv


@dataclass(frozen=True)
class DispersionSignal:
    """A dispersion snapshot placed in its historical context."""

    snapshot: DispersionSnapshot
    z_score: float | None
    mean: float | None
    stdev: float | None
    sample_size: int
    direction: str  # "sell_index_vol" | "buy_index_vol" | "neutral"

    @property
    def is_actionable(self) -> bool:
        return self.direction != "neutral"


def _validate_inputs(ivs: dict[str, float], weights: dict[str, float]) -> None:
    if not weights:
        raise DispersionError("no basket weights supplied")
    if not ivs:
        raise DispersionError("no constituent implied volatilities supplied")

    missing = set(weights) - set(ivs)
    if missing:
        raise DispersionError(f"missing implied volatility for: {sorted(missing)}")

    for symbol in weights:
        iv = ivs[symbol]
        if iv <= 0 or not math.isfinite(iv):
            raise DispersionError(f"invalid implied volatility for {symbol}: {iv}")

    for symbol, w in weights.items():
        if w < 0 or not math.isfinite(w):
            raise DispersionError(f"invalid weight for {symbol}: {w}")

    total = sum(weights.values())
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise DispersionError(f"weights must sum to 1.0, got {total}")


def _validate_index_iv(index_iv: float) -> None:
    if index_iv <= 0 or not math.isfinite(index_iv):
        raise DispersionError(f"invalid index implied volatility: {index_iv}")


def basket_implied_vol(ivs: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted average constituent implied volatility, ``sum_i w_i sigma_i``.

    This is the *upper bound* on index vol implied by the parts: the value the
    index would have if every constituent moved in lockstep.
    """
    _validate_inputs(ivs, weights)
    return sum(weights[sym] * ivs[sym] for sym in weights)


def implied_correlation(
    index_iv: float, ivs: dict[str, float], weights: dict[str, float]
) -> float | None:
    """Average pairwise correlation implied by index vol, given the parts.

        rho = (sigma_idx^2 - sum_i w_i^2 sigma_i^2)
              / (sum_i sum_{j!=i} w_i w_j sigma_i sigma_j)

    Returns ``None`` for a single-name basket, where the cross term vanishes and
    correlation is undefined -- never a fabricated value.

    The result is deliberately **not clamped** to [0, 1].  A reading outside that
    range is meaningful: it says index vol is inconsistent with the sub-basket,
    usually because the basket covers only part of the index or the feed is
    stale.  The risk engine treats an out-of-range correlation as a data-quality
    veto, which it could not do if this function quietly clipped the value.
    """
    _validate_inputs(ivs, weights)
    _validate_index_iv(index_iv)

    if len(weights) < 2:
        return None

    weighted = {sym: weights[sym] * ivs[sym] for sym in weights}
    total = sum(weighted.values())
    sum_of_squares = sum(v * v for v in weighted.values())

    # sum_i sum_{j != i} (w_i s_i)(w_j s_j) == (sum)^2 - sum(squares)
    cross_term = total * total - sum_of_squares
    if cross_term <= 0:
        return None

    return (index_iv * index_iv - sum_of_squares) / cross_term


def compute_snapshot(
    index_iv: float, ivs: dict[str, float], weights: dict[str, float]
) -> DispersionSnapshot:
    """Build a full dispersion observation from one set of implied volatilities."""
    _validate_index_iv(index_iv)
    basket_iv = basket_implied_vol(ivs, weights)
    if basket_iv <= 0:
        raise DispersionError(f"basket implied volatility must be positive, got {basket_iv}")

    return DispersionSnapshot(
        index_iv=index_iv,
        basket_iv=basket_iv,
        dispersion_ratio=index_iv / basket_iv,
        implied_correlation=implied_correlation(index_iv, ivs, weights),
        constituent_ivs={sym: ivs[sym] for sym in weights},
        weights=dict(weights),
    )


def build_signal(
    snapshot: DispersionSnapshot,
    history: list[float],
    z_entry: float,
    min_sample: int = 20,
) -> DispersionSignal:
    """Place a snapshot against its own history and decide whether to act.

    ``history`` is a series of past dispersion ratios, reconstructed by
    ``scripts/bootstrap_history.py`` by inverting Black-Scholes over historical
    closes (Alpaca does not serve historical implied volatility).

    With fewer than ``min_sample`` usable observations, or a degenerate standard
    deviation, the signal is reported as ``neutral`` with ``z_score=None``.  The
    desk refuses to trade on a z-score it cannot compute: an under-sampled mean
    is precisely how a strategy with no backtest fools itself into trading noise.
    """
    if z_entry <= 0:
        raise DispersionError(f"z_entry must be positive, got {z_entry}")
    if min_sample < 2:
        raise DispersionError(f"min_sample must be at least 2, got {min_sample}")

    usable = [x for x in history if math.isfinite(x) and x > 0]
    if len(usable) < min_sample:
        return DispersionSignal(
            snapshot=snapshot,
            z_score=None,
            mean=None,
            stdev=None,
            sample_size=len(usable),
            direction="neutral",
        )

    mean = statistics.fmean(usable)
    stdev = statistics.stdev(usable)
    if stdev <= 0 or not math.isfinite(stdev):
        return DispersionSignal(
            snapshot=snapshot,
            z_score=None,
            mean=mean,
            stdev=stdev,
            sample_size=len(usable),
            direction="neutral",
        )

    z = (snapshot.dispersion_ratio - mean) / stdev

    if z >= z_entry:
        # Index vol unusually rich versus the parts: sell index vol, buy names.
        direction = "sell_index_vol"
    elif z <= -z_entry:
        direction = "buy_index_vol"
    else:
        direction = "neutral"

    return DispersionSignal(
        snapshot=snapshot,
        z_score=z,
        mean=mean,
        stdev=stdev,
        sample_size=len(usable),
        direction=direction,
    )
