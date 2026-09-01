"""Realised volatility and realised correlation from daily closes.

Why this module exists
----------------------
The dispersion signal needs a baseline: is today's *implied* correlation high or
low compared with something meaningful?

The obvious answer -- a z-score of the implied reading against its own history --
requires a long series of historical implied volatilities, and Alpaca does not
serve those.  Reconstructing them by inverting Black-Scholes over historical
option bars is possible but fragile: on the free tier those bars are sparse for
anything off the money, and a signal silently resting on a dozen noisy
observations is worse than no signal at all.

So the primary baseline is **realised** correlation, computed from daily stock
returns.  Stock bars are dense and reliable even on the free tier, and the
comparison is the one real dispersion desks actually trade:

    correlation risk premium  =  implied correlation  -  realised correlation

When implied sits well above realised, index volatility is expensive relative to
what the constituents have actually been doing together, and the desk sells
index vol against the names.  When it sits below, the reverse.

The caveat is stated rather than buried: realised correlation is backward
looking and implied is forward looking, so a positive premium is *normal* on
average -- index options carry a structural bid because they hedge portfolios.
The desk therefore trades deviations from the premium's own recent level, never
the raw premium merely being positive.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

TRADING_DAYS_PER_YEAR = 252

# Below this many returns the estimates are too noisy to size a trade with.
MIN_RETURNS = 20


class RealizedError(ValueError):
    """Raised when realised statistics cannot be computed from the inputs."""


@dataclass(frozen=True)
class RealizedStats:
    """Backward-looking volatility and correlation for one basket."""

    volatilities: dict[str, float]  # annualised, decimal (0.25 == 25%)
    index_volatility: float
    average_correlation: float
    implied_index_volatility_from_parts: float
    sample_size: int

    @property
    def dispersion_ratio(self) -> float:
        """The DR realised behaviour implies.

        Directly comparable to the implied DR computed from today's chain.
        """
        return self.index_volatility / self.implied_index_volatility_from_parts


def log_returns(closes: list[float]) -> list[float]:
    """Daily log returns. A non-positive price is a data error, not a zero return."""
    if len(closes) < 2:
        return []

    out: list[float] = []
    for previous, current in zip(closes, closes[1:]):
        if previous <= 0 or current <= 0:
            raise RealizedError(f"non-positive close price in series: {previous} -> {current}")
        out.append(math.log(current / previous))
    return out


def realized_volatility(closes: list[float], min_returns: int = MIN_RETURNS) -> float | None:
    """Annualised realised volatility from daily closes.

    Returns ``None`` when there are too few observations, rather than a number
    computed from a handful of points that would look equally authoritative.

    ``min_returns`` exists for one legitimate case: *measuring* a realised
    outcome over a short forward window, as the backtester does. An estimate
    that feeds a trading decision must keep the strict default -- sizing a
    position off ten noisy returns is how a strategy convinces itself it has an
    edge it does not have.
    """
    returns = log_returns(closes)
    if len(returns) < max(2, min_returns):
        return None

    daily = statistics.stdev(returns)
    if daily <= 0 or not math.isfinite(daily):
        return None
    return daily * math.sqrt(TRADING_DAYS_PER_YEAR)


def pairwise_correlation(
    a: list[float], b: list[float], min_returns: int = MIN_RETURNS
) -> float | None:
    """Pearson correlation of two aligned return series."""
    if len(a) != len(b):
        raise RealizedError(f"return series must be the same length, got {len(a)} and {len(b)}")
    if len(a) < max(2, min_returns):
        return None

    try:
        rho = statistics.correlation(a, b)
    except statistics.StatisticsError:
        # Raised when a series has zero variance; correlation is undefined.
        return None
    return rho if math.isfinite(rho) else None


def average_pairwise_correlation(
    returns: dict[str, list[float]],
    weights: dict[str, float],
    min_returns: int = MIN_RETURNS,
) -> float | None:
    """Weighted average pairwise correlation across the basket.

    The weighting matches the implied-correlation formula in ``dispersion.py``:
    each pair contributes in proportion to ``w_i * w_j``, so the two numbers are
    directly comparable.  An unweighted average would let the smallest names
    dominate a metric the largest names actually drive.
    """
    symbols = [s for s in weights if s in returns]
    if len(symbols) < 2:
        return None

    numerator = 0.0
    denominator = 0.0

    for i, sym_i in enumerate(symbols):
        for sym_j in symbols[i + 1 :]:
            rho = pairwise_correlation(returns[sym_i], returns[sym_j], min_returns)
            if rho is None:
                continue
            pair_weight = weights[sym_i] * weights[sym_j]
            numerator += pair_weight * rho
            denominator += pair_weight

    if denominator <= 0:
        return None
    return numerator / denominator


def compute_realized_stats(
    closes: dict[str, list[float]], weights: dict[str, float]
) -> RealizedStats | None:
    """Realised volatility and correlation for a weighted basket.

    Returns ``None`` if any basket member lacks enough clean data.  Partial
    baskets are refused deliberately: a correlation computed over four of six
    names is not the correlation of the basket being traded, and quietly
    substituting one for the other is how a signal stops meaning what its name
    says.
    """
    if not weights:
        raise RealizedError("no basket weights supplied")

    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        raise RealizedError(f"weights must sum to 1.0, got {total_weight}")

    returns: dict[str, list[float]] = {}
    vols: dict[str, float] = {}

    for symbol in weights:
        series = closes.get(symbol)
        if not series:
            return None

        vol = realized_volatility(series)
        if vol is None:
            return None

        returns[symbol] = log_returns(series)
        vols[symbol] = vol

    # Align every series to the shortest so the correlation matrix is computed
    # over a common window rather than over mismatched date ranges.
    shortest = min(len(r) for r in returns.values())
    if shortest < MIN_RETURNS:
        return None
    returns = {sym: series[-shortest:] for sym, series in returns.items()}

    rho = average_pairwise_correlation(returns, weights)
    if rho is None:
        return None

    # Index volatility implied by the parts and their realised correlation:
    #   sigma_idx^2 = sum_i w_i^2 s_i^2 + rho * sum_{i != j} w_i w_j s_i s_j
    weighted = {sym: weights[sym] * vols[sym] for sym in weights}
    total = sum(weighted.values())
    sum_of_squares = sum(v * v for v in weighted.values())
    cross_term = total * total - sum_of_squares

    variance = sum_of_squares + rho * cross_term
    if variance <= 0:
        # A sufficiently negative average correlation can drive the implied
        # index variance non-positive. That is not a tradable state; it means
        # the estimate has broken down.
        return None

    return RealizedStats(
        volatilities=vols,
        index_volatility=math.sqrt(variance),
        average_correlation=rho,
        implied_index_volatility_from_parts=total,
        sample_size=shortest,
    )


def correlation_risk_premium(
    implied_correlation: float | None, realized_correlation: float | None
) -> float | None:
    """Implied minus realised correlation.

    Positive means index options price more co-movement than the constituents
    have recently shown -- the state in which selling index volatility against
    the names is compensated.  ``None`` propagates when either input is missing,
    so a missing estimate can never be read as a zero premium.
    """
    if implied_correlation is None or realized_correlation is None:
        return None
    if not (math.isfinite(implied_correlation) and math.isfinite(realized_correlation)):
        return None
    return implied_correlation - realized_correlation
