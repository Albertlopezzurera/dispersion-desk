"""The traded basket and its index weights.

Alpaca does not expose index composition, so the constituent weights below are a
**static, dated snapshot** that must be refreshed by hand.  This is a real
limitation, not an oversight, and it is surfaced three ways: in this docstring,
in the README, and through :func:`weights_age_days`, which the risk engine reads
so a stale basket can be flagged in the UI instead of silently trusted.

Why a sub-basket rather than all 500 SPY names
----------------------------------------------
Trading 500 option chains is neither feasible in a paper account nor sensible:
the long tail is illiquid and bid-ask cost would swamp any edge.  Real dispersion
desks trade a liquid subset too.  The consequence is a **basis error** -- our
basket is not SPY -- which we quantify rather than hide: :func:`basket_coverage`
reports the fraction of index weight represented, and the dispersion ratio is
interpreted as a signal about *this basket versus the index*, never as a
replication of SPY.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# Provenance: approximate index weights for the SPDR S&P 500 ETF Trust as of the
# date below. Weights drift daily with prices; refresh before relying on them.
WEIGHTS_AS_OF = date(2026, 8, 28)

# Raw index weights (fraction of the whole index, NOT of the basket).
_RAW_INDEX_WEIGHTS: dict[str, float] = {
    "NVDA": 0.0771,
    "MSFT": 0.0682,
    "AAPL": 0.0594,
    "AMZN": 0.0401,
    "META": 0.0286,
    "AVGO": 0.0259,
    "GOOGL": 0.0224,
    "TSLA": 0.0198,
    "GOOG": 0.0184,
    "BRK.B": 0.0163,
    "JPM": 0.0148,
    "LLY": 0.0131,
}

# BRK.B lacks an option chain suitable for this strategy and uses a different
# symbology; GOOG is dropped because GOOGL already carries the same risk factor
# and holding both would double-count one company's volatility.
_EXCLUDED: frozenset[str] = frozenset({"BRK.B", "GOOG"})

INDEX_SYMBOL = "SPY"


class UniverseError(ValueError):
    """Raised when a basket cannot be constructed."""


@dataclass(frozen=True)
class BasketMember:
    symbol: str
    index_weight: float  # share of the whole index
    basket_weight: float  # share of the basket, renormalised to sum to 1


def basket(max_names: int = 8) -> list[BasketMember]:
    """The tradable basket, largest index weight first.

    Weights are renormalised so ``sum(basket_weight) == 1``.  That renormalisation
    is what makes the dispersion ratio well defined on a subset: it compares the
    index's implied volatility against a properly weighted average of the names
    actually traded, rather than against an arbitrary partial sum.
    """
    if max_names < 1:
        raise UniverseError(f"max_names must be at least 1, got {max_names}")

    eligible = [(s, w) for s, w in _RAW_INDEX_WEIGHTS.items() if s not in _EXCLUDED]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    selected = eligible[:max_names]

    total = sum(w for _, w in selected)
    if total <= 0:
        raise UniverseError("selected basket has zero total weight")

    return [BasketMember(symbol=s, index_weight=w, basket_weight=w / total) for s, w in selected]


def basket_weights(max_names: int = 8) -> dict[str, float]:
    """Basket weights keyed by symbol, summing to 1. Shape expected by ``dispersion``."""
    return {m.symbol: m.basket_weight for m in basket(max_names)}


def basket_coverage(max_names: int = 8) -> float:
    """Fraction of total index weight the basket represents.

    Shown in the UI so the basis error stays visible: a coverage of 0.30 means
    70% of the index's volatility comes from names the desk does not hold.
    """
    return sum(m.index_weight for m in basket(max_names))


def weights_age_days(today: date | None = None) -> int:
    """Days since the weight snapshot was taken. Read by the staleness gate."""
    return ((today or date.today()) - WEIGHTS_AS_OF).days
