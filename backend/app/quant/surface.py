"""Turning a raw option chain into one trustworthy at-the-money volatility.

The dispersion signal compares a single implied volatility per underlying, so
this module's whole job is to reduce hundreds of noisy contracts to one number
per name -- and to refuse when it cannot do so honestly.

Why the desk re-solves implied volatility itself
------------------------------------------------
Alpaca publishes greeks and an ``impliedVolatility`` field on its snapshots, and
those are kept for cross-checking.  But the strategy trades on volatility we
derive ourselves, from the quoted mid, for three reasons:

* the indicative feed's published IV may be computed against inputs we cannot
  see or reproduce;
* a value we compute can be checked against put-call parity, and is checked;
* when the solver fails we learn the quote is unusable, whereas a supplied
  number always looks valid.

Both call and put at the same strike are solved and averaged.  In a consistent
market these agree; a large gap means the quotes violate put-call parity, which
is a data-quality signal the desk acts on rather than smooths over.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from app.alpaca.client import OptionQuote
from app.quant.black_scholes import implied_volatility

logger = logging.getLogger(__name__)

# Calendar years are the model's unit, and 365.0 matches how the pricer and the
# greeks are scaled elsewhere.
DAYS_PER_YEAR = 365.0

# Call and put IV at one strike should agree closely. A wider gap means the two
# sides of the market are inconsistent.
MAX_PARITY_DISAGREEMENT = 0.10  # 10 volatility points

# Strikes either side of spot contributing to the ATM estimate. Two gives some
# averaging without drifting into the skewed wings.
ATM_STRIKES_PER_SIDE = 2


class SurfaceError(ValueError):
    """Raised when a usable volatility cannot be extracted from a chain."""


@dataclass(frozen=True)
class StrikeVol:
    """Implied volatility at one strike, solved from both sides where possible."""

    strike: float
    call_iv: float | None
    put_iv: float | None
    parity_gap: float | None
    moneyness: float  # strike / spot

    @property
    def iv(self) -> float | None:
        """Best single estimate at this strike.

        Both sides present and agreeing -> their average.  Only one side present
        -> that one.  Both present but disagreeing beyond
        ``MAX_PARITY_DISAGREEMENT`` -> ``None``, because a parity violation means
        at least one quote is wrong and we cannot tell which.
        """
        if self.call_iv is not None and self.put_iv is not None:
            if self.parity_gap is not None and self.parity_gap > MAX_PARITY_DISAGREEMENT:
                return None
            return (self.call_iv + self.put_iv) / 2.0
        return self.call_iv if self.call_iv is not None else self.put_iv


@dataclass(frozen=True)
class AtmVolatility:
    """One underlying's at-the-money volatility, with its provenance."""

    underlying: str
    spot: float
    expiration: date
    days_to_expiry: int
    implied_volatility: float
    strikes_used: list[StrikeVol]
    contracts_used: list[str]
    max_quote_age_seconds: float | None

    @property
    def parity_gaps(self) -> list[float]:
        return [s.parity_gap for s in self.strikes_used if s.parity_gap is not None]


def year_fraction(today: date, expiration: date) -> float:
    """Time to expiry in years. Never negative."""
    return max((expiration - today).days, 0) / DAYS_PER_YEAR


def select_expiration(
    quotes: list[OptionQuote], today: date, dte_min: int, dte_max: int
) -> date | None:
    """Choose the expiry nearest the middle of the target window.

    Restricting to a window keeps the cross-underlying comparison honest: an
    index volatility at 30 days and a single-name volatility at 7 days are not
    comparable quantities, and mixing them would make the dispersion ratio
    meaningless.
    """
    target = (dte_min + dte_max) / 2.0
    candidates = {
        q.expiration for q in quotes if dte_min <= (q.expiration - today).days <= dte_max
    }
    if not candidates:
        return None
    return min(candidates, key=lambda e: abs((e - today).days - target))


def _solve_side(quote: OptionQuote, spot: float, t: float, r: float) -> float | None:
    mid = quote.mid
    if mid is None:
        return None
    return implied_volatility(mid, spot, quote.strike, t, r, 0.0, quote.option_type)


def build_strike_vols(
    quotes: list[OptionQuote], spot: float, expiration: date, today: date, risk_free_rate: float
) -> list[StrikeVol]:
    """Solve implied volatility at every strike of one expiry, both sides."""
    if spot <= 0:
        raise SurfaceError(f"spot must be positive, got {spot}")

    t = year_fraction(today, expiration)
    if t <= 0:
        raise SurfaceError(f"expiration {expiration} is not in the future relative to {today}")

    by_strike: dict[float, dict[str, OptionQuote]] = defaultdict(dict)
    for q in quotes:
        if q.expiration == expiration:
            by_strike[q.strike][q.option_type] = q

    out: list[StrikeVol] = []
    for strike, sides in sorted(by_strike.items()):
        call = sides.get("call")
        put = sides.get("put")
        call_iv = _solve_side(call, spot, t, risk_free_rate) if call else None
        put_iv = _solve_side(put, spot, t, risk_free_rate) if put else None

        gap = abs(call_iv - put_iv) if (call_iv is not None and put_iv is not None) else None
        out.append(
            StrikeVol(
                strike=strike,
                call_iv=call_iv,
                put_iv=put_iv,
                parity_gap=gap,
                moneyness=strike / spot,
            )
        )
    return out


def atm_volatility(
    underlying: str,
    quotes: list[OptionQuote],
    spot: float,
    today: date,
    dte_min: int,
    dte_max: int,
    risk_free_rate: float,
    strikes_per_side: int = ATM_STRIKES_PER_SIDE,
) -> AtmVolatility | None:
    """Extract one at-the-money implied volatility for an underlying.

    Returns ``None`` -- never a guess -- when the chain cannot support a
    trustworthy estimate: no expiry in the target window, no solvable strikes
    near the money, or a nonsensical spot.  A ``None`` propagates as a missing
    constituent, and the dispersion engine refuses to build a signal without
    every member of the basket.
    """
    if spot <= 0:
        logger.warning("%s: non-positive spot %s; skipping", underlying, spot)
        return None
    if not quotes:
        logger.warning("%s: empty option chain", underlying)
        return None

    expiration = select_expiration(quotes, today, dte_min, dte_max)
    if expiration is None:
        logger.warning(
            "%s: no expiry between %d and %d days out; cannot compare with the index",
            underlying,
            dte_min,
            dte_max,
        )
        return None

    strike_vols = build_strike_vols(quotes, spot, expiration, today, risk_free_rate)
    solvable = [s for s in strike_vols if s.iv is not None]
    if not solvable:
        logger.warning("%s: no strike produced a solvable implied volatility", underlying)
        return None

    # Take the strikes closest to spot, then average. A small neighbourhood
    # rather than a single strike stops one bad quote setting the whole
    # underlying's volatility.
    solvable.sort(key=lambda s: abs(s.moneyness - 1.0))
    chosen = solvable[: max(1, strikes_per_side * 2)]

    ivs = [s.iv for s in chosen if s.iv is not None]
    if not ivs:
        return None
    atm_iv = sum(ivs) / len(ivs)

    chosen_strikes = {s.strike for s in chosen}
    relevant = [q for q in quotes if q.expiration == expiration and q.strike in chosen_strikes]
    ages = [age for q in relevant if (age := q.age_seconds()) is not None]

    return AtmVolatility(
        underlying=underlying,
        spot=spot,
        expiration=expiration,
        days_to_expiry=(expiration - today).days,
        implied_volatility=atm_iv,
        strikes_used=chosen,
        contracts_used=[q.symbol for q in relevant],
        max_quote_age_seconds=max(ages) if ages else None,
    )
