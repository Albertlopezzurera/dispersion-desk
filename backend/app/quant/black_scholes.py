"""Black-Scholes-Merton pricing, greeks and implied-volatility inversion.

This module is the numerical foundation of the desk.  Everything downstream --
the volatility surface, the dispersion signal, position sizing and the P&L
attribution -- is derived from the functions here, so this file is deliberately
small, dependency-light and heavily tested (see ``backend/tests/test_black_scholes.py``).

Modelling note
--------------
Listed equity and ETF options in the US are *American* style, but the entire
options market -- Alpaca's own snapshot endpoint included -- quotes implied
volatility using the European Black-Scholes-Merton model.  We follow that
convention so our numbers are directly comparable with the venue's.  For the
out-of-the-money, 21-45 DTE contracts this desk trades, the early-exercise
premium is negligible; the approximation is documented in the README as a known
limitation rather than hidden.

Sign and scale conventions (chosen to match how trading desks quote greeks):

===========  ==========================================================
delta        per 1.0 move in the underlying, per share
gamma        delta change per 1.0 move in the underlying, per share
vega         P&L per **1 volatility point** (i.e. per 0.01 of sigma)
theta        P&L per **calendar day**
rho          P&L per **1 percentage point** of interest rate
===========  ==========================================================

All functions operate on a *single share*.  Contract-level values require
multiplying by the contract multiplier (normally 100); that is the caller's
job and is done explicitly in ``surface.py`` so the scaling stays visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]

# Numerical guards. A contract with only minutes of life, or a volatility below
# 0.0001%, is treated as expired/degenerate rather than fed to formulas that
# would divide by ~0.
_MIN_T = 1e-5  # years (~5 minutes)
_MIN_SIGMA = 1e-6

# Bounds for the implied-volatility search. 1e-4 = 0.01% vol, 10.0 = 1000% vol.
# Anything outside this bracket is not a real quote.
IV_LOWER_BOUND = 1e-4
IV_UPPER_BOUND = 10.0


@dataclass(frozen=True)
class Greeks:
    """Per-share risk sensitivities, in the desk conventions documented above."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class BlackScholesError(ValueError):
    """Raised when inputs cannot produce a meaningful result."""


def _validate(S: float, K: float, T: float) -> None:
    if S <= 0:
        raise BlackScholesError(f"spot must be positive, got {S}")
    if K <= 0:
        raise BlackScholesError(f"strike must be positive, got {K}")
    if T < 0:
        raise BlackScholesError(f"time to expiry cannot be negative, got {T}")


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def intrinsic_value(S: float, K: float, option_type: OptionType) -> float:
    """Payoff if the option were exercised right now."""
    return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """Black-Scholes-Merton price of a European option on one share.

    Args:
        S: spot price of the underlying.
        K: strike price.
        T: time to expiry in years.
        r: continuously-compounded risk-free rate (0.04 == 4%).
        sigma: volatility as a decimal (0.20 == 20%).
        q: continuous dividend yield.
        option_type: ``"call"`` or ``"put"``.

    At (or past) expiry, or at vanishing volatility, the option collapses to its
    discounted intrinsic value -- the limit of the formula, not a special case.
    """
    _validate(S, K, T)

    if T <= _MIN_T or sigma <= _MIN_SIGMA:
        fwd = S * math.exp(-q * T)
        discounted_k = K * math.exp(-r * T)
        if option_type == "call":
            return max(0.0, fwd - discounted_k)
        return max(0.0, discounted_k - fwd)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_s = S * math.exp(-q * T)
    disc_k = K * math.exp(-r * T)

    if option_type == "call":
        return disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
    return disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> Greeks:
    """Analytic greeks, scaled to the desk conventions.

    An expired or zero-vol option has no continuous sensitivities, so every
    greek is zero except delta, which becomes the step function of moneyness.
    """
    _validate(S, K, T)

    if T <= _MIN_T or sigma <= _MIN_SIGMA:
        if option_type == "call":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return Greeks(delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrt_t = math.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
    # Vega is per 1.0 of sigma here; /100 converts to "per volatility point".
    vega = S * disc_q * pdf_d1 * sqrt_t / 100.0

    # Theta is derived per year, then /365 for a calendar day.
    decay = -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta_year = decay - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
        rho = K * T * disc_r * norm.cdf(d2) / 100.0
    else:
        delta = disc_q * (norm.cdf(d1) - 1.0)
        theta_year = decay + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
        rho = -K * T * disc_r * norm.cdf(-d2) / 100.0

    return Greeks(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta_year / 365.0,
        rho=rho,
    )


def no_arbitrage_bounds(
    S: float, K: float, T: float, r: float, q: float, option_type: OptionType
) -> tuple[float, float]:
    """Lower and upper price bounds a European option must respect.

    A quote outside this band cannot be inverted to a real implied volatility;
    it means the quote is stale, crossed, or synthetic.  The desk treats that as
    a data-quality failure rather than clamping it into range.
    """
    disc_s = S * math.exp(-q * T)
    disc_k = K * math.exp(-r * T)
    if option_type == "call":
        return max(0.0, disc_s - disc_k), disc_s
    return max(0.0, disc_k - disc_s), disc_k


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> float | None:
    """Invert Black-Scholes for sigma using Brent's method.

    Returns ``None`` -- never a fabricated number -- when the quote cannot yield
    a real implied volatility.  Callers must treat ``None`` as "no data" and let
    the risk engine's IV-sanity gate reject the contract.  Silently returning a
    clamped guess here would inject fake volatility into the dispersion signal,
    which is precisely the failure this desk exists to avoid.

    Brent is used rather than Newton-Raphson because it is bracketed and so
    cannot diverge on the near-zero-vega wings, where deep OTM quotes live.
    """
    _validate(S, K, T)

    if price <= 0 or T <= _MIN_T:
        return None

    lower, upper = no_arbitrage_bounds(S, K, T, r, q, option_type)
    # A small tolerance absorbs rounding in penny-quoted prices; anything beyond
    # it is a genuine violation.
    tol = 1e-8
    if price < lower - tol or price > upper + tol:
        return None

    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, q, option_type) - price

    f_lo = objective(IV_LOWER_BOUND)
    f_hi = objective(IV_UPPER_BOUND)

    # Brent needs a sign change across the bracket. Without one, the price sits
    # outside what any volatility in [1e-4, 10.0] can produce.
    if f_lo * f_hi > 0:
        return None

    try:
        return float(brentq(objective, IV_LOWER_BOUND, IV_UPPER_BOUND, xtol=1e-8, maxiter=200))
    except (ValueError, RuntimeError):
        return None
