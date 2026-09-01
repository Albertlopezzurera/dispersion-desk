"""The deterministic risk engine: the only component with veto authority.

Design rules, in order of importance
------------------------------------
1. **No LLM touches this file.**  Every decision here is arithmetic against a
   declared limit.  A language model can propose a basket and can argue against
   one, but it can never authorise one.  That separation is the reason the desk
   can state what it will and will not do and be believed.

2. **Limits are checked post-trade, not pre-trade.**  The question is never "is
   this basket small?" but "if this basket fills, is the *portfolio* still
   inside its envelope?".  Checking the basket in isolation is how books drift
   past their limits one acceptable-looking trade at a time.

3. **Fail closed.**  Missing data is a rejection, not a pass.  If a quote has no
   timestamp, if implied volatility could not be solved, if net asset value is
   unknown -- the answer is no.  Every gate is written so that the absence of
   evidence blocks the trade.

4. **Every rejection carries its arithmetic.**  A ``RiskCheck`` records the
   observed value and the limit it was measured against, so the Risk Center can
   show *why* rather than just *that* something was refused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.config import Settings

# Standard US equity option contract multiplier: one contract covers 100 shares.
CONTRACT_MULTIPLIER = 100

# An implied correlation outside this band means the index vol and the basket
# vols are mutually inconsistent -- almost always a feed problem rather than a
# genuine opportunity. Correlation is bounded by [-1, 1] in theory; we allow a
# small margin before vetoing so that ordinary noise is not treated as an error.
_CORRELATION_SANITY_BAND = (-1.05, 1.05)


@dataclass(frozen=True)
class RiskCheck:
    """One gate's verdict, with the numbers that produced it."""

    name: str
    passed: bool
    message: str
    observed: float | None = None
    limit: float | None = None

    def describe(self) -> str:
        if self.observed is None or self.limit is None:
            return self.message
        return f"{self.message} (observed {self.observed:.4g}, limit {self.limit:.4g})"


@dataclass(frozen=True)
class PositionGreeks:
    """Aggregate greeks, already scaled to dollars per unit move."""

    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0

    def __add__(self, other: "PositionGreeks") -> "PositionGreeks":
        return PositionGreeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
        )


@dataclass(frozen=True)
class ProposedLeg:
    """One leg of a candidate basket, with its per-share greeks."""

    symbol: str
    underlying: str
    side: str  # "buy" | "sell"
    quantity: int  # contracts, always positive
    price: float  # per share, the limit price the desk intends to pay/receive
    delta: float
    gamma: float
    vega: float
    theta: float
    spread_pct_of_mid: float | None
    open_interest: int | None
    quote_age_seconds: float | None
    implied_volatility: float | None

    @property
    def signed_contracts(self) -> int:
        return self.quantity if self.side == "buy" else -self.quantity

    @property
    def scaled_greeks(self) -> PositionGreeks:
        """Per-share greeks scaled by direction, quantity and the multiplier."""
        factor = self.signed_contracts * CONTRACT_MULTIPLIER
        return PositionGreeks(
            delta=self.delta * factor,
            gamma=self.gamma * factor,
            vega=self.vega * factor,
            theta=self.theta * factor,
        )


@dataclass(frozen=True)
class BasketProposal:
    """A complete candidate trade, as handed to the risk engine."""

    basket_id: str
    direction: str  # "sell_index_vol" | "buy_index_vol"
    legs: list[ProposedLeg]
    max_loss: float  # worst-case dollar loss; must be finite and positive
    rationale: str = ""

    @property
    def greeks(self) -> PositionGreeks:
        total = PositionGreeks()
        for leg in self.legs:
            total = total + leg.scaled_greeks
        return total

    def risk_by_underlying(self) -> dict[str, float]:
        """Split ``max_loss`` across underlyings by gross notional contribution.

        Exact attribution of a multi-leg worst case is path dependent; gross
        notional is a deliberately conservative proxy that never understates a
        single name's share.
        """
        gross: dict[str, float] = {}
        for leg in self.legs:
            gross[leg.underlying] = gross.get(leg.underlying, 0.0) + abs(
                leg.price * leg.quantity * CONTRACT_MULTIPLIER
            )
        total = sum(gross.values())
        if total <= 0:
            return {}
        return {sym: self.max_loss * value / total for sym, value in gross.items()}


@dataclass(frozen=True)
class PortfolioState:
    """The book as it stands, before the proposed basket."""

    net_asset_value: float
    daily_pnl: float
    greeks: PositionGreeks
    open_defined_risk: float
    risk_by_underlying: dict[str, float] = field(default_factory=dict)
    market_is_open: bool = True
    weights_age_days: int = 0


@dataclass(frozen=True)
class RiskDecision:
    """The engine's verdict on one proposal."""

    approved: bool
    checks: list[RiskCheck]
    basket_id: str
    evaluated_at: datetime

    @property
    def failures(self) -> list[RiskCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def rejection_summary(self) -> str:
        failures = self.failures
        if not failures:
            return ""
        return "; ".join(c.describe() for c in failures)

    def render(self) -> str:
        """Human-readable verdict, as shown in the UI and written to the journal."""
        header = "TRADE APPROVED" if self.approved else "TRADE REJECTED"
        lines = [header, f"Basket: {self.basket_id}", ""]
        for check in self.checks:
            lines.append(f"  [{'PASS' if check.passed else 'FAIL'}] {check.describe()}")
        if not self.approved:
            lines += ["", "Reason:", f"  {self.rejection_summary}"]
        return "\n".join(lines)


class RiskEngine:
    """Evaluates proposals against the configured envelope. Stateless by design."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(self, proposal: BasketProposal, portfolio: PortfolioState) -> RiskDecision:
        """Run every gate and approve only if all of them pass.

        Gates are never short-circuited: the Risk Center is more useful showing
        all four reasons a basket failed than only the first one.
        """
        checks: list[RiskCheck] = []
        checks.extend(self._structural_checks(proposal))
        checks.extend(self._data_quality_checks(proposal, portfolio))
        checks.extend(self._circuit_breaker_checks(portfolio))
        checks.extend(self._greek_checks(proposal, portfolio))
        checks.extend(self._capital_checks(proposal, portfolio))

        return RiskDecision(
            approved=all(c.passed for c in checks),
            checks=checks,
            basket_id=proposal.basket_id,
            evaluated_at=datetime.now(timezone.utc),
        )

    # --- structure ---------------------------------------------------------

    def _structural_checks(self, proposal: BasketProposal) -> list[RiskCheck]:
        checks = [
            RiskCheck(
                name="has_legs",
                passed=bool(proposal.legs),
                message="Proposal contains at least one leg",
            ),
            RiskCheck(
                name="max_loss_is_defined",
                passed=(
                    math.isfinite(proposal.max_loss)
                    and proposal.max_loss > 0
                ),
                message=(
                    "Worst-case loss is finite and positive. The desk only trades "
                    "defined-risk structures; an undefined max loss is an automatic veto"
                ),
                observed=proposal.max_loss if math.isfinite(proposal.max_loss) else None,
            ),
        ]

        bad_quantities = [leg.symbol for leg in proposal.legs if leg.quantity < 1]
        checks.append(
            RiskCheck(
                name="positive_quantities",
                passed=not bad_quantities,
                message=(
                    "Every leg has a positive contract count"
                    if not bad_quantities
                    else f"Legs with non-positive quantity: {bad_quantities}"
                ),
            )
        )

        bad_prices = [leg.symbol for leg in proposal.legs if leg.price <= 0]
        checks.append(
            RiskCheck(
                name="positive_prices",
                passed=not bad_prices,
                message=(
                    "Every leg has a positive limit price"
                    if not bad_prices
                    else f"Legs priced at or below zero: {bad_prices}"
                ),
            )
        )
        return checks

    # --- data quality ------------------------------------------------------

    def _data_quality_checks(
        self, proposal: BasketProposal, portfolio: PortfolioState
    ) -> list[RiskCheck]:
        """Gates that reject on missing or stale inputs.

        These matter more than usual here: on the free indicative feed the desk
        is working with derived quotes, so trusting them blindly would mean
        trading on numbers that never existed in the market.
        """
        s = self.settings
        checks: list[RiskCheck] = []

        stale = [
            leg.symbol
            for leg in proposal.legs
            if leg.quote_age_seconds is None or leg.quote_age_seconds > s.max_quote_age_seconds
        ]
        checks.append(
            RiskCheck(
                name="quote_freshness",
                passed=not stale,
                message=(
                    "All quotes are fresh enough to act on"
                    if not stale
                    else f"Stale or untimestamped quotes: {stale}"
                ),
                limit=float(s.max_quote_age_seconds),
            )
        )

        wide = [
            leg.symbol
            for leg in proposal.legs
            if leg.spread_pct_of_mid is None or leg.spread_pct_of_mid > s.max_spread_pct_of_mid
        ]
        checks.append(
            RiskCheck(
                name="liquidity_spread",
                passed=not wide,
                message=(
                    "All legs trade inside the maximum bid-ask spread"
                    if not wide
                    else f"Legs too wide or with no two-sided market: {wide}"
                ),
                limit=s.max_spread_pct_of_mid,
            )
        )

        thin = [
            leg.symbol
            for leg in proposal.legs
            if leg.open_interest is None or leg.open_interest < s.min_open_interest
        ]
        checks.append(
            RiskCheck(
                name="open_interest",
                passed=not thin,
                message=(
                    "All legs meet the minimum open interest"
                    if not thin
                    else f"Legs below the open-interest floor: {thin}"
                ),
                limit=float(s.min_open_interest),
            )
        )

        bad_iv = [
            leg.symbol
            for leg in proposal.legs
            if leg.implied_volatility is None
            or not math.isfinite(leg.implied_volatility)
            or not (s.min_iv <= leg.implied_volatility <= s.max_iv)
        ]
        checks.append(
            RiskCheck(
                name="implied_volatility_sanity",
                passed=not bad_iv,
                message=(
                    "Implied volatility solved and in range for every leg"
                    if not bad_iv
                    else f"Legs with unsolvable or out-of-range IV: {bad_iv}"
                ),
            )
        )

        checks.append(
            RiskCheck(
                name="index_weights_freshness",
                passed=portfolio.weights_age_days <= s.max_weights_age_days,
                message=(
                    "Index constituent weights are recent enough to trust. They are a "
                    "manual snapshot and drift with prices"
                ),
                observed=float(portfolio.weights_age_days),
                limit=float(s.max_weights_age_days),
            )
        )

        checks.append(
            RiskCheck(
                name="market_hours",
                passed=portfolio.market_is_open,
                message=(
                    "Market is open. Option quotes outside regular hours are not "
                    "actionable and fills cannot be trusted"
                ),
            )
        )
        return checks

    # --- circuit breaker ---------------------------------------------------

    def _circuit_breaker_checks(self, portfolio: PortfolioState) -> list[RiskCheck]:
        s = self.settings
        nav_known = math.isfinite(portfolio.net_asset_value) and portfolio.net_asset_value > 0

        checks = [
            RiskCheck(
                name="nav_known",
                passed=nav_known,
                message="Net asset value is known; risk cannot be sized without it",
                observed=portfolio.net_asset_value if math.isfinite(portfolio.net_asset_value) else None,
            )
        ]

        if not nav_known:
            # Without NAV the loss limit is not computable. Fail closed rather
            # than skip the check.
            checks.append(
                RiskCheck(
                    name="daily_loss_circuit_breaker",
                    passed=False,
                    message="Cannot evaluate the daily loss limit without a known NAV",
                )
            )
            return checks

        loss_pct = -100.0 * portfolio.daily_pnl / portfolio.net_asset_value
        checks.append(
            RiskCheck(
                name="daily_loss_circuit_breaker",
                passed=loss_pct < s.max_daily_loss_pct,
                message=(
                    "Daily loss is within the circuit-breaker threshold. Once tripped, "
                    "the desk stops opening positions for the rest of the session"
                ),
                observed=loss_pct,
                limit=s.max_daily_loss_pct,
            )
        )
        return checks

    # --- greeks ------------------------------------------------------------

    def _greek_checks(
        self, proposal: BasketProposal, portfolio: PortfolioState
    ) -> list[RiskCheck]:
        """Post-trade greek limits.

        The net-delta band is the important one: this is a direction-neutral
        strategy, so a book that drifts directional is not merely over a limit,
        it is no longer running the strategy it claims to run.
        """
        s = self.settings
        post = portfolio.greeks + proposal.greeks

        return [
            RiskCheck(
                name="net_delta_band",
                passed=abs(post.delta) <= s.max_net_delta,
                message=(
                    "Post-trade net delta stays inside the neutrality band. This is the "
                    "correctness test for a direction-neutral desk"
                ),
                observed=abs(post.delta),
                limit=s.max_net_delta,
            ),
            RiskCheck(
                name="portfolio_vega",
                passed=abs(post.vega) <= s.max_portfolio_vega,
                message="Post-trade absolute vega is within limit. Vega is the risk the desk chooses to take",
                observed=abs(post.vega),
                limit=s.max_portfolio_vega,
            ),
            RiskCheck(
                name="portfolio_gamma",
                passed=abs(post.gamma) <= s.max_portfolio_gamma,
                message="Post-trade absolute gamma is within limit",
                observed=abs(post.gamma),
                limit=s.max_portfolio_gamma,
            ),
            RiskCheck(
                name="daily_theta",
                passed=abs(post.theta) <= s.max_daily_theta,
                message="Post-trade daily time decay is within limit",
                observed=abs(post.theta),
                limit=s.max_daily_theta,
            ),
        ]

    # --- capital -----------------------------------------------------------

    def _capital_checks(
        self, proposal: BasketProposal, portfolio: PortfolioState
    ) -> list[RiskCheck]:
        s = self.settings
        nav = portfolio.net_asset_value
        if not (math.isfinite(nav) and nav > 0):
            return [
                RiskCheck(
                    name="capital_limits",
                    passed=False,
                    message="Cannot evaluate capital limits without a known NAV",
                )
            ]

        basket_pct = 100.0 * proposal.max_loss / nav
        total_pct = 100.0 * (portfolio.open_defined_risk + proposal.max_loss) / nav

        checks = [
            RiskCheck(
                name="risk_per_basket",
                passed=basket_pct <= s.max_risk_per_basket_pct,
                message="Worst-case loss on this basket is within the per-trade cap",
                observed=basket_pct,
                limit=s.max_risk_per_basket_pct,
            ),
            RiskCheck(
                name="total_defined_risk",
                passed=total_pct <= s.max_total_risk_pct,
                message="Aggregate worst-case loss across all open baskets is within the cap",
                observed=total_pct,
                limit=s.max_total_risk_pct,
            ),
        ]

        # Concentration is evaluated per underlying, post-trade.
        post_by_symbol = dict(portfolio.risk_by_underlying)
        for symbol, risk in proposal.risk_by_underlying().items():
            post_by_symbol[symbol] = post_by_symbol.get(symbol, 0.0) + risk

        worst_symbol, worst_pct = "", 0.0
        for symbol, risk in post_by_symbol.items():
            pct = 100.0 * risk / nav
            if pct > worst_pct:
                worst_symbol, worst_pct = symbol, pct

        checks.append(
            RiskCheck(
                name="underlying_concentration",
                passed=worst_pct <= s.max_underlying_concentration_pct,
                message=(
                    "No single underlying exceeds the concentration cap"
                    if worst_pct <= s.max_underlying_concentration_pct
                    else f"{worst_symbol} exceeds the per-underlying concentration cap"
                ),
                observed=worst_pct,
                limit=s.max_underlying_concentration_pct,
            )
        )
        return checks


def correlation_is_sane(implied_correlation: float | None) -> RiskCheck:
    """Standalone gate on the dispersion signal's own consistency.

    Called by the orchestrator before a proposal is even built: an implied
    correlation far outside [-1, 1] means index and constituent volatilities
    disagree beyond what any correlation could explain, so the signal itself is
    unusable regardless of how attractive the resulting trade looks.
    """
    if implied_correlation is None or not math.isfinite(implied_correlation):
        return RiskCheck(
            name="implied_correlation_sanity",
            passed=False,
            message="Implied correlation could not be computed; the dispersion signal is unusable",
        )

    low, high = _CORRELATION_SANITY_BAND
    return RiskCheck(
        name="implied_correlation_sanity",
        passed=low <= implied_correlation <= high,
        message=(
            "Implied correlation is inside its theoretical bounds"
            if low <= implied_correlation <= high
            else "Implied correlation is outside [-1, 1]; index and basket volatilities "
            "are mutually inconsistent, which points to a data problem rather than an edge"
        ),
        observed=implied_correlation,
        limit=high,
    )
