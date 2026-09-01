"""The autonomous trading cycle.

One pass, in order:

    1. safety   -- kill switch, market hours, account state
    2. data     -- option chains and spots for the index and every basket name
    3. signal   -- implied correlation (options) vs realised correlation (returns)
    4. build    -- defined-risk structures expressing that view
    5. catalyst -- LLM veto on names whose vol is explained by a known event
    6. advocate -- LLM adversarial pre-mortem (advisory)
    7. risk     -- deterministic gates; the only component that can authorise
    8. execute  -- multi-leg orders via Alpaca, if and only if approved
    9. journal  -- everything above, traded or not

Every step is logged to the journal as it happens; that is what the live Agent
Activity view streams.

The cycle is designed so the *common* outcome is "no trade".  A dispersion
opportunity worth crossing two spreads for is rare, and a system that finds one
every fifteen minutes is not finding opportunities -- it is finding noise.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.agents.analysts import AdvocateOpinion, CatalystVerdict, build_agents
from app.alpaca.client import AlpacaClient, OptionQuote
from app.config import ConfigError, Settings
from app.execution.executor import Executor, build_basket
from app.journal.db import Journal
from app.quant import dispersion, realized, surface, universe
from app.risk.engine import (
    CONTRACT_MULTIPLIER,
    PortfolioState,
    PositionGreeks,
    RiskDecision,
    RiskEngine,
    correlation_is_sane,
)

logger = logging.getLogger(__name__)

# Daily bars pulled for the realised-correlation estimate. Long enough to be
# statistically meaningful, short enough to reflect the current regime.
REALIZED_LOOKBACK_DAYS = 90

# Strike window fetched around spot, as a fraction of spot. Wide enough to hold
# the condor wings, narrow enough to keep the chain request small.
CHAIN_STRIKE_WINDOW = 0.25


@dataclass
class CycleResult:
    """What one pass of the cycle concluded."""

    cycle_id: int
    outcome: str  # traded | proposed | no_signal | vetoed | rejected | halted | error
    detail: str = ""
    signal: dict[str, Any] | None = None
    decision: RiskDecision | None = None
    memo: str = ""
    orders: list[dict] = field(default_factory=list)


class Orchestrator:
    """Runs trading cycles. Owns no strategy logic of its own."""

    def __init__(self, settings: Settings, journal: Journal | None = None) -> None:
        self.settings = settings
        self.journal = journal or Journal(settings.database_url)
        self.risk = RiskEngine(settings)
        self._running = False

    # --- helpers -----------------------------------------------------------

    def _log(self, cycle_id: int | None, step: str, message: str, level: str = "info") -> None:
        logger.info("[%s] %s", step, message)
        self.journal.log(cycle_id, step, message, level)

    async def portfolio_state(self, client: AlpacaClient) -> PortfolioState:
        """Read the book from Alpaca and aggregate it into risk-engine terms.

        Greeks on *existing* positions come from Alpaca's snapshots.  The desk
        solves its own implied volatility for decisions, but for marking a book
        already on the tape the venue's greeks are adequate and far cheaper than
        re-fetching every chain.
        """
        account, positions, clock = await asyncio.gather(
            client.get_account(), client.get_positions(), client.get_clock()
        )

        nav = float(account.get("equity") or 0.0)
        last_equity = float(account.get("last_equity") or nav)
        daily_pnl = nav - last_equity

        option_positions = [p for p in positions if p.get("asset_class") == "us_option"]
        greeks = PositionGreeks()
        risk_by_underlying: dict[str, float] = {}
        open_risk = 0.0

        if option_positions:
            snapshots = await client.get_option_snapshots([p["symbol"] for p in option_positions])
            for pos in option_positions:
                snap = snapshots.get(pos["symbol"])
                if snap is None:
                    continue
                scale = float(pos.get("qty") or 0) * CONTRACT_MULTIPLIER
                greeks = greeks + PositionGreeks(
                    delta=(snap.delta or 0.0) * scale,
                    gamma=(snap.gamma or 0.0) * scale,
                    vega=(snap.vega or 0.0) * scale,
                    theta=(snap.theta or 0.0) * scale,
                )
                # Long options risk the premium paid; short options are only ever
                # opened inside a defined-risk structure, so cost basis is a
                # conservative stand-in for their contribution.
                exposure = abs(float(pos.get("cost_basis") or 0.0))
                open_risk += exposure
                risk_by_underlying[snap.underlying] = (
                    risk_by_underlying.get(snap.underlying, 0.0) + exposure
                )

        return PortfolioState(
            net_asset_value=nav,
            daily_pnl=daily_pnl,
            greeks=greeks,
            open_defined_risk=open_risk,
            risk_by_underlying=risk_by_underlying,
            market_is_open=bool(clock.get("is_open", False)),
            weights_age_days=universe.weights_age_days(),
        )

    async def _fetch_vols(
        self, client: AlpacaClient, symbols: list[str], today: date
    ) -> tuple[dict[str, surface.AtmVolatility], dict[str, list[OptionQuote]]]:
        """At-the-money volatility and raw chain for each symbol, fetched together."""
        s = self.settings

        async def one(symbol: str):
            spot = await client.get_stock_price(symbol)
            if not spot:
                return symbol, None, []
            chain = await client.get_option_chain(
                symbol,
                expiration_gte=today + timedelta(days=s.target_dte_min),
                expiration_lte=today + timedelta(days=s.target_dte_max),
                strike_gte=spot * (1 - CHAIN_STRIKE_WINDOW),
                strike_lte=spot * (1 + CHAIN_STRIKE_WINDOW),
            )
            atm = surface.atm_volatility(
                symbol, chain, spot, today, s.target_dte_min, s.target_dte_max, s.risk_free_rate
            )
            return symbol, atm, chain

        results = await asyncio.gather(*(one(sym) for sym in symbols))
        return (
            {sym: atm for sym, atm, _ in results if atm is not None},
            {sym: chain for sym, _, chain in results},
        )

    def _record_rejection(
        self, cycle_id: int, proposal, direction: str, verdicts, reason: str, check_name: str
    ) -> None:
        self.journal.record_decision(
            cycle_id,
            {
                "basket_id": proposal.basket_id,
                "direction": direction,
                "approved": False,
                "max_loss": proposal.max_loss,
                "rationale": proposal.rationale,
                "catalyst_verdicts": [v.__dict__ for v in verdicts],
                "advocate_opinion": {},
                "memo": reason,
                "legs": [leg.__dict__ for leg in proposal.legs],
                "checks": [
                    {
                        "name": check_name,
                        "passed": False,
                        "message": reason,
                        "observed": None,
                        "limit": None,
                    }
                ],
            },
        )

    # --- the cycle ---------------------------------------------------------

    async def run_cycle(self) -> CycleResult:
        """Execute one full pass. Never raises: failures become journal entries."""
        cycle_id = self.journal.start_cycle()

        if self.settings.kill_switch:
            self._log(cycle_id, "halted", "Kill switch engaged; no analysis performed.", "warning")
            self.journal.finish_cycle(cycle_id, "halted", "kill switch")
            return CycleResult(cycle_id, "halted", "Kill switch engaged.")

        try:
            return await self._run_cycle_inner(cycle_id)
        except ConfigError as exc:
            self._log(cycle_id, "error", str(exc), "error")
            self.journal.finish_cycle(cycle_id, "error", str(exc))
            return CycleResult(cycle_id, "error", str(exc))
        except Exception as exc:  # noqa: BLE001 - a bad cycle must not kill the service
            logger.exception("cycle %s failed", cycle_id)
            self._log(cycle_id, "error", f"Unhandled failure: {exc}", "error")
            self.journal.finish_cycle(cycle_id, "error", str(exc))
            return CycleResult(cycle_id, "error", str(exc))

    async def _run_cycle_inner(self, cycle_id: int) -> CycleResult:
        s = self.settings
        today = datetime.now(timezone.utc).date()
        names = [m.symbol for m in universe.basket(s.basket_size)]
        weights = universe.basket_weights(s.basket_size)

        async with AlpacaClient(s) as client:
            self._log(cycle_id, "start", f"Cycle started. Basket: {', '.join(names)}")

            portfolio = await self.portfolio_state(client)
            self._log(
                cycle_id,
                "portfolio",
                f"NAV ${portfolio.net_asset_value:,.0f}, day P&L ${portfolio.daily_pnl:+,.0f}, "
                f"net delta {portfolio.greeks.delta:+.0f}, vega {portfolio.greeks.vega:+.0f}",
            )

            if not portfolio.market_is_open:
                msg = "Market is closed; quotes are not actionable."
                self._log(cycle_id, "halted", msg)
                self.journal.finish_cycle(cycle_id, "no_signal", msg)
                return CycleResult(cycle_id, "no_signal", msg)

            # --- data ------------------------------------------------------
            self._log(
                cycle_id, "market_data", f"Fetching chains for {s.index_symbol} + {len(names)} names"
            )
            vols, chains = await self._fetch_vols(client, [s.index_symbol] + names, today)

            index_atm = vols.get(s.index_symbol)
            if index_atm is None:
                msg = f"No usable {s.index_symbol} volatility; cannot form a view."
                self._log(cycle_id, "no_signal", msg, "warning")
                self.journal.finish_cycle(cycle_id, "no_signal", msg)
                return CycleResult(cycle_id, "no_signal", msg)

            missing = [n for n in names if n not in vols]
            if missing:
                msg = f"Incomplete basket, missing {missing}; refusing a partial signal."
                self._log(cycle_id, "no_signal", msg, "warning")
                self.journal.finish_cycle(cycle_id, "no_signal", msg)
                return CycleResult(cycle_id, "no_signal", msg)

            # --- signal ----------------------------------------------------
            constituent_ivs = {n: vols[n].implied_volatility for n in names}
            snap = dispersion.compute_snapshot(
                index_atm.implied_volatility, constituent_ivs, weights
            )

            sanity = correlation_is_sane(snap.implied_correlation)
            if not sanity.passed:
                self._log(cycle_id, "vetoed", sanity.describe(), "warning")
                self.journal.finish_cycle(cycle_id, "vetoed", sanity.describe())
                return CycleResult(cycle_id, "vetoed", sanity.describe())

            start = (today - timedelta(days=REALIZED_LOOKBACK_DAYS)).isoformat()
            bars = await asyncio.gather(
                *(client.get_stock_bars(n, start, today.isoformat()) for n in names)
            )
            closes = {n: [b["c"] for b in series] for n, series in zip(names, bars)}
            rstats = realized.compute_realized_stats(closes, weights)
            premium = realized.correlation_risk_premium(
                snap.implied_correlation, rstats.average_correlation if rstats else None
            )

            signal_row = {
                "index_symbol": s.index_symbol,
                "index_iv": snap.index_iv,
                "basket_iv": snap.basket_iv,
                "dispersion_ratio": snap.dispersion_ratio,
                "implied_correlation": snap.implied_correlation,
                "realized_correlation": rstats.average_correlation if rstats else None,
                "correlation_premium": premium,
                "constituent_ivs": constituent_ivs,
                "direction": "neutral",
            }

            if premium is None:
                msg = "Realised correlation unavailable; no baseline, so no trade."
                self._log(cycle_id, "no_signal", msg, "warning")
                self.journal.record_signal(cycle_id, signal_row)
                self.journal.finish_cycle(cycle_id, "no_signal", msg)
                return CycleResult(cycle_id, "no_signal", msg, signal=signal_row)

            if premium >= s.correlation_premium_entry:
                direction = "sell_index_vol"
            elif premium <= -s.correlation_premium_entry:
                direction = "buy_index_vol"
            else:
                direction = "neutral"
            signal_row["direction"] = direction

            self._log(
                cycle_id,
                "signal",
                f"DR {snap.dispersion_ratio:.4f} | implied rho {snap.implied_correlation:.3f} | "
                f"realised rho {rstats.average_correlation:.3f} | premium {premium:+.3f} "
                f"(entry +/-{s.correlation_premium_entry:.2f}) -> {direction}",
            )
            self.journal.record_signal(cycle_id, signal_row)

            if direction == "neutral":
                msg = (
                    f"Correlation premium {premium:+.3f} is inside the band; "
                    "no edge worth crossing a spread for."
                )
                self._log(cycle_id, "no_signal", msg)
                self.journal.finish_cycle(cycle_id, "no_signal", msg)
                return CycleResult(cycle_id, "no_signal", msg, signal=signal_row)

            # --- build -----------------------------------------------------
            built = build_basket(
                direction,
                index_atm,
                chains.get(s.index_symbol, []),
                {n: vols[n] for n in names},
                chains,
                today,
                s,
            )
            if built is None:
                msg = "Signal present but no tradable defined-risk structure could be built."
                self._log(cycle_id, "no_signal", msg, "warning")
                self.journal.finish_cycle(cycle_id, "no_signal", msg)
                return CycleResult(cycle_id, "no_signal", msg, signal=signal_row)

            proposal, structures = built
            self._log(
                cycle_id,
                "proposal",
                f"{proposal.basket_id}: {proposal.rationale} | "
                f"max loss ${proposal.max_loss:,.0f} | net delta {proposal.greeks.delta:+.0f}",
            )

            # --- agents ----------------------------------------------------
            llm, catalyst_agent, advocate_agent, narrator = build_agents(s)
            try:
                traded = sorted({leg.underlying for leg in proposal.legs} - {s.index_symbol})
                self._log(cycle_id, "catalyst", f"Checking catalysts for {traded}")

                headline_sets = await asyncio.gather(
                    *(client.get_news([n], limit=12) for n in traded)
                )
                verdicts: list[CatalystVerdict] = list(
                    await asyncio.gather(
                        *(catalyst_agent.assess(n, h) for n, h in zip(traded, headline_sets))
                    )
                )

                blocking = [v for v in verdicts if v.blocks_trade]
                if blocking:
                    reason = "; ".join(f"{v.symbol}: {v.reason}" for v in blocking)
                    msg = f"Catalyst veto -- {reason}"
                    self._log(cycle_id, "vetoed", msg, "warning")
                    self._record_rejection(
                        cycle_id, proposal, direction, verdicts, msg, "catalyst_veto"
                    )
                    self.journal.finish_cycle(cycle_id, "vetoed", msg)
                    return CycleResult(cycle_id, "vetoed", msg, signal=signal_row, memo=msg)

                summary = (
                    f"Direction: {direction}. Implied correlation {snap.implied_correlation:.3f} "
                    f"vs realised {rstats.average_correlation:.3f} (premium {premium:+.3f}). "
                    f"Structures: {proposal.rationale}. Max loss ${proposal.max_loss:,.0f}."
                )
                opinion: AdvocateOpinion = await advocate_agent.challenge(summary)
                self._log(cycle_id, "advocate", opinion.strongest_objection)

                # --- risk --------------------------------------------------
                decision = self.risk.evaluate(proposal, portfolio)
                self._log(
                    cycle_id,
                    "risk",
                    ("APPROVED" if decision.approved else "REJECTED")
                    + ("" if decision.approved else f" -- {decision.rejection_summary}"),
                    "info" if decision.approved else "warning",
                )

                memo = await narrator.write(
                    summary
                    + f" Risk engine: {'approved' if decision.approved else 'rejected'}."
                    + ("" if decision.approved else f" Reason: {decision.rejection_summary}")
                )

                self.journal.record_decision(
                    cycle_id,
                    {
                        "basket_id": proposal.basket_id,
                        "direction": direction,
                        "approved": decision.approved,
                        "max_loss": proposal.max_loss,
                        "rationale": proposal.rationale,
                        "catalyst_verdicts": [v.__dict__ for v in verdicts],
                        "advocate_opinion": opinion.__dict__,
                        "memo": memo,
                        "legs": [leg.__dict__ for leg in proposal.legs],
                        "checks": [
                            {
                                "name": c.name,
                                "passed": c.passed,
                                "message": c.message,
                                "observed": c.observed,
                                "limit": c.limit,
                            }
                            for c in decision.checks
                        ],
                    },
                )

                if not decision.approved:
                    self.journal.finish_cycle(cycle_id, "rejected", decision.rejection_summary)
                    return CycleResult(
                        cycle_id,
                        "rejected",
                        decision.rejection_summary,
                        signal=signal_row,
                        decision=decision,
                        memo=memo,
                    )

                # --- execute -----------------------------------------------
                if s.propose_only:
                    msg = "Approved, but PROPOSE_ONLY is set; no order submitted."
                    self._log(cycle_id, "proposed", msg)
                    self.journal.finish_cycle(cycle_id, "proposed", msg)
                    return CycleResult(
                        cycle_id, "proposed", msg, signal=signal_row, decision=decision, memo=memo
                    )

                results = await Executor(client, s).submit(structures)
                self.journal.record_orders(proposal.basket_id, results)
                sent = sum(1 for r in results if "error" not in r)
                self._log(cycle_id, "executed", f"{sent}/{len(structures)} structures submitted")
                self.journal.finish_cycle(cycle_id, "traded", proposal.basket_id)

                return CycleResult(
                    cycle_id,
                    "traded",
                    proposal.basket_id,
                    signal=signal_row,
                    decision=decision,
                    memo=memo,
                    orders=results,
                )
            finally:
                await llm.aclose()

    # --- autonomous loop ---------------------------------------------------

    async def run_forever(self) -> None:
        """Run cycles on the configured interval until stopped."""
        self._running = True
        interval = self.settings.cycle_interval_seconds
        self._log(None, "agent", f"Autonomous loop started ({interval}s interval)")
        while self._running:
            await self.run_cycle()
            for _ in range(interval):
                if not self._running:
                    break
                await asyncio.sleep(1)
        self._log(None, "agent", "Autonomous loop stopped")

    def stop(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running
