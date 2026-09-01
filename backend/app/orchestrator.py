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
import json
import logging
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.agents.analysts import AdvocateOpinion, CatalystVerdict, build_agents
from app.alpaca import mcp as alpaca_mcp
from app.alpaca.client import AlpacaClient, OptionQuote
from app.config import ConfigError, Settings
from app.execution.executor import Executor, build_basket
from app.journal.db import Journal
from app.backtest.engine import build_observations
from app.quant import dispersion, evidence, realized, surface, universe
from app.quant.attribution import LegSnapshot, attribute_basket
from app.quant.black_scholes import bs_greeks, implied_volatility
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

# History pulled to assess whether the signal has proven predictive power.
# Long, because after correcting for the overlap between rolling windows a
# year of daily data is worth only a handful of independent samples.
EVIDENCE_LOOKBACK_DAYS = 6 * 365

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

            # Spot prices for every underlying we hold, fetched once each.
            underlyings = {s.underlying for s in snapshots.values()}
            spots = dict(
                zip(
                    underlyings,
                    await asyncio.gather(*(client.get_stock_price(u) for u in underlyings)),
                )
            )

            today = datetime.now(timezone.utc).date()
            for pos in option_positions:
                snap = snapshots.get(pos["symbol"])
                if snap is None:
                    continue
                scale = float(pos.get("qty") or 0) * CONTRACT_MULTIPLIER

                # The free `indicative` feed returns only latestQuote: no greeks,
                # no implied volatility. Reading them off the snapshot silently
                # produced a book that measured as delta 0 and vega 0 no matter
                # what was actually held, which would make every post-trade greek
                # limit meaningless. So the desk solves them itself, from the
                # same mid it would trade on.
                leg = self._position_greeks(snap, spots.get(snap.underlying), today)
                if leg is None:
                    logger.warning(
                        "cannot value %s: no spot or unsolvable IV; greek limits "
                        "will understate the book",
                        pos["symbol"],
                    )
                    continue

                greeks = greeks + PositionGreeks(
                    delta=leg.delta * scale,
                    gamma=leg.gamma * scale,
                    vega=leg.vega * scale,
                    theta=leg.theta * scale,
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
            # Open interest is not in the snapshot on the free feed, so it is
            # fetched separately and merged in. Without this the liquidity gate
            # evaluates None for every contract and refuses everything.
            oi = await client.get_open_interest(
                symbol,
                expiration_gte=today + timedelta(days=s.target_dte_min),
                expiration_lte=today + timedelta(days=s.target_dte_max),
            )
            if oi:
                chain = [
                    replace(q, open_interest=oi.get(q.symbol, q.open_interest)) for q in chain
                ]

            atm = surface.atm_volatility(
                symbol, chain, spot, today, s.target_dte_min, s.target_dte_max, s.risk_free_rate
            )
            return symbol, atm, chain

        results = await asyncio.gather(*(one(sym) for sym in symbols))
        return (
            {sym: atm for sym, atm, _ in results if atm is not None},
            {sym: chain for sym, _, chain in results},
        )

    def _position_greeks(self, snap, spot, today):
        """Greeks for one held contract, solved from its own quoted mid.

        Returns ``None`` when the contract cannot be valued, which the caller
        logs rather than silently treating as zero risk.
        """
        mid = snap.mid
        if not spot or mid is None:
            return None

        t = surface.year_fraction(today, snap.expiration)
        if t <= 0:
            return None

        iv = implied_volatility(
            mid, spot, snap.strike, t, self.settings.risk_free_rate, 0.0, snap.option_type
        )
        if iv is None:
            return None

        return bs_greeks(
            spot, snap.strike, t, self.settings.risk_free_rate, iv, 0.0, snap.option_type
        )

    async def _fetch_headlines(self, client, symbols, cycle_id):
        """Headlines per symbol, preferring the MCP server over REST.

        The source actually used is logged, because "uses the MCP server" is a
        claim that should be checkable from the audit trail rather than taken on
        trust from a config file.
        """
        via_mcp = await alpaca_mcp.get_news(self.settings, symbols, limit=12)
        if via_mcp is not None:
            by_symbol: dict[str, list] = {s: [] for s in symbols}
            for item in via_mcp:
                for sym in item.get("symbols") or []:
                    if sym in by_symbol:
                        by_symbol[sym].append(item)
            self._log(
                cycle_id,
                "mcp",
                f"Fetched {len(via_mcp)} headlines through Alpaca's MCP server "
                f"(tool get_news). Payload is flagged untrusted by the server and is "
                f"passed to the model as data, never as instructions.",
            )
            return [by_symbol[s] for s in symbols]

        self._log(cycle_id, "mcp", "MCP server unavailable; using the REST news endpoint")
        return list(await asyncio.gather(*(client.get_news([s], limit=12) for s in symbols)))

    async def _assess_evidence(self, client, names, weights):
        """Has this signal ever been shown to predict anything?

        Returns ``None`` when the history is too short to ask the question,
        which the caller treats as "unknown" rather than as a pass.
        """
        try:
            today = datetime.now(timezone.utc).date()
            start = (today - timedelta(days=EVIDENCE_LOOKBACK_DAYS)).isoformat()
            bars = await asyncio.gather(
                *(client.get_stock_bars(n, start, today.isoformat()) for n in names),
                client.get_stock_bars(self.settings.index_symbol, start, today.isoformat()),
            )
            closes = {n: [b["c"] for b in series] for n, series in zip(names, bars[:-1])}
            closes[self.settings.index_symbol] = [b["c"] for b in bars[-1]]
            dates = [today] * min(len(v) for v in closes.values())

            observations = build_observations(
                dates, closes, self.settings.index_symbol, weights, lookback=60, horizon=10
            )
            rows = [o for o in observations if o.forward_dispersion_ratio is not None]
            if len(rows) < 30:
                return None
            return evidence.assess(
                [o.dispersion_ratio for o in rows],
                [o.forward_dispersion_ratio - o.dispersion_ratio for o in rows],
            )
        except Exception as exc:  # noqa: BLE001 - evidence is advisory, never fatal
            logger.warning("evidence assessment unavailable: %s", exc)
            return None

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

    async def monitor_positions(self) -> list[dict]:
        """Mark every executed basket and decompose its P&L by greek.

        This is where the desk's claim gets checked against reality. The
        strategy says its profits come from vega; the attribution says where
        they actually came from. A basket showing delta as its dominant driver
        is not a lucky win, it is a hedge that is not working.

        Runs against live quotes, so it reflects the book as it stands rather
        than as it was modelled.
        """
        results: list[dict] = []
        decisions = self.journal.executed_decisions()
        if not decisions:
            return results

        async with AlpacaClient(self.settings) as client:
            today = datetime.now(timezone.utc).date()
            positions = {p["symbol"]: p for p in await client.get_positions()}

            for decision in decisions:
                try:
                    legs = json.loads(decision.get("legs") or "[]")
                    spots = json.loads(decision.get("spots") or "{}")
                except json.JSONDecodeError:
                    continue

                # Only legs still held can be marked; a closed leg has already
                # had its outcome recorded.
                held = [leg for leg in legs if leg.get("symbol") in positions]
                if not held or not spots:
                    continue

                snapshots = await client.get_option_snapshots([leg["symbol"] for leg in held])
                underlyings = {leg["underlying"] for leg in held}
                current_spots = dict(
                    zip(
                        underlyings,
                        await asyncio.gather(*(client.get_stock_price(u) for u in underlyings)),
                    )
                )

                pairs = []
                for leg in held:
                    snap = snapshots.get(leg["symbol"])
                    entry_spot = spots.get(leg["underlying"])
                    exit_spot = current_spots.get(leg["underlying"])
                    if snap is None or not entry_spot or not exit_spot:
                        continue

                    mid = snap.mid
                    if mid is None:
                        continue

                    t_exit = surface.year_fraction(today, snap.expiration)
                    if t_exit <= 0:
                        continue
                    # Time elapsed since the decision, recovered from the leg's
                    # own recorded expiry distance.
                    t_entry = t_exit + (
                        (datetime.now(timezone.utc) - datetime.fromisoformat(
                            decision["decided_at"]
                        )).days / 365.0
                    )

                    exit_iv = implied_volatility(
                        mid,
                        exit_spot,
                        snap.strike,
                        t_exit,
                        self.settings.risk_free_rate,
                        0.0,
                        snap.option_type,
                    )
                    if exit_iv is None:
                        continue

                    signed = int(float(positions[leg["symbol"]].get("qty") or 0))
                    if signed == 0:
                        continue

                    entry = LegSnapshot(
                        symbol=leg["symbol"],
                        option_type=snap.option_type,
                        signed_contracts=signed,
                        spot=entry_spot,
                        strike=snap.strike,
                        time_to_expiry=t_entry,
                        implied_volatility=leg["implied_volatility"],
                        price=leg["price"],
                        risk_free_rate=self.settings.risk_free_rate,
                    )
                    exit_ = LegSnapshot(
                        symbol=leg["symbol"],
                        option_type=snap.option_type,
                        signed_contracts=signed,
                        spot=exit_spot,
                        strike=snap.strike,
                        time_to_expiry=t_exit,
                        implied_volatility=exit_iv,
                        price=mid,
                        risk_free_rate=self.settings.risk_free_rate,
                    )
                    pairs.append((entry, exit_, None))

                if not pairs:
                    continue

                attribution = attribute_basket(pairs)
                payload = {
                    "total": attribution.total,
                    "delta_pnl": attribution.delta_pnl,
                    "gamma_pnl": attribution.gamma_pnl,
                    "vega_pnl": attribution.vega_pnl,
                    "theta_pnl": attribution.theta_pnl,
                    "slippage": attribution.slippage,
                    "residual": attribution.residual,
                    "dominant": attribution.dominant_driver,
                }
                self.journal.record_attribution(decision["basket_id"], payload)
                self._log(
                    None,
                    "attribution",
                    f"{decision['basket_id']}: {attribution.total:+,.2f} "
                    f"driven by {attribution.dominant_driver} "
                    f"(vega {attribution.vega_pnl:+,.2f}, delta {attribution.delta_pnl:+,.2f}, "
                    f"theta {attribution.theta_pnl:+,.2f})",
                )
                results.append({"basket_id": decision["basket_id"], **payload})

        return results

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

            # --- evidence -------------------------------------------------
            # Before acting on the signal, ask whether this signal has ever been
            # shown to work. The answer scales the position; it does not merely
            # decorate the report.
            report = await self._assess_evidence(client, names, weights)
            if report is not None:
                signal_row["evidence_verdict"] = report.verdict
                signal_row["evidence_scale"] = evidence.position_scale(report)
                self._log(
                    cycle_id,
                    "evidence",
                    f"{report.verdict.upper()} | {report.n_independent} independent samples "
                    f"({report.inflation_factor:.0f}x inflation) | hit {report.hit_rate:.1%} "
                    f"vs naive {report.naive_hit_rate:.1%} | p={report.p_value:.3f} "
                    f"-> position scale {evidence.position_scale(report):.0%}",
                    "info" if report.is_significant else "warning",
                )

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
            # The evidence verdict sizes the trade. An unproven signal is not
            # forbidden, it is capped -- refusing outright would mean never
            # gathering the data that settles the question.
            scale = float(signal_row.get("evidence_scale", 0.25))
            built = build_basket(
                direction,
                index_atm,
                chains.get(s.index_symbol, []),
                {n: vols[n] for n in names},
                chains,
                today,
                s,
                size_scale=scale,
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

                # Headlines come through Alpaca's official MCP server, which is
                # the integration the hackathon asks for and the one place a
                # tool call is genuinely semantic: fetching text for a model to
                # read. If the server cannot start, the REST path takes over --
                # an optional integration must never stop the desk trading.
                headline_sets = await self._fetch_headlines(client, traded, cycle_id)
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
                        # Entry spot per underlying: without it the attribution
                        # has no reference point to price the delta effect
                        # against when the position is marked later.
                        "spots": {
                            sym: atm.spot
                            for sym, atm in ({s.index_symbol: index_atm} | vols).items()
                        },
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
