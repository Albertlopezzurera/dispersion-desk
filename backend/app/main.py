"""HTTP API and live activity stream for the desk.

The frontend is a thin reader: every number it displays is computed by the
backend and persisted in the journal first, so what the screen shows and what
the audit trail records can never diverge.

Endpoints are grouped by the view they serve -- dashboard, activity, trade
detail, risk centre -- which keeps the contract obvious and stops the UI from
stitching together three calls to render one panel.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.alpaca.client import AlpacaClient
from app.config import ConfigError, get_settings
from app.journal.db import Journal
from app.orchestrator import Orchestrator
from app.quant import universe

logger = logging.getLogger(__name__)

settings = get_settings()
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)

journal = Journal(settings.database_url)
orchestrator = Orchestrator(settings, journal)

_loop_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """The autonomous loop is started explicitly, never on boot.

    Starting a trading loop as a side effect of launching a web server is how an
    agent ends up running when nobody meant it to. The operator turns it on.
    """
    logger.info(
        "Dispersion Desk API up. propose_only=%s kill_switch=%s paper=%s",
        settings.propose_only,
        settings.kill_switch,
        settings.alpaca_paper_trade,
    )
    yield
    orchestrator.stop()
    if _loop_task:
        _loop_task.cancel()


app = FastAPI(title="Dispersion Desk", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _limits() -> dict[str, float]:
    return {
        "max_net_delta": settings.max_net_delta,
        "max_portfolio_vega": settings.max_portfolio_vega,
        "max_portfolio_gamma": settings.max_portfolio_gamma,
        "max_daily_theta": settings.max_daily_theta,
        "max_risk_per_basket_pct": settings.max_risk_per_basket_pct,
        "max_total_risk_pct": settings.max_total_risk_pct,
        "max_underlying_concentration_pct": settings.max_underlying_concentration_pct,
        "max_daily_loss_pct": settings.max_daily_loss_pct,
    }


@app.get("/api/status")
async def status() -> dict[str, Any]:
    """Configuration, safety switches, and whether the agent loop is running."""
    return {
        "agent_running": orchestrator.is_running,
        "propose_only": settings.propose_only,
        "kill_switch": settings.kill_switch,
        "paper_trading": settings.alpaca_paper_trade,
        "has_alpaca_credentials": settings.has_alpaca_credentials,
        "llm_provider": settings.llm_provider,
        "options_feed": settings.alpaca_options_feed,
        "index_symbol": settings.index_symbol,
        "basket": [m.symbol for m in universe.basket(settings.basket_size)],
        "basket_coverage_pct": round(universe.basket_coverage(settings.basket_size) * 100, 2),
        "weights_as_of": universe.WEIGHTS_AS_OF.isoformat(),
        "weights_age_days": universe.weights_age_days(),
        "cycle_interval_seconds": settings.cycle_interval_seconds,
        "correlation_premium_entry": settings.correlation_premium_entry,
        "limits": _limits(),
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/dashboard")
async def dashboard() -> dict[str, Any]:
    """Portfolio value, P&L, positions and greek exposure, live from Alpaca."""
    if not settings.has_alpaca_credentials:
        raise HTTPException(
            status_code=503,
            detail="Alpaca credentials are not configured. Copy .env.example to .env.",
        )
    try:
        async with AlpacaClient(settings) as client:
            state = await orchestrator.portfolio_state(client)
            positions = await client.get_positions()
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("dashboard failed")
        raise HTTPException(status_code=502, detail=f"Alpaca error: {exc}") from exc

    nav = state.net_asset_value
    return {
        "net_asset_value": nav,
        "daily_pnl": state.daily_pnl,
        "daily_pnl_pct": (100.0 * state.daily_pnl / nav) if nav > 0 else 0.0,
        "market_is_open": state.market_is_open,
        "greeks": {
            "delta": state.greeks.delta,
            "gamma": state.greeks.gamma,
            "vega": state.greeks.vega,
            "theta": state.greeks.theta,
        },
        "limits": _limits(),
        "open_defined_risk": state.open_defined_risk,
        "open_defined_risk_pct": (100.0 * state.open_defined_risk / nav) if nav > 0 else 0.0,
        "risk_by_underlying": state.risk_by_underlying,
        "positions": [
            {
                "symbol": p.get("symbol"),
                "qty": p.get("qty"),
                "asset_class": p.get("asset_class"),
                "market_value": p.get("market_value"),
                "unrealized_pl": p.get("unrealized_pl"),
                "cost_basis": p.get("cost_basis"),
            }
            for p in positions
        ],
        "agent_running": orchestrator.is_running,
    }


@app.get("/api/activity")
async def activity(limit: int = 200) -> list[dict]:
    return journal.recent_activity(limit)


@app.get("/api/activity/stream")
async def activity_stream() -> StreamingResponse:
    """Server-sent events: new journal entries as the agent produces them.

    Polling the journal rather than pushing from the orchestrator keeps the two
    decoupled -- the agent can run headless, and any number of browsers can
    watch without the cycle knowing or caring.
    """

    async def events() -> AsyncIterator[str]:
        recent = journal.recent_activity(1)
        last_id = recent[0]["id"] if recent else 0
        while True:
            rows = [r for r in reversed(journal.recent_activity(50)) if r["id"] > last_id]
            for row in rows:
                last_id = max(last_id, row["id"])
                yield f"data: {json.dumps(row)}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/decisions")
async def decisions(limit: int = 50) -> list[dict]:
    return journal.recent_decisions(limit)


@app.get("/api/decisions/{basket_id}")
async def decision_detail(basket_id: str) -> dict:
    """Everything behind one basket: signal, agents, risk checks, orders, P&L."""
    found = journal.decision(basket_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"No decision with id {basket_id}")
    for field_name in ("catalyst_verdicts", "advocate_opinion", "legs"):
        raw = found.get(field_name)
        if isinstance(raw, str):
            try:
                found[field_name] = json.loads(raw)
            except json.JSONDecodeError:
                found[field_name] = None
    return found


@app.get("/api/signals")
async def signals(limit: int = 200) -> list[dict]:
    rows = journal.signal_history(limit)
    for row in rows:
        raw = row.get("constituent_ivs")
        if isinstance(raw, str):
            try:
                row["constituent_ivs"] = json.loads(raw)
            except json.JSONDecodeError:
                row["constituent_ivs"] = {}
    return rows


@app.get("/api/risk")
async def risk_centre(limit: int = 50) -> dict[str, Any]:
    """Limits, and the record of every basket the engine refused."""
    return {
        "limits": _limits(),
        "rejected": journal.rejected_decisions(limit),
        "cycles": journal.recent_cycles(limit),
        "safety": {
            "propose_only": settings.propose_only,
            "kill_switch": settings.kill_switch,
            "paper_trading": settings.alpaca_paper_trade,
        },
    }


@app.post("/api/cycle/run")
async def run_cycle() -> dict[str, Any]:
    """Run one cycle immediately. Used for demos and manual inspection."""
    result = await orchestrator.run_cycle()
    return {
        "cycle_id": result.cycle_id,
        "outcome": result.outcome,
        "detail": result.detail,
        "signal": result.signal,
        "memo": result.memo,
        "orders": result.orders,
        "risk": (
            {
                "approved": result.decision.approved,
                "rendered": result.decision.render(),
                "checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "message": c.message,
                        "observed": c.observed,
                        "limit": c.limit,
                    }
                    for c in result.decision.checks
                ],
            }
            if result.decision
            else None
        ),
    }


@app.post("/api/agent/start")
async def start_agent() -> dict[str, Any]:
    global _loop_task
    if orchestrator.is_running:
        return {"agent_running": True, "detail": "Already running."}
    if settings.kill_switch:
        raise HTTPException(status_code=409, detail="Kill switch is engaged.")
    _loop_task = asyncio.create_task(orchestrator.run_forever())
    return {"agent_running": True, "detail": "Autonomous loop started."}


@app.post("/api/agent/stop")
async def stop_agent() -> dict[str, Any]:
    orchestrator.stop()
    return {"agent_running": False, "detail": "Stop requested; the current cycle will finish."}


@app.post("/api/kill-switch")
async def kill_switch(engaged: bool = True) -> dict[str, Any]:
    """Engage or release the global kill switch for this process.

    Deliberately not persisted: a restart returns to whatever ``.env`` declares,
    so an emergency stop can never be silently inherited by a later run the
    operator believes is clean.
    """
    settings.kill_switch = engaged
    if engaged:
        orchestrator.stop()
    journal.log(
        None, "kill_switch", f"Kill switch {'engaged' if engaged else 'released'}", "warning"
    )
    return {"kill_switch": engaged, "agent_running": orchestrator.is_running}


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
