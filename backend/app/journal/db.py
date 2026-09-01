"""The decision journal: an append-only record of everything the desk did.

Why every cycle is written down, including the ones that traded nothing
----------------------------------------------------------------------
A trading system is only auditable if its *refusals* are as visible as its
fills.  Most of what this desk does is decline to trade -- the signal is inside
its band, a catalyst was found, a quote was stale -- and those are the records
proving the risk controls are real rather than decorative.  So a cycle row is
written whether or not an order follows, and every risk check is stored with the
number that produced it.

The journal is append-only by convention: nothing here rewrites a decision after
the fact.  Position outcomes arrive as new attribution rows keyed to the original
basket, so the record of what was believed at decision time survives contact
with how it turned out.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    outcome       TEXT NOT NULL,           -- running|traded|no_signal|vetoed|rejected|error
    detail        TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id              INTEGER NOT NULL REFERENCES cycles(id),
    observed_at           TEXT NOT NULL,
    index_symbol          TEXT NOT NULL,
    index_iv              REAL NOT NULL,
    basket_iv             REAL NOT NULL,
    dispersion_ratio      REAL NOT NULL,
    implied_correlation   REAL,
    realized_correlation  REAL,
    correlation_premium   REAL,
    direction             TEXT NOT NULL,
    constituent_ivs       TEXT NOT NULL,    -- JSON {symbol: iv}
    evidence_verdict      TEXT,             -- proven | unproven | underpowered
    evidence_scale        REAL              -- fraction of full size the verdict allows
);

CREATE TABLE IF NOT EXISTS decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id          INTEGER NOT NULL REFERENCES cycles(id),
    basket_id         TEXT NOT NULL UNIQUE,
    decided_at        TEXT NOT NULL,
    direction         TEXT NOT NULL,
    approved          INTEGER NOT NULL,
    max_loss          REAL NOT NULL,
    rationale         TEXT,
    catalyst_verdicts TEXT,                 -- JSON list
    advocate_opinion  TEXT,                 -- JSON object
    memo              TEXT,
    legs              TEXT NOT NULL,        -- JSON list
    spots             TEXT                  -- JSON {underlying: spot at decision}
);

CREATE TABLE IF NOT EXISTS risk_checks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    basket_id   TEXT NOT NULL,
    name        TEXT NOT NULL,
    passed      INTEGER NOT NULL,
    message     TEXT NOT NULL,
    observed    REAL,
    limit_value REAL
);

CREATE TABLE IF NOT EXISTS orders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    basket_id    TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    structure    TEXT NOT NULL,
    payload      TEXT NOT NULL,             -- JSON: Alpaca response
    error        TEXT
);

CREATE TABLE IF NOT EXISTS attributions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    basket_id   TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    total       REAL NOT NULL,
    delta_pnl   REAL NOT NULL,
    gamma_pnl   REAL NOT NULL,
    vega_pnl    REAL NOT NULL,
    theta_pnl   REAL NOT NULL,
    slippage    REAL NOT NULL,
    residual    REAL NOT NULL,
    dominant    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS activity (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    at       TEXT NOT NULL,
    cycle_id INTEGER,
    level    TEXT NOT NULL,
    step     TEXT NOT NULL,
    message  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_activity_at ON activity(id DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_cycle ON decisions(cycle_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_from_url(database_url: str) -> Path:
    """Accept a SQLAlchemy-style URL so the setting reads like every other project."""
    raw = database_url.removeprefix("sqlite:///").removeprefix("sqlite://")
    return Path(raw or "dispersion_desk.db")


class Journal:
    """Thin SQLite wrapper. One method per thing the desk records."""

    def __init__(self, database_url: str) -> None:
        self.path = _path_from_url(database_url)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns introduced after a database was first created.

        CREATE TABLE IF NOT EXISTS silently does nothing to an existing table,
        so a journal written by an earlier version would keep working while
        quietly missing the column, and the attribution would find no entry spot
        to price against.
        """
        existing = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
        if "spots" not in existing:
            conn.execute("ALTER TABLE decisions ADD COLUMN spots TEXT")
            logger.info("journal: added decisions.spots")

        signal_cols = {row[1] for row in conn.execute("PRAGMA table_info(signals)")}
        for column, kind in (("evidence_verdict", "TEXT"), ("evidence_scale", "REAL")):
            if column not in signal_cols:
                conn.execute(f"ALTER TABLE signals ADD COLUMN {column} {kind}")
                logger.info("journal: added signals.%s", column)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- writes ------------------------------------------------------------

    def start_cycle(self) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO cycles (started_at, outcome) VALUES (?, 'running')", (_now(),)
            )
            return int(cur.lastrowid)

    def finish_cycle(self, cycle_id: int, outcome: str, detail: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE cycles SET finished_at = ?, outcome = ?, detail = ? WHERE id = ?",
                (_now(), outcome, detail, cycle_id),
            )

    def log(self, cycle_id: int | None, step: str, message: str, level: str = "info") -> dict:
        """Record one step of the cycle. Backs the live Agent Activity feed."""
        entry = {
            "at": _now(),
            "cycle_id": cycle_id,
            "level": level,
            "step": step,
            "message": message,
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO activity (at, cycle_id, level, step, message) VALUES (?,?,?,?,?)",
                (entry["at"], cycle_id, level, step, message),
            )
        return entry

    def record_signal(self, cycle_id: int, signal: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO signals (cycle_id, observed_at, index_symbol, index_iv,
                   basket_iv, dispersion_ratio, implied_correlation, realized_correlation,
                   correlation_premium, direction, constituent_ivs,
                   evidence_verdict, evidence_scale)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    cycle_id,
                    _now(),
                    signal["index_symbol"],
                    signal["index_iv"],
                    signal["basket_iv"],
                    signal["dispersion_ratio"],
                    signal.get("implied_correlation"),
                    signal.get("realized_correlation"),
                    signal.get("correlation_premium"),
                    signal["direction"],
                    json.dumps(signal.get("constituent_ivs", {})),
                    signal.get("evidence_verdict"),
                    signal.get("evidence_scale"),
                ),
            )

    def record_decision(self, cycle_id: int, decision: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO decisions (cycle_id, basket_id, decided_at, direction, approved,
                   max_loss, rationale, catalyst_verdicts, advocate_opinion, memo, legs,
                   spots)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    cycle_id,
                    decision["basket_id"],
                    _now(),
                    decision["direction"],
                    int(decision["approved"]),
                    decision["max_loss"],
                    decision.get("rationale", ""),
                    json.dumps(decision.get("catalyst_verdicts", [])),
                    json.dumps(decision.get("advocate_opinion", {})),
                    decision.get("memo", ""),
                    json.dumps(decision.get("legs", [])),
                    json.dumps(decision.get("spots", {})),
                ),
            )
            conn.executemany(
                """INSERT INTO risk_checks
                   (basket_id, name, passed, message, observed, limit_value)
                   VALUES (?,?,?,?,?,?)""",
                [
                    (
                        decision["basket_id"],
                        c["name"],
                        int(c["passed"]),
                        c["message"],
                        c.get("observed"),
                        c.get("limit"),
                    )
                    for c in decision.get("checks", [])
                ],
            )

    def record_orders(self, basket_id: str, results: list[dict[str, Any]]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO orders (basket_id, submitted_at, structure, payload, error)
                   VALUES (?,?,?,?,?)""",
                [
                    (
                        basket_id,
                        _now(),
                        r.get("structure", "?"),
                        json.dumps(r.get("order", {})),
                        r.get("error"),
                    )
                    for r in results
                ],
            )

    def record_attribution(self, basket_id: str, a: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO attributions (basket_id, computed_at, total, delta_pnl, gamma_pnl,
                   vega_pnl, theta_pnl, slippage, residual, dominant)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    basket_id,
                    _now(),
                    a["total"],
                    a["delta_pnl"],
                    a["gamma_pnl"],
                    a["vega_pnl"],
                    a["theta_pnl"],
                    a["slippage"],
                    a["residual"],
                    a["dominant"],
                ),
            )

    # --- reads -------------------------------------------------------------

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def recent_activity(self, limit: int = 200) -> list[dict]:
        return self._rows("SELECT * FROM activity ORDER BY id DESC LIMIT ?", (limit,))

    def recent_decisions(self, limit: int = 50) -> list[dict]:
        return self._rows("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))

    def decision(self, basket_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM decisions WHERE basket_id = ?", (basket_id,))
        if not rows:
            return None
        decision = rows[0]
        decision["checks"] = self._rows(
            "SELECT * FROM risk_checks WHERE basket_id = ? ORDER BY id", (basket_id,)
        )
        decision["orders"] = self._rows(
            "SELECT * FROM orders WHERE basket_id = ? ORDER BY id", (basket_id,)
        )
        decision["attributions"] = self._rows(
            "SELECT * FROM attributions WHERE basket_id = ? ORDER BY id DESC", (basket_id,)
        )
        return decision

    def executed_decisions(self, limit: int = 20) -> list[dict]:
        """Approved baskets that actually reached the market.

        An approval that never filled has nothing to attribute, so the monitor
        works from orders rather than from decisions.
        """
        return self._rows(
            """SELECT DISTINCT d.* FROM decisions d
               JOIN orders o ON o.basket_id = d.basket_id
               WHERE d.approved = 1 AND o.error IS NULL
               ORDER BY d.id DESC LIMIT ?""",
            (limit,),
        )

    def rejected_decisions(self, limit: int = 50) -> list[dict]:
        """Refusals, for the Risk Center. The record that the gates actually bite."""
        return self._rows(
            "SELECT * FROM decisions WHERE approved = 0 ORDER BY id DESC LIMIT ?", (limit,)
        )

    def signal_history(self, limit: int = 500) -> list[dict]:
        return self._rows("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,))

    def dispersion_ratio_history(self, limit: int = 500) -> list[float]:
        """Past dispersion ratios, oldest first. Feeds the secondary z-score signal."""
        rows = self._rows("SELECT dispersion_ratio FROM signals ORDER BY id DESC LIMIT ?", (limit,))
        return [r["dispersion_ratio"] for r in reversed(rows)]

    def recent_cycles(self, limit: int = 50) -> list[dict]:
        return self._rows("SELECT * FROM cycles ORDER BY id DESC LIMIT ?", (limit,))
