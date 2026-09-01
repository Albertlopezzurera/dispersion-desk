"""Async HTTP client for the Alpaca Trading and Market Data APIs.

Scope and safety
----------------
This client talks to the **paper** trading host by default, and
:meth:`AlpacaClient.submit_mleg_order` re-checks the safety switches at the last
point before a request leaves the process.  The desk is built for simulated
funds; nothing here should ever be pointed at a live account.

Endpoints used (all verified against Alpaca's published documentation):

======================================  ==================================================
account / positions / orders            ``https://paper-api.alpaca.markets/v2/...``
option chain w/ greeks + IV             ``GET /v1beta1/options/snapshots/{underlying}``
option snapshots by contract            ``GET /v1beta1/options/snapshots?symbols=...``
historical option bars                  ``GET /v1beta1/options/bars``
stock bars / latest trade               ``GET /v2/stocks/...``
news                                    ``GET /v1beta1/news``
multi-leg order                         ``POST /v2/orders`` with ``order_class="mleg"``
======================================  ==================================================

A note on data quality
----------------------
On the free ``indicative`` feed the quotes are *derived*, not consolidated OPRA
quotes, and trades are delayed.  This client therefore preserves the raw quote
timestamp on every contract it returns, so the risk engine's staleness gate can
reject data too old to act on.  Discarding that timestamp would make the gate
impossible to enforce.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

PAPER_TRADING_HOST = "https://paper-api.alpaca.markets"
LIVE_TRADING_HOST = "https://api.alpaca.markets"
DATA_HOST = "https://data.alpaca.markets"

# Alpaca caps the `symbols` query parameter at 100 contracts per request.
MAX_SYMBOLS_PER_REQUEST = 100

# Alpaca's multi-leg order class accepts at most four legs.
MAX_MLEG_LEGS = 4

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 4


class AlpacaError(RuntimeError):
    """A call to Alpaca failed in a way the desk cannot recover from."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


@dataclass(frozen=True)
class OptionQuote:
    """One option contract as the desk sees it.

    ``implied_volatility`` and the greeks come from Alpaca when the feed supplies
    them.  They may be ``None`` -- the desk recomputes both from the mid price
    with its own Black-Scholes inversion, and that is the number the strategy
    trades on.  Alpaca's values are retained for cross-checking.
    """

    symbol: str
    underlying: str
    strike: float
    expiration: date
    option_type: str  # "call" | "put"
    bid: float | None
    ask: float | None
    last: float | None
    implied_volatility: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    rho: float | None
    open_interest: int | None
    quote_timestamp: datetime | None

    @property
    def mid(self) -> float | None:
        """Mid price, or ``None`` when the quote is one-sided or crossed.

        A zero or missing bid means nobody is willing to buy; treating that as a
        tradable price is how backtests manufacture profits that do not exist.
        """
        if self.bid is None or self.ask is None:
            return None
        if self.bid <= 0 or self.ask <= 0 or self.ask < self.bid:
            return None
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pct_of_mid(self) -> float | None:
        mid = self.mid
        if mid is None or mid <= 0 or self.bid is None or self.ask is None:
            return None
        return 100.0 * (self.ask - self.bid) / mid

    def age_seconds(self, now: datetime | None = None) -> float | None:
        if self.quote_timestamp is None:
            return None
        reference = now or datetime.now(timezone.utc)
        return (reference - self.quote_timestamp).total_seconds()


@dataclass(frozen=True)
class OrderLeg:
    """One leg of a multi-leg order, in the shape Alpaca's API expects."""

    symbol: str
    side: str  # "buy" | "sell"
    ratio_qty: int
    position_intent: str  # buy_to_open | sell_to_open | buy_to_close | sell_to_close

    def to_payload(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "ratio_qty": str(self.ratio_qty),
            "position_intent": self.position_intent,
        }


def _parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        # Alpaca returns RFC-3339 with a trailing Z and nanosecond precision;
        # fromisoformat accepts at most microseconds.
        cleaned = raw.replace("Z", "+00:00")
        if "." in cleaned:
            head, rest = cleaned.split(".", 1)
            frac, plus, tz = rest.partition("+")
            cleaned = f"{head}.{frac[:6]}+{tz}" if plus else f"{head}.{frac[:6]}"
        parsed = datetime.fromisoformat(cleaned)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        logger.warning("could not parse Alpaca timestamp: %r", raw)
        return None


def parse_occ_symbol(symbol: str) -> tuple[str, date, str, float]:
    """Decompose an OCC contract symbol, e.g. ``SPY260918C00450000``.

    Layout: root (variable length) + YYMMDD + C/P + strike in thousandths,
    zero-padded to 8 digits.  Parsing right-to-left is what makes variable-length
    roots unambiguous.
    """
    if len(symbol) < 16:
        raise AlpacaError(f"malformed OCC option symbol: {symbol!r}")

    tail = symbol[-15:]
    root = symbol[:-15]
    if not root:
        raise AlpacaError(f"option symbol has no underlying root: {symbol!r}")

    yy, mm, dd = tail[0:2], tail[2:4], tail[4:6]
    kind_char = tail[6].upper()
    strike_raw = tail[7:]

    if kind_char not in ("C", "P"):
        raise AlpacaError(f"unknown option type {kind_char!r} in {symbol!r}")
    if not (yy + mm + dd).isdigit() or not strike_raw.isdigit():
        raise AlpacaError(f"non-numeric date or strike in {symbol!r}")

    try:
        expiration = date(2000 + int(yy), int(mm), int(dd))
    except ValueError as exc:
        raise AlpacaError(f"invalid expiry date in {symbol!r}: {exc}") from exc

    return root, expiration, "call" if kind_char == "C" else "put", int(strike_raw) / 1000.0


def quote_from_snapshot(symbol: str, snap: dict[str, Any]) -> OptionQuote:
    """Map one Alpaca snapshot entry onto an :class:`OptionQuote`."""
    underlying, expiration, option_type, strike = parse_occ_symbol(symbol)
    quote = snap.get("latestQuote") or {}
    trade = snap.get("latestTrade") or {}
    greeks = snap.get("greeks") or {}

    return OptionQuote(
        symbol=symbol,
        underlying=underlying,
        strike=strike,
        expiration=expiration,
        option_type=option_type,
        bid=quote.get("bp"),
        ask=quote.get("ap"),
        last=trade.get("p"),
        implied_volatility=snap.get("impliedVolatility"),
        delta=greeks.get("delta"),
        gamma=greeks.get("gamma"),
        theta=greeks.get("theta"),
        vega=greeks.get("vega"),
        rho=greeks.get("rho"),
        open_interest=snap.get("openInterest"),
        quote_timestamp=_parse_timestamp(quote.get("t")),
    )


class AlpacaClient:
    """Thin, explicit async wrapper. One method per thing the desk needs."""

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self.settings = settings or get_settings()
        self.settings.require_alpaca()
        self._external_client = client is not None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))

    @property
    def trading_host(self) -> str:
        return PAPER_TRADING_HOST if self.settings.alpaca_paper_trade else LIVE_TRADING_HOST

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.settings.alpaca_secret_key,
            "accept": "application/json",
        }

    async def __aenter__(self) -> "AlpacaClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if not self._external_client:
            await self._client.aclose()

    # --- transport ---------------------------------------------------------

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue one request, retrying only on transient failures.

        A 4xx other than 429 means the desk asked for something wrong; retrying
        would just repeat the mistake, so it is raised immediately with the
        response body attached for diagnosis.
        """
        last_error: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = await self._client.request(
                    method, url, params=params, json=json_body, headers=self._headers
                )
            except httpx.RequestError as exc:
                last_error = exc
                logger.warning("network error calling %s (attempt %d): %s", url, attempt, exc)
            else:
                if response.status_code < 400:
                    if not response.content:
                        return {}
                    payload = response.json()
                    return payload if isinstance(payload, dict) else {"data": payload}

                body = response.text[:500]
                if response.status_code not in _RETRYABLE_STATUS:
                    raise AlpacaError(
                        f"{method} {url} failed with {response.status_code}: {body}",
                        status_code=response.status_code,
                        body=body,
                    )
                last_error = AlpacaError(
                    f"{method} {url} transient {response.status_code}: {body}",
                    status_code=response.status_code,
                    body=body,
                )
                logger.warning(
                    "retryable %d from %s (attempt %d/%d)",
                    response.status_code,
                    url,
                    attempt,
                    _MAX_ATTEMPTS,
                )

            if attempt < _MAX_ATTEMPTS:
                await asyncio.sleep(min(2 ** (attempt - 1), 8) * 0.5)

        raise AlpacaError(f"{method} {url} failed after {_MAX_ATTEMPTS} attempts: {last_error}")

    # --- account and positions --------------------------------------------

    async def get_account(self) -> dict[str, Any]:
        return await self._request("GET", f"{self.trading_host}/v2/account")

    async def get_positions(self) -> list[dict[str, Any]]:
        payload = await self._request("GET", f"{self.trading_host}/v2/positions")
        return payload.get("data", [])

    async def get_orders(self, status: str = "all", limit: int = 100) -> list[dict[str, Any]]:
        payload = await self._request(
            "GET",
            f"{self.trading_host}/v2/orders",
            params={"status": status, "limit": limit, "nested": "true"},
        )
        return payload.get("data", [])

    # --- market data -------------------------------------------------------

    async def get_stock_price(self, symbol: str) -> float | None:
        """Latest trade price for the underlying, used as spot in the pricer."""
        payload = await self._request(
            "GET", f"{DATA_HOST}/v2/stocks/{symbol}/trades/latest", params={"feed": "iex"}
        )
        return (payload.get("trade") or {}).get("p")

    async def get_stock_bars(
        self, symbol: str, start: str, end: str, timeframe: str = "1Day"
    ) -> list[dict[str, Any]]:
        """Daily bars for the underlying. Feeds the historical bootstrap."""
        bars: list[dict[str, Any]] = []
        page_token: str | None = None

        while True:
            params: dict[str, Any] = {
                "symbols": symbol,
                "start": start,
                "end": end,
                "timeframe": timeframe,
                "limit": 10000,
                "feed": "iex",
            }
            if page_token:
                params["page_token"] = page_token

            payload = await self._request("GET", f"{DATA_HOST}/v2/stocks/bars", params=params)
            bars.extend((payload.get("bars") or {}).get(symbol, []))

            page_token = payload.get("next_page_token")
            if not page_token:
                return bars

    async def get_option_chain(
        self,
        underlying: str,
        *,
        expiration_gte: date | None = None,
        expiration_lte: date | None = None,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
        option_type: str | None = None,
        max_pages: int = 10,
    ) -> list[OptionQuote]:
        """Full option chain for one underlying, greeks and IV included.

        ``max_pages`` bounds the walk so a wide filter cannot spin forever.  When
        the cap is hit it is logged loudly, because a silently truncated chain
        would bias the at-the-money volatility the whole signal rests on.
        """
        quotes: list[OptionQuote] = []
        page_token: str | None = None

        for page in range(max_pages):
            params: dict[str, Any] = {"feed": self.settings.alpaca_options_feed, "limit": 1000}
            if expiration_gte:
                params["expiration_date_gte"] = expiration_gte.isoformat()
            if expiration_lte:
                params["expiration_date_lte"] = expiration_lte.isoformat()
            if strike_gte is not None:
                params["strike_price_gte"] = strike_gte
            if strike_lte is not None:
                params["strike_price_lte"] = strike_lte
            if option_type:
                params["type"] = option_type
            if page_token:
                params["page_token"] = page_token

            payload = await self._request(
                "GET", f"{DATA_HOST}/v1beta1/options/snapshots/{underlying}", params=params
            )

            for symbol, snap in (payload.get("snapshots") or {}).items():
                try:
                    quotes.append(quote_from_snapshot(symbol, snap))
                except AlpacaError as exc:
                    # One unparseable contract must not sink the whole chain.
                    logger.warning("skipping contract %s: %s", symbol, exc)

            page_token = payload.get("next_page_token")
            if not page_token:
                return quotes

        logger.warning(
            "option chain for %s hit the %d-page cap; results may be truncated",
            underlying,
            max_pages,
        )
        return quotes

    async def get_open_interest(
        self,
        underlying: str,
        expiration_gte: date | None = None,
        expiration_lte: date | None = None,
        max_pages: int = 10,
    ) -> dict[str, int]:
        """Open interest per contract, keyed by OCC symbol.

        This lives on the *trading* API rather than the market-data snapshot, and
        the distinction is not academic: on the free ``indicative`` feed a
        snapshot returns only ``latestQuote`` -- no greeks, no implied
        volatility, and no open interest. Gating on a field that endpoint never
        populates would reject every contract forever while looking like a
        working liquidity check.

        Open interest is also stale by construction; it is published once a day,
        after the close. It is a filter for "does anyone hold this contract",
        never a live measure of what is currently trading.
        """
        out: dict[str, int] = {}
        page_token: str | None = None

        for _ in range(max_pages):
            params: dict[str, Any] = {
                "underlying_symbols": underlying,
                "limit": 10000,
                "status": "active",
            }
            if expiration_gte:
                params["expiration_date_gte"] = expiration_gte.isoformat()
            if expiration_lte:
                params["expiration_date_lte"] = expiration_lte.isoformat()
            if page_token:
                params["page_token"] = page_token

            payload = await self._request(
                "GET", f"{self.trading_host}/v2/options/contracts", params=params
            )
            for contract in payload.get("option_contracts") or []:
                symbol = contract.get("symbol")
                raw = contract.get("open_interest")
                if symbol and raw is not None:
                    try:
                        out[symbol] = int(raw)
                    except (TypeError, ValueError):
                        continue

            page_token = payload.get("next_page_token")
            if not page_token:
                break

        return out

    async def get_option_snapshots(self, symbols: list[str]) -> dict[str, OptionQuote]:
        """Refresh specific contracts, chunked to Alpaca's 100-symbol limit."""
        out: dict[str, OptionQuote] = {}

        for i in range(0, len(symbols), MAX_SYMBOLS_PER_REQUEST):
            chunk = symbols[i : i + MAX_SYMBOLS_PER_REQUEST]
            payload = await self._request(
                "GET",
                f"{DATA_HOST}/v1beta1/options/snapshots",
                params={"symbols": ",".join(chunk), "feed": self.settings.alpaca_options_feed},
            )
            for symbol, snap in (payload.get("snapshots") or {}).items():
                try:
                    out[symbol] = quote_from_snapshot(symbol, snap)
                except AlpacaError as exc:
                    logger.warning("skipping contract %s: %s", symbol, exc)

        return out

    async def get_option_bars(
        self, symbols: list[str], start: str, end: str, timeframe: str = "1Day"
    ) -> dict[str, list[dict[str, Any]]]:
        """Historical option bars. Raw material for the DR history bootstrap."""
        out: dict[str, list[dict[str, Any]]] = {}

        for i in range(0, len(symbols), MAX_SYMBOLS_PER_REQUEST):
            chunk = symbols[i : i + MAX_SYMBOLS_PER_REQUEST]
            payload = await self._request(
                "GET",
                f"{DATA_HOST}/v1beta1/options/bars",
                params={
                    "symbols": ",".join(chunk),
                    "start": start,
                    "end": end,
                    "timeframe": timeframe,
                    "limit": 10000,
                },
            )
            for symbol, bars in (payload.get("bars") or {}).items():
                out.setdefault(symbol, []).extend(bars)

        return out

    async def get_news(self, symbols: list[str], limit: int = 20) -> list[dict[str, Any]]:
        """Recent headlines. Consumed by the Catalyst agent, never by the maths."""
        payload = await self._request(
            "GET",
            f"{DATA_HOST}/v1beta1/news",
            params={"symbols": ",".join(symbols), "limit": min(limit, 50), "sort": "desc"},
        )
        return payload.get("news", [])

    # --- execution ---------------------------------------------------------

    async def submit_mleg_order(
        self,
        legs: list[OrderLeg],
        qty: int,
        limit_price: float,
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        """Submit one multi-leg options order.

        The safety switches are re-checked here, at the last point before the
        request leaves the process, so no caller can bypass them.
        """
        self.settings.require_live_execution_allowed()

        if not legs:
            raise AlpacaError("cannot submit an order with no legs")
        if len(legs) > MAX_MLEG_LEGS:
            raise AlpacaError(
                f"Alpaca accepts at most {MAX_MLEG_LEGS} legs per multi-leg order, got "
                f"{len(legs)}. Split the basket into several orders and account for the "
                "legging risk in the attribution."
            )
        if qty < 1:
            raise AlpacaError(f"order quantity must be at least 1, got {qty}")
        if limit_price <= 0:
            raise AlpacaError(
                f"limit price must be positive, got {limit_price}. The desk never sends "
                "market orders on options: the spread is the dominant cost."
            )

        body = {
            "order_class": "mleg",
            "qty": str(qty),
            "type": "limit",
            "limit_price": f"{limit_price:.2f}",
            "time_in_force": time_in_force,
            "legs": [leg.to_payload() for leg in legs],
        }
        logger.info("submitting mleg order: %s", body)
        return await self._request("POST", f"{self.trading_host}/v2/orders", json_body=body)

    async def cancel_order(self, order_id: str) -> None:
        await self._request("DELETE", f"{self.trading_host}/v2/orders/{order_id}")

    async def get_clock(self) -> dict[str, Any]:
        """Market clock. Backs the trading-hours gate."""
        return await self._request("GET", f"{self.trading_host}/v2/clock")
