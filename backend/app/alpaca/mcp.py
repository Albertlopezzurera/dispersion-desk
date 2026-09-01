"""Talking to Alpaca through its official MCP server.

Why this exists
---------------
The hackathon requires projects to use Alpaca's MCP server or CLI. Declaring one
in a config file is not using it, so the desk routes a real step of its trading
cycle through the MCP server and falls back to REST only if that fails.

The step chosen is the Catalyst agent's news lookup, and the choice is
deliberate. MCP earns its place where a tool call is *semantic* -- fetching
headlines for a language model to read is exactly that. The quantitative path
stays on REST, because a desk that refuses to trade on stale quotes should not
put a subprocess handshake between itself and the prices it values from.

How it runs
-----------
The server is launched on demand with ``uvx alpaca-mcp-server`` over stdio, with
credentials passed through the environment. It is started per call rather than
held open: a cycle runs every fifteen minutes, so a long-lived subprocess would
spend nearly all its life idle and would need supervising for no benefit.

Everything here fails soft. If ``uvx`` is missing, the server will not start, or
a tool call errors, the caller receives ``None`` and uses the REST path. An
optional integration must never be able to stop the desk from trading.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from app.config import Settings

logger = logging.getLogger(__name__)

# Only the toolsets the desk actually calls. A narrower surface starts faster and
# makes it obvious from the configuration what this integration may do.
TOOLSETS = "account,options-data,news"


class MCPUnavailable(RuntimeError):
    """The MCP server could not be reached. Always caught by the caller."""


def is_available() -> bool:
    """Whether the MCP server can plausibly be launched at all."""
    return shutil.which("uvx") is not None


@asynccontextmanager
async def session(settings: Settings) -> AsyncIterator[Any]:
    """An initialised MCP session against Alpaca's server.

    Credentials go through the environment, never on the command line, where
    they would be visible in the process table to anyone on the machine.
    """
    if not is_available():
        raise MCPUnavailable("uvx is not on PATH; cannot launch the Alpaca MCP server")

    settings.require_alpaca()

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:  # pragma: no cover - only without the dependency
        raise MCPUnavailable(f"the mcp client package is not installed: {exc}") from exc

    params = StdioServerParameters(
        command="uvx",
        args=["--from", "alpaca-mcp-server", "alpaca-mcp-server", "--transport", "stdio"],
        env={
            **os.environ,
            "ALPACA_API_KEY": settings.alpaca_api_key,
            "ALPACA_SECRET_KEY": settings.alpaca_secret_key,
            # Belt and braces: the desk is paper-only, and the MCP server is told
            # so explicitly rather than left to a default.
            "ALPACA_PAPER_TRADE": "true",
            "ALPACA_TOOLSETS": TOOLSETS,
        },
    )

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                yield client
    except Exception as exc:  # noqa: BLE001 - any failure means "use REST instead"
        raise MCPUnavailable(f"MCP session failed: {exc}") from exc


def _extract_payload(result: Any) -> Any:
    """Pull usable data out of an MCP tool result.

    Servers return content blocks rather than plain values, and the useful part
    is usually JSON inside a text block. Anything unparseable comes back as raw
    text so the caller can still see what arrived.
    """
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text

        # Alpaca's MCP server wraps every response as
        # {"_alpaca_mcp_security": {...}, "data": {...}} and marks the payload
        # "untrusted_tool_output". The envelope is unwrapped here, and the
        # warning is taken seriously rather than merely unwrapped: headlines
        # reach a language model, so the Catalyst agent's prompt treats them as
        # data to judge and never as instructions to follow.
        if isinstance(payload, dict) and "data" in payload:
            marker = payload.get("_alpaca_mcp_security") or {}
            if marker.get("trust") == "untrusted_tool_output":
                logger.debug("MCP payload marked untrusted by the server, as expected")
            return payload["data"]
        return payload
    return None


async def list_tools(settings: Settings) -> list[str]:
    """Names of the tools the server exposes. Used to prove the link works."""
    async with session(settings) as client:
        listing = await client.list_tools()
        return [tool.name for tool in listing.tools]


async def get_news(settings: Settings, symbols: list[str], limit: int = 12) -> list[dict] | None:
    """Recent headlines for ``symbols``, fetched through the MCP server.

    Returns ``None`` when the MCP path is unavailable, which the caller treats as
    "use REST" and never as "there is no news" -- the Catalyst agent vetoes on an
    empty headline list, so confusing the two would silently block trading.
    """
    try:
        async with session(settings) as client:
            result = await client.call_tool(
                "get_news",
                {"symbols": ",".join(symbols), "limit": min(limit, 50), "sort": "desc"},
            )
    except MCPUnavailable as exc:
        logger.info("MCP news unavailable (%s); falling back to REST", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP get_news failed: %s; falling back to REST", exc)
        return None

    payload = _extract_payload(result)
    if isinstance(payload, dict):
        items = payload.get("news")
        if isinstance(items, list):
            return items
    if isinstance(payload, list):
        return payload

    logger.warning("MCP get_news returned an unexpected shape: %s", type(payload).__name__)
    return None
