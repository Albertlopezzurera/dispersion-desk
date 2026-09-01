"""The three places where a language model earns its seat at this desk.

The rule the whole architecture rests on: **the LLM never does arithmetic and
never authorises a trade.**  Implied volatility, correlation, sizing and the
risk limits are deterministic code.  A model is used only where judgement over
unstructured text adds something maths cannot:

* :class:`CatalystAgent` -- reads the news and decides whether a name's rich
  volatility is *explained* by a known event.  A dispersion signal on a company
  reporting earnings tomorrow is not a mispricing; it is event risk, correctly
  priced.  Telling those apart requires reading, and getting it wrong is the
  most expensive mistake this strategy can make.  This agent can veto.
* :class:`DevilsAdvocate` -- argues the strongest case *against* the basket.
  Advisory: it lowers confidence, it does not block.
* :class:`Narrator` -- writes the decision memo. No influence on the trade.

Every agent fails soft in one direction only: if the model is unreachable or
returns nonsense, the Catalyst agent reports a catalyst with zero confidence,
which the orchestrator treats as a veto.  An LLM outage must never silently
become an approval.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

_MAX_HEADLINES = 12
_MAX_HEADLINE_CHARS = 180


class LLMError(RuntimeError):
    """The model could not be reached or produced unusable output."""


class LLMClient:
    """Minimal OpenAI-compatible chat client.

    Featherless, and any other OpenAI-compatible endpoint, speak the same
    protocol, so one implementation covers both.  ``mock`` returns deterministic
    responses so the whole pipeline is testable with no network and no key.
    """

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        settings.require_llm()
        self._external = client is not None
        self._client = client or httpx.AsyncClient(timeout=settings.llm_timeout_seconds)

    async def aclose(self) -> None:
        if not self._external:
            await self._client.aclose()

    @property
    def is_mock(self) -> bool:
        return self.settings.llm_provider == "mock"

    async def complete_json(self, system: str, user: str, mock_reply: dict[str, Any]) -> dict:
        """Ask for a JSON object back.

        ``mock_reply`` is what the offline provider returns, so tests exercise
        the real parsing and validation path rather than bypassing it.
        """
        if self.is_mock:
            return mock_reply

        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = await self._client.post(
                f"{self.settings.llm_base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"LLM call failed: {exc}") from exc

        return extract_json(content)


def extract_json(content: str) -> dict:
    """Pull a JSON object out of a model reply.

    Small open models routinely wrap JSON in prose or code fences.  Rather than
    demand perfection, the first balanced ``{...}`` block is extracted; if none
    parses, that is an :class:`LLMError` and the caller fails closed.
    """
    text = content.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1]
        text = text.removeprefix("json").strip()

    start = text.find("{")
    if start == -1:
        raise LLMError(f"no JSON object in model reply: {content[:200]!r}")

    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            try:
                parsed = json.loads(text[start : i + 1])
            except json.JSONDecodeError as exc:
                raise LLMError(f"malformed JSON in model reply: {exc}") from exc
            if not isinstance(parsed, dict):
                raise LLMError("model returned JSON that is not an object")
            return parsed

    raise LLMError(f"unbalanced JSON in model reply: {content[:200]!r}")


@dataclass(frozen=True)
class CatalystVerdict:
    """Per-symbol judgement on whether rich volatility is event-explained."""

    symbol: str
    has_catalyst: bool
    confidence: float  # 0..1
    reason: str
    headlines_seen: int

    @property
    def blocks_trade(self) -> bool:
        """A known catalyst, or an unusable answer, both stop the trade.

        Being long a name's volatility into its own earnings is not a dispersion
        trade -- it is an event bet wearing a dispersion costume.
        """
        return self.has_catalyst or self.confidence <= 0.0


@dataclass(frozen=True)
class AdvocateOpinion:
    """The adversarial pre-mortem. Advisory only: it never blocks."""

    strongest_objection: str
    failure_mode: str
    confidence_adjustment: float  # -1..0, applied to the desk's own confidence


class CatalystAgent:
    """Decides whether a name's volatility is rich for a knowable reason."""

    SYSTEM = (
        "You are a risk analyst on an options dispersion desk. You are given recent "
        "headlines for one stock. Your only job is to decide whether a KNOWN, DATED, "
        "UPCOMING event would justify that stock's options being expensive: scheduled "
        "earnings, a pending merger or acquisition, an FDA decision, a court ruling, a "
        "product launch, an analyst day. Ordinary market commentary, price-move stories, "
        "opinion pieces and old news are NOT catalysts. "
        'Reply with JSON only: {"has_catalyst": bool, "confidence": 0.0-1.0, '
        '"reason": "one sentence"}'
    )

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def assess(self, symbol: str, headlines: list[dict[str, Any]]) -> CatalystVerdict:
        if not headlines:
            # No news is not evidence of no catalyst; it is absence of evidence.
            # Fail closed: the desk does not trade a name it could not check.
            return CatalystVerdict(
                symbol=symbol,
                has_catalyst=True,
                confidence=0.0,
                reason="No headlines available; cannot rule out a scheduled event.",
                headlines_seen=0,
            )

        listed = "\n".join(
            f"- [{str(h.get('created_at', '?'))[:10]}] "
            f"{str(h.get('headline', ''))[:_MAX_HEADLINE_CHARS]}"
            for h in headlines[:_MAX_HEADLINES]
        )
        prompt = f"Stock: {symbol}\n\nRecent headlines:\n{listed}"

        try:
            reply = await self.llm.complete_json(
                self.SYSTEM,
                prompt,
                mock_reply={
                    "has_catalyst": False,
                    "confidence": 0.8,
                    "reason": "Mock provider: no catalyst detected.",
                },
            )
            return CatalystVerdict(
                symbol=symbol,
                has_catalyst=bool(reply.get("has_catalyst", True)),
                confidence=_clamp(float(reply.get("confidence", 0.0)), 0.0, 1.0),
                reason=str(reply.get("reason", ""))[:400],
                headlines_seen=len(headlines),
            )
        except (LLMError, TypeError, ValueError) as exc:
            logger.warning("catalyst check failed for %s: %s", symbol, exc)
            return CatalystVerdict(
                symbol=symbol,
                has_catalyst=True,
                confidence=0.0,
                reason=f"Catalyst check unavailable ({exc}); refusing to trade blind.",
                headlines_seen=len(headlines),
            )


class DevilsAdvocate:
    """Argues against the basket. Cannot block, only reduce confidence."""

    SYSTEM = (
        "You are the risk devil's advocate on an options dispersion desk. The desk is "
        "about to trade the spread between index implied volatility and the implied "
        "volatility of its constituents. Argue the STRONGEST case AGAINST the trade. Be "
        "specific and quantitative where you can. Do not hedge or list generic risks. "
        'Reply with JSON only: {"strongest_objection": "...", "failure_mode": "...", '
        '"confidence_adjustment": -1.0 to 0.0}'
    )

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def challenge(self, summary: str) -> AdvocateOpinion:
        try:
            reply = await self.llm.complete_json(
                self.SYSTEM,
                summary,
                mock_reply={
                    "strongest_objection": "Mock provider: realised correlation may be "
                    "understated because the sample excludes the last macro shock.",
                    "failure_mode": "Correlation spikes and the short index-vol leg loses "
                    "faster than the long single-name legs gain.",
                    "confidence_adjustment": -0.15,
                },
            )
            return AdvocateOpinion(
                strongest_objection=str(reply.get("strongest_objection", ""))[:600],
                failure_mode=str(reply.get("failure_mode", ""))[:600],
                confidence_adjustment=_clamp(
                    float(reply.get("confidence_adjustment", 0.0)), -1.0, 0.0
                ),
            )
        except (LLMError, TypeError, ValueError) as exc:
            logger.warning("devil's advocate unavailable: %s", exc)
            # Advisory only, so an outage costs confidence rather than the trade.
            return AdvocateOpinion(
                strongest_objection=f"Adversarial review unavailable ({exc}).",
                failure_mode="Unreviewed.",
                confidence_adjustment=-0.25,
            )


class Narrator:
    """Writes the human-readable memo. Zero influence on the decision."""

    SYSTEM = (
        "You are writing the decision memo for an options dispersion desk. Explain, in "
        "plain English and under 150 words, what the desk observed, what it decided, and "
        "why. Do not invent numbers. Do not give investment advice. "
        'Reply with JSON only: {"memo": "..."}'
    )

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def write(self, facts: str) -> str:
        try:
            reply = await self.llm.complete_json(self.SYSTEM, facts, mock_reply={"memo": facts})
            return str(reply.get("memo", ""))[:2000]
        except LLMError as exc:
            logger.warning("narrator unavailable: %s", exc)
            # Fall back to the raw facts: never block a decision on prose.
            return facts


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_agents(
    settings: Settings, client: httpx.AsyncClient | None = None
) -> tuple[LLMClient, CatalystAgent, DevilsAdvocate, Narrator]:
    """Construct the three agents over one shared LLM client."""
    llm = LLMClient(settings, client)
    return llm, CatalystAgent(llm), DevilsAdvocate(llm), Narrator(llm)
