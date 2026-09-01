# Submission pack — Dispersion Desk

Everything needed to file the hackathon entry: the required one-page write-up, the demo
script, and the submission checklist.

---

# Part 1 — One-page write-up

*(This is the "one-page write-up covering your AI logic, risk gates, and Alpaca
infrastructure implementation" the hackathon requires. Paste as-is.)*

## Dispersion Desk

**An autonomous options agent that trades the gap between index volatility and the
volatility of its parts — and proves its profits came from the risk it chose to take.**

### The idea

Almost every trading agent bets on direction. Over three trading sessions that question
is close to a coin flip, and worse, a profitable run proves nothing: a bot that made
money and a bot that got lucky produce identical evidence.

Dispersion Desk trades **correlation** instead. An index's implied volatility is not the
average of its constituents' volatilities — it is that average damped by how much they
actually move together. Since every pairwise correlation is at most 1, subadditivity
guarantees `σ_index ≤ Σ wᵢσᵢ`, and the gap between the two sides is pure correlation.

The signal is the **correlation risk premium**: implied correlation, backed out of
today's option chains, minus realised correlation, computed from 90 days of daily
returns. When index options price more co-movement than the constituents have shown, the
desk sells index volatility and buys single-name volatility. Nothing here is a view on
whether the market goes up or down.

**Why this works on free data.** Alpaca's `indicative` options feed publishes derived,
not consolidated OPRA, quotes. A strategy valuing contracts in absolute terms inherits
that bias in full. Expressing the signal as a *ratio of one implied volatility to another
from the same feed* cancels most systematic multiplicative bias. That is the central
design bet, and it is what makes the strategy viable without a paid subscription.

### AI logic — three agents, and why not more

**The LLM never does arithmetic and never authorises a trade.** Implied volatility,
correlation, position sizing and every risk limit are deterministic, tested code. Models
appear in exactly three places where judgement over unstructured text adds something
maths cannot:

- **Catalyst Agent — can veto.** Reads recent headlines per name and decides whether that
  name's expensive volatility is *explained* by a known, dated event: earnings, a pending
  merger, an FDA decision. A dispersion signal on a company reporting tomorrow is not a
  mispricing — it is event risk, correctly priced. Confusing the two is the most expensive
  mistake this strategy can make, and separating them requires reading. If the model is
  unreachable or returns nonsense, the verdict is "catalyst present, zero confidence",
  which blocks the trade. **An LLM outage can never become an approval.**
- **Devil's Advocate — advisory.** Argues the strongest case against the basket. Lowers
  confidence; cannot block.
- **Narrator — zero influence.** Writes the human-readable decision memo.

Six competing projects wire five models into a debate and call it multi-agent. This desk
uses three and justifies each one. The maths is deliberately not delegated to a model.

### Risk gates — expressed in greek space

Limits match what the strategy actually risks, rather than generic dollar caps:

- **Net delta band** — the correctness test for a direction-neutral desk
- **Portfolio vega / gamma / daily theta** — caps on the risk deliberately taken
- **Defined risk per basket and in aggregate**, as a % of NAV
- **Per-underlying concentration**, post-trade, counting existing exposure
- **Daily loss circuit breaker** — trips once, stops the session
- **Quote freshness** — critical on a delayed indicative feed
- **Bid-ask spread and open interest** — refuses illiquid or one-sided markets
- **IV sanity** — rejects any contract whose implied volatility could not be solved
- **Implied correlation sanity** — a ρ outside [-1, 1] means the data is wrong, not that
  there is an edge
- **Index weight staleness** — the constituent weights are a manual snapshot

Four principles, enforced in code: no LLM touches the risk engine; limits are checked
**post-trade** on the resulting book, not on the basket in isolation; missing data is a
**rejection**, never a pass; every refusal carries the arithmetic that produced it.

The invariant is proven over 4,000 randomised portfolios: **if the engine approves a
basket, no configured limit is breached.**

### Why the P&L attribution is the real result

The desk is delta-neutral by construction, so its profits are supposed to come from
**vega** and be eroded by **theta**. Every closed position is decomposed:

```
P&L = delta + gamma + vega + theta + slippage + residual
```

A worked example from the test suite:

```
Total realised   +789.10
  Delta            +6.38    <- not what we bet on
  Gamma            +2.08
  Vega           +902.08    <- 91.8% of gross: the risk we chose
  Theta           -72.18
  Slippage         +0.00
  Residual        -49.27
Dominant driver: vega
```

If a profitable run attributes its P&L to delta, the desk did not earn that money the way
it claims to — the hedge is broken or it got lucky. That is a bug report, not a result.
Over a handful of sessions this decomposition says far more about whether the system works
than the P&L number, which at that sample size is mostly noise.

### Alpaca infrastructure

| Purpose | Endpoint |
|---|---|
| Option chain (quotes only on the free feed) | `GET /v1beta1/options/snapshots/{underlying}` |
| Open interest | `GET /v2/options/contracts` |
| Contract snapshots | `GET /v1beta1/options/snapshots?symbols=…` |
| Underlying spot and daily bars | `GET /v2/stocks/…` |
| News (Catalyst agent) | `GET /v1beta1/news` |
| Account, positions, market clock | `GET /v2/account`, `/v2/positions`, `/v2/clock` |
| Multi-leg orders | `POST /v2/orders`, `order_class: "mleg"` |

The desk re-solves implied volatility itself from quoted mids via Brent inversion rather
than trusting the feed's published figure, and cross-checks both sides of every strike
against put-call parity. Alpaca's **MCP server** is called on every cycle: the Catalyst agent's news lookup runs
through `uvx alpaca-mcp-server` over stdio, and the audit trail records which path was
used. The server flags its payload as untrusted output, and the desk honours that — those
headlines reach a language model, so the prompt treats them as data and never as
instructions. The quantitative path stays on REST, where a subprocess handshake between
the desk and its prices would be a liability rather than a feature.

Structures are always defined-risk: a short iron condor on the index (4 legs, one order)
against long strangles on the constituents. Alpaca caps multi-leg orders at four legs, so
a basket cannot fill atomically — the resulting legging risk is measured as `slippage` in
the attribution rather than assumed away.

### Honest limitations

Free indicative feed (derived quotes, 15-minute delayed trades). Basket covers ~30% of
the index by weight — six liquid names, not a replication of SPY. Index weights are a
dated manual snapshot with a staleness gate. Paper fills are simulated. Black-Scholes on
American options, as the whole market quotes IV.

**269 tests.** Put-call parity exact to 1e-10, greeks against finite differences, implied
correlation recovered to 1e-12, attribution reconciling exactly, and every risk gate
failing closed.

---

# Part 2 — Demo script (2–3 minutes)

The goal: someone who has never seen the project understands the problem, the mechanism,
the controls, and the result — in that order.

### 0:00–0:25 — The problem

> "Almost every agent in this hackathon bets on direction. Over three trading days that's
> a coin flip — and a profitable run proves nothing, because luck and skill look
> identical. We built an agent that trades a different question, and can prove which one
> it got."

Show the Dashboard. Point at **net delta near zero**.

### 0:25–1:00 — The mechanism

> "An index's volatility isn't the average of its parts — it's that average damped by
> correlation. We back implied correlation out of today's option chains, compute realised
> correlation from ninety days of returns, and trade the gap. When index options price
> more co-movement than the stocks have actually shown, we sell index vol and buy
> single-name vol."

Show *Strategy configuration*: index, basket, coverage %, entry threshold.

> "And because it's a ratio of two implied vols from the same feed, the bias in Alpaca's
> free derived-quote feed largely cancels. That's what makes it work without paying for
> OPRA."

### 1:00–1:40 — The agent working

Click **Run cycle now**, switch to **Agent Activity**. Let the feed stream.

> "Every step is journalled as it happens: chains fetched, signal computed, structure
> built, catalyst check, adversarial review, risk verdict."

Point at the Catalyst step.

> "This is the only place an LLM can stop a trade — deciding whether a name's expensive
> volatility is explained by a known event. Earnings tomorrow isn't a mispricing, it's
> event risk priced correctly. That judgement needs reading. The maths doesn't — and no
> model touches it."

### 1:40–2:15 — The controls

Open **Risk Centre**, then a rejection.

> "Most of what this desk does is refuse to trade. Every refusal carries the arithmetic:
> observed value, limit, gate name. Limits are in greek space — net delta, vega, gamma —
> and checked on the resulting book, not on the basket alone. Missing data is a rejection,
> never a pass."

> "This invariant is tested over four thousand randomised portfolios: if the engine
> approves, no limit is breached."

### 2:15–2:50 — The result

Open a decision → **P&L attribution**.

> "Here's the part nobody else has. We claim our profits come from volatility, not
> direction. So we decompose every closed position by greek."

Point at the vega row.

> "Vega: the risk we chose. Delta: essentially nothing. We didn't get lucky — we got paid
> for the risk we decided to take. And if that ever flips, it's a bug report, not a
> result."

### 2:50–3:00 — Close

> "Direction-neutral, defined-risk, fully audited, and honest about what it doesn't know.
> Everything runs on Alpaca paper trading."

---

# Part 3 — Submission checklist

| Item | Status |
|---|---|
| Autonomous agent on Alpaca Trading API | done |
| Uses Alpaca MCP server or CLI | done — MCP server called every  cycle for the Catalyst news lookup, logged in the audit trail |
| All strategies incorporate options | done — every structure is options-only |
| Brand-new paper account, $100,000 balance | **pending — you must create this** |
| Alpaca Account ID in submission | pending — blocked on the above |
| Public GitHub repository | pending — awaiting authorisation to create and commit |
| One-page write-up | done — Part 1 above |
| Video presentation | pending — script in Part 2 |
| Slide presentation | pending |
| Cover image | pending |
| Application URL / demo platform | pending — runs locally, needs deployment |
| Social posts (up to 5, tagging lablab.ai + Alpaca) | optional — separate $500 prize |

**Before submitting, confirm:** `.env` is not committed, the account is fresh and funded
at $100,000, and `PROPOSE_ONLY=false` if the agent is meant to place real paper orders
during judging.
