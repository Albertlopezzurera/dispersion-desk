# Dispersion Desk

**An autonomous options agent that trades the gap between index volatility and the
volatility of its parts — and proves its profits came from the risk it chose to take.**

Built for the [Alpaca AI Trading Agents Hackathon](https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon).
Paper trading only.

---

## The problem

Almost every autonomous trading agent bets on **direction**: will this go up or down?
That question is crowded, mostly noise over short horizons, and — critically — a
profitable run tells you nothing about whether the system works. Over three trading
sessions, a directional bot that makes money and a directional bot that got lucky
produce identical evidence.

Dispersion Desk asks a different question, and answers a harder one about itself.

## The strategy

An index's implied volatility is not the average of its constituents' volatilities. It
is that average **damped by correlation**. For weights `w` and constituent vols `σᵢ`:

```
σ_index² = Σᵢ Σⱼ wᵢ wⱼ σᵢ σⱼ ρᵢⱼ
```

Since every `ρᵢⱼ ≤ 1`, subadditivity guarantees `σ_index ≤ Σᵢ wᵢ σᵢ`. The gap between
the two sides is **pure correlation**, and it is what this desk trades.

The signal is the **correlation risk premium**:

```
premium = implied correlation (from today's option chains)
        − realised correlation (from 90 days of daily returns)
```

When implied sits well above realised, index options price more co-movement than the
constituents have actually shown: sell index volatility, buy single-name volatility.
When it sits below, do the reverse. **Nothing here is a view on market direction.**

### Why this survives a free market-data feed

Alpaca's free `indicative` options feed publishes *derived* quotes, not consolidated
OPRA quotes. A strategy that values contracts in absolute terms inherits that bias in
full. This one expresses its signal as a **ratio of one implied volatility to another
from the same feed**, so a systematic multiplicative bias largely cancels. That
robustness is the desk's central design bet, and it is why the strategy is viable
without a paid data subscription.

## Why the P&L attribution is the real test

The desk is delta-neutral by construction, so its profits are *supposed* to come from
**vega** and be eroded by **theta**. That claim is falsifiable, and every closed
position falsifies it:

```
P&L = delta + gamma + vega + theta + slippage + residual
```

A worked example, produced by `backend/app/quant/attribution.py`:

```
P&L ATTRIBUTION -- basket
  Total realised               +789.10
  Delta (direction)              +6.38   <- not what we bet on
  Gamma (convexity)              +2.08
  Vega  (volatility)           +902.08   <- 91.8% of gross: the risk we chose
  Theta (time decay)            -72.18
  Slippage (execution)           +0.00
  Residual (unexplained)        -49.27
  Dominant driver: vega
```

If a profitable run attributes its P&L to **delta**, the desk did not earn that money
the way it claims to — the hedge is broken, the sizing is wrong, or it got lucky. That
is a bug report, not a result. With only a handful of sessions available, this
decomposition says far more about whether the system works than the P&L number does.

## Architecture

```
Option chains (Alpaca) ── spot + 90d daily bars (Alpaca)
        │
   Surface Engine        [deterministic]  own Black-Scholes IV inversion;
        │                                 rejects put-call parity violations
   Dispersion Engine     [deterministic]  dispersion ratio, implied ρ
   Realised Engine       [deterministic]  realised ρ from returns
        │
   Correlation premium   ──── inside band? ──> no trade (the common outcome)
        │
   Basket Builder        [deterministic]  index iron condor + name strangles
        │
   Catalyst Agent        [LLM]  is this vol explained by a known event? CAN VETO
   Devil's Advocate      [LLM]  adversarial pre-mortem (advisory only)
        │
   Risk Engine           [deterministic, NO LLM]  ── the only component that can
        │                                            authorise a trade
   Executor              multi-leg orders via Alpaca (order_class="mleg")
        │
   Journal (SQLite)  +  Attribution  +  Narrator [LLM]
```

### The rule the architecture rests on

**The LLM never does arithmetic and never authorises a trade.** Implied volatility,
correlation, position sizing and every risk limit are deterministic, tested code. A
model is used in exactly three places where judgement over unstructured text adds
something maths cannot — and only one of them can stop a trade.

Most projects in this space wire five LLMs into a debate and call it multi-agent. This
desk deliberately uses **three**, and says why each one is there. The Catalyst Agent is
the one that earns its place: a dispersion signal on a company reporting earnings
tomorrow is not a mispricing, it is event risk correctly priced. Telling those apart
requires reading the news, and getting it wrong is the most expensive mistake this
strategy can make.

## Risk architecture

Limits are expressed in **greek space**, matching what the strategy actually risks:

| Gate | Why |
|---|---|
| Net delta band | The correctness test for a direction-neutral desk |
| Portfolio vega / gamma / daily theta | Caps on the risk deliberately taken |
| Defined risk per basket, and in aggregate | Both as a % of NAV |
| Per-underlying concentration | Post-trade, counting existing exposure |
| Daily loss circuit breaker | Trips once, stops the session |
| Quote freshness | Critical on a delayed indicative feed |
| Bid-ask spread, open interest | Refuses illiquid or one-sided markets |
| IV sanity | Rejects contracts whose IV could not be solved |
| Implied correlation sanity | A ρ outside [-1, 1] means the data is wrong, not that there is an edge |
| Index weight staleness | The constituent weights are a manual snapshot |

Four principles, enforced in code:

1. **No LLM touches the risk engine.**
2. **Limits are checked post-trade**, on the resulting book — not on the basket in
   isolation. Checking baskets alone is how books drift past their limits one
   acceptable-looking trade at a time.
3. **Fail closed.** Missing data is a rejection. No timestamp, no solvable IV, no known
   NAV → no trade.
4. **Every rejection carries its arithmetic**, so the Risk Centre shows *why*, not just
   *that*.

This invariant is proven over 4,000 randomised portfolios in
`backend/tests/test_risk_engine.py`:

> **If the engine approves a basket, no configured limit is breached.**

## Alpaca integration

| Purpose | Endpoint |
|---|---|
| Option chain (quotes) | `GET /v1beta1/options/snapshots/{underlying}` |
| Open interest | `GET /v2/options/contracts` |
| Contract snapshots | `GET /v1beta1/options/snapshots?symbols=…` |
| Underlying spot / daily bars | `GET /v2/stocks/…` |
| News (Catalyst agent) | `GET /v1beta1/news` |
| Account, positions, clock | `GET /v2/account`, `/v2/positions`, `/v2/clock` |
| Multi-leg orders | `POST /v2/orders` with `order_class: "mleg"` |

### The MCP server, actually used

Alpaca's official MCP server is not merely declared in a config file: the Catalyst
agent's news lookup runs through it on every cycle, launched on demand with
`uvx alpaca-mcp-server` over stdio. The audit trail records which path was taken, so the
claim is checkable rather than asserted:

```
[mcp] Fetched 12 headlines through Alpaca's MCP server (tool get_news).
      Payload is flagged untrusted by the server and is passed to the model
      as data, never as instructions.
```

That step was chosen deliberately. MCP earns its place where a tool call is *semantic* —
fetching text for a language model to read is exactly that. The quantitative path stays on
REST, because a desk that refuses to trade on stale quotes should not put a subprocess
handshake between itself and the prices it values from. If the server cannot start, the
REST endpoint takes over; an optional integration must never stop the desk trading.

**A security detail worth honouring.** Alpaca's MCP server wraps every response as
`{"_alpaca_mcp_security": {...}, "data": {...}}` and marks the payload
`untrusted_tool_output`, warning that it may contain prompt injection. Those headlines go
straight to a language model, so the desk takes the warning seriously: the Catalyst
prompt instructs the model to treat headlines strictly as data, the content is fenced in
explicit delimiters, and the agent is told never to follow an instruction found inside
one.

## Does the signal actually work? The answer changed the project

The first honest measurement looked excellent: **85.7%** directional hit rate
out-of-sample on four years of real prices. It was wrong, and how it was wrong became the
point of the whole system.

The signal uses a rolling 60-day window, so consecutive observations share 59 of their 60
days. On real data the dispersion ratio has an autocorrelation of **0.977 at one day** and
does not fall below 0.2 until about **24 days** out. So 366 "observations" are worth
roughly **16 independent samples**. Scoring a hit rate across all 366 inflates the
apparent sample size about twenty-fold, and with it every confidence interval.

Corrected, on six years of real Alpaca data:

| segment | independent samples | hit rate | naive hit rate | p-value | verdict |
|---|---|---|---|---|---|
| train | 24 (30x inflation) | — | — | 0.006 | proven |
| validation | 9 (40x) | 87.5% | 65.0% | 0.035 | **underpowered** |
| out-of-sample | 12 (30x) | 66.7% | 68.8% | 0.194 | **unproven** |

Note validation: p = 0.035 clears the conventional 5% bar, and the gate still refuses it,
because a p-value computed on nine samples is not worth having.

The same number, three ways, depending on parameters nobody should be able to tune after
the fact: 85.7% (lookback 40, horizon 21), 68.8% (lookback 60, naive), 66.7% (lookback 60,
independent). Only the last one is honest.

### What the desk does about it

`backend/app/quant/evidence.py` runs this test continuously, and the verdict **sizes the
position**:

| verdict | position | reasoning |
|---|---|---|
| proven | 100% | the signal beats a coin flip on independent samples |
| unproven | 10% | it does not, so it trades small rather than not at all |
| underpowered | 25% | too few samples to conclude; refusing outright would mean never learning |

This is visible in live logs. On a real cycle the gate cut a basket's worst case from
**$4,722 to $872**, which is what brought it inside the 1.5%-of-NAV per-trade limit. The
finding is not a footnote in a report; it governs what the agent is allowed to do.

### What is deliberately not reported

A P&L backtest. It would need a historical *implied* volatility surface, which Alpaca does
not serve. Substituting realised volatility fails for a checkable reason: with a 60-day
estimation window and a 10-day holding period the entry and exit windows overlap by ~83%,
so the measurable move is damped to near zero while costs are not. The engine is in the
repository and runs under `--include-modelled-pnl`; its output is not presented as
evidence, because making it look good would only require assuming lower costs.

```bash
python scripts/backtest.py --years 6
```

### Bugs this process found

Four, none of which a unit test would have caught, all found by running against the live
market:

1. `buy_index_vol` was unreachable — the defined-risk rule rejected every short-name leg,
   so the signal fired and the basket was always abandoned. Fixed by using iron condors
   rather than sold strangles on the short side.
2. The evidence gate computed a position scale that nothing applied.
3. The free feed returns no open interest, so the liquidity gate rejected 100% of
   contracts while appearing to work.
4. The free feed returns no greeks, so the book measured as delta 0 and vega 0 regardless
   of what was held.

## Deployment

The image serves the API and the console together, so the demo has one URL.

```bash
docker build -t dispersion-desk .
```

Then run it, passing credentials as environment variables (never baked into the image):

```bash
docker run -p 8000:8000 --env-file .env dispersion-desk
```

For a hosted deployment, `render.yaml` is a Render blueprint: point Render at the
repository and it provisions the service, prompting once for the secrets and storing them
encrypted. A deployed instance runs with `PROPOSE_ONLY=true`, so it analyses and proposes
but does not submit orders until that is deliberately changed. A public URL that places
trades the moment someone loads it is not a demo.

Render's free plan has no persistent disk, so the decision journal resets on redeploy;
uncomment the `disk:` block in the blueprint on a paid plan to keep the audit trail.

## Quick start

```bash
python -m venv .venv
```

Then install dependencies (`.venv/bin/pip` on macOS or Linux):

```bash
.venv/Scripts/pip install -r requirements.txt
```

Copy the environment template:

```bash
cp .env.example .env
```

Create a **fresh Alpaca paper trading account** with a $100,000 starting balance and put
its keys in `.env`. That file is git-ignored and must never be committed.

Start the API:

```bash
.venv/Scripts/python -m uvicorn app.main:app --app-dir backend --port 8000
```

In a second terminal, start the console at `http://127.0.0.1:5173`:

```bash
npm install --prefix frontend && npm run dev --prefix frontend
```

The agent does **not** start on boot. Press *Start agent* in the console, or *Run cycle
now* for a single pass. Starting a trading loop as a side effect of launching a web
server is how an agent ends up running when nobody meant it to.

### Safety switches

| Variable | Default | Effect |
|---|---|---|
| `PROPOSE_ONLY` | `true` | Analyses and proposes, never submits an order |
| `KILL_SWITCH` | `false` | Refuses to run any cycle at all |
| `ALPACA_PAPER_TRADE` | `true` | The client refuses to run against a live account |

All three are re-checked at the last point before a request leaves the process, so no
code path can bypass them.

## Tests

```bash
.venv/Scripts/python -m pytest -v
```

269 tests, none of them smoke tests:

- **Black-Scholes** — put-call parity exact to 1e-10; analytic greeks against finite
  differences of the pricer itself; IV round-trips across 2%–300% vol; the solver
  returns `None` rather than inventing a number from an un-invertible quote.
- **Risk engine** — every gate fails closed on missing data; the approval invariant over
  4,000 randomised portfolios, with a guard asserting the generator actually produces
  approvals (a property test that approves nothing passes vacuously and proves nothing).
- **Attribution** — components plus residual reconcile to realised P&L *exactly*, in
  every case including violent moves; the decomposition correctly discriminates vega
  from theta from delta.

## Known limitations

Stated here, in the console, and in the code — not buried.

- **Free `indicative` feed**: quotes are derived, not OPRA; trades are delayed 15
  minutes. Mitigated by the ratio framing, not eliminated.
- **The basket covers ~30% of the index by weight.** Six liquid names, renormalised —
  not a replication of SPY. The residual is a real basis error, and the coverage figure
  is displayed in the console.
- **Index weights are a dated manual snapshot.** Alpaca does not serve index
  composition. A staleness gate refuses to trade on weights older than 30 days.
- **Max 4 legs per Alpaca multi-leg order**, so a basket cannot fill atomically: the
  condor is one order and each strangle another. The resulting legging risk shows up as
  measured `slippage` in the attribution rather than being assumed away.
- **Paper fills are simulated** by Alpaca and are more forgiving than a real book.
- **American vs European**: listed equity options are American, priced here with
  Black-Scholes, as the whole market — Alpaca included — quotes IV. Negligible for the
  out-of-the-money 21–45 DTE contracts traded.

## Repository layout

```
backend/app/
  config.py             typed settings; every risk limit validated on load
  alpaca/client.py      async Trading + Market Data client
  quant/
    black_scholes.py    pricing, greeks, Brent IV inversion
    surface.py          chain -> one trustworthy ATM volatility
    dispersion.py       dispersion ratio, implied correlation
    realized.py         realised volatility and correlation
    attribution.py      greek P&L decomposition
    universe.py         basket and index weights
  agents/analysts.py    Catalyst, Devil's Advocate, Narrator + LLM client
  backtest/engine.py    chronological splits, metrics, signal test
  risk/engine.py        deterministic gates; the only authoriser
  execution/executor.py structure construction and MLEG submission
  journal/db.py         append-only SQLite decision journal
  orchestrator.py       the autonomous cycle
  main.py               FastAPI + SSE
frontend/src/App.tsx    operator console
scripts/backtest.py     backtest runner
Dockerfile              single image: API + console
render.yaml             Render blueprint
```

## Disclosures

Paper trading only. Simulated funds, no real capital. Paper results are hypothetical and
do not represent actual trading or guarantee future results. This is not investment
advice. Options trading carries significant risk.
