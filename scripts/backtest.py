"""Run the dispersion backtest against real Alpaca price history.

    python scripts/backtest.py --years 4

The protocol, and why it is arranged this way
---------------------------------------------
1. **Train** -- the whole parameter grid is searched here. This segment is
   allowed to be overfitted; that is what it is for.
2. **Validation** -- the top candidates from train are re-run. The winner is
   chosen on *validation* Sharpe, never on train Sharpe.
3. **Out-of-sample** -- run exactly once, at the very end, with the single chosen
   parameter set. Nothing is selected here. Whatever it says is the answer.

The gap between train and out-of-sample is reported explicitly, because that gap
*is* the overfitting measurement. A strategy that halves out-of-sample is fragile
regardless of how good the training numbers look.

What is being measured
----------------------
The signal test uses real prices only and is the trustworthy result. The strategy
metrics use modelled option prices (see ``engine.py``) and are labelled as such
wherever they appear. Neither is live trading performance.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.alpaca.client import AlpacaClient  # noqa: E402
from app.backtest.engine import (  # noqa: E402
    BacktestError,
    Params,
    build_observations,
    make_split,
    run_segment,
    test_signal,
)
from app.config import get_settings  # noqa: E402
from app.quant.evidence import assess as assess_evidence  # noqa: E402
from app.quant import universe  # noqa: E402

# The grid is deliberately coarse. A fine grid over many dimensions finds a
# spectacular combination by chance alone; every extra parameter multiplies the
# number of chances the search gets to fool itself.
ENTRY_THRESHOLDS = [1.0, 1.25, 1.5, 2.0]
HOLDING_DAYS = [5, 10, 21]
LOOKBACKS = [40, 60]

# A candidate must trade at least this often in a segment to be considered.
# Anything rarer is a handful of lucky observations wearing a strategy costume.
MIN_TRADES_TO_CONSIDER = 12

# Report lines are joined with this; kept as a name so the join reads clearly.
NEWLINE = chr(10)


async def fetch_closes(symbols: list[str], start: date, end: date):
    """Daily closes and their dates, straight from Alpaca. Real data."""
    settings = get_settings()
    settings.require_alpaca()

    async with AlpacaClient(settings) as client:
        series = await asyncio.gather(
            *(client.get_stock_bars(s, start.isoformat(), end.isoformat()) for s in symbols)
        )

    closes: dict[str, list[float]] = {}
    dates: list[date] = []
    for symbol, bars in zip(symbols, series):
        if not bars:
            raise BacktestError(f"Alpaca returned no bars for {symbol}")
        closes[symbol] = [b["c"] for b in bars]
        if len(bars) > len(dates):
            dates = [datetime.fromisoformat(b["t"].replace("Z", "+00:00")).date() for b in bars]

    # Align every series to the shortest, so a late listing cannot shift the
    # calendar out from under the others.
    shortest = min(len(v) for v in closes.values())
    return {k: v[-shortest:] for k, v in closes.items()}, dates[-shortest:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Dispersion strategy backtest")
    parser.add_argument("--years", type=float, default=4.0, help="history to pull")
    parser.add_argument("--basket", type=int, default=6, help="constituents to trade")
    parser.add_argument("--cost", type=float, default=0.03, help="cost per option leg")
    parser.add_argument("--out", type=str, default="docs/backtest-results.md")
    parser.add_argument(
        "--include-modelled-pnl",
        action="store_true",
        help=(
            "Also run the modelled option P&L. OFF by default: with only free "
            "stock data the entry and exit volatility windows overlap by ~83%%, "
            "which damps the measurable move to near zero and makes the P&L "
            "numbers uninformative rather than merely noisy. The engine is kept "
            "for when a real implied-volatility history is available."
        ),
    )
    args = parser.parse_args()

    settings = get_settings()
    index = settings.index_symbol
    weights = universe.basket_weights(args.basket)
    symbols = [index] + list(weights)

    end = date.today()
    start = end - timedelta(days=int(args.years * 365))

    print(f"Fetching {args.years:.0f}y of daily bars for {', '.join(symbols)} ...")
    closes, dates = asyncio.run(fetch_closes(symbols, start, end))
    print(f"  {len(dates)} sessions, {dates[0]} -> {dates[-1]}  [REAL Alpaca data]\n")

    report: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        report.append(line)

    # ---- observations and split ------------------------------------------
    by_lookback: dict[int, list] = {}
    for lookback in LOOKBACKS:
        by_lookback[lookback] = build_observations(
            dates, closes, index, weights, lookback=lookback, horizon=max(HOLDING_DAYS)
        )

    reference = by_lookback[LOOKBACKS[0]]
    split = make_split([o.day for o in reference])
    split.assert_no_leakage()

    emit("# Backtest results")
    emit()
    emit(f"Generated {date.today().isoformat()} from real Alpaca daily bars.")
    emit()
    emit("## Data")
    emit()
    emit("```")
    emit(f"index      {index}")
    emit(f"basket     {', '.join(weights)}")
    emit(f"sessions   {len(dates)}  ({dates[0]} -> {dates[-1]})")
    emit("```")
    emit()
    emit("## Chronological split (no leakage)")
    emit()
    emit("```")
    emit(split.describe())
    emit("```")
    emit()

    # ---- signal test: REAL data, corrected for overlap --------------------
    emit("## Does the signal have proven predictive power?")
    emit()
    emit(
        "Computed from actual closes only: no option model, no cost assumption, no "
        "fitted parameters. Two numbers are reported for each segment, and the gap "
        "between them is the point."
    )
    emit()
    emit(
        "The signal uses a rolling 60-day window, so neighbouring observations share 59 "
        "of their 60 days. Scoring a hit rate across all of them counts the same few "
        "underlying events over and over. The **independent** figure spaces observations "
        "by the measured decorrelation lag; the **naive** figure is what the overlapping "
        "sample would have claimed."
    )
    emit()
    for name, seg in (
        ("train", split.train),
        ("validation", split.validation),
        ("out-of-sample", split.out_of_sample),
    ):
        rows = [o for o in reference[seg] if o.forward_dispersion_ratio is not None]
        if len(rows) < 8:
            emit(f"`{name}`: too few observations to assess.")
            emit()
            continue
        evidence = assess_evidence(
            [o.dispersion_ratio for o in rows],
            [o.forward_dispersion_ratio - o.dispersion_ratio for o in rows],
        )
        emit("```")
        emit(name)
        emit(evidence.render())
        emit("```")
        emit()

    out_path_early = Path(args.out)
    out_path_early.parent.mkdir(parents=True, exist_ok=True)

    if not args.include_modelled_pnl:
        emit("## Modelled option P&L — deliberately not reported")
        emit()
        emit(
            "A P&L backtest of this strategy would require a historical **implied** "
            "volatility surface. Alpaca does not serve one, and its historical option "
            "bars are too sparse away from the money to reconstruct it."
        )
        emit()
        emit(
            "Substituting realised volatility does not work either, and the reason is "
            "worth stating precisely: with a 60-day estimation window and a 10-day "
            "holding period the entry and exit windows overlap by about 83%, so only "
            "a sixth of the data changes. The measurable volatility move is damped "
            "almost to zero while transaction costs are not, and the resulting numbers "
            "describe the estimator rather than the market."
        )
        emit()
        emit(
            "The engine is in the repository (`backend/app/backtest/engine.py`) and runs "
            "under `--include-modelled-pnl`, but its output is not presented as evidence. "
            "Producing an attractive backtest here would only require assuming lower "
            "costs or a longer holding period, which is exactly why it is omitted."
        )
        emit()
        emit(
            "**What is reported instead is the signal test above**, computed from actual "
            "closes with no option model, no cost assumption, and no fitted parameters."
        )
        emit()
        out_path_early.write_text(NEWLINE.join(report), encoding="utf-8")
        print("Written to " + args.out)
        return 0

    # ---- parameter search on TRAIN ONLY ----------------------------------
    emit("## Parameter search")
    emit()
    emit(
        "The grid is searched on **train** only. The winner is picked on **validation**. "
        "Out-of-sample is run once, at the end, and selects nothing."
    )
    emit()

    candidates: list[tuple[float, Params]] = []
    for lookback in LOOKBACKS:
        train_obs = by_lookback[lookback][split.train]
        for threshold in ENTRY_THRESHOLDS:
            for hold in HOLDING_DAYS:
                params = Params(
                    entry_threshold=threshold,
                    holding_days=hold,
                    lookback=lookback,
                    cost_per_leg=args.cost,
                )
                res = run_segment(train_obs, params, "train")
                if res.metrics.n_trades >= MIN_TRADES_TO_CONSIDER:
                    candidates.append((res.metrics.sharpe, params))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not candidates:
        emit("No parameter set traded often enough in train to be considered.")
        emit()
        emit(
            "That is a result, not a failure: with this data and these costs the "
            "strategy does not produce a usable number of opportunities."
        )
        out_path.write_text(NEWLINE.join(report), encoding="utf-8")
        return 0

    candidates.sort(key=lambda kv: kv[0], reverse=True)
    shortlist = [p for _, p in candidates[:5]]

    emit("Top 5 by train Sharpe, then re-scored on validation:")
    emit()
    emit("| params | train Sharpe | validation Sharpe | validation trades |")
    emit("|---|---|---|---|")

    scored: list[tuple[float, Params]] = []
    for params in shortlist:
        observations = by_lookback[params.lookback]
        train_res = run_segment(observations[split.train], params, "train")
        val_res = run_segment(observations[split.validation], params, "validation")
        emit(
            f"| {params.label()} | {train_res.metrics.sharpe:+.2f} | "
            f"{val_res.metrics.sharpe:+.2f} | {val_res.metrics.n_trades} |"
        )
        if val_res.metrics.n_trades >= max(4, MIN_TRADES_TO_CONSIDER // 2):
            scored.append((val_res.metrics.sharpe, params))
    emit()

    if not scored:
        emit("No candidate traded often enough in validation to be selected. Stopping here.")
        out_path.write_text(NEWLINE.join(report), encoding="utf-8")
        return 0

    scored.sort(key=lambda kv: kv[0], reverse=True)
    chosen = scored[0][1]

    emit(f"**Selected on validation:** `{chosen.label()}`")
    emit()

    # ---- the single out-of-sample run ------------------------------------
    observations = by_lookback[chosen.lookback]
    train_res = run_segment(observations[split.train], chosen, "train")
    val_res = run_segment(observations[split.validation], chosen, "validation")
    oos_res = run_segment(observations[split.out_of_sample], chosen, "out-of-sample")

    emit("## Results — modelled option prices")
    emit()
    emit(
        "> **These are modelled, not live results.** Option P&L is simulated with "
        "Black-Scholes straddles priced off realised volatility, charging "
        f"{args.cost:.0%} per leg. Alpaca does not serve a historical implied "
        "volatility surface, so a real-price options backtest is not possible with "
        "this data. Nothing below is live trading performance."
    )
    emit()
    for res in (train_res, val_res, oos_res):
        emit("```")
        emit(res.render())
        emit("```")
        emit()

    # ---- the overfitting measurement -------------------------------------
    decay = train_res.metrics.sharpe - oos_res.metrics.sharpe
    emit("## Overfitting check")
    emit()
    emit("```")
    emit(f"train Sharpe          {train_res.metrics.sharpe:+.2f}")
    emit(f"validation Sharpe     {val_res.metrics.sharpe:+.2f}")
    emit(f"out-of-sample Sharpe  {oos_res.metrics.sharpe:+.2f}")
    emit(f"decay (train - OOS)   {decay:+.2f}")
    emit("```")
    emit()

    if not oos_res.metrics.is_credible:
        verdict = (
            f"**Inconclusive.** Only {oos_res.metrics.n_trades} out-of-sample trades — "
            "too few to support any claim. Reported as-is rather than padded by "
            "loosening the entry threshold until the sample looks respectable."
        )
    elif oos_res.metrics.sharpe <= 0:
        verdict = (
            "**The strategy does not survive out-of-sample.** Whatever the training "
            "numbers show, this parameter set does not generalise. Reported rather "
            "than retuned, because retuning on out-of-sample is exactly the error "
            "this protocol exists to prevent."
        )
    elif decay > 1.0:
        verdict = (
            f"**Fragile.** Sharpe decays by {decay:.2f} from train to out-of-sample, "
            "which is the signature of a fitted parameter set rather than a real edge."
        )
    else:
        verdict = (
            f"**Holds up.** Out-of-sample Sharpe {oos_res.metrics.sharpe:+.2f} with "
            f"{oos_res.metrics.n_trades} trades and a maximum drawdown of "
            f"{oos_res.metrics.max_drawdown_pct:.1f}%. Decay of {decay:+.2f} is within "
            "what honest parameter selection costs."
        )
    emit(verdict)
    emit()

    out_path.write_text(NEWLINE.join(report), encoding="utf-8")
    print("Written to " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
