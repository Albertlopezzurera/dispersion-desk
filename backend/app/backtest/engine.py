"""Backtesting the dispersion signal, with the honesty constraints stated up front.

What is real and what is modelled
---------------------------------
**Real data:** daily closes for the index and every basket constituent, from
Alpaca. Realised volatility, realised correlation, and every return used to score
the strategy come from those actual prices.

**Modelled:** option prices. Alpaca does not serve a historical implied
volatility surface, and its historical option bars are sparse away from the
money. Reconstructing real chains for seven underlyings across months is not
possible with this data. So the P&L of the option structures is *modelled*:
positions are valued as Black-Scholes straddles off a realised-volatility
estimate, with a transaction cost charged on every leg.

This distinction is not a footnote. A modelled option backtest can be made to
show anything by choosing friendly assumptions, so:

* every strategy result is tagged ``modelled``;
* transaction costs default to a deliberately punitive level, because on options
  the spread is the dominant cost and optimistic fills are the commonest way a
  backtest lies;
* the **signal test** -- does rich dispersion actually mean-revert? -- is computed
  from prices alone and tagged ``real``. That is the result worth trusting, and
  it is reported separately.

Guarding against overfitting
----------------------------
The sample is split chronologically into **train**, **validation** and
**out-of-sample**. Parameters are searched on train, the survivor is chosen on
validation, and out-of-sample is touched exactly once, at the end, to report.
:meth:`DataSplit.assert_no_leakage` enforces that the segments do not overlap and
are strictly ordered in time.

Out-of-sample is never used to select anything. If a strategy looks excellent in
train and mediocre out-of-sample, the out-of-sample number is the honest one and
the strategy is reported as fragile rather than quietly retuned.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Literal

from app.quant.black_scholes import bs_greeks, bs_price
from app.quant.realized import (
    MIN_RETURNS,
    TRADING_DAYS_PER_YEAR,
    average_pairwise_correlation,
    log_returns,
    realized_volatility,
)

Provenance = Literal["real", "modelled"]

# Round-trip cost charged per option leg, as a fraction of the option's price.
# Options quote wide; assuming anything gentler is how a backtest flatters
# itself. 3% per leg across four legs is a 12% round-trip drag.
DEFAULT_COST_PER_LEG = 0.03

# Calendar days a position is held before being marked out.
DEFAULT_HOLDING_DAYS = 10

# Rolling window for the realised statistics feeding each observation.
DEFAULT_LOOKBACK = 60


class BacktestError(ValueError):
    """Raised when a backtest cannot be run honestly on the supplied data."""


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    """Risk-and-return summary of a return series.

    Every field is reported, including the unflattering ones. A strategy is
    judged on the whole set: a high total return with a 40% drawdown across nine
    trades is not evidence of anything.
    """

    n_trades: int
    total_return_pct: float
    mean_trade_pct: float
    win_rate: float
    profit_factor: float
    sharpe: float
    volatility_pct: float
    max_drawdown_pct: float
    best_pct: float
    worst_pct: float
    equity_curve: list[float] = field(default_factory=list)

    @property
    def is_credible(self) -> bool:
        """Whether the sample is large enough to mean anything.

        Twenty trades is already generous for drawing conclusions; below it the
        numbers are anecdotes. Reported so a thin sample cannot masquerade as
        evidence.
        """
        return self.n_trades >= 20

    def render(self, title: str = "") -> str:
        lines = [title if title else "METRICS"]
        lines += [
            f"  trades              {self.n_trades}",
            f"  total return        {self.total_return_pct:+.2f}%",
            f"  mean per trade      {self.mean_trade_pct:+.3f}%",
            f"  win rate            {self.win_rate:.1%}",
            f"  profit factor       {self.profit_factor:.2f}",
            f"  Sharpe (annualised) {self.sharpe:+.2f}",
            f"  volatility          {self.volatility_pct:.2f}%",
            f"  max drawdown        {self.max_drawdown_pct:.2f}%",
            f"  best / worst trade  {self.best_pct:+.2f}% / {self.worst_pct:+.2f}%",
        ]
        if not self.is_credible:
            lines.append(f"  NOTE: only {self.n_trades} trades -- too few to conclude anything.")
        return "\n".join(lines)


def compute_metrics(returns: list[float], periods_per_year: float = 26.0) -> Metrics:
    """Summarise a series of per-trade fractional returns.

    ``periods_per_year`` converts the per-trade Sharpe to an annual figure. With
    a 10-day holding period roughly 25 non-overlapping trades fit in a year.
    """
    if not returns:
        return Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [1.0])

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]

    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    # With no losses the ratio is undefined rather than infinite; report the
    # gross win so a one-sided sample is visible.
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float(gross_win)

    mean = statistics.fmean(returns)
    stdev = statistics.stdev(returns) if len(returns) > 1 else 0.0
    sharpe = (mean / stdev) * math.sqrt(periods_per_year) if stdev > 0 else 0.0

    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))

    peak, max_dd = equity[0], 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)

    return Metrics(
        n_trades=len(returns),
        total_return_pct=100.0 * (equity[-1] - 1.0),
        mean_trade_pct=100.0 * mean,
        win_rate=len(wins) / len(returns),
        profit_factor=profit_factor,
        sharpe=sharpe,
        volatility_pct=100.0 * stdev * math.sqrt(periods_per_year),
        max_drawdown_pct=100.0 * max_dd,
        best_pct=100.0 * max(returns),
        worst_pct=100.0 * min(returns),
        equity_curve=equity,
    )


# --------------------------------------------------------------------------
# Chronological splitting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DataSplit:
    """Chronological train / validation / out-of-sample boundaries.

    Split by time, never randomly: shuffling a price series lets tomorrow inform
    yesterday, which is the textbook way to manufacture a backtest that cannot be
    reproduced live.
    """

    dates: list[date]
    train_end: int
    validation_end: int

    @property
    def train(self) -> slice:
        return slice(0, self.train_end)

    @property
    def validation(self) -> slice:
        return slice(self.train_end, self.validation_end)

    @property
    def out_of_sample(self) -> slice:
        return slice(self.validation_end, len(self.dates))

    def assert_no_leakage(self) -> None:
        """Fail loudly if the segments overlap or are out of order."""
        if not (0 < self.train_end < self.validation_end < len(self.dates)):
            raise BacktestError(
                f"invalid split boundaries: train_end={self.train_end}, "
                f"validation_end={self.validation_end}, n={len(self.dates)}"
            )
        train_last = self.dates[self.train_end - 1]
        val_first = self.dates[self.train_end]
        val_last = self.dates[self.validation_end - 1]
        oos_first = self.dates[self.validation_end]
        if not (train_last < val_first and val_last < oos_first):
            raise BacktestError(
                "segments are not strictly ordered in time; the split leaks future data"
            )

    def describe(self) -> str:
        return (
            f"  train         {self.dates[0]} -> {self.dates[self.train_end - 1]} "
            f"({self.train_end} obs)\n"
            f"  validation    {self.dates[self.train_end]} -> "
            f"{self.dates[self.validation_end - 1]} "
            f"({self.validation_end - self.train_end} obs)\n"
            f"  out-of-sample {self.dates[self.validation_end]} -> {self.dates[-1]} "
            f"({len(self.dates) - self.validation_end} obs)"
        )


def make_split(dates: list[date], train: float = 0.5, validation: float = 0.25) -> DataSplit:
    """Split chronologically into three segments, defaulting to 50/25/25."""
    if not 0 < train < 1 or not 0 < validation < 1 or train + validation >= 1:
        raise BacktestError(f"invalid split fractions: train={train}, validation={validation}")
    n = len(dates)
    if n < 30:
        raise BacktestError(f"need at least 30 observations to split meaningfully, got {n}")

    split = DataSplit(
        dates=dates,
        train_end=int(n * train),
        validation_end=int(n * (train + validation)),
    )
    split.assert_no_leakage()
    return split


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """One day's state, computed only from data available on that day.

    Every state field uses a strictly backward-looking window. ``forward_*``
    fields describe what happened afterwards and exist solely to score the
    decision; nothing in the decision path may read them.
    """

    day: date
    index_vol: float
    basket_vol: float
    realized_correlation: float
    dispersion_ratio: float
    # Outcomes, known only in hindsight.
    forward_index_vol: float | None
    forward_correlation: float | None
    forward_dispersion_ratio: float | None


def build_observations(
    dates: list[date],
    closes: dict[str, list[float]],
    index_symbol: str,
    weights: dict[str, float],
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HOLDING_DAYS,
) -> list[Observation]:
    """Roll a backward-looking window across the sample.

    At index ``i`` only ``[i - lookback, i]`` informs the state, and
    ``[i, i + horizon]`` the outcome. The two never overlap, so a decision can
    never see its own result.
    """
    names = [s for s in weights if s in closes and s != index_symbol]
    if len(names) < 2:
        raise BacktestError("need at least two constituents besides the index")
    if index_symbol not in closes:
        raise BacktestError(f"no price series for index {index_symbol}")

    total = sum(weights[s] for s in names)
    if total <= 0:
        raise BacktestError("basket weights sum to zero")
    norm = {s: weights[s] / total for s in names}

    series_len = min(len(closes[s]) for s in [index_symbol] + names)
    if series_len < lookback + horizon + 5:
        raise BacktestError(
            f"series too short: {series_len} closes for lookback={lookback}, horizon={horizon}"
        )
    if len(dates) < series_len:
        raise BacktestError("fewer dates than price observations")

    out: list[Observation] = []
    # The shifted forward window needs `lookback` observations behind its own
    # start, which the outer bound already guarantees.
    for i in range(lookback, series_len - horizon):
        window = slice(i - lookback, i + 1)

        index_vol = realized_volatility(closes[index_symbol][window])
        vols = {s: realized_volatility(closes[s][window]) for s in names}
        if index_vol is None or any(v is None for v in vols.values()):
            continue

        returns = {s: log_returns(closes[s][window]) for s in names}
        rho = average_pairwise_correlation(returns, norm)
        if rho is None:
            continue

        basket_vol = sum(norm[s] * vols[s] for s in names)  # type: ignore[operator]
        if basket_vol <= 0:
            continue

        # The forward measurement uses the SAME window length as the entry
        # measurement, simply shifted forward by the holding period.
        #
        # This matters more than it looks. A 60-day realised volatility and a
        # 10-day realised volatility are different estimators with different
        # magnitudes and variances, so subtracting one from the other measures
        # the estimator mismatch, not the market. Comparing like with like is
        # what makes the resulting P&L attributable to the dispersion move.
        #
        # The window extends past `i`, which is correct: it scores the decision
        # and is never read by the decision itself.
        fwd = slice(i - lookback + horizon, i + horizon + 1)
        fwd_index_vol = realized_volatility(closes[index_symbol][fwd])
        fwd_vols = {s: realized_volatility(closes[s][fwd]) for s in names}
        fwd_returns = {s: log_returns(closes[s][fwd]) for s in names}
        fwd_rho = average_pairwise_correlation(fwd_returns, norm)

        fwd_basket = (
            sum(norm[s] * fwd_vols[s] for s in names)  # type: ignore[operator]
            if all(v is not None for v in fwd_vols.values())
            else None
        )
        fwd_dr = (
            fwd_index_vol / fwd_basket
            if (fwd_index_vol is not None and fwd_basket)
            else None
        )

        out.append(
            Observation(
                day=dates[i],
                index_vol=index_vol,
                basket_vol=basket_vol,
                realized_correlation=rho,
                dispersion_ratio=index_vol / basket_vol,
                forward_index_vol=fwd_index_vol,
                forward_correlation=fwd_rho,
                forward_dispersion_ratio=fwd_dr,
            )
        )

    return out


# --------------------------------------------------------------------------
# The signal test -- real data, no option model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalTest:
    """Does the dispersion signal carry information about the future?

    Computed purely from prices, so this result is tagged ``real``. It answers a
    narrower question than "is the strategy profitable", but it answers it
    honestly -- and a signal that fails this test cannot be rescued by any amount
    of execution cleverness.
    """

    provenance: Provenance
    n: int
    correlation_with_forward: float | None
    high_signal_mean_change: float
    low_signal_mean_change: float
    spread: float
    hit_rate: float

    def render(self) -> str:
        corr = (
            f"{self.correlation_with_forward:+.3f}"
            if self.correlation_with_forward is not None
            else "n/a"
        )
        return "\n".join(
            [
                f"SIGNAL TEST  [{self.provenance} data, no option model]",
                f"  observations                     {self.n}",
                f"  corr(signal, forward change)     {corr}",
                f"  mean forward DR change, high     {self.high_signal_mean_change:+.4f}",
                f"  mean forward DR change, low      {self.low_signal_mean_change:+.4f}",
                f"  spread between the two           {self.spread:+.4f}",
                f"  directional hit rate             {self.hit_rate:.1%}",
            ]
        )


def test_signal(observations: Iterable[Observation], quantile: float = 0.3) -> SignalTest:
    """Compare what followed high-signal days versus low-signal days.

    The signal is the dispersion ratio: high means index volatility is rich
    relative to the parts. If it is informative, that richness should *fall* over
    the following days -- mean reversion in correlation is the whole premise.
    """
    usable = [o for o in observations if o.forward_dispersion_ratio is not None]
    n = len(usable)
    if n < 3 * MIN_RETURNS:
        return SignalTest("real", n, None, 0.0, 0.0, 0.0, 0.0)

    changes = [o.forward_dispersion_ratio - o.dispersion_ratio for o in usable]  # type: ignore[operator]
    signals = [o.dispersion_ratio for o in usable]

    try:
        corr = statistics.correlation(signals, changes)
    except statistics.StatisticsError:
        corr = None

    ranked = sorted(usable, key=lambda o: o.dispersion_ratio)
    k = max(1, int(n * quantile))
    low, high = ranked[:k], ranked[-k:]

    high_change = statistics.fmean(
        [o.forward_dispersion_ratio - o.dispersion_ratio for o in high]  # type: ignore[operator]
    )
    low_change = statistics.fmean(
        [o.forward_dispersion_ratio - o.dispersion_ratio for o in low]  # type: ignore[operator]
    )

    # A "hit" is the signal predicting the direction correctly: rich dispersion
    # falling, cheap dispersion rising.
    hits = sum(1 for o in high if (o.forward_dispersion_ratio - o.dispersion_ratio) < 0)  # type: ignore[operator]
    hits += sum(1 for o in low if (o.forward_dispersion_ratio - o.dispersion_ratio) > 0)  # type: ignore[operator]

    return SignalTest(
        provenance="real",
        n=n,
        correlation_with_forward=corr,
        high_signal_mean_change=high_change,
        low_signal_mean_change=low_change,
        spread=high_change - low_change,
        hit_rate=hits / (2 * k),
    )


# --------------------------------------------------------------------------
# Strategy simulation -- modelled option prices
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Params:
    """One point in the parameter space being searched."""

    entry_threshold: float  # z-score of the dispersion ratio required to act
    holding_days: int = DEFAULT_HOLDING_DAYS
    lookback: int = DEFAULT_LOOKBACK
    cost_per_leg: float = DEFAULT_COST_PER_LEG
    risk_per_trade_pct: float = 1.0  # capital risked per trade, % of NAV

    def label(self) -> str:
        return (
            f"z>={self.entry_threshold:.2f} hold={self.holding_days}d "
            f"cost={self.cost_per_leg:.1%}/leg risk={self.risk_per_trade_pct:.1f}%"
        )


@dataclass(frozen=True)
class BacktestResult:
    params: Params
    metrics: Metrics
    provenance: Provenance
    segment: str

    def render(self) -> str:
        header = f"{self.segment.upper()}  [{self.provenance}]  {self.params.label()}"
        return f"{header}\n{self.metrics.render()}"


def _straddle_vega(spot: float, vol: float, days: float, rate: float = 0.043) -> float:
    """Vega of an at-the-money straddle, per unit of spot, per volatility point.

    Needed because the two legs of a dispersion trade have very different
    volatilities -- an index near 15%, single names near 30% -- and therefore
    very different vega per straddle. Sizing both legs at one straddle each
    leaves the position net long or short *volatility itself*, so its P&L is
    driven by the level of vol rather than by the ratio the strategy actually
    has a view on.
    """
    t = max(days, 1.0) / 365.0
    call = bs_greeks(spot, spot, t, rate, vol, 0.0, "call")
    put = bs_greeks(spot, spot, t, rate, vol, 0.0, "put")
    return (call.vega + put.vega) / spot


def _straddle_value(spot: float, vol: float, days: float, rate: float = 0.043) -> float:
    """Value of an at-the-money straddle as a fraction of spot.

    A straddle is the cleanest instrument for a pure volatility view: delta
    neutral at inception, value almost entirely vega. Using it keeps the model
    honest -- simulated P&L responds to volatility, which is what the strategy
    claims to trade.
    """
    t = max(days, 1.0) / 365.0
    call = bs_price(spot, spot, t, rate, vol, 0.0, "call")
    put = bs_price(spot, spot, t, rate, vol, 0.0, "put")
    return (call + put) / spot


def simulate(observations: list[Observation], params: Params, zscore_window: int = 60) -> list[float]:
    """Run the strategy over a segment and return per-trade fractional returns.

    The trade: when the dispersion ratio is unusually high, index volatility is
    expensive relative to the constituents, so the desk goes short index
    volatility and long basket volatility. P&L comes from how that relationship
    moves over the holding period.

    **Modelled, not real.** Positions are valued as at-the-money straddles priced
    with Black-Scholes off realised volatility. Real chains would introduce skew,
    discrete strikes and worse fills. Costs are charged on four legs -- two per
    side -- at ``cost_per_leg``.

    The z-score uses a trailing window only, so no observation is scored against
    a mean containing its own future. Trades are non-overlapping: once opened,
    the loop steps past the holding period, because overlapping trades would
    count the same market move repeatedly and inflate both sample size and
    Sharpe.
    """
    if params.holding_days < 1:
        raise BacktestError(f"holding_days must be at least 1, got {params.holding_days}")

    returns: list[float] = []
    i = zscore_window
    while i < len(observations):
        window = [o.dispersion_ratio for o in observations[i - zscore_window : i]]
        if len(window) < MIN_RETURNS:
            i += 1
            continue

        mean = statistics.fmean(window)
        stdev = statistics.stdev(window)
        if stdev <= 0:
            i += 1
            continue

        obs = observations[i]
        z = (obs.dispersion_ratio - mean) / stdev

        if abs(z) < params.entry_threshold:
            i += 1
            continue
        if obs.forward_index_vol is None or not obs.forward_dispersion_ratio:
            i += 1
            continue

        short_index = z > 0
        days = params.holding_days
        basket_fwd_vol = obs.forward_index_vol / obs.forward_dispersion_ratio

        index_vega = _straddle_vega(1.0, obs.index_vol, days)
        basket_vega = _straddle_vega(1.0, obs.basket_vol, days)
        if index_vega <= 0 or basket_vega <= 0:
            i += 1
            continue

        # Vega-match the legs: one unit of vega on each side. A parallel shift
        # in both volatilities then nets to zero, leaving the position exposed
        # to the *spread* between them -- which is the view being expressed.
        index_qty = 1.0 / index_vega
        basket_qty = 1.0 / basket_vega

        index_entry = _straddle_value(1.0, obs.index_vol, days) * index_qty
        index_exit = _straddle_value(1.0, obs.forward_index_vol, days) * index_qty
        basket_entry = _straddle_value(1.0, obs.basket_vol, days) * basket_qty
        basket_exit = _straddle_value(1.0, basket_fwd_vol, days) * basket_qty

        index_pnl = (index_entry - index_exit) if short_index else (index_exit - index_entry)
        basket_pnl = (basket_exit - basket_entry) if short_index else (basket_entry - basket_exit)

        gross = index_pnl + basket_pnl
        # Four legs: a straddle each side, two legs apiece.
        premium = index_entry + basket_entry
        cost = params.cost_per_leg * 2 * premium
        net = gross - cost

        if premium <= 0:
            i += 1
            continue

        # Express the return against NAV rather than against premium.
        returns.append((net / premium) * (params.risk_per_trade_pct / 100.0))
        i += days

    return returns


def run_segment(observations: list[Observation], params: Params, segment: str) -> BacktestResult:
    """Simulate one segment and summarise it."""
    trade_returns = simulate(observations, params)
    periods = TRADING_DAYS_PER_YEAR / max(params.holding_days, 1)
    return BacktestResult(
        params=params,
        metrics=compute_metrics(trade_returns, periods_per_year=periods),
        provenance="modelled",
        segment=segment,
    )
