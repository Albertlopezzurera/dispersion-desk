# Backtest results

Generated 2026-09-01 from real Alpaca daily bars.

## Data

```
index      SPY
basket     NVDA, MSFT, AAPL, AMZN, META, AVGO
sessions   1506  (2020-09-02 -> 2026-09-01)
```

## Chronological split (no leakage)

```
  train         2020-10-29 -> 2023-09-13 (722 obs)
  validation    2023-09-14 -> 2025-02-21 (361 obs)
  out-of-sample 2025-02-24 -> 2026-08-03 (362 obs)
```

## Does the signal have proven predictive power?

Computed from actual closes only: no option model, no cost assumption, no fitted parameters. Two numbers are reported for each segment, and the gap between them is the point.

The signal uses a rolling 60-day window, so neighbouring observations share 59 of their 60 days. Scoring a hit rate across all of them counts the same few underlying events over and over. The **independent** figure spaces observations by the measured decorrelation lag; the **naive** figure is what the overlapping sample would have claimed.

```
train
SIGNAL EVIDENCE
  observations (overlapping)   722
  decorrelates after           26 days
  independent samples          28 (26x inflation)
  hit rate, independent        75.0%
  hit rate, naive              69.8%  <- overstated
  p-value vs coin flip         0.006
  verdict                      PROVEN
```

```
validation
SIGNAL EVIDENCE
  observations (overlapping)   361
  decorrelates after           42 days
  independent samples          9 (40x inflation)
  hit rate, independent        87.5%
  hit rate, naive              65.0%  <- overstated
  p-value vs coin flip         0.035
  verdict                      UNDERPOWERED
  Fewer than 12 independent samples: this is
  absence of evidence, not evidence of absence.
```

```
out-of-sample
SIGNAL EVIDENCE
  observations (overlapping)   362
  decorrelates after           31 days
  independent samples          12 (30x inflation)
  hit rate, independent        66.7%
  hit rate, naive              68.8%  <- overstated
  p-value vs coin flip         0.194
  verdict                      UNPROVEN
  The signal does not beat a coin flip on independent samples.
```

## Modelled option P&L — deliberately not reported

A P&L backtest of this strategy would require a historical **implied** volatility surface. Alpaca does not serve one, and its historical option bars are too sparse away from the money to reconstruct it.

Substituting realised volatility does not work either, and the reason is worth stating precisely: with a 60-day estimation window and a 10-day holding period the entry and exit windows overlap by about 83%, so only a sixth of the data changes. The measurable volatility move is damped almost to zero while transaction costs are not, and the resulting numbers describe the estimator rather than the market.

The engine is in the repository (`backend/app/backtest/engine.py`) and runs under `--include-modelled-pnl`, but its output is not presented as evidence. Producing an attractive backtest here would only require assuming lower costs or a longer holding period, which is exactly why it is omitted.

**What is reported instead is the signal test above**, computed from actual closes with no option model, no cost assumption, and no fitted parameters.
