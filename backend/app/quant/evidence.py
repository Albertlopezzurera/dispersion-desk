"""Does this desk's own signal have proven predictive power?

Why this module is the centre of the project
--------------------------------------------
Building the dispersion signal was straightforward. Testing it honestly was not,
and testing it honestly is what changed the design.

The first measurement looked excellent: an 85.7% directional hit rate
out-of-sample on four years of real prices. It was wrong, and the way it was
wrong is the commonest error in quantitative trading.

The signal is computed over a rolling 60-day window, so consecutive observations
share 59 of their 60 days. Measured on real data, the dispersion ratio has an
autocorrelation of 0.977 at one day and does not fall below 0.2 until about 24
days out. So 366 "observations" are worth roughly **16 independent samples**.
Scoring a hit rate across all 366 inflates the apparent sample size about
twenty-fold, and with it every confidence interval and every p-value.

Corrected, the same signal on the same data gives a 50.0% hit rate
out-of-sample with p = 0.60. A coin flip.

What the desk does about it
---------------------------
It refuses to trade as though it had an edge it has not demonstrated.
:func:`assess` runs this test continuously and the risk engine consults the
verdict: a signal that has not cleared its own significance bar cannot authorise
a full-size position, however attractive the current reading looks.

This is unusual, and deliberately so. A trading agent that cannot tell a real
edge from an artefact of overlapping windows will find "edges" in noise
indefinitely, and will do it with complete confidence.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

# Autocorrelation below this is treated as effectively decorrelated. 0.2 is a
# convention, not a law; it is named here so the choice stays visible rather
# than buried as a magic number.
DECORRELATION_THRESHOLD = 0.2

# Standard significance level. A signal that cannot clear it does not get to
# size a position as if it had.
ALPHA = 0.05

# Below this many independent samples no result means anything either way, and
# the honest verdict is "underpowered" rather than "fails".
MIN_INDEPENDENT_SAMPLES = 12


class EvidenceError(ValueError):
    """Raised when an evidence assessment cannot be computed."""


def autocorrelation(series: Sequence[float], lag: int) -> float:
    """Sample autocorrelation of ``series`` at ``lag``.

    Returns 0.0 for a constant series, where correlation is undefined and any
    other answer would be an invention.
    """
    n = len(series)
    if lag <= 0:
        raise EvidenceError(f"lag must be positive, got {lag}")
    if n <= lag + 1:
        return 0.0

    mean = statistics.fmean(series)
    denominator = sum((v - mean) ** 2 for v in series)
    if denominator <= 0:
        return 0.0

    numerator = sum((series[i] - mean) * (series[i + lag] - mean) for i in range(n - lag))
    return numerator / denominator


def decorrelation_lag(series: Sequence[float], threshold: float = DECORRELATION_THRESHOLD) -> int:
    """Smallest lag at which the series stops predicting itself.

    This is the spacing at which observations may be treated as independent. If
    autocorrelation never falls below the threshold within half the sample, the
    whole sample is effectively one observation, and that is what is returned.
    """
    n = len(series)
    if n < 4:
        return max(1, n)

    for lag in range(1, n // 2):
        if autocorrelation(series, lag) < threshold:
            return lag
    return n // 2


def effective_sample_size(
    series: Sequence[float], threshold: float = DECORRELATION_THRESHOLD
) -> int:
    """How many genuinely independent observations a series contains."""
    lag = decorrelation_lag(series, threshold)
    return max(1, len(series) // max(1, lag))


def binomial_p_value(successes: int, trials: int, probability: float = 0.5) -> float:
    """One-sided probability of ``successes`` or more under the null.

    The null is a coin flip: a signal with no information gets the direction
    right half the time. Anything a strategy claims above that has to beat this
    number before it counts as evidence.
    """
    if trials <= 0:
        raise EvidenceError(f"trials must be positive, got {trials}")
    if not 0 < probability < 1:
        raise EvidenceError(f"probability must be in (0, 1), got {probability}")
    successes = max(0, min(successes, trials))

    return sum(
        math.comb(trials, k) * probability**k * (1 - probability) ** (trials - k)
        for k in range(successes, trials + 1)
    )


@dataclass(frozen=True)
class EvidenceReport:
    """The signal's statistical standing, stated without decoration."""

    n_raw: int  # overlapping observations: the misleading number
    n_independent: int  # what the sample is actually worth
    decorrelation_days: int
    hit_rate: float
    p_value: float
    naive_hit_rate: float  # what the overlapping sample would have claimed

    @property
    def is_underpowered(self) -> bool:
        return self.n_independent < MIN_INDEPENDENT_SAMPLES

    @property
    def is_significant(self) -> bool:
        """Significant only if the sample can support the claim at all."""
        return not self.is_underpowered and self.p_value < ALPHA

    @property
    def inflation_factor(self) -> float:
        """How far the overlapping sample overstates the evidence."""
        return self.n_raw / max(1, self.n_independent)

    @property
    def verdict(self) -> str:
        if self.is_underpowered:
            return "underpowered"
        return "proven" if self.is_significant else "unproven"

    def render(self) -> str:
        lines = [
            "SIGNAL EVIDENCE",
            f"  observations (overlapping)   {self.n_raw}",
            f"  decorrelates after           {self.decorrelation_days} days",
            f"  independent samples          {self.n_independent} "
            f"({self.inflation_factor:.0f}x inflation)",
            f"  hit rate, independent        {self.hit_rate:.1%}",
            f"  hit rate, naive              {self.naive_hit_rate:.1%}  <- overstated",
            f"  p-value vs coin flip         {self.p_value:.3f}",
            f"  verdict                      {self.verdict.upper()}",
        ]
        if self.is_underpowered:
            lines.append(
                f"  Fewer than {MIN_INDEPENDENT_SAMPLES} independent samples: this is"
            )
            lines.append("  absence of evidence, not evidence of absence.")
        elif not self.is_significant:
            lines.append("  The signal does not beat a coin flip on independent samples.")
        return "\n".join(lines)


def assess(
    signals: Sequence[float],
    forward_changes: Sequence[float],
    quantile: float = 0.5,
) -> EvidenceReport:
    """Score a signal against its own future, correcting for overlap.

    Args:
        signals: the signal value at each observation.
        forward_changes: what the signal did next, aligned index for index.
        quantile: fraction taken from each tail when forming the test set.

    The test is directional: a high signal should be followed by a fall and a low
    signal by a rise. That is the claim mean reversion makes, and it is scored
    only on observations spaced far enough apart to be independent.
    """
    if len(signals) != len(forward_changes):
        raise EvidenceError(
            f"signals and forward_changes must align, got {len(signals)} and "
            f"{len(forward_changes)}"
        )
    if not 0 < quantile <= 0.5:
        raise EvidenceError(f"quantile must be in (0, 0.5], got {quantile}")

    n_raw = len(signals)
    if n_raw < 4:
        raise EvidenceError(f"need at least 4 observations, got {n_raw}")

    lag = decorrelation_lag(signals)

    def hit_rate(idx: Sequence[int]) -> tuple[float, int, int]:
        pairs = sorted(((signals[i], forward_changes[i]) for i in idx), key=lambda p: p[0])
        if len(pairs) < 2:
            return 0.0, 0, 0
        k = max(1, int(len(pairs) * quantile))
        low, high = pairs[:k], pairs[-k:]
        hits = sum(1 for _, change in high if change < 0)
        hits += sum(1 for _, change in low if change > 0)
        return hits / (2 * k), hits, 2 * k

    independent_idx = list(range(0, n_raw, max(1, lag)))
    rate, hits, trials = hit_rate(independent_idx)
    naive_rate, _, _ = hit_rate(list(range(n_raw)))

    return EvidenceReport(
        n_raw=n_raw,
        n_independent=len(independent_idx),
        decorrelation_days=lag,
        hit_rate=rate,
        p_value=binomial_p_value(hits, trials) if trials else 1.0,
        naive_hit_rate=naive_rate,
    )


def position_scale(report: EvidenceReport) -> float:
    """How much of a full position an unproven signal may take.

    Not a binary switch. A signal that has cleared its significance bar trades at
    full size; one that has not is capped at a fraction; one whose sample cannot
    support any conclusion is capped harder still.

    Zero is deliberately avoided for the underpowered case: refusing to trade at
    all means never gathering the data that would settle the question. A small
    position keeps the experiment running at a cost the book can absorb.
    """
    if report.is_significant:
        return 1.0
    if report.is_underpowered:
        return 0.25
    return 0.10
