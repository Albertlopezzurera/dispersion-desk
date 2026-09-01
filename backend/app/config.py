"""Typed configuration for the desk, loaded from the environment.

Two rules govern this module:

1. **Secrets never get defaults.**  ``ALPACA_API_KEY`` and friends are optional
   at import time so the test-suite and the pure-quant modules run without
   credentials, but :meth:`Settings.require_alpaca` fails loudly the moment
   something actually needs to talk to the broker.  A silent fallback to an
   empty key would surface as a confusing 401 deep inside a trading cycle.

2. **Risk limits are configuration, not code.**  Every hard cap the risk engine
   enforces is declared here with a conservative default, so the operating
   envelope of the desk can be read in one place and changed without touching
   the engine.  All limits are validated on load: a negative or zero limit is a
   configuration error, never a permissive setting.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["featherless", "openai_compatible", "mock"]
OptionsFeed = Literal["indicative", "opra"]


class ConfigError(RuntimeError):
    """Raised when the desk is asked to do something its configuration forbids."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Alpaca ------------------------------------------------------------
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_paper_trade: bool = True
    alpaca_account_id: str = ""
    alpaca_options_feed: OptionsFeed = "indicative"

    # --- LLM ---------------------------------------------------------------
    llm_provider: LLMProvider = "mock"
    llm_base_url: str = "https://api.featherless.ai/v1"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    featherless_api_key: str = ""
    llm_timeout_seconds: float = Field(default=45.0, gt=0)

    # --- Safety switches ---------------------------------------------------
    propose_only: bool = True
    kill_switch: bool = False

    # --- Risk limits (hard caps enforced by risk/engine.py) ----------------
    max_net_delta: float = Field(default=150.0, gt=0)
    max_portfolio_vega: float = Field(default=400.0, gt=0)
    max_portfolio_gamma: float = Field(default=50.0, gt=0)
    max_daily_theta: float = Field(default=250.0, gt=0)
    max_risk_per_basket_pct: float = Field(default=1.5, gt=0, le=100)
    max_total_risk_pct: float = Field(default=10.0, gt=0, le=100)
    max_underlying_concentration_pct: float = Field(default=4.0, gt=0, le=100)
    max_daily_loss_pct: float = Field(default=3.0, gt=0, le=100)

    # --- Liquidity and data-quality gates ----------------------------------
    max_spread_pct_of_mid: float = Field(default=12.0, gt=0)
    min_open_interest: int = Field(default=25, ge=0)
    max_quote_age_seconds: int = Field(default=900, gt=0)
    min_iv: float = Field(default=0.01, gt=0)
    max_iv: float = Field(default=5.0, gt=0)
    max_weights_age_days: int = Field(default=30, gt=0)

    # --- Strategy ----------------------------------------------------------
    index_symbol: str = "SPY"
    basket_size: int = Field(default=6, ge=2, le=12)
    dispersion_z_entry: float = Field(default=1.5, gt=0)
    # Primary signal threshold: how far implied correlation must sit above (or
    # below) realised correlation before the desk acts. A positive premium is
    # normal -- index options carry a structural hedging bid -- so the entry
    # level sits well above zero.
    correlation_premium_entry: float = Field(default=0.12, gt=0)
    dispersion_z_exit: float = Field(default=0.4, ge=0)
    min_history_samples: int = Field(default=20, ge=2)
    target_dte_min: int = Field(default=21, gt=0)
    target_dte_max: int = Field(default=45, gt=0)
    cycle_interval_seconds: int = Field(default=900, gt=0)
    risk_free_rate: float = Field(default=0.043, ge=0, le=1)

    # --- Persistence -------------------------------------------------------
    database_url: str = "sqlite:///./data/dispersion_desk.db"
    log_level: str = "INFO"

    @field_validator("index_symbol")
    @classmethod
    def _normalise_symbol(cls, v: str) -> str:
        return v.strip().upper()

    @model_validator(mode="after")
    def _check_coherence(self) -> "Settings":
        if self.target_dte_min >= self.target_dte_max:
            raise ValueError(
                f"target_dte_min ({self.target_dte_min}) must be below "
                f"target_dte_max ({self.target_dte_max})"
            )
        if self.min_iv >= self.max_iv:
            raise ValueError(f"min_iv ({self.min_iv}) must be below max_iv ({self.max_iv})")
        if self.dispersion_z_exit >= self.dispersion_z_entry:
            raise ValueError(
                f"dispersion_z_exit ({self.dispersion_z_exit}) must be below "
                f"dispersion_z_entry ({self.dispersion_z_entry}); otherwise a position "
                "would be closed the instant it is opened"
            )
        if self.max_risk_per_basket_pct > self.max_total_risk_pct:
            raise ValueError(
                f"max_risk_per_basket_pct ({self.max_risk_per_basket_pct}) cannot exceed "
                f"max_total_risk_pct ({self.max_total_risk_pct})"
            )
        return self

    # --- Guards ------------------------------------------------------------

    @property
    def has_alpaca_credentials(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)

    def require_alpaca(self) -> None:
        """Fail fast, with an actionable message, before any broker call."""
        if not self.has_alpaca_credentials:
            raise ConfigError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY are not set. Copy .env.example "
                "to .env and fill in the keys from your paper trading dashboard."
            )

    def require_live_execution_allowed(self) -> None:
        """The three switches standing between a proposal and a real order.

        They are checked here rather than at each call site so no execution path
        can forget one of them.
        """
        if self.kill_switch:
            raise ConfigError("KILL_SWITCH is engaged; the desk refuses to trade.")
        if self.propose_only:
            raise ConfigError(
                "PROPOSE_ONLY is true; the desk analyses and proposes but will not "
                "submit orders. Set PROPOSE_ONLY=false to enable paper execution."
            )
        if not self.alpaca_paper_trade:
            raise ConfigError(
                "ALPACA_PAPER_TRADE is false. This project is built and tested for "
                "paper trading only and refuses to run against a live account."
            )

    @property
    def llm_api_key(self) -> str:
        return self.featherless_api_key

    def require_llm(self) -> None:
        if self.llm_provider == "mock":
            return
        if not self.llm_api_key:
            raise ConfigError(
                f"LLM_PROVIDER is '{self.llm_provider}' but no API key is set. "
                "Set FEATHERLESS_API_KEY, or use LLM_PROVIDER=mock for offline runs."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Cached so .env is read exactly once."""
    return Settings()
