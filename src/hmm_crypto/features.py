"""Feature engineering on OHLCV data.

Every feature at date t only uses information available at the close of t
(rolling / exponentially-weighted windows looking backwards). This causality
is enforced by unit tests (`tests/test_features.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]


def log_returns(close: pd.Series) -> pd.Series:
    """r_t = ln(P_t / P_{t-1})."""
    return np.log(close).diff()


def realized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of log-returns (non-annualised)."""
    return returns.rolling(window, min_periods=window).std()


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Parkinson (1980) range-based estimator: sigma^2 = E[ln(H/L)^2] / (4 ln 2).

    More efficient than close-to-close volatility because it uses the intraday range.
    """
    hl2 = np.log(high / low) ** 2
    return np.sqrt(hl2.rolling(window, min_periods=window).mean() / (4.0 * np.log(2.0)))


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing (alpha = 1/window). In [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    # avg_loss == 0 -> only gains -> RSI = 100
    out = out.where(avg_loss != 0, 100.0)
    return out.where(avg_gain.notna())


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line and histogram (Appel, 1979)."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def volume_zscore(volume: pd.Series, window: int = 30) -> pd.Series:
    """Standardised log-volume relative to its trailing window (detects panic/euphoria)."""
    lv = np.log(volume.replace(0, np.nan))
    mu = lv.rolling(window, min_periods=window).mean()
    sd = lv.rolling(window, min_periods=window).std()
    return (lv - mu) / sd


def build_features(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """Compute the full feature set from a clean OHLCV DataFrame.

    Returns a DataFrame aligned on df.index (leading rows contain NaN during
    the warm-up of rolling windows; drop them with `.dropna()`).
    """
    missing = set(OHLCV) - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    close = df["close"]
    feats = pd.DataFrame(index=df.index)
    feats["log_ret"] = log_returns(close)
    feats["log_ret_5"] = np.log(close).diff(5)  # weekly momentum
    feats[f"vol_{vol_window}"] = realized_volatility(feats["log_ret"], vol_window)
    feats[f"park_vol_{vol_window}"] = parkinson_volatility(df["high"], df["low"], vol_window)
    # Instantaneous (single-day) volatility proxy: log of the intraday range.
    # Unlike rolling windows it is not autocorrelated by construction, so it
    # respects the HMM assumption that x_t | s_t is independent of x_{t-1}.
    # Rolling features make BIC over-select K (see docs/notes_methodo.md).
    hl = np.log(df["high"] / df["low"])
    feats["log_range"] = np.log(hl.clip(lower=1e-6))
    feats["rsi_14"] = rsi(close, 14)
    m = macd(close)
    # Normalise the MACD histogram by price so it is comparable across assets/eras
    feats["macd_hist_norm"] = m["macd_hist"] / close
    feats["volume_z"] = volume_zscore(df["volume"])
    return feats
