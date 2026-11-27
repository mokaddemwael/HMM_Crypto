"""Data acquisition and cleaning.

Three sources, one output format
--------------------------------
* `load_csv`      : any OHLCV file (the dataset provided by the supervisors,
                    Binance exports, Kaggle dumps...). Column names are
                    normalised automatically.
* `fetch_yahoo`   : downloads from Yahoo Finance (cached on disk).
* `simulate.py`   : synthetic regime-switching market (for validation).

All return a DataFrame indexed by a tz-naive DatetimeIndex with lowercase
columns ["open", "high", "low", "close", "volume"].
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .features import OHLCV

log = logging.getLogger(__name__)

# Aliases met in common exports (Yahoo, Binance, Kaggle, CoinMarketCap, French files)
_ALIASES = {
    "date": "date", "datetime": "date", "timestamp": "date", "time": "date",
    "open_time": "date", "opentime": "date", "open time": "date",
    "open": "open", "ouverture": "open",
    "high": "high", "haut": "high", "plus haut": "high",
    "low": "low", "bas": "low", "plus bas": "low",
    "close": "close", "cloture": "close", "clôture": "close", "price": "close",
    "adj close": "adj_close", "adj_close": "adj_close",
    "volume": "volume", "vol": "volume", "volume usdt": "volume", "volume_usd": "volume",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in _ALIASES and _ALIASES[key] not in mapping.values():
            mapping[c] = _ALIASES[key]
    return df.rename(columns=mapping)


def _parse_dates(s: pd.Series) -> pd.DatetimeIndex:
    if np.issubdtype(s.dtype, np.number):
        # Unix epoch: Binance uses milliseconds, others seconds
        unit = "ms" if s.median() > 1e11 else "s"
        idx = pd.to_datetime(s, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(s, utc=True, errors="coerce")
    return pd.DatetimeIndex(idx).tz_convert(None)


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, deduplicate and sanity-check an OHLCV frame. Logs what was removed."""
    out = df[OHLCV].apply(pd.to_numeric, errors="coerce")
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()

    n0 = len(out)
    out = out.dropna(subset=["close"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    # Repair inconsistent candles instead of dropping them
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)
    if len(out) < n0:
        log.warning("clean_ohlcv: removed %d invalid rows out of %d", n0 - len(out), n0)
    return out


def load_csv(path: str | Path, date_col: str | None = None, **read_kw) -> pd.DataFrame:
    """Load an OHLCV CSV/Excel file with automatic column detection."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, **read_kw)
    else:
        raw = pd.read_csv(path, sep=None, engine="python", **read_kw)
    raw = _normalise_columns(raw)
    dcol = date_col or "date"
    if dcol not in raw.columns:
        raise ValueError(f"No date column found in {path.name}; columns = {list(raw.columns)}")
    raw.index = _parse_dates(raw[dcol])
    raw.index.name = "date"
    missing = set(OHLCV) - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    return clean_ohlcv(raw)


def fetch_yahoo(
    ticker: str = "BTC-USD",
    start: str = "2017-01-01",
    end: str | None = None,
    interval: str = "1d",
    cache_dir: str | Path = "data/raw",
    refresh: bool = False,
) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance, with a CSV cache for reproducibility."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{ticker}_{interval}_{start}_{end or 'latest'}.csv"
    if cache.exists() and not refresh:
        log.info("Loading cached %s", cache)
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return clean_ohlcv(df)

    import yfinance as yf  # optional dependency, imported lazily

    raw = yf.download(ticker, start=start, end=end, interval=interval,
                      auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):  # yfinance >= 0.2 returns (field, ticker)
        raw.columns = raw.columns.get_level_values(0)
    raw = _normalise_columns(raw)
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw.index.name = "date"
    df = clean_ohlcv(raw)
    if interval == "1d":
        # Crypto trades 24/7: today's daily candle is still forming -> drop it.
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        df = df[df.index < today]
    df.to_csv(cache)
    return df
