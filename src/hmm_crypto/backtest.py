"""Vectorised backtest engine and performance metrics.

Timing convention
-----------------
position_t is decided with information up to the close of day t and earns the
simple return of day t+1:   R^strat_{t+1} = position_t * R_{t+1} - c * |position_t - position_{t-1}|
where c is the proportional transaction cost (fees + slippage) paid on turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    close: pd.Series,
    position: pd.Series,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Simulate the strategy and a Buy & Hold benchmark on the same dates."""
    asset_ret = close.pct_change()
    pos = position.reindex(close.index).astype(float)
    first = pos.first_valid_index()
    if first is None:
        raise ValueError("position is empty")
    pos = pos.loc[first:].fillna(0.0)
    asset_ret = asset_ret.loc[first:]

    held = pos.shift(1).fillna(0.0)             # position held during day t
    turnover = pos.diff().abs().fillna(pos.abs())  # trade executed at close of t
    cost = (cost_bps / 1e4) * turnover.shift(1).fillna(0.0)

    strat = held * asset_ret.fillna(0.0) - cost
    bh = asset_ret.fillna(0.0).copy()
    bh.iloc[0] = 0.0  # both start invested at the first close

    return pd.DataFrame({
        "position": held,
        "asset_ret": asset_ret.fillna(0.0),
        "strategy_ret": strat,
        "buyhold_ret": bh,
        "strategy_equity": (1 + strat).cumprod(),
        "buyhold_equity": (1 + bh).cumprod(),
        "turnover": turnover,
    })


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def performance_metrics(ret: pd.Series, periods_per_year: int = 365) -> dict[str, float]:
    """Standard risk/return metrics of a daily simple-return series (rf = 0)."""
    ret = ret.dropna()
    n = len(ret)
    equity = (1 + ret).cumprod()
    total = float(equity.iloc[-1] - 1)
    years = n / periods_per_year
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else np.nan
    vol = float(ret.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(periods_per_year)) if ret.std() > 0 else np.nan
    downside = ret[ret < 0].std(ddof=1)
    sortino = float(ret.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else np.nan
    mdd = max_drawdown(equity)
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "hit_rate": float((ret[ret != 0] > 0).mean()) if (ret != 0).any() else np.nan,
    }


def summary_table(backtests: dict[str, pd.DataFrame], periods_per_year: int = 365) -> pd.DataFrame:
    """Metrics for each strategy backtest + the Buy & Hold benchmark (same dates)."""
    cols = {}
    for name, bt in backtests.items():
        s = performance_metrics(bt["strategy_ret"], periods_per_year)
        s["avg_exposure"] = float(bt["position"].abs().mean())
        s["ann_turnover"] = float(bt["turnover"].sum() / len(bt) * periods_per_year)
        cols[name] = s
    bt = next(iter(backtests.values()))
    b = performance_metrics(bt["buyhold_ret"], periods_per_year)
    b["avg_exposure"], b["ann_turnover"] = 1.0, 0.0
    cols["Buy & Hold"] = b
    return pd.DataFrame(cols)
