import numpy as np
import pandas as pd
import pytest

from hmm_crypto.backtest import max_drawdown, performance_metrics, run_backtest


@pytest.fixture
def close():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 300))), index=idx)


def test_always_long_without_cost_equals_buy_and_hold(close):
    bt = run_backtest(close, pd.Series(1.0, index=close.index), cost_bps=0)
    np.testing.assert_allclose(bt["strategy_equity"], bt["buyhold_equity"])


def test_position_is_applied_next_day(close):
    """A signal built from today's return must NOT earn today's return."""
    r = close.pct_change()
    pos = (r > 0).astype(float)
    bt = run_backtest(close, pos, cost_bps=0)
    expected = pos.shift(1).fillna(0) * r.fillna(0)
    np.testing.assert_allclose(bt["strategy_ret"], expected)


def test_transaction_costs(close):
    pos = pd.Series(0.0, index=close.index)
    pos.iloc[10:20] = 1.0  # one round trip
    bt0 = run_backtest(close, pos, cost_bps=0)
    bt1 = run_backtest(close, pos, cost_bps=25)
    diff = (bt0["strategy_ret"] - bt1["strategy_ret"]).sum()
    assert diff == pytest.approx(2 * 25e-4)


def test_metrics():
    eq = pd.Series([1, 1.2, 0.9, 1.1, 0.6, 1.0])
    assert max_drawdown(eq) == pytest.approx(0.6 / 1.2 - 1)
    m = performance_metrics(pd.Series([0.01] * 365))
    assert m["cagr"] == pytest.approx(1.01**365 - 1)
    assert m["max_drawdown"] == 0
