import pytest
import pandas as pd
import numpy as np
from roostoo_bot_template.strategy import add_indicators, latest_signal

BASE_CFG = {
    'ema_fast': 3,
    'ema_slow': 6,
    'rsi_period': 14,
    'atr_period': 14,
    'adx_period': 14,
    'adx_trend_threshold': 25,
    'rsi_buy': 30,
    'rsi_sell': 70
}

def make_df(n=30):
    return pd.DataFrame({
        'high': np.linspace(10, 20, n) + np.random.random(n),
        'low': np.linspace(9, 19, n) - np.random.random(n),
        'close': np.linspace(9.5, 19.5, n) + np.random.random(n)
    })

def test_add_indicators_happy_path():
    df = make_df(50)
    out = add_indicators(df, BASE_CFG)
    assert 'ema_fast' in out.columns
    assert 'rsi' in out.columns
    assert 'adx' in out.columns

def test_add_indicators_missing_cols():
    df = pd.DataFrame({'open': [1,2,3]})
    with pytest.raises(ValueError):
        add_indicators(df, BASE_CFG)

def test_latest_signal_empty():
    df = pd.DataFrame(columns=['high','low','close'])
    with pytest.raises(ValueError):
        latest_signal(df, BASE_CFG)

def test_latest_signal_values():
    df = make_df(60)
    out = add_indicators(df, BASE_CFG)
    sig = latest_signal(out, BASE_CFG)
    assert sig['signal'] in ('BUY','SELL','HOLD')
    for k in ('rsi','ema_fast','ema_slow','atr','adx','close'):
        assert isinstance(sig[k], float)
