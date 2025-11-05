import traceback
import sys
import os
import numpy as np
import pandas as pd

# Ensure package importable when running test directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PKG_DIR = os.path.join(ROOT, 'roostoo_bot_template')
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

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
    try:
        add_indicators(df, BASE_CFG)
        print('ERROR: expected ValueError for missing cols')
        return 1
    except ValueError:
        pass


def test_latest_signal_empty():
    df = pd.DataFrame(columns=['high','low','close'])
    try:
        latest_signal(df, BASE_CFG)
        print('ERROR: expected ValueError for empty df')
        return 1
    except ValueError:
        pass


def test_latest_signal_values():
    df = make_df(60)
    out = add_indicators(df, BASE_CFG)
    sig = latest_signal(out, BASE_CFG)
    assert sig['signal'] in ('BUY','SELL','HOLD')
    for k in ('rsi','ema_fast','ema_slow','atr','adx','close'):
        assert isinstance(sig[k], float)


if __name__ == '__main__':
    tests = [
        test_add_indicators_happy_path,
        test_add_indicators_missing_cols,
        test_latest_signal_empty,
        test_latest_signal_values
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f'{t.__name__}: PASS')
        except Exception:
            failed += 1
            print(f'{t.__name__}: FAIL')
            traceback.print_exc()
    if failed:
        print(f'{failed} tests failed')
        sys.exit(2)
    else:
        print('All smoke tests passed')
        sys.exit(0)
