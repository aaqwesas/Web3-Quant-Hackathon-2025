import os
import sys
import pandas as pd

# Ensure project root is on sys.path so package imports work when running the test directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from roostoo_bot_template.strategy import add_indicators, latest_signal

# Small sample dataset
df = pd.DataFrame({
    'high': [10 + i for i in range(20)],
    'low': [9 + 0.5*i for i in range(20)],
    'close': [9.5 + i for i in range(20)],
})

cfg = {
    'ema_fast': 3,
    'ema_slow': 6,
    'rsi_period': 14,
    'atr_period': 14,
    'adx_period': 14,
    'adx_trend_threshold': 25,
    'rsi_buy': 30,
    'rsi_sell': 70
}

print('Running add_indicators...')
try:
    df2 = add_indicators(df, cfg)
    print('Computed indicators (tail):')
    print(df2.tail()[['ema_fast','ema_slow','rsi','atr','adx','regime']])
    print('\nLatest signal:')
    print(latest_signal(df2, cfg))
except Exception as e:
    print('Error while running indicators:')
    raise
