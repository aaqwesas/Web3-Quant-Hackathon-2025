import pandas as pd
import numpy as np


def ema(series: pd.Series, span: int):
    if series is None or len(series) == 0:
        raise ValueError("EMA requires a non-empty series")
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14):
    close = close.dropna()
    if len(close) < length + 1:
        return pd.Series([np.nan] * len(close), index=close.index)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def atr(df: pd.DataFrame, length: int = 14):
    if not {'high','low','close'}.issubset(set(df.columns)):
        raise ValueError('ATR requires high, low, close columns')
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr


def adx(df: pd.DataFrame, length: int = 14):
    if not {'high','low','close'}.issubset(set(df.columns)):
        raise ValueError('ADX requires high, low, close columns')
    up_move = df['high'] - df['high'].shift()
    down_move = df['low'].shift() - df['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([(df['high'] - df['low']), (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1/length, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/length, adjust=False).mean() / atr_val)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/length, adjust=False).mean() / atr_val)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/length, adjust=False).mean()
    return adx.fillna(0)


def add_indicators(df: pd.DataFrame, cfg: dict):
    df = df.copy()
    df['ema_short'] = ema(df['close'], cfg.get('ema_short', 8))
    df['ema_long'] = ema(df['close'], cfg.get('ema_long', 21))
    df['rsi'] = rsi(df['close'], cfg.get('rsi_length', 14))
    df['atr'] = atr(df, cfg.get('atr_length', 14))
    df['adx'] = adx(df, cfg.get('adx_length', 14))
    # regime: 1 if ema_short > ema_long else -1
    df['regime'] = np.where(df['ema_short'] > df['ema_long'], 1, -1)
    return df


def signal_row(df: pd.DataFrame, row_idx: int, cfg: dict):
    row = df.iloc[row_idx]
    if np.isnan(row['close']) or np.isnan(row['rsi']) or np.isnan(row['atr']):
        return {'signal': 'HOLD', 'reason': 'insufficient_data'}
    if row['regime'] == 1 and row['rsi'] < cfg.get('rsi_buy', 40) and row['adx'] > cfg.get('adx_threshold', 20):
        return {'signal': 'BUY', 'close': row['close'], 'rsi': row['rsi'], 'atr': row['atr'], 'adx': row['adx'], 'regime': row['regime']}
    if row['regime'] == -1 and row['rsi'] > cfg.get('rsi_sell', 60) and row['adx'] > cfg.get('adx_threshold', 20):
        return {'signal': 'SELL', 'close': row['close'], 'rsi': row['rsi'], 'atr': row['atr'], 'adx': row['adx'], 'regime': row['regime']}
    return {'signal': 'HOLD', 'close': row['close'], 'rsi': row['rsi'], 'atr': row['atr'], 'adx': row['adx'], 'regime': row['regime']}


def latest_signal(df: pd.DataFrame, cfg: dict):
    if len(df) == 0:
        return {'signal': 'HOLD', 'reason': 'empty_df'}
    # prefer last valid row
    for i in range(len(df)-1, -1, -1):
        if not np.isnan(df['close'].iloc[i]):
            sig = signal_row(df, i, cfg)
            # enrich with timestamp
            sig['ts'] = int(df['open_time'].iloc[i]) if 'open_time' in df.columns else None
            return sig
    return {'signal': 'HOLD', 'reason': 'no_valid_price'}
