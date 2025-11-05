import pandas as pd
import numpy as np

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']; low = df['low']; close = df['close']
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']; low = df['low']; close = df['close']
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr_ + 1e-12))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr_ + 1e-12))
    dx = ( (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12) ) * 100
    return dx.ewm(alpha=1/period, adjust=False).mean()

def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # Validate inputs
    required_cols = {'high', 'low', 'close'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    df = df.copy()
    # minimal length check: need at least 2 rows and prefer more than the largest period
    max_period = max(int(cfg.get('ema_slow', 0)), int(cfg.get('rsi_period', 0)), int(cfg.get('atr_period', 0)), int(cfg.get('adx_period', 0)))
    if len(df) < 2:
        raise ValueError(f"DataFrame too short: need at least 2 rows, got {len(df)}")
    if max_period and len(df) < max(2, max_period // 2):
        # allow proceeding for small inputs but warn via NaNs and later checks
        # we don't raise here to remain flexible, but calculations may have NaNs
        pass

    df['ema_fast'] = ema(df['close'], int(cfg['ema_fast']))
    df['ema_slow'] = ema(df['close'], int(cfg['ema_slow']))
    df['rsi'] = rsi(df['close'], int(cfg['rsi_period']))
    df['atr'] = atr(df, int(cfg['atr_period']))
    df['adx'] = adx(df, int(cfg['adx_period']))
    df['regime'] = np.where(df['adx'] >= cfg['adx_trend_threshold'], 'trending', 'ranging')
    return df

def signal_row(row, rsi_buy: float, rsi_sell: float):
    if row['regime'] == 'ranging':
        if row['rsi'] < rsi_buy: return 'BUY'
        if row['rsi'] > rsi_sell: return 'SELL'
        return 'HOLD'
    else:
        if row['ema_fast'] > row['ema_slow']: return 'BUY'
        else: return 'SELL'

def latest_signal(df: pd.DataFrame, cfg: dict) -> dict:
    if df is None or len(df) == 0:
        raise ValueError("Empty DataFrame passed to latest_signal")

    row = df.iloc[-1]

    # Ensure required cols present in row
    for c in ['regime', 'rsi', 'ema_fast', 'ema_slow', 'atr', 'adx', 'close']:
        if c not in df.columns:
            raise ValueError(f"Missing indicator column: {c}")

    sig = signal_row(row, float(cfg['rsi_buy']), float(cfg['rsi_sell']))

    # convert NaNs to sensible defaults or raise
    def safe_float(x, name):
        try:
            v = float(x)
            if np.isnan(v):
                raise ValueError
            return v
        except Exception:
            raise ValueError(f"Invalid numeric value for {name}: {x}")

    return {
        "signal": sig,
        "regime": str(row['regime']),
        "rsi": safe_float(row['rsi'], 'rsi'),
        "ema_fast": safe_float(row['ema_fast'], 'ema_fast'),
        "ema_slow": safe_float(row['ema_slow'], 'ema_slow'),
        "atr": safe_float(row['atr'], 'atr'),
        "adx": safe_float(row['adx'], 'adx'),
        "close": safe_float(row['close'], 'close')
    }
