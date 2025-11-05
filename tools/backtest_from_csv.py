import os
import sys
import csv
import pandas as pd
from datetime import datetime

# Ensure package importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PKG_DIR = os.path.join(ROOT, 'roostoo_bot_template')
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from roostoo_bot_template import bot
from roostoo_bot_template import metrics as met
from roostoo_bot_template.strategy import add_indicators, latest_signal


def load_price_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'timestamp' in df.columns and 'price' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df[['ts', 'price']].copy()
        df = df.sort_values('ts').reset_index(drop=True)
        return df
    else:
        raise ValueError('CSV must contain timestamp and price columns')


def build_candles_from_close(df_price: pd.DataFrame) -> pd.DataFrame:
    # Create OHLC where close is price, open is previous close, high/low are ±0.5%
    df = df_price.copy()
    df['open'] = df['price'].shift(1)
    df['open'].iloc[0] = df['price'].iloc[0]
    df['close'] = df['price']
    df['high'] = df['price'] * 1.005
    df['low'] = df['price'] * 0.995
    df['volume'] = 1.0
    df = df.rename(columns={'ts': 'open_time'})
    return df[['open_time', 'open', 'high', 'low', 'close', 'volume']]


def backtest(df_candles: pd.DataFrame, cfg: dict, initial_equity: float = 10000.0, fee_pct: float = 0.001, slippage_pct: float = 0.0005):
    df = df_candles.copy().reset_index(drop=True)
    # compute indicators
    ind_cfg = cfg.get('strategy', cfg)
    df = add_indicators(df, ind_cfg)

    cash = initial_equity
    position = 0.0
    entry_price = None
    trades = []
    equity_curve = []

    lookback = 0
    for i in range(len(df)-1):
        row = df.iloc[i]
        next_row = df.iloc[i+1]
        sig = None
        try:
            sig = latest_signal(df.iloc[:i+1], ind_cfg)
        except Exception:
            # skip until indicators available
            equity = cash + position * row['close']
            equity_curve.append(equity)
            continue

        # execute at next close price
        exec_price = float(next_row['close'])

        if sig['signal'] == 'BUY' and position == 0:
            # simulate slippage: buyer pays a slightly worse price
            buy_price = exec_price * (1.0 + slippage_pct)
            qty = bot.position_size(cash, buy_price, float(cfg['risk']['position_fraction']))
            cost = qty * buy_price * (1.0 + fee_pct)
            if qty > 0 and cost <= cash:
                cash -= cost
                position += qty
                entry_price = buy_price
                trades.append({'ts': next_row['open_time'].isoformat(), 'side': 'BUY', 'qty': qty, 'price': buy_price, 'pnl': 0.0, 'fee': fee_pct * qty * buy_price, 'slippage': slippage_pct})

        elif sig['signal'] == 'SELL' and position > 0:
            # simulate slippage: seller receives slightly worse price
            sell_price = exec_price * (1.0 - slippage_pct)
            qty = position
            proceeds = qty * sell_price * (1.0 - fee_pct)
            pnl = proceeds - (qty * entry_price)
            cash += proceeds
            trades.append({'ts': next_row['open_time'].isoformat(), 'side': 'SELL', 'qty': qty, 'price': sell_price, 'pnl': pnl, 'fee': fee_pct * qty * sell_price, 'slippage': slippage_pct})
            position = 0.0
            entry_price = None

        # mark-to-market equity
        equity = cash + position * exec_price
        equity_curve.append(equity)

    # final equity at last close
    last_price = float(df['close'].iloc[-1])
    equity = cash + position * last_price
    equity_curve.append(equity)

    eq_ser = pd.Series(equity_curve)
    ret = eq_ser.pct_change().fillna(0)
    m = met.composite_score(ret, eq_ser)

    return trades, eq_ser, ret, m


def save_trades_and_metrics(trades, eq_ser, ret, metrics_out, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    trades_csv = os.path.join(out_dir, 'backtest_trades.csv')
    metrics_csv = os.path.join(out_dir, 'backtest_metrics.csv')
    with open(trades_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ts','side','qty','price','pnl'])
        for t in trades:
            w.writerow([t['ts'], t['side'], t['qty'], t['price'], t['pnl']])

    dfm = pd.DataFrame({'equity': eq_ser.values})
    dfm['ret'] = ret.values
    dfm.to_csv(metrics_csv, index=False)
    print('Saved trades:', trades_csv)
    print('Saved metrics:', metrics_csv)


def main(csv_path: str):
    # ensure current working directory is package dir so config.yaml is found
    os.chdir(PKG_DIR)
    cfg = bot.load_config()
    df_price = load_price_csv(csv_path)
    df_candles = build_candles_from_close(df_price)

    # optional CLI args: fee_pct, slippage_pct
    fee_pct = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
    slippage_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0005
    trades, eq_ser, ret, m = backtest(df_candles, cfg, fee_pct=fee_pct, slippage_pct=slippage_pct)
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'roostoo_bot_template', 'data')
    save_trades_and_metrics(trades, eq_ser, ret, m, out_dir)
    print('Metrics summary:', m)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python backtest_from_csv.py <path_to_price_csv>')
        sys.exit(1)
    main(sys.argv[1])
