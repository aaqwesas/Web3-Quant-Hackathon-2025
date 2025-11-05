import os
import sys
import time
import random
from datetime import datetime

# ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Also add package dir so imports like `from utils.logger import ...` work
PKG_DIR = os.path.join(ROOT, 'roostoo_bot_template')
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from roostoo_bot_template import bot
from roostoo_bot_template.strategy import add_indicators, latest_signal
import csv

class MockRoostooClient:
    def __init__(self, **kwargs):
        self._now = int(time.time() * 1000)

    def get_balance(self):
        # Return a fake USDT balance
        return {"USDT": {"free": 10000.0}}

    def get_candles(self, symbol: str, interval: str, limit: int = 500):
        # Produce `limit` synthetic candles with increasing price and some noise
        out = []
        base = 20000.0 if symbol.startswith('BTC') else 1000.0
        for i in range(limit):
            t = self._now - (limit - i) * 60 * 1000
            close = base + i * 0.5 + random.uniform(-1, 1)
            open_p = close - random.uniform(-0.5, 0.5)
            high = max(open_p, close) + random.uniform(0, 1.0)
            low = min(open_p, close) - random.uniform(0, 1.0)
            vol = random.uniform(0.1, 5.0)
            out.append({
                'open_time': int(t),
                'open': open_p,
                'high': high,
                'low': low,
                'close': close,
                'volume': vol
            })
        return out

    def place_order(self, symbol: str, side: str, qty: float, order_type: str = "MARKET"):
        # Fake order response
        return {"orderId": f"MOCK-{int(time.time())}", "status": "FILLED", "avgPrice": None}

    def get_open_orders(self, symbol: str = None):
        return []

    def cancel_order(self, symbol: str, order_id: str):
        return {"canceled": True}


def dry_run_one_loop():
    # Ensure current working directory is the package dir so config.yaml is found
    os.chdir(PKG_DIR)
    cfg = bot.load_config()
    client = MockRoostooClient()
    symbols = cfg["bot"]["symbols"]
    interval = cfg["bot"]["interval"]
    lookback = int(cfg["bot"]["lookback"])

    # prepare data paths
    data_dir = os.path.join(PKG_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)
    trades_csv = os.path.join(data_dir, 'trades.csv')
    metrics_csv = os.path.join(data_dir, 'metrics.csv')
    bot.ensure_csv(trades_csv, ["ts","symbol","side","qty","price","pnl"])
    bot.ensure_csv(metrics_csv, ["ts","equity","ret","sharpe","sortino","calmar","mdd","score"])

    for sym in symbols:
        kl = client.get_candles(sym, interval, limit=lookback)
        df = bot.to_df(kl)
        ind_cfg = cfg["strategy"]
        df = add_indicators(df, ind_cfg)
        sig = latest_signal(df, ind_cfg)

        print(f"Symbol: {sym}")
        print("Latest signal:", sig)

        # Simulate order decision without sending anything and record a fake trade
        price = sig["close"]
        pos_frac = float(cfg["risk"]["position_fraction"])
        qty = bot.position_size(10000.0, price, pos_frac)
        print(f"Calculated position qty for {sym}: {qty}\n")

        # write trade if BUY/SELL
        if sig["signal"] in ("BUY", "SELL") and qty > 0:
            ts = datetime.utcnow().isoformat()
            pnl = 0.0
            try:
                with open(trades_csv, 'a', newline='') as f:
                    csv.writer(f).writerow([ts, sym, sig["signal"], qty, price, pnl])
            except PermissionError:
                alt = trades_csv.replace('.csv', '_dryrun.csv')
                print(f"Permission denied writing {trades_csv}, falling back to {alt}")
                with open(alt, 'a', newline='') as f:
                    csv.writer(f).writerow([ts, sym, sig["signal"], qty, price, pnl])

    # write a simple metrics row summarizing the loop
    # compute naive equity using last-close returns
    equity = 10000.0
    returns = []
    for sym in symbols:
        kl = client.get_candles(sym, interval, limit=lookback)
        df = bot.to_df(kl)
        if len(df) >= 2:
            ret = (df['close'].iloc[-1] / df['close'].iloc[-2]) - 1.0
            returns.append(ret)
            equity *= (1.0 + ret)

    if returns:
        last_ret = returns[-1]
    else:
        last_ret = 0.0

    try:
        with open(metrics_csv, 'a', newline='') as f:
            csv.writer(f).writerow([datetime.utcnow().isoformat(), equity, last_ret, 0.0, 0.0, 0.0, 0.0, 0.0])
    except PermissionError:
        altm = metrics_csv.replace('.csv', '_dryrun.csv')
        print(f"Permission denied writing {metrics_csv}, falling back to {altm}")
        with open(altm, 'a', newline='') as f:
            csv.writer(f).writerow([datetime.utcnow().isoformat(), equity, last_ret, 0.0, 0.0, 0.0, 0.0, 0.0])

if __name__ == '__main__':
    print('Starting dry run (single loop) at', datetime.utcnow().isoformat())
    dry_run_one_loop()
    print('Dry run complete.')
