import os, time, csv, yaml, math
# Standard libs
import argparse
import pandas as pd
from datetime import datetime

# Use package-qualified imports so the package is importable from outside the repo root
from roostoo_bot_template.utils.logger import get_logger
from roostoo_bot_template.roostoo_api import RoostooClient
from roostoo_bot_template import strategy as strat
from roostoo_bot_template import metrics as met

LOG = get_logger("bot")

def load_config():
    cfg_path = "config.local.yaml" if os.path.exists("config.local.yaml") else "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def ensure_csv(path: str, headers: list):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def to_df(klines):
    # Adapt this to actual Roostoo kline schema. Expecting list of candles.
    # Placeholder assumes dict with keys: open, high, low, close, volume, open_time
    df = pd.DataFrame(klines)
    # Rename if needed
    cols = {c:c for c in df.columns}
    for k in ["open","high","low","close","volume","open_time"]:
        if k not in df.columns:
            raise ValueError("Candle field missing: " + k)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", errors="ignore")
    df = df.sort_values("open_time").reset_index(drop=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def position_size(balance_usdt: float, price: float, frac: float) -> float:
    usd_alloc = balance_usdt * frac
    qty = usd_alloc / max(price, 1e-6)
    return max(0.0, round(qty, 6))

def main():
    cfg = load_config()
    client = RoostooClient(**cfg["roostoo"])
    symbols = cfg["bot"]["symbols"]
    interval = cfg["bot"]["interval"]
    lookback = int(cfg["bot"]["lookback"])
    loop_seconds = int(cfg["bot"]["loop_seconds"])

    trades_csv = os.path.join("data", "trades.csv")
    metrics_csv = os.path.join("data", "metrics.csv")
    ensure_csv(trades_csv, ["ts","symbol","side","qty","price","pnl"])
    ensure_csv(metrics_csv, ["ts","equity","ret","sharpe","sortino","calmar","mdd","score"])

    # Local state
    equity = 10000.0  # start equity (for metrics curve only; replace with account equity if provided)
    eq_curve = [equity]
    returns = []

    last_metrics_ts = 0.0

    LOG.info("Starting bot...")
    while True:
        loop_start = time.time()
        try:
            bal = client.get_balance()
            # Normalize balance: docs show response contains 'Wallet' with currency keys and 'Free'/'Lock'
            balance_usdt = 10000.0
            try:
                if isinstance(bal, dict) and 'Wallet' in bal:
                    w = bal['Wallet']
                    # prefer USD then USDT
                    for k in ('USD', 'USDT', 'USDT '):
                        if k in w and isinstance(w[k], dict):
                            balance_usdt = float(w[k].get('Free', w[k].get('free', balance_usdt)))
                            break
                elif isinstance(bal, dict) and 'Success' in bal and not bal['Success']:
                    # If error, fallback
                    balance_usdt = 10000.0
            except Exception:
                balance_usdt = 10000.0

            for sym in symbols:
                kl = client.get_candles(sym, interval, limit=lookback)
                df = to_df(kl)
                ind_cfg = cfg["strategy"]
                df = strat.add_indicators(df, ind_cfg)
                sig = strat.latest_signal(df, ind_cfg)

                price = sig["close"]
                atrv = sig["atr"]

                # Risk
                pos_frac = float(cfg["risk"]["position_fraction"])
                qty = position_size(balance_usdt, price, pos_frac)

                side = None
                if sig["signal"] == "BUY":
                    side = "BUY"
                elif sig["signal"] == "SELL":
                    side = "SELL"
                else:
                    LOG.info(f"{sym} HOLD | reg={sig['regime']} rsi={sig['rsi']:.1f} adx={sig['adx']:.1f} close={price:.2f}")
                
                if side and qty > 0:
                    try:
                        # Place an order on Roostoo: pair naming in our system should match API, e.g. 'BTC/USD'
                        resp = client.place_order(pair=sym, side=side, quantity=qty)
                        LOG.info(f"Order {side} {sym} qty={qty} price~{price} resp={resp}")
                        # Simplified PnL impact (placeholder). For real PnL, fetch fills/executions.
                        pnl = 0.0
                        with open(trades_csv, "a", newline="") as f:
                            csv.writer(f).writerow([datetime.utcnow().isoformat(), sym, side, qty, price, pnl])
                        # Fee (0.1%) impact on equity approximation
                        equity *= (1.0 - 0.001)
                    except Exception as e:
                        LOG.error(f"Order failed for {sym}: {e}")
                        continue

                # Update naive equity curve with price move proxy (very rough; replace with true positions)
                if len(df) >= 2:
                    ret = (df['close'].iloc[-1] / df['close'].iloc[-2]) - 1.0
                    returns.append(ret)
                    equity *= (1.0 + ret)
                    eq_curve.append(equity)

            # Metrics hourly
            if time.time() - last_metrics_ts > 3600 and len(returns) > 30:
                ser_ret = pd.Series(returns[-1000:])
                ser_eq = pd.Series(eq_curve[-1000:])
                m = met.composite_score(ser_ret, ser_eq)
                with open(metrics_csv, "a", newline="") as f:
                    csv.writer(f).writerow([datetime.utcnow().isoformat(), equity, ser_ret.iloc[-1], m["sharpe"], m["sortino"], m["calmar"], m["max_drawdown"], m["score"]])
                LOG.info(f"Metrics | Eq={equity:.2f} Sharpe={m['sharpe']:.2f} Sortino={m['sortino']:.2f} Calmar={m['calmar']:.2f} MDD={m['max_drawdown']:.2%} Score={m['score']:.3f}")
                last_metrics_ts = time.time()

        except Exception as e:
            LOG.exception(f"Loop error: {e}")

        # Sleep remaining time
        dt = time.time() - loop_start
        sleep_s = max(0, loop_seconds - dt)
        time.sleep(sleep_s)


def run_one_iteration(cfg, client):
    """Run one iteration of the bot loop (safe for dry-run). Returns a dict of signals per symbol."""
    results = {}
    symbols = cfg["bot"]["symbols"]
    lookback = int(cfg["bot"]["lookback"])
    equity = 10000.0
    for sym in symbols:
        kl = client.get_candles(sym, cfg["bot"]["interval"], limit=lookback)
        df = to_df(kl)
        ind_cfg = cfg["strategy"]
        df = strat.add_indicators(df, ind_cfg)
        sig = strat.latest_signal(df, ind_cfg)
        results[sym] = sig
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Run a single dry iteration with a mock client (no network/no orders)')
    args = parser.parse_args()

    if args.dry_run:
        # lightweight mock client similar to tests
        class MockRoostooClient:
            def __init__(self):
                self._now = int(time.time() * 1000)

            def get_balance(self):
                return {"USDT": {"free": 10000.0}}

            def get_candles(self, symbol: str, interval: str, limit: int = 500):
                out = []
                base = 20000.0 if symbol.startswith('BTC') else 1000.0
                for i in range(limit):
                    t = self._now - (limit - i) * 60 * 1000
                    close = base + i * 0.5
                    open_p = close - 0.2
                    high = max(open_p, close) + 0.5
                    low = min(open_p, close) - 0.5
                    vol = 1.0
                    out.append({
                        'open_time': int(t),
                        'open': open_p,
                        'high': high,
                        'low': low,
                        'close': close,
                        'volume': vol
                    })
                return out

            def place_order(self, *args, **kwargs):
                return {"orderId": "MOCK", "status": "FILLED"}

        cfg = load_config()
        client = MockRoostooClient()
        res = run_one_iteration(cfg, client)
        for s, v in res.items():
            print(f"{s}: {v}")
    else:
        main()
