# Roostoo Web3 Quant Bot (Hybrid MR + Momentum)

A production-ready **starter template** for the HKU x Roostoo Web3 Quant Hackathon.
- Strategy: **Hybrid Mean Reversion + Momentum** with regime detection.
- Risk-first design to maximize **Sharpe, Sortino, Calmar**.
- Clean logs, metrics, and trade records.
- Ready for **AWS EC2** deployment.

## 📦 Structure
```
roostoo_bot_template/
├── bot.py                # Main loop (scheduler + execution)
├── strategy.py           # Signals (RSI + EMA crossover + ATR/ADX regime)
├── roostoo_api.py        # Thin Roostoo REST client
├── metrics.py            # Rolling metrics (Sharpe/Sortino/Calmar/MDD)
├── utils/
│   └── logger.py         # Logging setup
├── config.yaml           # API keys, symbols, params
├── requirements.txt
├── run.sh
├── data/
│   ├── trades.csv
│   └── metrics.csv
└── logs/
    └── bot.log
```

## 🚀 Quickstart
```bash
# 1) Python 3.10+ recommended
python -V

# 2) Create venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install deps
pip install -r requirements.txt

# 4) Configure
cp config.yaml config.local.yaml
# Edit config.local.yaml: add API keys/URLs/symbol list and risk params

# 5) Run locally
python bot.py
# or as a background process
nohup python bot.py > logs/bot.log 2>&1 &
tail -f logs/bot.log
```

## ☁️ AWS (EC2) Notes
- Instance: `t3.medium` (2 vCPU, 4GB RAM) is fine.
- Keep bot always-on using `nohup` (above) or `systemd` service.
- Use `tmux`/`screen` for interactive sessions.
- Ensure you **rotate logs** and **persist data/** (CSV) for the final report.

## ⚙️ Configuration
Edit `config.local.yaml` (preferred). If absent, `config.yaml` is used.

```yaml
roostoo:
  base_url: "https://api.roostoo.com"   # replace per docs
  api_key: "YOUR_API_KEY"
  api_secret: "YOUR_SECRET"             # if required
  timeout_sec: 10
  min_request_interval_sec: 0.8         # avoid rate limits and HFT

bot:
  symbols: ["BTCUSDT", "ETHUSDT"]
  interval: "1m"                         # 1m or 5m
  lookback: 500                          # candles to pull each loop
  loop_seconds: 60                       # schedule frequency

risk:
  position_fraction: 0.05                # 5% per trade
  max_concurrent_positions: 2
  stop_atr_mult: 2.5
  take_atr_mult: 4.0
  daily_loss_stop_pct: 0.05              # stop trading for the day at -5%

strategy:
  rsi_period: 14
  rsi_buy: 30
  rsi_sell: 70
  ema_fast: 21
  ema_slow: 55
  atr_period: 14
  adx_period: 14
  adx_trend_threshold: 22                # regime switch
```

## 📊 Data & Metrics
- `data/trades.csv` — every fill (time, symbol, side, qty, price, PnL).
- `data/metrics.csv` — rolling return, Sharpe, Sortino, Calmar, MDD (per hour).

## 🧪 Strategy (High-level)
- **Regime**: ADX/ATR detects `trending` vs `ranging`.
- **Ranging**: RSI(14) mean-reversion: buy < 30; sell > 70.
- **Trending**: EMA(21) > EMA(55) ⇒ long bias; else flat.
- **Risk**: ATR-based stop/take; cap positions; daily loss stop.

## 🧰 Notes
- This client is a minimal wrapper. Verify endpoints/params against **Roostoo API docs**.
- Placeholders for auth headers/signatures are included—adapt to your keys.
- Keep the **request interval** conservative. No HFT.

Good luck and ship it.
