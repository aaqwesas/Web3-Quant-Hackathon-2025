#!/usr/bin/env bash
set -euo pipefail

# If CONFIG_FROM_ENV is set, create config.local.yaml from environment variables
if [ "${CONFIG_FROM_ENV:-0}" != "0" ]; then
  cat > /app/config.local.yaml <<EOF
roostoo:
  base_url: "${ROOSTOO_BASE_URL:-https://mock-api.roostoo.com}"
  api_key: "${ROOSTOO_API_KEY:-}"
  api_secret: "${ROOSTOO_API_SECRET:-}"
  timeout_sec: ${ROOSTOO_TIMEOUT_SEC:-10}
  min_request_interval_sec: ${ROOSTOO_MIN_INTERVAL:-0.2}
bot:
  symbols: ${BOT_SYMBOLS:-'["BTC/USD"]'}
  interval: ${BOT_INTERVAL:-"1m"}
  lookback: ${BOT_LOOKBACK:-200}
  loop_seconds: ${BOT_LOOP_SECONDS:-60}
strategy:
  # example defaults - override via file mount or environment
  ema_fast: 12
  ema_slow: 26
  rsi_period: 14
risk:
  position_fraction: ${RISK_POS_FRAC:-0.01}
EOF
  echo "Wrote /app/config.local.yaml from environment"
fi

exec "$@"
