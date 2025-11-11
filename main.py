import os
import time
import hmac
import hashlib
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict, Optional, Dict, List

import aiohttp
import pandas as pd
import requests
from datetime import datetime
from functools import wraps
from dotenv import load_dotenv

# ------------------------------
# Setup
# ------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ------------------------------
# Types
# ------------------------------
class TradeSignal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class MarketData(TypedDict):
    price: float
    volume: float
    high: float
    low: float
    open: float
    close: float
    volume_30d_avg: float


class Performance(TypedDict):
    total_iterations: int
    current_positions: int
    trades_today: int
    available_cash: float
    portfolio_value: float
    total_equity: float


@dataclass
class Position:
    symbol: str
    entry_price: float
    size: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    half_sold: bool = False


def apply_delay(func):
    """Delay wrapper for SYNC functions only (e.g., HTTP POST to Roostoo)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        time.sleep(1)  # small cooldown between API calls
        return result
    return wrapper


# ------------------------------
# Strategy
# ------------------------------
class Web3MeanReversionStrategy:
    def __init__(self, config: dict):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.trade_count_today = 0
        self._current_day = datetime.utcnow().date()

        # API configuration
        self.base_url = "https://mock-api.roostoo.com"
        self.api_key = os.environ.get("ROOSTOO_API")
        self.secret_key = os.environ.get("SECRET_KEY")

        # Strategy parameters
        self.z_score_threshold = config.get('z_score_threshold', -1.75)
        self.z_score_reversal = config.get('z_score_reversal', -1.5)
        self.volume_multiplier = config.get('volume_multiplier', 1.5)
        self.rsi_threshold = config.get('rsi_threshold', 30)
        self.position_size = config.get('position_size', 0.015)
        self.stop_loss_pct = config.get('stop_loss_pct', 0.04)
        self.take_profit_1 = config.get('take_profit_1', 0.06)
        self.take_profit_2 = config.get('take_profit_2', 0.12)
        self.max_positions = config.get('max_positions', 10)
        self.max_daily_trades = config.get('max_daily_trades', 15)

        # Equity & cash
        self.total_equity = self.get_balance()
        self.available_cash = float(self.total_equity)

        # Historical data cache
        self.historical_data_cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------
    # Core Methods
    # ------------------------------
    def get_balance(self) -> float:
        """Get USD balance from Roostoo (SpotWallet.USD.Free)."""
        url = f"{self.base_url}/v3/balance"
        headers, payload, _ = self._get_signed_headers({})
        try:
            res = requests.get(url, headers=headers, params=payload, timeout=10)
            res.raise_for_status()
            data = res.json()
            usd_free = data["SpotWallet"]["USD"]["Free"]
            logger.info(f"[Roostoo] USD Free balance: {usd_free}")
            return float(usd_free)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting balance: {e}")
            logger.error(f"Response text: {e.response.text if getattr(e, 'response', None) else 'N/A'}")
            # Fallback so the bot can continue in paper style if balance fails
            return 50000.0

    async def run_continuous_strategy(self, watchlist: List[str]):
        """Run strategy continuously every 5 minutes."""
        logger.info("Starting continuous strategy execution")

        iteration = 0
        while True:
            # Reset daily counter at UTC day change
            today = datetime.utcnow().date()
            if today != self._current_day:
                self.trade_count_today = 0
                self._current_day = today
                logger.info("New UTC day: trade counter reset.")

            iteration += 1
            logger.info(f"=== Iteration #{iteration} at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC ===")

            try:
                await self.run_strategy(watchlist)
            except Exception as e:
                logger.exception(f"run_strategy crashed: {e}")

            self._log_portfolio_status()
            logger.info("Waiting 5 minutes until next scan...")
            await asyncio.sleep(300)

    async def run_strategy(self, watchlist: List[str]):
        for symbol in watchlist:
            # Entry
            try:
                entry_signal = await self.check_entry_signal(symbol)
            except Exception as e:
                logger.exception(f"check_entry_signal({symbol}) failed: {e}")
                entry_signal = TradeSignal.HOLD

            if entry_signal == TradeSignal.BUY:
                await self.execute_trade(symbol, entry_signal)

            # Exit
            try:
                exit_signal = await self.check_exit_signals(symbol)
            except Exception as e:
                logger.exception(f"check_exit_signals({symbol}) failed: {e}")
                exit_signal = TradeSignal.HOLD

            if exit_signal == TradeSignal.SELL:
                await self.execute_trade(symbol, exit_signal)

    async def check_entry_signal(self, symbol: str) -> TradeSignal:
        if len(self.positions) >= self.max_positions:
            return TradeSignal.HOLD
        if self.trade_count_today >= self.max_daily_trades:
            return TradeSignal.HOLD
        if symbol in self.positions:
            return TradeSignal.HOLD

        market_data = await self.get_market_data([symbol])
        if symbol not in market_data:
            return TradeSignal.HOLD

        historical_data = await self.get_historical_data(symbol, periods=48)
        if len(historical_data) < 24:
            return TradeSignal.HOLD

        historical_data = self.calculate_indicators(historical_data)

        overshoot_condition = self.check_overshoot_condition(market_data[symbol], historical_data)
        reversal_confirmation = self.check_reversal_confirmation(historical_data)

        if overshoot_condition and reversal_confirmation:
            logger.info(f"BUY signal for {symbol}")
            return TradeSignal.BUY

        return TradeSignal.HOLD

    async def check_exit_signals(self, symbol: str) -> TradeSignal:
        if symbol not in self.positions:
            return TradeSignal.HOLD

        position = self.positions[symbol]
        market_data = await self.get_market_data([symbol])
        if symbol not in market_data:
            return TradeSignal.HOLD

        current_price = market_data[symbol]["price"]

        # Stop loss
        if current_price <= position.stop_loss:
            logger.info(f"Stop loss triggered for {symbol}")
            return TradeSignal.SELL

        # Take profit ladder
        change = (current_price - position.entry_price) / position.entry_price
        if not position.half_sold and change >= self.take_profit_1:
            logger.info(f"Take profit 1 reached for {symbol}")
            return TradeSignal.SELL
        if position.half_sold and change >= self.take_profit_2:
            logger.info(f"Take profit 2 reached for {symbol}")
            return TradeSignal.SELL

        return TradeSignal.HOLD

    def _get_timestamp(self) -> str:
        """13-digit millisecond timestamp as string."""
        return str(int(time.time() * 1000))

    async def execute_trade(self, symbol: str, signal: TradeSignal):
        """Execute trade based on signal. (async but NO decorator)"""
        market_data = await self.get_market_data([symbol])
        if symbol not in market_data:
            return

        current_price = market_data[symbol]["price"]

        if signal == TradeSignal.BUY:
            units = self.calculate_position_size(current_price)
            cost = units * current_price

            if cost > self.available_cash:
                logger.warning(f"Insufficient cash for {symbol}")
                return

            if not self.submit_trade_order(symbol, "BUY", units):
                logger.error("Order failed")
                return

            position = Position(
                symbol=symbol,
                entry_price=current_price,
                size=units,
                stop_loss=current_price * (1 - self.stop_loss_pct),
                take_profit_1=current_price * (1 + self.take_profit_1),
                take_profit_2=current_price * (1 + self.take_profit_2),
            )
            self.positions[symbol] = position
            self.available_cash -= cost
            self.trade_count_today += 1
            logger.info(f"Opened {symbol}: {units:.4f} units at ${current_price:.2f}")

        elif signal == TradeSignal.SELL:
            position = self.positions[symbol]
            if position.half_sold:
                sell_size = position.size
                del self.positions[symbol]
            else:
                sell_size = position.size / 2.0
                position.size = sell_size
                position.half_sold = True
                position.stop_loss = position.entry_price * 1.02  # ratchet up

            if not self.submit_trade_order(symbol, "SELL", sell_size):
                logger.error("Order failed")
                return

            self.available_cash += sell_size * current_price
            logger.info(f"Realized P&L for {symbol} part/close at ${current_price:.2f}")

    # ------------------------------
    # Indicators
    # ------------------------------
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 24:
            return df

        df = df.copy()
        # Z-score (24 samples)
        df["price_mean_24h"] = df["close"].rolling(window=24, min_periods=1).mean()
        df["price_std_24h"] = df["close"].rolling(window=24, min_periods=1).std()
        df["z_score"] = (df["close"] - df["price_mean_24h"]) / df["price_std_24h"]

        # 5-period EMA
        df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()

        # RSI(14)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, pd.NA)
        df["rsi"] = 100 - (100 / (1 + rs))

        return df.bfill().ffill()

    def check_overshoot_condition(self, symbol_data: MarketData, historical_data: pd.DataFrame) -> bool:
        latest = historical_data.iloc[-1]
        z_ok = latest["z_score"] < self.z_score_threshold
        vol_ok = symbol_data["volume"] >= (symbol_data["volume_30d_avg"] * self.volume_multiplier)
        rsi_ok = latest["rsi"] < self.rsi_threshold
        return bool(z_ok and vol_ok and rsi_ok)

    def check_reversal_confirmation(self, historical_data: pd.DataFrame) -> bool:
        current = historical_data.iloc[-1]
        previous = historical_data.iloc[-2]
        ema_ok = current["close"] > current["ema_5"]
        z_improving = current["z_score"] > (previous["z_score"] - 0.1)  # allow small slack
        z_reversal = current["z_score"] > self.z_score_reversal
        return bool(ema_ok and z_improving and z_reversal)

    def calculate_position_size(self, current_price: float) -> float:
        return max(0.0, (self.total_equity * self.position_size) / max(current_price, 1e-12))

    # ------------------------------
    # Data
    # ------------------------------
    async def get_market_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Fetch real-time market data (Binance 24hr ticker)."""
        market: Dict[str, MarketData] = {}

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for symbol in symbols:
                try:
                    data = await self._get_binance_ticker_data(session, symbol)
                    if data:
                        market[symbol] = data
                except Exception as e:
                    logger.exception(f"_get_binance_ticker_data({symbol}) failed: {e}")

        return market

    async def _get_binance_ticker_data(
        self, session: aiohttp.ClientSession, symbol: str
    ) -> Optional[MarketData]:
        """Get individual symbol data from Binance."""
        binance_symbol = self._convert_to_binance_symbol(symbol)
        url = "https://api.binance.com/api/v3/ticker/24hr"

        async with session.get(url, params={"symbol": binance_symbol}) as response:
            if response.status != 200:
                return None

            data = await response.json()

            hist = await self.get_historical_data(symbol, periods=24 * 30)
            if hist.empty:
                vol30 = float(data.get("volume", "0"))
            else:
                vol30 = float(hist["volume"].mean())

            return MarketData(
                price=float(data["lastPrice"]),
                volume=float(data["volume"]),
                high=float(data["highPrice"]),
                low=float(data["lowPrice"]),
                open=float(data["openPrice"]),
                close=float(data["lastPrice"]),
                volume_30d_avg=vol30,
            )

    async def get_historical_data(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        """Fetch historical OHLC from Binance (15m)."""
        cache_key = f"{symbol}_{periods}"
        if cache_key in self.historical_data_cache:
            return self.historical_data_cache[cache_key].copy()

        timeout = aiohttp.ClientTimeout(total=15)
        binance_symbol = self._convert_to_binance_symbol(symbol)
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": binance_symbol, "interval": "15m", "limit": periods}

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        return pd.DataFrame()

                    data = await response.json()

            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
                ],
            )

            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)

            self.historical_data_cache[cache_key] = df.copy()
            return df
        except Exception as e:
            logger.exception(f"get_historical_data({symbol}) failed: {e}")
            return pd.DataFrame()

    def _convert_to_binance_symbol(self, symbol: str) -> str:
        base = symbol.split("/")[0] if "/" in symbol else symbol
        return f"{base}USDT"

    # ------------------------------
    # Trading (Roostoo)
    # ------------------------------
    @apply_delay
    def submit_trade_order(self, pair_or_coin: str, side: str, quantity: float) -> bool:
        """Submit a MARKET order to Roostoo."""
        # Build pair in Roostoo format
        pair = f"{pair_or_coin}/USD" if "/" not in pair_or_coin else pair_or_coin
        url = f"{self.base_url}/v3/place_order"

        payload = {
            "pair": pair,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": f"{quantity:.6f}",
            "timestamp": self._get_timestamp(),
        }

        headers, _, total_params = self._get_signed_headers(payload)
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        try:
            res = requests.post(url, headers=headers, data=total_params, timeout=10)
            res.raise_for_status()
            logger.info(f"[Roostoo] Order OK: {res.json()}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error placing order: {e}")
            logger.error(f"Response text: {e.response.text if getattr(e, 'response', None) else 'N/A'}")
            return False

    def _get_signed_headers(self, payload: dict):
        """Generate signed headers for Roostoo."""
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Missing ROOSTOO_API or SECRET_KEY env variables")

        payload = dict(payload)  # copy to avoid side-effects
        payload["timestamp"] = payload.get("timestamp", self._get_timestamp())
        sorted_keys = sorted(payload.keys())
        total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)

        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            total_params.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature,
        }
        return headers, payload, total_params

    # ------------------------------
    # Utility
    # ------------------------------
    def _log_portfolio_status(self):
        portfolio_value = sum(p.size * p.entry_price for p in self.positions.values())
        total_value = self.available_cash + portfolio_value
        logger.info(
            f"Portfolio: Cash=${self.available_cash:.2f}, "
            f"Positions=${portfolio_value:.2f}, Total=${total_value:.2f}, "
            f"Active={len(self.positions)}, Trades Today={self.trade_count_today}"
        )


# ------------------------------
# Main
# ------------------------------
async def main():
    strategy_config = {
        "z_score_threshold": -1.5,
        "z_score_reversal": -1.25,
        "volume_multiplier": 1.2,
        "rsi_threshold": 25,
        "position_size": 0.03,
        "stop_loss_pct": 0.06,
        "take_profit_1": 0.08,
        "take_profit_2": 0.15,
        "max_positions": 15,
        "max_daily_trades": 25,
    }

    watchlist = ["BNB", "SOL", "ADA", "XRP", "DOT", "ICP", "LTC", "ZEC", "UNI", "TRUMP", "SUI"]

    strategy = Web3MeanReversionStrategy(strategy_config)
    await strategy.run_continuous_strategy(watchlist)


if __name__ == "__main__":
    asyncio.run(main())
