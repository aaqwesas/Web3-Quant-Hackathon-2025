import os
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict, Callable
import aiohttp
import asyncio
import hmac
import hashlib
import time
import requests
from datetime import datetime
from functools import wraps
from dotenv import load_dotenv


load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



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

# def apply_delay[T, **P](func: Callable[P, T]) -> Callable[P, T]:
#     @wraps(func)
#     def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
#         result = func(*args, **kwargs)
#         time.sleep(1)
#         return result
#     return wrapper

def apply_delay(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        time.sleep(1)
        return result
    return wrapper

class Web3MeanReversionStrategy:
    def __init__(self, config: dict):
        self.config = config
        self.positions = {}
        self.trade_count_today = 0
        
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
        

        self.total_equity = self.get_balance()
        self.available_cash = self.total_equity
        
        # Historical data cache
        self.historical_data_cache = {}

    # ------------------------------
    # Core Strategy Methods
    # ------------------------------


    def get_balance(self):
        """Get wallet balances (RCL_TopLevelCheck)."""
        url = f"{self.base_url}/v3/balance"
        headers, payload, _ = self._get_signed_headers({})
        try:
            res = requests.get(url, headers=headers, params=payload)
            res.raise_for_status()
            res = res.json()["SpotWallet"]["USD"]["Free"]
            return int(res)

        except requests.exceptions.RequestException as e:
            print(f"Error getting balance: {e}")
            print(f"Response text: {e.response.text if e.response else 'N/A'}")
            return 50000
        
    async def run_continuous_strategy(self, watchlist: list[str]):
        """Run strategy continuously every 5 minutes"""
        logger.info("Starting continuous strategy execution")
        
        iteration = 0
        while True:
            iteration += 1
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"=== Iteration #{iteration} at {current_time} ===")
            
            await self.run_strategy(watchlist)
            self._log_portfolio_status()
            
            logger.info("Waiting 5 minutes until next scan...")
            await asyncio.sleep(300)

    async def run_strategy(self, watchlist: list[str]):
        """Main strategy execution loop"""
        for symbol in watchlist:
            # Check for entry signals
            entry_signal = await self.check_entry_signal(symbol)
            if entry_signal == TradeSignal.BUY:
                await self.execute_trade(symbol, entry_signal)
            
            # Check for exit signals
            exit_signal = await self.check_exit_signals(symbol)
            if exit_signal == TradeSignal.SELL:
                await self.execute_trade(symbol, exit_signal)

    async def check_entry_signal(self, symbol: str) -> TradeSignal:
        """Check if entry conditions are met"""
        if (len(self.positions) >= self.max_positions or 
            self.trade_count_today >= self.max_daily_trades or
            symbol in self.positions):
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
        """Check exit conditions for existing positions"""
        if symbol not in self.positions:
            return TradeSignal.HOLD
            
        position = self.positions[symbol]
        market_data = await self.get_market_data([symbol])
        
        if symbol not in market_data:
            return TradeSignal.HOLD
            
        current_price = market_data[symbol]['price']
        
        # Check stop loss
        if current_price <= position.stop_loss:
            logger.info(f"Stop loss triggered for {symbol}")
            return TradeSignal.SELL
        
        # Check take profit levels
        price_change = (current_price - position.entry_price) / position.entry_price
        
        if not position.half_sold and price_change >= self.take_profit_1:
            logger.info(f"Take profit 1 reached for {symbol}")
            return TradeSignal.SELL
            
        if position.half_sold and price_change >= self.take_profit_2:
            logger.info(f"Take profit 2 reached for {symbol}")
            return TradeSignal.SELL
            
        return TradeSignal.HOLD
    
    def _get_timestamp(self):
        """Return a 13-digit millisecond timestamp as string."""
        return str(int(time.time() * 1000))
    

    @apply_delay
    async def execute_trade(self, symbol: str, signal: TradeSignal):
        """Execute trade based on signal"""
        market_data = await self.get_market_data([symbol])
        if symbol not in market_data:
            return
            
        current_price = market_data[symbol]['price']
        
        if signal == TradeSignal.BUY:
            units = self.calculate_position_size(current_price)
            cost = units * current_price
            
            if cost > self.available_cash:
                logger.warning(f"Insufficient cash for {symbol}")
                return
                
            result = self.submit_trade_order(symbol, 'BUY', units)
            
            if not result:
                logger.error("Order failed")
                return
            position = Position(
                symbol=symbol,
                entry_price=current_price,
                size=units,
                stop_loss=current_price * (1 - self.stop_loss_pct),
                take_profit_1=current_price * (1 + self.take_profit_1),
                take_profit_2=current_price * (1 + self.take_profit_2)
            )
            
            self.positions[symbol] = position
            self.available_cash -= cost
            self.trade_count_today += 1
            logger.info(f"Opened position in {symbol}: {units:.4f} units at ${current_price:.2f}")
                
        elif signal == TradeSignal.SELL:
            position = self.positions[symbol]
            
            if position.half_sold:
                sell_size = position.size
                del self.positions[symbol]
            else:
                sell_size = position.size / 2
                position.size = sell_size
                position.half_sold = True
                position.stop_loss = position.entry_price * 1.02
            
            result = self.submit_trade_order(symbol, 'SELL', sell_size)
            
            if not result:
                logger.error("order failed")
            self.available_cash += sell_size * current_price
            logger.info(f"Closed position in {symbol}")

    # ------------------------------
    # Technical Analysis Methods
    # ------------------------------

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if len(df) < 24:
            return df
            
        # Z-score (24-hour)
        df['price_mean_24h'] = df['close'].rolling(window=24, min_periods=1).mean()
        df['price_std_24h'] = df['close'].rolling(window=24, min_periods=1).std()
        df['z_score'] = (df['close'] - df['price_mean_24h']) / df['price_std_24h']
        
        # 5-period EMA
        df['ema_5'] = df['close'].ewm(span=5).mean()
        
        # RSI (14 periods)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df.bfill().ffill()

    def check_overshoot_condition(self, symbol_data: MarketData, historical_data: pd.DataFrame) -> bool:
        """Check if asset meets overshoot conditions"""
        latest = historical_data.iloc[-1]
        
        z_score_condition = latest['z_score'] < self.z_score_threshold
        volume_condition = symbol_data['volume'] >= (symbol_data['volume_30d_avg'] * self.volume_multiplier)
        rsi_condition = latest['rsi'] < self.rsi_threshold
        
        return z_score_condition and volume_condition and rsi_condition

    def check_reversal_confirmation(self, historical_data: pd.DataFrame) -> bool:
        """Check for reversal confirmation"""
        current = historical_data.iloc[-1]
        previous = historical_data.iloc[-2]
        
        ema_condition = current['close'] > current['ema_5']
        z_score_improving = current['z_score'] > previous['z_score'] - 0.1 # allow some range
        z_score_threshold = current['z_score'] > self.z_score_reversal
        
        return ema_condition and z_score_improving and z_score_threshold 

    def calculate_position_size(self, current_price: float) -> float:
        """Calculate position size based on available equity"""
        return (self.total_equity * self.position_size) / current_price

    # ------------------------------
    # Data Methods
    # ------------------------------

    async def get_market_data(self, symbols: list[str]) -> dict[str, MarketData]:
        """Get real-time market data from Binance"""
        market_data = {}
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                data = await self._get_binance_ticker_data(session, symbol)
                if data:
                    market_data[symbol] = data
        
        return market_data

    async def _get_binance_ticker_data(self, session: aiohttp.ClientSession, symbol: str) -> MarketData | None:
        """Get individual symbol data from Binance"""
        binance_symbol = self._convert_to_binance_symbol(symbol)
        
        url = "https://api.binance.com/api/v3/ticker/24hr"
        async with session.get(url, params={'symbol': binance_symbol}) as response:
            if response.status == 200:
                data = await response.json()
                
                historical_data = await self.get_historical_data(symbol, periods=24*30)
                volume_30d_avg = historical_data['volume'].mean() if not historical_data.empty else float(data['volume'])
                
                return MarketData(
                    price=float(data['lastPrice']),
                    volume=float(data['volume']),
                    high=float(data['highPrice']),
                    low=float(data['lowPrice']),
                    open=float(data['openPrice']),
                    close=float(data['lastPrice']),
                    volume_30d_avg=volume_30d_avg
                )

    async def get_historical_data(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        """Get historical OHLC data from Binance"""
        cache_key = f"{symbol}_{periods}"
        if cache_key in self.historical_data_cache:
            return self.historical_data_cache[cache_key].copy()
        
        binance_symbol = self._convert_to_binance_symbol(symbol)
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': binance_symbol, 'interval': '15m', 'limit': periods}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    
                    self.historical_data_cache[cache_key] = df.copy()
                    return df
        
        return pd.DataFrame()

    def _convert_to_binance_symbol(self, symbol: str) -> str:
        """Convert symbol to Binance format"""
        if '/' in symbol:
            base = symbol.split('/')[0]
        else:
            base = symbol
        return f"{base}USDT"

    # ------------------------------
    # Trading Methods (with apply_delay)
    # ------------------------------

    @apply_delay
    def submit_trade_order(self, pair_or_coin: str, side: str, quantity: float) -> bool:
        """
        Place a LIMIT or MARKET order.
        """
        url = f"{self.base_url}/v3/place_order"
        pair = f"{pair_or_coin}/USD" if "/" not in pair_or_coin else pair_or_coin


        payload = {
            'pair': pair,
            'side': side.upper(),
            'type': "MARKET",
            'quantity': f"{quantity:.1f}",
            "timestamp": self._get_timestamp()
        }

        headers, _, total_params = self._get_signed_headers(payload)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'

        try:
            res = requests.post(url, headers=headers, data=total_params)
            res.raise_for_status()
            print(res.json())
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error placing order: {e}")
            print(f"Response text: {e.response.text if e.response else 'N/A'}")
            return False

    def _get_signed_headers(self, payload: dict):
        """Generate signed headers for Roostoo API"""
        payload['timestamp'] = str(int(time.time() * 1000))
        sorted_keys = sorted(payload.keys())
        total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)

        signature = hmac.new(
            self.secret_key.encode('utf-8'), # type: ignore
            total_params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        headers = {
            'RST-API-KEY': self.api_key,
            'MSG-SIGNATURE': signature
        }

        return headers, payload, total_params

    # ------------------------------
    # Utility Methods
    # ------------------------------

    def _log_portfolio_status(self):
        """Log portfolio status"""
        portfolio_value = sum(pos.size * pos.entry_price for pos in self.positions.values())
        total_value = self.available_cash + portfolio_value
        
        logger.info(f"Portfolio: Cash=${self.available_cash:.2f}, "
                   f"Positions=${portfolio_value:.2f}, Total=${total_value:.2f}, "
                   f"Active={len(self.positions)}, Trades Today={self.trade_count_today}")

# ------------------------------
# Main Execution
# ------------------------------


async def main():
    strategy_config = {
        'z_score_threshold': -2.0,
        'z_score_reversal': -1.0,
        'volume_multiplier': 1.2,
        'rsi_threshold': 25,
        'position_size': 0.03,
        'stop_loss_pct': 0.06,
        'take_profit_1': 0.08,
        'take_profit_2': 0.15,
        'max_positions': 15,
        'max_daily_trades': 25,
    }
    
    watchlist = ["BNB", "SOL", "ADA", "XRP", "DOT", "ICP", "LTC", "ZEC", "UNI", "TRUMP", "SUI"]
    
    strategy = Web3MeanReversionStrategy(strategy_config)
    await strategy.run_continuous_strategy(watchlist)

if __name__ == "__main__":
    asyncio.run(main())