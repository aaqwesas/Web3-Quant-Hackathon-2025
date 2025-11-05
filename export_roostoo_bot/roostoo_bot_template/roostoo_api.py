import time
import hashlib
import hmac
import requests
import urllib.parse


class RoostooClient:
    """Minimal signed client for Roostoo API.

    Usage:
        client = RoostooClient(api_key="...", api_secret="...", base_url="https://api.roostoo.com")
    """
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.roostoo.com"):
        self.api_key = api_key
        self.api_secret = api_secret.encode('utf-8')
        self.base_url = base_url.rstrip('/')
        self.s = requests.Session()
        self.s.headers.update({'Content-Type': 'application/x-www-form-urlencoded'})

    def _timestamp(self):
        return str(int(time.time() * 1000))

    def _sign(self, params: dict) -> str:
        # sort params by key, url-encode values, produce key=val&... string
        items = []
        for k in sorted(params.keys()):
            v = params[k]
            if isinstance(v, (list, tuple)):
                v = ','.join(map(str, v))
            items.append(f"{k}={urllib.parse.quote_plus(str(v))}")
        data = '&'.join(items)
        h = hmac.new(self.api_secret, data.encode('utf-8'), hashlib.sha256)
        return h.hexdigest()

    def _post(self, path: str, params: dict):
        params = params.copy()
        params['timestamp'] = self._timestamp()
        sig = self._sign(params)
        headers = {
            'RST-API-KEY': self.api_key,
            'MSG-SIGNATURE': sig,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        url = self.base_url + path
        resp = self.s.post(url, data=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: dict = None):
        params = params.copy() if params else {}
        params['timestamp'] = self._timestamp()
        sig = self._sign(params)
        headers = {
            'RST-API-KEY': self.api_key,
            'MSG-SIGNATURE': sig,
        }
        url = self.base_url + path
        resp = self.s.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # Public wrappers
    def get_exchange_info(self):
        return self._get('/v3/exchangeInfo')

    def get_ticker(self, symbol: str):
        return self._get('/v3/ticker/24hr', {'symbol': symbol})

    def get_balance(self):
        return self._get('/v3/account')

    def get_candles(self, symbol: str, interval: str, limit: int = 500):
        # endpoint returns list of candles or a structured dict depending on API; normalize to list of dicts
        return self._get('/v3/klines', {'symbol': symbol, 'interval': interval, 'limit': limit})

    def place_order(self, pair: str, side: str, quantity: float, price: float = None, order_type: str = 'MARKET'):
        payload = {
            'symbol': pair,
            'side': side,
            'type': order_type,
            'quantity': quantity
        }
        if price is not None:
            payload['price'] = price
        return self._post('/v3/order', payload)

    def query_order(self, symbol: str, order_id: str):
        return self._get('/v3/order', {'symbol': symbol, 'orderId': order_id})

    def cancel_order(self, symbol: str, order_id: str):
        return self._post('/v3/order', {'symbol': symbol, 'orderId': order_id, 'side': 'CANCEL'})
