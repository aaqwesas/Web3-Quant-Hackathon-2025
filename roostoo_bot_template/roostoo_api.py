import time
import hmac
import hashlib
import requests
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple


class RoostooClient:
    """Roostoo API client implementing the signed request pattern from the official docs.

    Usage:
        client = RoostooClient(base_url, api_key=..., api_secret=...)
        client.place_order('BTC/USD', 'BUY', 0.01)
    """

    def __init__(self, base_url: str, api_key: str = "", api_secret: str = "",
                 timeout_sec: int = 10, min_request_interval_sec: float = 0.2):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout_sec
        self.min_dt = min_request_interval_sec
        self._last = 0.0

    # ---- Helpers for signing requests ----
    def _get_timestamp(self) -> str:
        return str(int(time.time() * 1000))

    def _generate_signature(self, params: Dict[str, Any]) -> Tuple[str, str]:
        """Given a params dict, produce (total_params_string, signature_hex).

        total_params_string is sorted by key: key=value&key2=value2 ...
        signature = HMAC_SHA256(secret, total_params_string)
        """
        # ensure all values are stringifiable
        sorted_items = sorted(params.items())
        total = '&'.join(f"{k}={params[k]}" for k, _ in sorted_items)
        sig = hmac.new(self.api_secret.encode('utf-8'), total.encode('utf-8'), hashlib.sha256).hexdigest()
        return total, sig

    def _get_signed_headers(self, params: Dict[str, Any]) -> Tuple[Dict[str, str], str, Dict[str, Any]]:
        """Return (headers, total_params_string, full_payload) ready to be sent.

        For POST requests we send data=total_params_string and Content-Type application/x-www-form-urlencoded
        For GET requests the params are sent as query params (requests will URL-encode them).
        """
        # Add timestamp if not present
        if 'timestamp' not in params:
            params = dict(params)
            params['timestamp'] = self._get_timestamp()

        total, sig = self._generate_signature(params)
        headers = {
            'RST-API-KEY': self.api_key,
            'MSG-SIGNATURE': sig,
            # POST callers should set Content-Type when sending body
        }
        return headers, total, params

    def _throttle(self):
        dt = time.time() - self._last
        if dt < self.min_dt:
            time.sleep(self.min_dt - dt)
        self._last = time.time()

    def _safe_post(self, path: str, payload: Dict[str, Any]) -> Any:
        self._throttle()
        url = f"{self.base_url}{path}"
        headers, total_str, params = self._get_signed_headers(payload)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
        try:
            r = requests.post(url, headers=headers, data=total_str, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            # Return structured error similar to docs
            return {'Success': False, 'ErrMsg': str(e), 'http_error': getattr(e, 'response', None).text if getattr(e, 'response', None) else None}

    def _safe_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._throttle()
        url = f"{self.base_url}{path}"
        p = params or {}
        # attach timestamp and signature in headers; for GET, params are sent in query string
        headers, total_str, full_params = self._get_signed_headers(p)
        try:
            r = requests.get(url, headers=headers, params=full_params, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            return {'Success': False, 'ErrMsg': str(e), 'http_error': getattr(e, 'response', None).text if getattr(e, 'response', None) else None}

    # ---- API methods (based on Roostoo docs) ----
    def get_exchange_info(self) -> Any:
        return self._safe_get('/v3/exchangeInfo', params={})

    def get_candles(self, pair: str, interval: str = '1m', limit: int = 500) -> Any:
        """Get historical klines/candles. Endpoint may vary by deployment; docs do not strictly define.

        Returns a list of candles; format depends on exchange. Caller should handle to_df conversion.
        """
        params = {'pair': pair, 'interval': interval, 'limit': str(limit)}
        return self._safe_get('/v3/klines', params=params)

    def get_ticker(self, pair: Optional[str] = None) -> Any:
        payload = {}
        if pair:
            payload['pair'] = pair
        return self._safe_get('/v3/ticker', params=payload)

    def get_balance(self) -> Any:
        return self._safe_get('/v3/balance', params={})

    def pending_count(self) -> Any:
        return self._safe_get('/v3/pending_count', params={})

    def place_order(self, pair: str, side: str, quantity: float, price: Optional[float] = None, order_type: Optional[str] = None) -> Any:
        """Place a new order. pair e.g. 'BTC/USD'. If price is None -> MARKET else LIMIT.

        Returns parsed JSON response from API.
        """
        if order_type is None:
            order_type = 'LIMIT' if price is not None else 'MARKET'
        payload = {
            'pair': pair,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': str(quantity),
        }
        if order_type.upper() == 'LIMIT':
            if price is None:
                return {'Success': False, 'ErrMsg': 'LIMIT order requires price'}
            payload['price'] = str(price)

        return self._safe_post('/v3/place_order', payload)

    def query_order(self, order_id: Optional[str] = None, pair: Optional[str] = None, pending_only: Optional[bool] = None,
                    offset: Optional[int] = None, limit: Optional[int] = None) -> Any:
        payload = {}
        if order_id:
            payload['order_id'] = str(order_id)
        elif pair:
            payload['pair'] = pair
            if pending_only is not None:
                payload['pending_only'] = 'TRUE' if pending_only else 'FALSE'
        if offset is not None:
            payload['offset'] = str(offset)
        if limit is not None:
            payload['limit'] = str(limit)
        return self._safe_post('/v3/query_order', payload)

    def cancel_order(self, order_id: Optional[str] = None, pair: Optional[str] = None) -> Any:
        payload = {}
        if order_id:
            payload['order_id'] = str(order_id)
        elif pair:
            payload['pair'] = pair
        # if neither provided, cancels all pending per docs
        return self._safe_post('/v3/cancel_order', payload)
