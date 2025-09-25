# provider_configs.py
"""
WebSocket provider configurations for popular stock data feeds.
Adapt the main script by importing the appropriate provider config.
"""

import json

class AlpacaConfig:
    """Alpaca WebSocket configuration"""
    WEBSOCKET_URL = "wss://stream.data.alpaca.markets/v2/iex"
    
    @staticmethod
    def get_auth_message(api_key, secret_key):
        return json.dumps({
            "action": "auth",
            "key": api_key,
            "secret": secret_key
        })
    
    @staticmethod
    def get_subscribe_message(symbols):
        return json.dumps({
            "action": "subscribe",
            "trades": symbols
        })
    
    @staticmethod
    def parse_message(msg_text):
        try:
            data = json.loads(msg_text)
            if isinstance(data, list):
                for msg in data:
                    if msg.get("T") == "t":  # trade message
                        return {
                            "symbol": msg.get("S"),
                            "price": float(msg.get("p")),
                            "size": float(msg.get("s")),
                            "ts": msg.get("t")  # timestamp in nanoseconds
                        }
        except:
            pass
        return None

class BinanceConfig:
    """Binance WebSocket configuration for crypto"""
    WEBSOCKET_URL_TEMPLATE = "wss://stream.binance.com:9443/ws/{symbol}@trade"
    
    @staticmethod
    def get_websocket_url(symbol):
        return f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"
    
    @staticmethod
    def parse_message(msg_text):
        try:
            data = json.loads(msg_text)
            if data.get("e") == "trade":
                return {
                    "symbol": data.get("s"),
                    "price": float(data.get("p")),
                    "size": float(data.get("q")),
                    "ts": int(data.get("T"))  # timestamp in milliseconds
                }
        except:
            pass
        return None

class PolygonConfig:
    """Polygon.io WebSocket configuration"""
    WEBSOCKET_URL = "wss://socket.polygon.io/stocks"
    
    @staticmethod
    def get_auth_message(api_key):
        return json.dumps({
            "action": "auth",
            "params": api_key
        })
    
    @staticmethod
    def get_subscribe_message(symbols):
        return json.dumps({
            "action": "subscribe",
            "params": f"T.{','.join(symbols)}"
        })
    
    @staticmethod
    def parse_message(msg_text):
        try:
            data = json.loads(msg_text)
            if isinstance(data, list):
                for msg in data:
                    if msg.get("ev") == "T":  # trade message
                        return {
                            "symbol": msg.get("sym"),
                            "price": float(msg.get("p")),
                            "size": float(msg.get("s")),
                            "ts": int(msg.get("t"))  # timestamp in milliseconds
                        }
        except:
            pass
        return None

class IEXConfig:
    """IEX Cloud WebSocket configuration"""
    WEBSOCKET_URL = "wss://cloud-sse.iexapis.com/stable/stocksUS"
    
    @staticmethod
    def get_websocket_url(api_token, symbols):
        symbols_str = ",".join(symbols)
        return f"wss://cloud-sse.iexapis.com/stable/stocksUS?symbols={symbols_str}&token={api_token}"
    
    @staticmethod
    def parse_message(msg_text):
        try:
            # IEX sends Server-Sent Events, need to parse differently
            if msg_text.startswith("data: "):
                data = json.loads(msg_text[6:])
                if isinstance(data, dict) and "price" in data:
                    return {
                        "symbol": data.get("symbol"),
                        "price": float(data.get("price")),
                        "size": float(data.get("size", 0)),
                        "ts": int(data.get("time", 0))
                    }
        except:
            pass
        return None

# Example usage configurations
PROVIDER_CONFIGS = {
    "alpaca": AlpacaConfig,
    "binance": BinanceConfig,
    "polygon": PolygonConfig,
    "iex": IEXConfig
}