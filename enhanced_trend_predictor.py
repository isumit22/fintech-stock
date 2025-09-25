# enhanced_trend_predictor.py
"""
Enhanced 5-minute trend prediction system with multiple provider support
and simulation mode for testing
"""

import asyncio
import json
import os
import argparse
from collections import deque
from datetime import datetime, timedelta
import math

import numpy as np
from river import forest, linear_model, preprocessing, metrics
import websockets

# Import our custom modules
from provider_configs import PROVIDER_CONFIGS
from simulation_mode import (
    simulate_websocket_feed, 
    create_mock_data_source, 
    create_historical_data_source,
    create_sample_csv
)

class TrendPredictor:
    """5-minute stock trend prediction system"""
    
    def __init__(self, symbol="AAPL", config=None):
        self.symbol = symbol
        self.config = config or self._default_config()
        
        # Initialize model
        if self.config["model_type"] == "random_forest":
            self.model = preprocessing.StandardScaler() | ensemble.AdaptiveRandomForestClassifier(
                n_estimators=self.config["n_estimators"], 
                seed=42
            )
        else:  # logistic regression
            self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
            
        # Metrics
        self.metric_acc = metrics.Accuracy()
        self.metric_bal = metrics.BalancedAccuracy()
        
        # Data storage
        self.bars = deque(maxlen=500)
        self.current_bar = None
        self.pending_snapshots = deque()
        
        # Logging
        self.tick_count = 0
        self.prediction_count = 0
        
    def _default_config(self):
        return {
            "bar_seconds": 60,
            "pred_horizon_minutes": 5,
            "feature_window_minutes": 15,
            "label_threshold": 0.001,  # 0.1%
            "model_type": "random_forest",  # or "logistic"
            "n_estimators": 10
        }
    
    def start_new_bar(self, ts, price, size):
        return {
            "ts": ts,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": size if size is not None else 0.0,
            "vwap_numer": price * (size if size is not None else 0.0),
            "vwap_denom": (size if size is not None else 0.0)
        }

    def update_bar(self, bar, price, size):
        bar["high"] = max(bar["high"], price)
        bar["low"] = min(bar["low"], price)
        bar["close"] = price
        if size is not None:
            bar["volume"] += size
            bar["vwap_numer"] += price * size
            bar["vwap_denom"] += size

    def finalize_bar(self, bar):
        vwap = bar["vwap_numer"] / bar["vwap_denom"] if bar["vwap_denom"] > 0 else bar["close"]
        return {
            "ts": bar["ts"],
            "open": float(bar["open"]),
            "high": float(bar["high"]),
            "low": float(bar["low"]),
            "close": float(bar["close"]),
            "volume": float(bar["volume"]),
            "vwap": float(vwap)
        }

    def compute_features_from_bars(self, bars_deque):
        """Enhanced feature engineering"""
        arr_close = np.array([b["close"] for b in bars_deque])
        arr_vol = np.array([b["volume"] for b in bars_deque])
        arr_vwap = np.array([b["vwap"] for b in bars_deque])
        arr_high = np.array([b["high"] for b in bars_deque])
        arr_low = np.array([b["low"] for b in bars_deque])

        n = len(arr_close)
        if n == 0:
            return {}

        last = float(arr_close[-1])
        features = {}
        
        # Basic features
        features["n_minutes"] = n
        features["last_price"] = last
        
        # Returns at multiple timeframes
        for period, name in [(1, "1m"), (3, "3m"), (5, "5m"), (10, "10m")]:
            if n >= period + 1:
                features[f"ret_{name}"] = float((arr_close[-1] / arr_close[-period-1]) - 1.0)
            else:
                features[f"ret_{name}"] = 0.0

        # Moving averages
        for period in [3, 5, 10]:
            if n >= period:
                features[f"ma_{period}"] = float(np.mean(arr_close[-period:]))
            else:
                features[f"ma_{period}"] = float(np.mean(arr_close))
        
        # MA crossovers
        features["ma_gap_3_10"] = features.get("ma_3", 0) - features.get("ma_10", 0)
        features["ma_gap_5_10"] = features.get("ma_5", 0) - features.get("ma_10", 0)

        # Volatility measures
        features["std_3"] = float(np.std(arr_close[-3:])) if n >= 3 else 0.0
        features["std_10"] = float(np.std(arr_close[-10:])) if n >= 10 else float(np.std(arr_close))
        
        # High-Low range features
        features["hl_range_1m"] = float(arr_high[-1] - arr_low[-1]) if n >= 1 else 0.0
        features["hl_avg_5m"] = float(np.mean(arr_high[-5:] - arr_low[-5:])) if n >= 5 else 0.0

        # Volume features
        features["vol_1m"] = float(arr_vol[-1])
        features["vol_avg_5m"] = float(np.mean(arr_vol[-5:])) if n >= 5 else float(np.mean(arr_vol))
        features["vol_ratio_1m_avg5m"] = features["vol_1m"] / (features["vol_avg_5m"] + 1e-9)
        
        # Volume trend
        if n >= 5:
            vol_trend = np.polyfit(range(5), arr_vol[-5:], 1)[0]  # slope of volume trend
            features["vol_trend_5m"] = float(vol_trend)
        else:
            features["vol_trend_5m"] = 0.0

        # VWAP features
        features["vwap_last"] = float(arr_vwap[-1])
        features["vwap_gap_price"] = features["vwap_last"] - last
        
        # Price position within recent range
        if n >= 10:
            recent_high = float(np.max(arr_high[-10:]))
            recent_low = float(np.min(arr_low[-10:]))
            if recent_high > recent_low:
                features["price_position_10m"] = (last - recent_low) / (recent_high - recent_low)
            else:
                features["price_position_10m"] = 0.5
        else:
            features["price_position_10m"] = 0.5

        # Momentum indicators
        ups = downs = 0
        for i in range(1, min(n, 6)):
            if arr_close[-i] > arr_close[-i-1]:
                ups += 1
            elif arr_close[-i] < arr_close[-i-1]:
                downs += 1
        features["momentum_up_count_5"] = ups
        features["momentum_down_count_5"] = downs
        features["momentum_ratio"] = ups / (ups + downs + 1e-9)

        # RSI approximation
        if n >= 14:
            gains = np.maximum(np.diff(arr_close[-14:]), 0)
            losses = np.maximum(-np.diff(arr_close[-14:]), 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                features["rsi_14"] = float(rsi)
            else:
                features["rsi_14"] = 50.0
        else:
            features["rsi_14"] = 50.0

        return features

    def make_label(self, price_at_creation, future_price):
        ret = (future_price / price_at_creation) - 1.0
        threshold = self.config["label_threshold"]
        if ret > threshold:
            return "up"
        if ret < -threshold:
            return "down"
        return "neutral"

    def parse_message(self, msg_text):
        """Generic message parser"""
        try:
            m = json.loads(msg_text)
        except Exception:
            return None
        
        # Handle different message formats
        sym = m.get("symbol") or m.get("s") or m.get("sym")
        if not sym:
            return None
        
        price = m.get("price") or m.get("p")
        size = m.get("size") or m.get("v") or m.get("volume")
        ts = m.get("t") or m.get("timestamp") or m.get("time")
        
        if price is None:
            return None
            
        # Parse timestamp
        if ts is None:
            dt = datetime.utcnow()
        else:
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                else:
                    # Assume epoch milliseconds
                    dt = datetime.utcfromtimestamp(int(ts)/1000.0)
            except Exception:
                dt = datetime.utcnow()
                
        return {
            "symbol": sym, 
            "price": float(price), 
            "size": float(size) if size is not None else None, 
            "ts": dt
        }

    async def process_tick(self, msg_text):
        """Process a single tick message"""
        parsed = self.parse_message(msg_text)
        if not parsed or parsed["symbol"] != self.symbol:
            return
            
        self.tick_count += 1
        
        tick_time = parsed["ts"]
        price = parsed["price"]
        size = parsed["size"]

        # Align to minute start
        minute_ts = tick_time.replace(second=0, microsecond=0)

        # Handle bar creation and finalization
        if (self.current_bar is None) or (minute_ts != self.current_bar["ts"]):
            # Finalize previous bar
            if self.current_bar is not None:
                finalized = self.finalize_bar(self.current_bar)
                self.bars.append(finalized)
                
                # Process pending snapshots for labeling
                await self._process_pending_labels(finalized)
                
            # Start new bar
            self.current_bar = self.start_new_bar(minute_ts, price, size)
        else:
            # Update current bar
            self.update_bar(self.current_bar, price, size)

        # Generate features and prediction
        await self._make_prediction(minute_ts, price)
        
        # Log progress
        if self.tick_count % 100 == 0:
            acc = self.metric_acc.get() if hasattr(self.metric_acc, 'get') else 0
            print(f"Processed {self.tick_count} ticks, {self.prediction_count} predictions, accuracy: {acc:.4f}")

    async def _process_pending_labels(self, finalized_bar):
        """Process pending snapshots for labeling"""
        now_bar_time = finalized_bar["ts"]
        
        while (self.pending_snapshots and 
               self.pending_snapshots[0]["target_time"] <= now_bar_time):
            
            snapshot = self.pending_snapshots.popleft()
            price_then = snapshot["price_at_creation"]
            future_price = finalized_bar["close"]
            y = self.make_label(price_then, future_price)
            x = snapshot["features_at_creation"]
            
            # Train model
            try:
                # Get prediction before learning
                y_pred = self.model.predict_one(x)
                
                # Learn from this example
                self.model.learn_one(x, y)
                
                # Update metrics
                self.metric_acc.update(y, y_pred)
                self.metric_bal.update(y, y_pred)
                
                print(f"[LEARN] {snapshot['created_at'].strftime('%H:%M:%S')} "
                      f"label={y} pred={y_pred} acc={self.metric_acc.get():.4f}")
                      
            except Exception as e:
                print(f"Learning error: {e}")

    async def _make_prediction(self, minute_ts, price):
        """Generate prediction for current state"""
        # Build feature window
        window_size = self.config["feature_window_minutes"]
        window_bars = list(self.bars)[-(window_size-1):] if len(self.bars) >= (window_size-1) else list(self.bars)
        
        # Add current bar simulation
        if self.current_bar:
            simulated_current = {
                "ts": self.current_bar["ts"],
                "open": self.current_bar["open"],
                "high": self.current_bar["high"],
                "low": self.current_bar["low"],
                "close": self.current_bar["close"],
                "volume": self.current_bar["volume"],
                "vwap": (self.current_bar["vwap_numer"] / self.current_bar["vwap_denom"]) 
                        if self.current_bar["vwap_denom"] > 0 else self.current_bar["close"]
            }
            window_bars.append(simulated_current)
        
        if len(window_bars) > window_size:
            window_bars = window_bars[-window_size:]

        # Compute features
        feats = self.compute_features_from_bars(window_bars)
        
        # Make prediction
        try:
            pred = self.model.predict_one(feats)
            proba = self.model.predict_proba_one(feats)
        except Exception as e:
            pred = "neutral"
            proba = {"up": 0.33, "neutral": 0.34, "down": 0.33}
            
        self.prediction_count += 1
        
        # Create snapshot for future labeling
        pred_horizon = timedelta(minutes=self.config["pred_horizon_minutes"])
        snapshot = {
            "created_at": minute_ts,
            "features_at_creation": feats,
            "price_at_creation": float(price),
            "target_time": minute_ts + pred_horizon
        }
        self.pending_snapshots.append(snapshot)
        
        # Keep pending snapshots bounded
        while len(self.pending_snapshots) > 500:
            self.pending_snapshots.popleft()
            
        # Log prediction
        proba_str = f"{proba}" if isinstance(proba, dict) else str(proba)
        print(f"[PRED] {minute_ts.strftime('%H:%M:%S')} price={price:.4f} "
              f"pred={pred} proba={proba_str} features={len(window_bars)}min")

    def get_stats(self):
        """Get current statistics"""
        return {
            "ticks_processed": self.tick_count,
            "predictions_made": self.prediction_count,
            "accuracy": self.metric_acc.get() if hasattr(self.metric_acc, 'get') else 0,
            "balanced_accuracy": self.metric_bal.get() if hasattr(self.metric_bal, 'get') else 0,
            "pending_labels": len(self.pending_snapshots),
            "bars_stored": len(self.bars)
        }

async def run_live_websocket(provider_name, symbol, api_key=None, secret_key=None):
    """Run with live WebSocket data"""
    if provider_name not in PROVIDER_CONFIGS:
        raise ValueError(f"Unknown provider: {provider_name}")
        
    provider_config = PROVIDER_CONFIGS[provider_name]
    predictor = TrendPredictor(symbol)
    
    print(f"Connecting to {provider_name} for {symbol}")
    
    async with websockets.connect(provider_config.WEBSOCKET_URL, ping_interval=20) as ws:
        # Authentication if needed
        if hasattr(provider_config, 'get_auth_message') and api_key:
            auth_msg = provider_config.get_auth_message(api_key, secret_key)
            await ws.send(auth_msg)
            
        # Subscribe to symbol
        if hasattr(provider_config, 'get_subscribe_message'):
            sub_msg = provider_config.get_subscribe_message([symbol])
            await ws.send(sub_msg)
            
        # Process messages
        async for raw_msg in ws:
            await predictor.process_tick(raw_msg)

async def run_simulation(mode="mock", csv_path=None, symbol="AAPL", max_ticks=1000):
    """Run in simulation mode"""
    predictor = TrendPredictor(symbol)
    
    if mode == "mock":
        data_source = create_mock_data_source(symbol)
    elif mode == "csv" and csv_path:
        data_source = create_historical_data_source(csv_path, playback_speed=10.0)
    else:
        # Create sample CSV first
        csv_path = "sample_stock_data.csv"
        create_sample_csv(csv_path, num_rows=max_ticks)
        data_source = create_historical_data_source(csv_path, playback_speed=10.0)
    
    await simulate_websocket_feed(data_source, predictor.process_tick, max_ticks)
    
    # Print final statistics
    stats = predictor.get_stats()
    print("\n=== SIMULATION COMPLETE ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

def main():
    parser = argparse.ArgumentParser(description="5-minute stock trend predictor")
    parser.add_argument("--mode", choices=["live", "simulation"], default="simulation",
                      help="Run mode: live WebSocket or simulation")
    parser.add_argument("--provider", choices=["alpaca", "binance", "polygon", "iex"], 
                      default="alpaca", help="WebSocket provider")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("--api-key", help="API key for WebSocket provider")
    parser.add_argument("--secret-key", help="Secret key for WebSocket provider")
    parser.add_argument("--csv-path", help="Path to CSV file for historical simulation")
    parser.add_argument("--max-ticks", type=int, default=1000, 
                      help="Maximum ticks for simulation mode")
    parser.add_argument("--sim-mode", choices=["mock", "csv"], default="mock",
                      help="Simulation data source")
    
    args = parser.parse_args()
    
    if args.mode == "live":
        asyncio.run(run_live_websocket(
            args.provider, args.symbol, args.api_key, args.secret_key
        ))
    else:
        asyncio.run(run_simulation(
            args.sim_mode, args.csv_path, args.symbol, args.max_ticks
        ))

if __name__ == "__main__":
    main()