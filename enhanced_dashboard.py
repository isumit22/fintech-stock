#!/usr/bin/env python3
"""
ENHANCED Fintech Stock Prediction Dashboard
Production-ready version with professional features for fintech-stock repo
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import yfinance as yf
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime
from collections import deque
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fintech-stock-enhanced-predictor'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class EnhancedStockPredictor:
    def __init__(self):
        self.current_symbol = None
        self.display_symbol = None
        self.is_running = False
        self.thread = None
        self.price_history = deque(maxlen=50)
        self.predictions = deque(maxlen=100)
        self.recent_predictions = deque(maxlen=50)  # Rolling window
        
        # Enhanced performance tracking
        self.performance = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'rolling_accuracy': 0.0,
            'rolling_predictions': 0,
            'rolling_correct': 0,
            'is_warming_up': True,
            'warmup_threshold': 10
        }
        
        self.last_prediction_time = None
        self.prediction_interval = 12  # Optimized for fintech
        
    def start_prediction(self, symbol):
        """Start prediction with enhanced error handling"""
        logger.info(f"🚀 FINTECH ENHANCED START for {symbol}")
        
        try:
            # Stop current if running
            if self.is_running:
                logger.info(f"⏹️ Stopping current prediction")
                self.stop_prediction()
                time.sleep(1)
            
            # Validate symbol for Indian market
            if '.' not in symbol:
                yahoo_symbol = f"{symbol}.NS"
                display_symbol = symbol
            else:
                yahoo_symbol = symbol
                display_symbol = symbol.split('.')[0]
                
            # Quick validation
            test_ticker = yf.Ticker(yahoo_symbol)
            test_hist = test_ticker.history(period="1d", interval="5m")
            if test_hist.empty:
                test_hist = test_ticker.history(period="1d", interval="1d")
                if test_hist.empty:
                    return False, f"No data available for {symbol}"
            
            self.current_symbol = yahoo_symbol
            self.display_symbol = display_symbol
            self.is_running = True
            
            # Clear history for new symbol
            self.price_history.clear()
            self.performance['is_warming_up'] = True
            
            # Start enhanced thread
            self.thread = threading.Thread(target=self._enhanced_prediction_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"✅ Fintech enhanced prediction started for {display_symbol}")
            return True, f"Started enhanced prediction for {display_symbol}"
            
        except Exception as e:
            logger.error(f"❌ Error starting prediction: {e}")
            self.is_running = False
            return False, f"Error: {str(e)}"
    
    def stop_prediction(self):
        """Stop prediction gracefully"""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        
        # Update session stats
        if self.performance['total_predictions'] > 0:
            logger.info(f"📊 Fintech session ended: {self.performance['total_predictions']} predictions, "
                       f"{self.performance['accuracy']:.1%} accuracy")
        
        logger.info("⏹️ Enhanced fintech prediction stopped")
        return True, "Stopped"
    
    def _enhanced_prediction_loop(self):
        """Enhanced prediction loop with fintech-grade features"""
        logger.info(f"🔄 FINTECH ENHANCED LOOP STARTED for {self.current_symbol}")
        
        iteration = 0
        prev_price = None
        last_prediction = None
        consecutive_errors = 0
        max_errors = 3
        
        while self.is_running:
            try:
                iteration += 1
                logger.info(f"📊 Fintech Iteration {iteration} - {self.current_symbol}")
                
                # Fetch enhanced data
                market_data = self._fetch_enhanced_data()
                if not market_data:
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        logger.error("🛑 Too many errors, stopping fintech loop")
                        break
                    time.sleep(5)
                    continue
                
                consecutive_errors = 0
                current_price = market_data['price']
                
                # Store price history
                price_point = {
                    'timestamp': market_data['timestamp'].isoformat(),
                    'price': current_price,
                    'volume': market_data['volume']
                }
                self.price_history.append(price_point)
                
                # Enhanced technical analysis
                features = self._compute_enhanced_features(market_data['history'])
                
                # Enhanced prediction with calibration
                prediction, raw_confidence, calibrated_confidence = self._make_enhanced_prediction(features)
                
                # Update enhanced performance
                if last_prediction and prev_price:
                    self._update_enhanced_performance(current_price, last_prediction, prev_price)
                
                # Update prediction time
                self.last_prediction_time = datetime.now()
                
                # Prepare enhanced update for fintech
                update_data = {
                    'symbol': self.display_symbol,
                    'yahoo_symbol': self.current_symbol,
                    'price': current_price,
                    'volume': market_data['volume'],
                    'high': market_data['high'],
                    'low': market_data['low'],
                    'open': market_data['open'],
                    'timestamp': market_data['timestamp'].isoformat(),
                    'prediction': prediction,
                    'confidence': calibrated_confidence,
                    'raw_confidence': raw_confidence,
                    'features': features,
                    'performance': self.performance.copy(),
                    'market_cap': market_data.get('market_cap', 'N/A'),
                    'pe_ratio': market_data.get('pe_ratio', 'N/A'),
                    'sector': market_data.get('sector', 'N/A'),
                    'last_prediction_time': self.last_prediction_time.isoformat(),
                    'next_prediction_in': self.prediction_interval,
                    'iteration': iteration,
                    'system_status': 'WARMING_UP' if self.performance['is_warming_up'] else 'ACTIVE'
                }
                
                logger.info(f"💰 {self.display_symbol}: ₹{current_price:.2f} -> {prediction} "
                           f"({calibrated_confidence:.1%}, raw: {raw_confidence:.1%})")
                logger.info(f"📡 Sending fintech update...")
                
                # Send to frontend
                socketio.emit('market_update', update_data)
                
                # Store for next iteration
                prev_price = current_price
                last_prediction = prediction
                
                # Enhanced wait
                logger.info(f"⏱️ Waiting {self.prediction_interval}s... (fintech iteration {iteration})")
                time.sleep(self.prediction_interval)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"💥 Fintech enhanced loop error: {e}")
                if consecutive_errors >= max_errors:
                    break
                time.sleep(5)
        
        logger.info(f"🏁 FINTECH ENHANCED LOOP ENDED for {self.current_symbol}")
        self.is_running = False
    
    def _fetch_enhanced_data(self):
        """Enhanced data fetching for fintech"""
        try:
            ticker = yf.Ticker(self.current_symbol)
            
            # Try multiple intervals for best data
            hist = None
            for interval in ["5m", "1m", "15m"]:
                for period in ["1d", "2d"]:
                    try:
                        hist = ticker.history(period=period, interval=interval)
                        if not hist.empty:
                            break
                    except:
                        continue
                if hist is not None and not hist.empty:
                    break
            
            if hist is None or hist.empty:
                return None
            
            latest = hist.iloc[-1]
            
            # Get enhanced company info for fintech
            try:
                info = ticker.info
            except:
                info = {}
            
            return {
                'price': float(latest['Close']),
                'volume': int(latest['Volume']) if pd.notna(latest['Volume']) else 0,
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'open': float(latest['Open']),
                'timestamp': datetime.now(),
                'history': hist,
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'sector': info.get('sector', 'N/A')
            }
            
        except Exception as e:
            logger.error(f"Fintech data fetch error: {e}")
            return None
    
    def _compute_enhanced_features(self, hist):
        """Enhanced technical indicators for fintech"""
        if len(hist) < 5:
            return {'basic': True}
        
        close_prices = hist['Close'].values
        volumes = hist['Volume'].values
        
        # Moving averages (fintech optimized)
        ma_5 = np.mean(close_prices[-5:])
        ma_10 = np.mean(close_prices[-10:]) if len(close_prices) >= 10 else ma_5
        ma_20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else ma_5
        
        # Returns analysis
        ret_1 = (close_prices[-1] / close_prices[-2] - 1) if len(close_prices) >= 2 else 0
        ret_5 = (close_prices[-1] / close_prices[-6] - 1) if len(close_prices) >= 6 else 0
        
        # RSI (Relative Strength Index)
        def calc_rsi(prices, window=14):
            if len(prices) < window + 1:
                return 50
            deltas = np.diff(prices[-window-1:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss == 0:
                return 100
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        
        rsi = calc_rsi(close_prices)
        
        # Volume analysis (fintech specific)
        vol_ma = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
        vol_ratio = volumes[-1] / vol_ma if vol_ma > 0 else 1
        
        return {
            'price_inr': float(close_prices[-1]),
            'ma_5': float(ma_5),
            'ma_10': float(ma_10),
            'ma_20': float(ma_20),
            'return_1period': float(ret_1),
            'return_5period': float(ret_5),
            'rsi': float(rsi),
            'volume_ratio': float(vol_ratio),
            'price_ma5_ratio': float(close_prices[-1] / ma_5),
            'ma5_ma20_ratio': float(ma_5 / ma_20) if ma_20 != 0 else 1.0
        }
    
    def _make_enhanced_prediction(self, features):
        """Enhanced prediction algorithm for fintech"""
        if not features or features.get('basic'):
            return "NEUTRAL", 0.35, 0.35
        
        # Enhanced scoring for fintech
        score = 0
        
        # Recent returns (fintech calibrated)
        score += features.get('return_1period', 0) * 3.5
        score += features.get('return_5period', 0) * 2.2
        
        # Moving average signals
        price_ma_signal = (features.get('price_ma5_ratio', 1.0) - 1.0) * 2.2
        ma_trend_signal = (features.get('ma5_ma20_ratio', 1.0) - 1.0) * 1.8
        score += price_ma_signal + ma_trend_signal
        
        # RSI signals (fintech tuned)
        rsi = features.get('rsi', 50)
        if rsi > 70:
            score -= 1.2
        elif rsi < 30:
            score += 1.2
        
        # Volume confirmation (fintech enhanced)
        vol_ratio = features.get('volume_ratio', 1.0)
        if vol_ratio > 1.5:
            score *= 1.4
        elif vol_ratio < 0.6:
            score *= 0.6
        
        # Convert to prediction (fintech thresholds)
        threshold = 0.004  # Fintech optimized threshold
        if score > threshold:
            prediction = "UP"
            raw_confidence = min(0.88, 0.52 + abs(score) * 12)
        elif score < -threshold:
            prediction = "DOWN"
            raw_confidence = min(0.88, 0.52 + abs(score) * 12)
        else:
            prediction = "NEUTRAL"
            raw_confidence = 0.42
        
        # Calibrate confidence
        calibrated_confidence = self._calibrate_confidence(raw_confidence)
        
        return prediction, raw_confidence, calibrated_confidence
    
    def _calibrate_confidence(self, raw_confidence):
        """Calibrate confidence for fintech reliability"""
        if self.performance['is_warming_up']:
            return min(raw_confidence * 0.75, 0.68)
        
        rolling_acc = self.performance.get('rolling_accuracy', 0.5)
        
        # Fintech calibration factors
        if rolling_acc > 0.70:
            factor = 1.15
        elif rolling_acc > 0.60:
            factor = 1.05
        elif rolling_acc > 0.50:
            factor = 0.95
        else:
            factor = 0.75
        
        return max(0.32, min(0.88, raw_confidence * factor))
    
    def _update_enhanced_performance(self, actual_price, predicted_direction, prev_price):
        """Enhanced performance tracking for fintech"""
        try:
            actual_change = actual_price - prev_price
            # Fintech threshold for direction changes
            actual_direction = "UP" if actual_change > 0.8 else "DOWN" if actual_change < -0.8 else "NEUTRAL"
            
            # Global counters
            self.performance['total_predictions'] += 1
            is_correct = (predicted_direction == actual_direction)
            
            if is_correct:
                self.performance['correct_predictions'] += 1
            
            # Global accuracy
            self.performance['accuracy'] = (
                self.performance['correct_predictions'] / self.performance['total_predictions']
            )
            
            # Rolling window for fintech
            prediction_result = {
                'prediction': predicted_direction,
                'actual': actual_direction,
                'correct': is_correct,
                'timestamp': datetime.now()
            }
            
            self.recent_predictions.append(prediction_result)
            
            # Rolling metrics
            rolling_correct = sum(1 for p in self.recent_predictions if p['correct'])
            self.performance['rolling_predictions'] = len(self.recent_predictions)
            self.performance['rolling_correct'] = rolling_correct
            self.performance['rolling_accuracy'] = rolling_correct / len(self.recent_predictions)
            
            # Warm-up check
            self.performance['is_warming_up'] = (
                self.performance['total_predictions'] < self.performance['warmup_threshold']
            )
            
            logger.info(f"📈 Fintech Performance: {predicted_direction} vs {actual_direction} "
                       f"(Rolling: {self.performance['rolling_accuracy']:.1%})")
            
        except Exception as e:
            logger.error(f"Fintech performance update error: {e}")

# Global predictor
predictor = EnhancedStockPredictor()

@app.route('/')
def index():
    return render_template('enhanced_dashboard.html')

@app.route('/api/start/<symbol>')
def start_prediction(symbol):
    logger.info(f"📞 Fintech Enhanced API call: start/{symbol}")
    success, message = predictor.start_prediction(symbol)
    return jsonify({
        'status': 'success' if success else 'error',
        'message': message,
        'is_running': predictor.is_running,
        'symbol': predictor.display_symbol
    })

@app.route('/api/stop')
def stop_prediction():
    logger.info(f"📞 Fintech Enhanced API call: stop")
    success, message = predictor.stop_prediction()
    return jsonify({
        'status': 'success' if success else 'error',
        'message': message
    })

@app.route('/api/status')
def get_status():
    return jsonify({
        'is_running': predictor.is_running,
        'symbol': predictor.current_symbol,
        'display_symbol': predictor.display_symbol,
        'performance': predictor.performance,
        'last_prediction_time': predictor.last_prediction_time.isoformat() if predictor.last_prediction_time else None
    })

@socketio.on('connect')
def handle_connect():
    logger.info("🔌 Fintech Enhanced client connected")
    emit('connected', {'message': 'Connected to Fintech Stock Predictor'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("🔌 Fintech Enhanced client disconnected")

if __name__ == '__main__':
    print("🏦 Fintech Stock Prediction Dashboard - Enhanced")
    print("🌐 Open: http://localhost:5003")
    print("✨ Production-ready with professional fintech features")
    
    socketio.run(app, debug=False, host='0.0.0.0', port=5003)