# real_time_dashboard.py
"""
Real-time stock prediction dashboard with live market data
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import yfinance as yf
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'stock-predictor-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class RealTimePredictor:
    def __init__(self):
        self.current_symbol = None
        self.is_running = False
        self.price_history = deque(maxlen=50)
        self.predictions = deque(maxlen=20)
        self.performance = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0
        }
        self.prediction_thread = None
        
    def start_prediction(self, symbol):
        """Start real-time prediction for a symbol"""
        if self.is_running:
            return False, "Already running"
            
        self.current_symbol = symbol.upper()
        self.is_running = True
        self.price_history.clear()
        self.predictions.clear()
        
        # Start prediction thread
        self.prediction_thread = threading.Thread(
            target=self._prediction_loop, 
            daemon=True
        )
        self.prediction_thread.start()
        
        logger.info(f"Started prediction for {symbol}")
        return True, f"Started prediction for {symbol}"
    
    def stop_prediction(self):
        """Stop real-time prediction"""
        self.is_running = False
        self.current_symbol = None
        logger.info("Stopped prediction")
        return True, "Prediction stopped"
    
    def _get_real_time_data(self, symbol):
        """Fetch real-time market data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get recent data (last 2 days, 1-minute intervals)
            hist = ticker.history(period="2d", interval="1m")
            if hist.empty:
                return None
                
            # Get the latest price
            latest = hist.iloc[-1]
            
            # Get basic info
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': float(latest['Close']),
                'volume': int(latest['Volume']) if pd.notna(latest['Volume']) else 0,
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'open': float(latest['Open']),
                'timestamp': datetime.now(),
                'history': hist,
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A')
            }
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _compute_technical_indicators(self, price_data):
        """Compute technical indicators from price history"""
        if len(price_data) < 10:
            return {}
            
        try:
            df = price_data.copy()
            close_prices = df['Close'].values
            volumes = df['Volume'].values
            
            # Moving averages
            ma_5 = np.mean(close_prices[-5:]) if len(close_prices) >= 5 else close_prices[-1]
            ma_10 = np.mean(close_prices[-10:]) if len(close_prices) >= 10 else close_prices[-1]
            ma_20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else close_prices[-1]
            
            # Returns
            ret_1 = (close_prices[-1] / close_prices[-2] - 1) if len(close_prices) >= 2 else 0
            ret_5 = (close_prices[-1] / close_prices[-6] - 1) if len(close_prices) >= 6 else 0
            ret_10 = (close_prices[-1] / close_prices[-11] - 1) if len(close_prices) >= 11 else 0
            
            # Volatility (standard deviation of returns)
            if len(close_prices) >= 10:
                returns = np.diff(close_prices[-10:]) / close_prices[-11:-1]
                volatility = np.std(returns)
            else:
                volatility = 0
            
            # RSI calculation
            def calculate_rsi(prices, window=14):
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
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            rsi = calculate_rsi(close_prices)
            
            # Volume indicators
            vol_ma = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1] if len(volumes) > 0 else 1
            vol_ratio = volumes[-1] / vol_ma if vol_ma > 0 else 1
            
            return {
                'price': float(close_prices[-1]),
                'ma_5': float(ma_5),
                'ma_10': float(ma_10),
                'ma_20': float(ma_20),
                'return_1min': float(ret_1),
                'return_5min': float(ret_5),
                'return_10min': float(ret_10),
                'volatility': float(volatility),
                'rsi': float(rsi),
                'volume_ratio': float(vol_ratio),
                'price_ma5_ratio': float(close_prices[-1] / ma_5) if ma_5 != 0 else 1.0,
                'ma5_ma20_ratio': float(ma_5 / ma_20) if ma_20 != 0 else 1.0
            }
        except Exception as e:
            logger.error(f"Error computing indicators: {e}")
            return {}
    
    def _make_prediction(self, features):
        """Make prediction based on technical indicators"""
        if not features:
            return "NEUTRAL", 0.33
            
        try:
            # Simple rule-based prediction model
            score = 0
            
            # Recent return signal
            score += features.get('return_1min', 0) * 3.0
            score += features.get('return_5min', 0) * 2.0
            
            # Moving average signals
            price_ma_signal = (features.get('price_ma5_ratio', 1.0) - 1.0) * 2.0
            ma_trend_signal = (features.get('ma5_ma20_ratio', 1.0) - 1.0) * 1.0
            score += price_ma_signal + ma_trend_signal
            
            # RSI signal (mean reversion)
            rsi = features.get('rsi', 50)
            if rsi > 70:  # Overbought
                score -= 0.5
            elif rsi < 30:  # Oversold
                score += 0.5
            
            # Volume confirmation
            vol_ratio = features.get('volume_ratio', 1.0)
            if vol_ratio > 1.5:  # High volume confirms signal
                score *= 1.2
            elif vol_ratio < 0.5:  # Low volume weakens signal
                score *= 0.8
            
            # Volatility adjustment
            volatility = features.get('volatility', 0)
            if volatility > 0.02:  # High volatility = less confidence
                score *= 0.9
            
            # Convert score to prediction
            threshold = 0.002
            if score > threshold:
                prediction = "UP"
                confidence = min(0.85, 0.5 + abs(score) * 10)
            elif score < -threshold:
                prediction = "DOWN" 
                confidence = min(0.85, 0.5 + abs(score) * 10)
            else:
                prediction = "NEUTRAL"
                confidence = 0.4
                
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return "NEUTRAL", 0.33
    
    def _update_performance(self, actual_price, predicted_direction, prev_price):
        """Update performance metrics"""
        if prev_price is None:
            return
            
        try:
            actual_change = actual_price - prev_price
            actual_direction = "UP" if actual_change > 0.01 else "DOWN" if actual_change < -0.01 else "NEUTRAL"
            
            self.performance['total_predictions'] += 1
            
            if predicted_direction == actual_direction:
                self.performance['correct_predictions'] += 1
            
            if self.performance['total_predictions'] > 0:
                self.performance['accuracy'] = (
                    self.performance['correct_predictions'] / 
                    self.performance['total_predictions']
                )
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def _prediction_loop(self):
        """Main prediction loop"""
        logger.info(f"Starting prediction loop for {self.current_symbol}")
        prev_price = None
        last_prediction = None
        
        while self.is_running:
            try:
                # Get real-time market data
                market_data = self._get_real_time_data(self.current_symbol)
                if not market_data:
                    time.sleep(10)
                    continue
                
                current_price = market_data['price']
                
                # Store price history
                price_point = {
                    'timestamp': market_data['timestamp'].isoformat(),
                    'price': current_price,
                    'volume': market_data['volume']
                }
                self.price_history.append(price_point)
                
                # Compute technical indicators
                features = self._compute_technical_indicators(market_data['history'])
                
                # Make prediction
                prediction, confidence = self._make_prediction(features)
                
                # Update performance if we have a previous prediction
                if last_prediction and prev_price:
                    self._update_performance(current_price, last_prediction, prev_price)
                
                # Prepare data to send to frontend
                update_data = {
                    'symbol': self.current_symbol,
                    'price': current_price,
                    'volume': market_data['volume'],
                    'high': market_data['high'],
                    'low': market_data['low'],
                    'open': market_data['open'],
                    'timestamp': market_data['timestamp'].isoformat(),
                    'prediction': prediction,
                    'confidence': confidence,
                    'features': features,
                    'performance': self.performance.copy(),
                    'market_cap': market_data.get('market_cap', 'N/A'),
                    'pe_ratio': market_data.get('pe_ratio', 'N/A')
                }
                
                # Send update to frontend
                socketio.emit('market_update', update_data)
                
                # Log prediction
                logger.info(f"{self.current_symbol}: ${current_price:.2f} -> {prediction} ({confidence:.2f})")
                
                # Store for next iteration
                prev_price = current_price
                last_prediction = prediction
                
                # Wait before next update (adjust frequency as needed)
                time.sleep(15)  # Update every 15 seconds
                
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                time.sleep(10)
        
        logger.info("Prediction loop stopped")

# Global predictor instance
predictor = RealTimePredictor()

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('index.html')

@app.route('/api/start/<symbol>')
def start_prediction(symbol):
    """Start prediction for a symbol"""
    success, message = predictor.start_prediction(symbol)
    return jsonify({
        'status': 'success' if success else 'error',
        'message': message
    })

@app.route('/api/stop')
def stop_prediction():
    """Stop prediction"""
    success, message = predictor.stop_prediction()
    return jsonify({
        'status': 'success' if success else 'error',
        'message': message
    })

@app.route('/api/status')
def get_status():
    """Get current status"""
    return jsonify({
        'is_running': predictor.is_running,
        'symbol': predictor.current_symbol,
        'performance': predictor.performance
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected to WebSocket")
    emit('connected', {'message': 'Connected to real-time stock prediction server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected from WebSocket")

if __name__ == '__main__':
    print("🚀 Starting Real-time Stock Prediction Dashboard")
    print("🌐 Open your browser to: http://localhost:5000")
    print("📊 Select a stock symbol to start live predictions!")
    print("⚡ Using real market data from Yahoo Finance")
    
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)