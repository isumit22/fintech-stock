# app.py
"""
Flask web application for real-time stock trend prediction visualization
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import asyncio
import threading
import json
import time
from datetime import datetime, timedelta
from collections import deque
import yfinance as yf
import pandas as pd
import numpy as np
from enhanced_trend_predictor import EnhancedTrendPredictor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global data storage
class DataManager:
    def __init__(self):
        self.current_data = {}
        self.price_history = deque(maxlen=100)
        self.predictions = deque(maxlen=50)
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'daily_return': 0.0
        }
        self.predictor = None
        self.is_running = False
        
data_manager = DataManager()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/symbols')
def get_symbols():
    """Get list of available symbols"""
    popular_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
        'NVDA', 'META', 'NFLX', 'DIS', 'UBER'
    ]
    return jsonify(popular_symbols)

@app.route('/api/start/<symbol>')
def start_prediction(symbol):
    """Start real-time prediction for a symbol"""
    try:
        if data_manager.is_running:
            return jsonify({'status': 'error', 'message': 'Already running'})
        
        # Initialize predictor
        data_manager.predictor = EnhancedTrendPredictor()
        data_manager.is_running = True
        
        # Start background thread for real-time data
        thread = threading.Thread(
            target=run_realtime_prediction, 
            args=(symbol,),
            daemon=True
        )
        thread.start()
        
        return jsonify({'status': 'success', 'message': f'Started prediction for {symbol}'})
    except Exception as e:
        logger.error(f"Error starting prediction: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stop')
def stop_prediction():
    """Stop real-time prediction"""
    data_manager.is_running = False
    return jsonify({'status': 'success', 'message': 'Stopped prediction'})

@app.route('/api/current_data')
def get_current_data():
    """Get current market data and prediction"""
    return jsonify(data_manager.current_data)

@app.route('/api/history')
def get_history():
    """Get price history for charts"""
    return jsonify({
        'prices': list(data_manager.price_history),
        'predictions': list(data_manager.predictions),
        'performance': data_manager.performance_metrics
    })

def get_real_market_data(symbol):
    """Fetch real market data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get current price and recent data
        info = ticker.info
        hist = ticker.history(period="1d", interval="1m")
        
        if hist.empty:
            return None
            
        latest = hist.iloc[-1]
        current_price = float(latest['Close'])
        volume = float(latest['Volume'])
        
        # Get extended data for features
        hist_5d = ticker.history(period="5d", interval="1m")
        
        return {
            'symbol': symbol,
            'price': current_price,
            'volume': volume,
            'timestamp': datetime.now(),
            'history': hist_5d,
            'info': info
        }
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

def compute_technical_indicators(price_data):
    """Compute technical indicators from price data"""
    if len(price_data) < 20:
        return {}
    
    df = pd.DataFrame(price_data)
    close_prices = df['Close'].values
    
    # Simple indicators
    ma_5 = np.mean(close_prices[-5:])
    ma_10 = np.mean(close_prices[-10:])
    ma_20 = np.mean(close_prices[-20:])
    
    # Returns
    ret_1 = (close_prices[-1] / close_prices[-2] - 1) if len(close_prices) >= 2 else 0
    ret_5 = (close_prices[-1] / close_prices[-6] - 1) if len(close_prices) >= 6 else 0
    
    # Volatility
    volatility = np.std(close_prices[-10:]) if len(close_prices) >= 10 else 0
    
    # RSI (simplified)
    def simple_rsi(prices, window=14):
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
    
    rsi = simple_rsi(close_prices)
    
    return {
        'ma_5': float(ma_5),
        'ma_10': float(ma_10),
        'ma_20': float(ma_20),
        'return_1': float(ret_1),
        'return_5': float(ret_5),
        'volatility': float(volatility),
        'rsi': float(rsi),
        'price_ma5_ratio': float(close_prices[-1] / ma_5) if ma_5 != 0 else 1.0,
        'ma5_ma20_ratio': float(ma_5 / ma_20) if ma_20 != 0 else 1.0
    }

def run_realtime_prediction(symbol):
    """Run real-time prediction in background thread"""
    logger.info(f"Starting real-time prediction for {symbol}")
    
    prediction_history = deque(maxlen=20)
    
    while data_manager.is_running:
        try:
            # Get real market data
            market_data = get_real_market_data(symbol)
            if not market_data:
                time.sleep(5)
                continue
            
            # Store price data
            price_point = {
                'timestamp': market_data['timestamp'].isoformat(),
                'price': market_data['price'],
                'volume': market_data['volume']
            }
            data_manager.price_history.append(price_point)
            
            # Compute technical features
            if not market_data['history'].empty:
                features = compute_technical_indicators(market_data['history'])
                
                if features:
                    # Make prediction using simple rule-based approach
                    prediction_score = (
                        features.get('return_1', 0) * 0.4 +
                        features.get('return_5', 0) * 0.3 +
                        (features.get('rsi', 50) - 50) / 100 * 0.3
                    )
                    
                    if prediction_score > 0.001:
                        prediction = "UP"
                        confidence = min(0.8, 0.5 + abs(prediction_score) * 10)
                    elif prediction_score < -0.001:
                        prediction = "DOWN"
                        confidence = min(0.8, 0.5 + abs(prediction_score) * 10)
                    else:
                        prediction = "NEUTRAL"
                        confidence = 0.4
                    
                    # Store prediction
                    pred_data = {
                        'timestamp': market_data['timestamp'].isoformat(),
                        'prediction': prediction,
                        'confidence': confidence,
                        'price': market_data['price']
                    }
                    
                    data_manager.predictions.append(pred_data)
                    prediction_history.append(pred_data)
                    
                    # Update performance metrics (simplified)
                    if len(prediction_history) >= 2:
                        data_manager.performance_metrics['total_predictions'] += 1
                        
                        # Simple accuracy check (if price moved in predicted direction)
                        prev_pred = prediction_history[-2]
                        price_change = market_data['price'] - prev_pred['price']
                        
                        if prev_pred['prediction'] == 'UP' and price_change > 0:
                            data_manager.performance_metrics['correct_predictions'] += 1
                        elif prev_pred['prediction'] == 'DOWN' and price_change < 0:
                            data_manager.performance_metrics['correct_predictions'] += 1
                        elif prev_pred['prediction'] == 'NEUTRAL' and abs(price_change) < 0.1:
                            data_manager.performance_metrics['correct_predictions'] += 1
                        
                        # Update accuracy
                        if data_manager.performance_metrics['total_predictions'] > 0:
                            data_manager.performance_metrics['accuracy'] = (
                                data_manager.performance_metrics['correct_predictions'] / 
                                data_manager.performance_metrics['total_predictions']
                            )
                    
                    # Update current data
                    data_manager.current_data = {
                        'symbol': symbol,
                        'price': market_data['price'],
                        'volume': market_data['volume'],
                        'timestamp': market_data['timestamp'].isoformat(),
                        'prediction': prediction,
                        'confidence': confidence,
                        'features': features,
                        'performance': data_manager.performance_metrics
                    }
                    
                    # Emit real-time update to frontend
                    socketio.emit('market_update', data_manager.current_data)
                    
                    logger.info(f"{symbol}: ${market_data['price']:.2f} -> {prediction} ({confidence:.2f})")
            
            # Wait before next update
            time.sleep(10)  # Update every 10 seconds
            
        except Exception as e:
            logger.error(f"Error in real-time prediction: {e}")
            time.sleep(5)
    
    logger.info("Real-time prediction stopped")

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info("Client connected")
    emit('connected', {'status': 'Connected to real-time feed'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info("Client disconnected")

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)