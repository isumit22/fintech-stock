# indian_stocks_dashboard.py
"""
Real-time Indian stock prediction dashboard (NSE/BSE) with live market data
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
app.config['SECRET_KEY'] = 'indian-stock-predictor-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class IndianStockPredictor:
    def __init__(self):
        self.current_symbol = None
        self.is_running = False
        self.price_history = deque(maxlen=50)
        self.predictions = deque(maxlen=100)  # Store more predictions for rolling metrics
        self.recent_predictions = deque(maxlen=50)  # Rolling window for recent performance
        self.performance = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'rolling_accuracy': 0.0,
            'rolling_predictions': 0,
            'rolling_correct': 0,
            'is_warming_up': True,
            'warmup_threshold': 10,
            'confidence_calibration': {'low': 0.4, 'medium': 0.6, 'high': 0.8}
        }
        self.prediction_thread = None
        self.last_prediction_time = None
        self.prediction_interval = 30  # seconds
        
    def start_prediction(self, symbol):
        """Start real-time prediction for an Indian stock"""
        try:
            # If already running, stop first
            if self.is_running:
                logger.info(f"Stopping current prediction for {self.current_symbol} to start {symbol}")
                self.stop_prediction()
                time.sleep(2)  # Give more time for cleanup
                
            # Convert to Yahoo Finance format for Indian stocks
            if '.' not in symbol:
                if symbol.endswith('NS'):
                    yahoo_symbol = symbol
                else:
                    yahoo_symbol = f"{symbol}.NS"  # NSE suffix
            else:
                yahoo_symbol = symbol
                
            # Test if symbol is valid by fetching data
            test_ticker = yf.Ticker(yahoo_symbol)
            test_hist = test_ticker.history(period="1d", interval="5m")
            if test_hist.empty:
                test_hist = test_ticker.history(period="1d", interval="1d")
                if test_hist.empty:
                    return False, f"No data available for {symbol}"
            
            self.current_symbol = yahoo_symbol
            self.display_symbol = symbol
            self.is_running = True
            self.price_history.clear()
            self.predictions.clear()
            self.recent_predictions.clear()
            
            # Reset performance for new stock
            self.performance['is_warming_up'] = True
            self.last_prediction_time = None
            
            # Start prediction thread
            self.prediction_thread = threading.Thread(
                target=self._prediction_loop, 
                daemon=True
            )
            self.prediction_thread.start()
            
            logger.info(f"Started prediction for Indian stock {symbol} ({yahoo_symbol})")
            return True, f"Started prediction for {symbol}"
            
        except Exception as e:
            logger.error(f"Error starting prediction for {symbol}: {e}")
            self.is_running = False  # Ensure we're not stuck in running state
            return False, f"Error starting prediction: {str(e)}"
    
    def stop_prediction(self):
        """Stop real-time prediction"""
        self.is_running = False
        self.current_symbol = None
        # Reset performance but persist learned patterns
        if not hasattr(self, 'session_performance'):
            self.session_performance = {
                'sessions': 0,
                'total_session_accuracy': 0.0,
                'best_accuracy': 0.0,
                'worst_accuracy': 1.0
            }
        
        # Update session stats
        if self.performance['total_predictions'] > 0:
            session_acc = self.performance['accuracy']
            self.session_performance['sessions'] += 1
            self.session_performance['total_session_accuracy'] += session_acc
            self.session_performance['best_accuracy'] = max(
                self.session_performance['best_accuracy'], session_acc
            )
            self.session_performance['worst_accuracy'] = min(
                self.session_performance['worst_accuracy'], session_acc
            )
        
        # Reset current session but keep rolling data partially
        self.performance['total_predictions'] = 0
        self.performance['correct_predictions'] = 0
        self.performance['accuracy'] = 0.0
        self.performance['is_warming_up'] = True
        
        logger.info("Stopped prediction - Session performance saved")
        return True, "Prediction stopped"
    
    def _get_indian_stock_data(self, symbol):
        """Fetch real-time Indian stock data"""
        try:
            logger.info(f"Fetching data for symbol: {symbol}")
            ticker = yf.Ticker(symbol)
            
            # Get recent data (try multiple intervals for Indian markets)
            hist = None
            intervals = ["5m", "1m", "15m", "1h"]
            periods = ["5d", "2d", "1d"]
            
            for period in periods:
                for interval in intervals:
                    try:
                        logger.info(f"Trying {period} period with {interval} interval for {symbol}")
                        hist = ticker.history(period=period, interval=interval)
                        if not hist.empty and len(hist) > 0:
                            logger.info(f"Successfully got {len(hist)} data points for {symbol}")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to get data with {period}/{interval}: {e}")
                        continue
                if hist is not None and not hist.empty:
                    break
                    
            if hist is None or hist.empty:
                logger.error(f"No historical data available for {symbol}")
                return None
                    
            # Get the latest price
            latest = hist.iloc[-1]
            
            # Get basic info (with timeout)
            try:
                info = ticker.info
                if not info:
                    info = {}
            except Exception as e:
                logger.warning(f"Could not fetch ticker info for {symbol}: {e}")
                info = {}
            
            # Convert price to INR format
            current_price = float(latest['Close'])
            
            if current_price <= 0:
                logger.error(f"Invalid price ({current_price}) for {symbol}")
                return None
            
            result = {
                'symbol': self.display_symbol,
                'yahoo_symbol': symbol,
                'price': current_price,
                'volume': int(latest['Volume']) if pd.notna(latest['Volume']) else 0,
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'open': float(latest['Open']),
                'timestamp': datetime.now(),
                'history': hist,
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A')
            }
            
            logger.info(f"Successfully prepared data for {symbol}: Price=₹{current_price:.2f}, Volume={result['volume']}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching Indian stock data for {symbol}: {e}", exc_info=True)
            return None
    
    def _compute_indian_technical_indicators(self, price_data):
        """Compute technical indicators for Indian stocks"""
        if len(price_data) < 10:
            return {}
            
        try:
            df = price_data.copy()
            close_prices = df['Close'].values
            volumes = df['Volume'].values
            high_prices = df['High'].values
            low_prices = df['Low'].values
            
            # Moving averages
            ma_5 = np.mean(close_prices[-5:]) if len(close_prices) >= 5 else close_prices[-1]
            ma_10 = np.mean(close_prices[-10:]) if len(close_prices) >= 10 else close_prices[-1]
            ma_20 = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else close_prices[-1]
            
            # Returns (adjusted for Indian market volatility)
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
            
            # High-Low indicators
            hl_ratio = (high_prices[-1] - low_prices[-1]) / close_prices[-1] if close_prices[-1] > 0 else 0
            
            # Price position within day's range
            day_position = ((close_prices[-1] - low_prices[-1]) / 
                           (high_prices[-1] - low_prices[-1])) if high_prices[-1] > low_prices[-1] else 0.5
            
            return {
                'price_inr': float(close_prices[-1]),
                'ma_5': float(ma_5),
                'ma_10': float(ma_10),
                'ma_20': float(ma_20),
                'return_1period': float(ret_1),
                'return_5period': float(ret_5),
                'return_10period': float(ret_10),
                'volatility': float(volatility),
                'rsi': float(rsi),
                'volume_ratio': float(vol_ratio),
                'high_low_ratio': float(hl_ratio),
                'day_position': float(day_position),
                'price_ma5_ratio': float(close_prices[-1] / ma_5) if ma_5 != 0 else 1.0,
                'ma5_ma20_ratio': float(ma_5 / ma_20) if ma_20 != 0 else 1.0
            }
        except Exception as e:
            logger.error(f"Error computing Indian stock indicators: {e}")
            return {}
    
    def _make_indian_stock_prediction(self, features):
        """Make prediction for Indian stocks with calibrated confidence"""
        if not features:
            return "NEUTRAL", 0.33, 0.35
            
        try:
            # Indian market specific prediction model
            score = 0
            
            # Recent return signals (Indian markets are more volatile)
            score += features.get('return_1period', 0) * 2.5
            score += features.get('return_5period', 0) * 1.5
            
            # Moving average signals (important for Indian stocks)
            price_ma_signal = (features.get('price_ma5_ratio', 1.0) - 1.0) * 1.5
            ma_trend_signal = (features.get('ma5_ma20_ratio', 1.0) - 1.0) * 1.0
            score += price_ma_signal + ma_trend_signal
            
            # RSI signal (adjusted for Indian market behavior)
            rsi = features.get('rsi', 50)
            if rsi > 75:  # Overbought (adjusted for Indian markets)
                score -= 0.7
            elif rsi < 25:  # Oversold (adjusted for Indian markets)
                score += 0.7
            elif 30 <= rsi <= 70:  # Neutral zone gets slight boost
                score *= 1.1
            
            # Volume confirmation (very important in Indian markets)
            vol_ratio = features.get('volume_ratio', 1.0)
            if vol_ratio > 2.0:  # Very high volume confirms signal
                score *= 1.4
            elif vol_ratio > 1.3:  # High volume confirms signal
                score *= 1.2
            elif vol_ratio < 0.7:  # Low volume weakens signal
                score *= 0.8
            
            # Volatility adjustment (Indian markets are volatile)
            volatility = features.get('volatility', 0)
            if volatility > 0.03:  # High volatility = less confidence
                score *= 0.85
            elif volatility < 0.01:  # Low volatility = more confidence
                score *= 1.1
            
            # Day position signal
            day_pos = features.get('day_position', 0.5)
            if day_pos > 0.8:  # Near day's high
                score += 0.2
            elif day_pos < 0.2:  # Near day's low
                score -= 0.2
            
            # Convert score to prediction (adjusted thresholds for Indian markets)
            threshold = 0.003  # Slightly higher threshold for Indian market noise
            if score > threshold:
                prediction = "UP"
                raw_confidence = min(0.85, 0.45 + abs(score) * 8)
            elif score < -threshold:
                prediction = "DOWN" 
                raw_confidence = min(0.85, 0.45 + abs(score) * 8)
            else:
                prediction = "NEUTRAL"
                raw_confidence = 0.35
            
            # Calibrate confidence based on historical performance
            calibrated_confidence = self._calibrate_confidence(raw_confidence, prediction)
            
            return prediction, raw_confidence, calibrated_confidence
            
        except Exception as e:
            logger.error(f"Error making Indian stock prediction: {e}")
            return "NEUTRAL", 0.33, 0.35
    
    def _calibrate_confidence(self, raw_confidence, prediction):
        """Calibrate confidence based on historical accuracy"""
        try:
            # If still warming up, return conservative confidence
            if self.performance['is_warming_up']:
                return min(raw_confidence * 0.8, 0.65)
            
            # Get rolling accuracy for calibration
            rolling_acc = self.performance.get('rolling_accuracy', 0.5)
            
            # Adjust confidence based on recent performance
            if rolling_acc > 0.65:  # Good performance
                calibration_factor = 1.1
            elif rolling_acc > 0.55:  # Average performance
                calibration_factor = 1.0
            elif rolling_acc > 0.45:  # Below average
                calibration_factor = 0.9
            else:  # Poor performance
                calibration_factor = 0.8
            
            calibrated = raw_confidence * calibration_factor
            
            # Cap confidence to reasonable bounds
            return max(0.35, min(0.85, calibrated))
            
        except Exception as e:
            logger.error(f"Error calibrating confidence: {e}")
            return raw_confidence
    
    def _update_performance(self, actual_price, predicted_direction, prev_price):
        """Update performance metrics with rolling window"""
        if prev_price is None:
            return
            
        try:
            actual_change = actual_price - prev_price
            # Adjusted thresholds for Indian market (in rupees)
            actual_direction = "UP" if actual_change > 0.5 else "DOWN" if actual_change < -0.5 else "NEUTRAL"
            
            # Update global counters
            self.performance['total_predictions'] += 1
            is_correct = (predicted_direction == actual_direction)
            
            if is_correct:
                self.performance['correct_predictions'] += 1
            
            # Update global accuracy
            if self.performance['total_predictions'] > 0:
                self.performance['accuracy'] = (
                    self.performance['correct_predictions'] / 
                    self.performance['total_predictions']
                )
            
            # Update rolling window
            prediction_result = {
                'prediction': predicted_direction,
                'actual': actual_direction,
                'correct': is_correct,
                'timestamp': datetime.now()
            }
            
            self.recent_predictions.append(prediction_result)
            
            # Calculate rolling metrics
            if len(self.recent_predictions) > 0:
                rolling_correct = sum(1 for p in self.recent_predictions if p['correct'])
                self.performance['rolling_predictions'] = len(self.recent_predictions)
                self.performance['rolling_correct'] = rolling_correct
                self.performance['rolling_accuracy'] = rolling_correct / len(self.recent_predictions)
            
            # Check if still warming up
            self.performance['is_warming_up'] = (
                self.performance['total_predictions'] < self.performance['warmup_threshold']
            )
            
            logger.info(f"Performance update: {predicted_direction} vs {actual_direction} "
                       f"(Correct: {is_correct}) - "
                       f"Rolling: {self.performance['rolling_accuracy']:.3f} "
                       f"({self.performance['rolling_correct']}/{self.performance['rolling_predictions']})")
                       
        except Exception as e:
            logger.error(f"Error updating performance: {e}")
    
    def _prediction_loop(self):
        """Main prediction loop for Indian stocks"""
        logger.info(f"Starting Indian stock prediction loop for {self.current_symbol}")
        prev_price = None
        last_prediction = None
        consecutive_errors = 0
        max_errors = 5
        
        while self.is_running:
            try:
                logger.info(f"Fetching data for {self.current_symbol}...")
                
                # Get real-time Indian market data
                market_data = self._get_indian_stock_data(self.current_symbol)
                if not market_data:
                    consecutive_errors += 1
                    logger.warning(f"Failed to fetch data for {self.current_symbol} (attempt {consecutive_errors})")
                    
                    if consecutive_errors >= max_errors:
                        logger.error(f"Too many consecutive errors, stopping prediction for {self.current_symbol}")
                        break
                    
                    time.sleep(10)  # Wait before retry
                    continue
                
                consecutive_errors = 0  # Reset error counter on success
                current_price = market_data['price']
                
                logger.info(f"Successfully fetched data: {self.current_symbol} = ₹{current_price:.2f}")
                
                # Store price history
                price_point = {
                    'timestamp': market_data['timestamp'].isoformat(),
                    'price': current_price,
                    'volume': market_data['volume']
                }
                self.price_history.append(price_point)
                
                # Compute technical indicators
                features = self._compute_indian_technical_indicators(market_data['history'])
                logger.info(f"Computed features for {self.current_symbol}: {len(features)} indicators")
                
                # Make prediction
                prediction, raw_confidence, calibrated_confidence = self._make_indian_stock_prediction(features)
                
                # Update performance if we have a previous prediction
                if last_prediction and prev_price:
                    self._update_performance(current_price, last_prediction, prev_price)
                
                # Store current prediction for next evaluation
                current_prediction = {
                    'prediction': prediction,
                    'price': current_price,
                    'timestamp': market_data['timestamp'],
                    'confidence': calibrated_confidence
                }
                self.predictions.append(current_prediction)
                
                # Update last prediction time
                self.last_prediction_time = datetime.now()
                
                # Prepare data to send to frontend
                update_data = {
                    'symbol': market_data['symbol'],
                    'yahoo_symbol': market_data['yahoo_symbol'],
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
                    'industry': market_data.get('industry', 'N/A'),
                    'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
                    'next_prediction_in': self.prediction_interval
                }
                
                # Send update to frontend
                logger.info(f"Sending update to frontend for {market_data['symbol']}")
                socketio.emit('market_update', update_data)
                
                # Log prediction
                logger.info(f"{market_data['symbol']}: ₹{current_price:.2f} -> {prediction} "
                           f"({calibrated_confidence:.2f}, raw: {raw_confidence:.2f})")
                
                # Store for next iteration
                prev_price = current_price
                last_prediction = prediction
                
                # Wait before next update (30 seconds for Indian markets)
                logger.info(f"Waiting {self.prediction_interval} seconds for next update...")
                time.sleep(self.prediction_interval)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in Indian stock prediction loop: {e}", exc_info=True)
                
                if consecutive_errors >= max_errors:
                    logger.error(f"Too many consecutive errors, stopping prediction loop")
                    break
                
                time.sleep(10)  # Wait before retry on error
        
        logger.info("Indian stock prediction loop stopped")
        self.is_running = False

# Global predictor instance
predictor = IndianStockPredictor()

@app.route('/')
def index():
    """Serve the Indian stocks dashboard"""
    return render_template('indian_index_new.html')

@app.route('/api/indian_symbols')
def get_indian_symbols():
    """Get list of popular Indian stocks"""
    indian_stocks = [
        {'symbol': 'RELIANCE.NS', 'name': 'Reliance Industries', 'exchange': 'NSE'},
        {'symbol': 'TCS.NS', 'name': 'Tata Consultancy Services', 'exchange': 'NSE'},
        {'symbol': 'HDFCBANK.NS', 'name': 'HDFC Bank', 'exchange': 'NSE'},
        {'symbol': 'INFY.NS', 'name': 'Infosys', 'exchange': 'NSE'},
        {'symbol': 'HINDUNILVR.NS', 'name': 'Hindustan Unilever', 'exchange': 'NSE'},
        {'symbol': 'ICICIBANK.NS', 'name': 'ICICI Bank', 'exchange': 'NSE'},
        {'symbol': 'SBIN.NS', 'name': 'State Bank of India', 'exchange': 'NSE'},
        {'symbol': 'BHARTIARTL.NS', 'name': 'Bharti Airtel', 'exchange': 'NSE'},
        {'symbol': 'ASIANPAINT.NS', 'name': 'Asian Paints', 'exchange': 'NSE'},
        {'symbol': 'MARUTI.NS', 'name': 'Maruti Suzuki', 'exchange': 'NSE'},
        {'symbol': 'KOTAKBANK.NS', 'name': 'Kotak Mahindra Bank', 'exchange': 'NSE'},
        {'symbol': 'LT.NS', 'name': 'Larsen & Toubro', 'exchange': 'NSE'},
        {'symbol': 'ADANIENTERPRISE.NS', 'name': 'Adani Enterprises', 'exchange': 'NSE'},
        {'symbol': 'TATAMOTORS.NS', 'name': 'Tata Motors', 'exchange': 'NSE'},
        {'symbol': 'SUNPHARMA.NS', 'name': 'Sun Pharmaceutical', 'exchange': 'NSE'}
    ]
    return jsonify(indian_stocks)

@app.route('/api/start/<symbol>')
def start_prediction(symbol):
    """Start prediction for an Indian stock"""
    try:
        logger.info(f"Received start request for symbol: {symbol}")
        
        # If already running, stop first then start new
        if predictor.is_running:
            logger.info(f"Already running {predictor.current_symbol}, switching to {symbol}")
            predictor.stop_prediction()
            time.sleep(1)  # Give time for cleanup
        
        success, message = predictor.start_prediction(symbol)
        logger.info(f"Start prediction result: success={success}, message={message}")
        
        return jsonify({
            'status': 'success' if success else 'error',
            'message': message,
            'previous_symbol': getattr(predictor, 'display_symbol', None) if success else None,
            'new_symbol': symbol if success else None,
            'is_running': predictor.is_running,
            'current_symbol': predictor.current_symbol
        })
    except Exception as e:
        logger.error(f"Error in start_prediction endpoint: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
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
    """Get current status with enhanced information"""
    return jsonify({
        'is_running': predictor.is_running,
        'symbol': predictor.current_symbol,
        'display_symbol': getattr(predictor, 'display_symbol', predictor.current_symbol),
        'performance': predictor.performance,
        'session_performance': getattr(predictor, 'session_performance', {}),
        'last_prediction_time': predictor.last_prediction_time.isoformat() if predictor.last_prediction_time else None,
        'prediction_interval': predictor.prediction_interval,
        'predictions_count': len(predictor.predictions),
        'recent_predictions_count': len(predictor.recent_predictions)
    })

@app.route('/api/test/<symbol>')
def test_symbol(symbol):
    """Test if a symbol works and return sample data"""
    try:
        logger.info(f"Testing symbol: {symbol}")
        
        # Convert to Yahoo Finance format
        if '.' not in symbol:
            yahoo_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
        else:
            yahoo_symbol = symbol
            
        # Test data fetch
        ticker = yf.Ticker(yahoo_symbol)
        hist = ticker.history(period="1d", interval="5m")
        
        if hist.empty:
            hist = ticker.history(period="1d", interval="1d")
            
        if hist.empty:
            return jsonify({
                'status': 'error',
                'message': f'No data available for {symbol} ({yahoo_symbol})',
                'symbol': symbol,
                'yahoo_symbol': yahoo_symbol
            })
        
        latest = hist.iloc[-1]
        sample_data = {
            'status': 'success',
            'symbol': symbol,
            'yahoo_symbol': yahoo_symbol,
            'price': float(latest['Close']),
            'volume': int(latest['Volume']) if pd.notna(latest['Volume']) else 0,
            'data_points': len(hist),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Symbol test successful: {sample_data}")
        return jsonify(sample_data)
        
    except Exception as e:
        logger.error(f"Error testing symbol {symbol}: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'symbol': symbol
        })

@app.route('/api/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': 'running',
        'predictor_status': 'active' if predictor.is_running else 'idle',
        'version': '2.0.0-enhanced'
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected to Indian stocks WebSocket")
    emit('connected', {'message': 'Connected to Indian stock prediction server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected from Indian stocks WebSocket")

if __name__ == '__main__':
    print("🇮🇳 Starting Indian Stock Prediction Dashboard (NSE/BSE)")
    print("🌐 Open your browser to: http://localhost:5001")
    print("📊 Select Indian stocks like RELIANCE.NS, TCS.NS, etc.")
    print("⚡ Using real Indian market data from Yahoo Finance")
    
    socketio.run(app, debug=False, host='0.0.0.0', port=5001)