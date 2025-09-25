#!/usr/bin/env python3
"""
Minimal working Indian stock dashboard for fintech-stock repo
Latest version with proven stability
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import yfinance as yf
import pandas as pd
import threading
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fintech-stock-predictor'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

class SimpleStockPredictor:
    def __init__(self):
        self.current_symbol = None
        self.display_symbol = None
        self.is_running = False
        self.thread = None
        
    def start_prediction(self, symbol):
        """Start prediction with enhanced symbol handling"""
        logger.info(f"🚀 START REQUEST for {symbol}")
        
        if self.is_running:
            logger.info(f"⏹️ Stopping current prediction for {self.current_symbol}")
            self.stop_prediction()
            time.sleep(1)
        
        # Handle symbol format
        if '.' not in symbol:
            yahoo_symbol = f"{symbol}.NS"
            display_symbol = symbol
        else:
            yahoo_symbol = symbol
            display_symbol = symbol.split('.')[0]
            
        self.current_symbol = yahoo_symbol
        self.display_symbol = display_symbol
        self.is_running = True
        
        # Start thread
        self.thread = threading.Thread(target=self._prediction_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"✅ Started prediction for {display_symbol} ({yahoo_symbol})")
        return True, f"Started {display_symbol}"
    
    def stop_prediction(self):
        """Stop prediction"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("⏹️ Stopped prediction")
        return True, "Stopped"
    
    def _prediction_loop(self):
        """Enhanced prediction loop with better error handling"""
        logger.info(f"🔄 PREDICTION LOOP STARTED for {self.current_symbol}")
        
        iteration = 0
        consecutive_errors = 0
        max_errors = 3
        
        while self.is_running:
            try:
                iteration += 1
                logger.info(f"📊 Iteration {iteration} - Fetching {self.current_symbol}")
                
                # Get data with multiple intervals
                ticker = yf.Ticker(self.current_symbol)
                hist = None
                
                # Try different intervals
                for interval in ["5m", "1m", "15m"]:
                    try:
                        hist = ticker.history(period="1d", interval=interval)
                        if not hist.empty:
                            break
                    except:
                        continue
                
                if hist is None or hist.empty:
                    # Fallback to daily data
                    hist = ticker.history(period="2d", interval="1d")
                
                if hist.empty:
                    logger.error(f"❌ No data for {self.current_symbol}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        logger.error(f"🛑 Too many errors, stopping")
                        break
                    time.sleep(5)
                    continue
                
                consecutive_errors = 0  # Reset error count
                
                # Get latest price
                latest = hist.iloc[-1]
                current_price = float(latest['Close'])
                volume = int(latest['Volume']) if pd.notna(latest['Volume']) else 0
                
                # Enhanced prediction
                prediction, confidence = self._make_prediction(hist)
                
                # Prepare update
                update_data = {
                    'symbol': self.display_symbol,
                    'yahoo_symbol': self.current_symbol,
                    'price': current_price,
                    'volume': volume,
                    'high': float(latest['High']),
                    'low': float(latest['Low']),
                    'open': float(latest['Open']),
                    'timestamp': datetime.now().isoformat(),
                    'prediction': prediction,
                    'confidence': confidence,
                    'iteration': iteration
                }
                
                logger.info(f"💰 {self.display_symbol}: ₹{current_price:.2f} -> {prediction} ({confidence:.1%})")
                logger.info(f"📡 Sending update to frontend...")
                
                # Send to frontend
                socketio.emit('market_update', update_data)
                
                # Wait
                logger.info(f"⏱️ Waiting 10 seconds... (iteration {iteration})")
                time.sleep(10)
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"💥 Error in loop: {e}")
                if consecutive_errors >= max_errors:
                    logger.error(f"🛑 Too many consecutive errors, stopping")
                    break
                time.sleep(5)
        
        logger.info(f"🏁 PREDICTION LOOP ENDED for {self.current_symbol}")
        self.is_running = False
    
    def _make_prediction(self, hist):
        """Enhanced prediction algorithm"""
        if len(hist) < 2:
            return "NEUTRAL", 0.35
        
        # Get recent prices
        current_price = float(hist.iloc[-1]['Close'])
        prev_price = float(hist.iloc[-2]['Close'])
        
        # Calculate change
        change = ((current_price - prev_price) / prev_price) * 100
        
        # Volume analysis
        current_volume = hist.iloc[-1]['Volume']
        avg_volume = hist['Volume'].tail(10).mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Price trend (if we have enough data)
        if len(hist) >= 5:
            recent_prices = hist['Close'].tail(5).values
            trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
        else:
            trend = change
        
        # Prediction logic
        base_confidence = 0.45
        
        if change > 0.1 or trend > 0.5:
            prediction = "UP"
            confidence = min(0.85, base_confidence + abs(change) * 0.05 + abs(trend) * 0.02)
            # Volume confirmation
            if volume_ratio > 1.5:
                confidence += 0.1
        elif change < -0.1 or trend < -0.5:
            prediction = "DOWN" 
            confidence = min(0.85, base_confidence + abs(change) * 0.05 + abs(trend) * 0.02)
            # Volume confirmation
            if volume_ratio > 1.5:
                confidence += 0.1
        else:
            prediction = "NEUTRAL"
            confidence = base_confidence
        
        return prediction, min(confidence, 0.85)

# Global instance
predictor = SimpleStockPredictor()

@app.route('/')
def index():
    return render_template('simple_dashboard.html')

@app.route('/api/start/<symbol>')
def start_prediction(symbol):
    logger.info(f"📞 API call: start/{symbol}")
    success, message = predictor.start_prediction(symbol)
    return jsonify({
        'status': 'success' if success else 'error',
        'message': message,
        'is_running': predictor.is_running,
        'symbol': predictor.display_symbol
    })

@app.route('/api/stop')
def stop_prediction():
    logger.info(f"📞 API call: stop")
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
        'display_symbol': predictor.display_symbol
    })

@socketio.on('connect')
def handle_connect():
    logger.info("🔌 Client connected")
    emit('connected', {'message': 'Connected to Fintech Stock Predictor'})

@socketio.on('disconnect') 
def handle_disconnect():
    logger.info("🔌 Client disconnected")

if __name__ == '__main__':
    print("🏦 Fintech Stock Prediction Dashboard")
    print("🌐 Open: http://localhost:5002")
    print("✨ Latest working version with enhanced stability")
    
    socketio.run(app, debug=False, host='0.0.0.0', port=5002)