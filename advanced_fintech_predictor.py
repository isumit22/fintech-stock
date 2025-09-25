"""
ADVANCED FINTECH STOCK PREDICTOR
Personal project by Sumit - Professional-grade real-time stock analysis

Features:
- Real-time stock price prediction with AI insights
- Advanced technical analysis (RSI, Moving averages, Volatility)
- Live streaming data processing with dynamic indexing
- AI-powered market analysis and risk assessment
- Professional fintech-grade dashboard with WebSocket updates
- RESTful API for programmatic access
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from flask import Flask, jsonify, request, render_template
from flask_socketio import SocketIO, emit
import requests
import os
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StockAnalysis:
    """Complete stock analysis data structure"""
    symbol: str
    company_name: str
    current_price: float
    volume: int
    price_change: float
    price_change_percent: float
    rsi: float
    volatility: float
    sma_5: float
    sma_20: float
    sma_50: float
    prediction: str
    confidence: float
    reasoning: str
    risk_score: float
    market_sentiment: str
    timestamp: datetime

@dataclass
class MarketInsight:
    """AI-generated market insights"""
    symbol: str
    trend_analysis: str
    support_level: float
    resistance_level: float
    volume_analysis: str
    momentum_indicator: str
    recommendation: str
    target_price: float
    stop_loss: float
    time_horizon: str
    confidence_score: float

class AdvancedStockPredictor:
    """
    Advanced Fintech Stock Predictor - Personal Project
    
    A sophisticated real-time stock analysis system with AI-powered insights,
    technical analysis, and professional-grade dashboard interface.
    """
    
    def __init__(self):
        # Flask application setup
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'advanced-fintech-predictor-2025'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Stock universe - Indian market focus
        self.stocks = {
            'RELIANCE.NS': 'Reliance Industries Ltd',
            'TCS.NS': 'Tata Consultancy Services',
            'INFY.NS': 'Infosys Limited', 
            'KOTAKBANK.NS': 'Kotak Mahindra Bank',
            'ITC.NS': 'ITC Limited',
            'HDFCBANK.NS': 'HDFC Bank Limited',
            'ICICIBANK.NS': 'ICICI Bank Limited',
            'HINDUNILVR.NS': 'Hindustan Unilever',
            'LT.NS': 'Larsen & Toubro',
            'SBIN.NS': 'State Bank of India'
        }
        
        self.current_stock = 'RELIANCE.NS'
        
        # Data storage and caching
        self.live_data = {}
        self.market_insights = {}
        self.analysis_history = []
        self.data_cache = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy_rate': 0.0,
            'avg_confidence': 0.0,
            'processing_time_avg': 0.0
        }
        
        # System state
        self.system_active = False
        self.last_update = None
        
        # Initialize components
        self.setup_routes()
        self.setup_websockets()
        
    def start_real_time_analysis(self):
        """Start the real-time stock analysis engine"""
        self.system_active = True
        
        while self.system_active:
            try:
                start_time = time.time()
                
                # Fetch and analyze current stock
                stock_data = self.fetch_comprehensive_data()
                
                if stock_data:
                    # Generate complete analysis
                    analysis = self.perform_advanced_analysis(stock_data)
                    
                    # Generate AI insights
                    insights = self.generate_market_insights(analysis)
                    
                    # Update data stores
                    self.live_data[self.current_stock] = analysis
                    self.market_insights[self.current_stock] = insights
                    
                    # Track performance
                    processing_time = (time.time() - start_time) * 1000
                    self.update_performance_metrics(processing_time)
                    
                    # Broadcast updates
                    self.broadcast_live_update(analysis, insights)
                    
                    # Store in history
                    self.analysis_history.append({
                        'timestamp': datetime.now(),
                        'symbol': self.current_stock,
                        'price': analysis.current_price,
                        'prediction': analysis.prediction,
                        'confidence': analysis.confidence
                    })
                    
                    # Keep history manageable
                    if len(self.analysis_history) > 1000:
                        self.analysis_history = self.analysis_history[-500:]
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Analysis engine error: {e}")
                time.sleep(5)
    
    def fetch_comprehensive_data(self) -> Optional[Dict]:
        """Fetch comprehensive stock data with error handling"""
        try:
            ticker = yf.Ticker(self.current_stock)
            
            # Get various timeframes for comprehensive analysis
            daily_data = ticker.history(period='3mo', interval='1d')
            hourly_data = ticker.history(period='5d', interval='1h')
            info = ticker.info
            
            if daily_data.empty:
                return None
            
            # Calculate comprehensive metrics
            latest_daily = daily_data.iloc[-1]
            latest_hourly = hourly_data.iloc[-1] if not hourly_data.empty else latest_daily
            
            # Price metrics
            current_price = float(latest_hourly['Close'])
            price_change = float(latest_hourly['Close'] - latest_hourly['Open'])
            price_change_percent = (price_change / latest_hourly['Open']) * 100
            
            # Technical indicators
            rsi_14 = self.calculate_rsi(daily_data['Close'], 14)
            rsi_current = float(rsi_14.iloc[-1]) if not rsi_14.empty else 50.0
            
            # Moving averages
            sma_5 = float(daily_data['Close'].rolling(window=5).mean().iloc[-1])
            sma_20 = float(daily_data['Close'].rolling(window=20).mean().iloc[-1])
            sma_50 = float(daily_data['Close'].rolling(window=50).mean().iloc[-1]) if len(daily_data) >= 50 else sma_20
            
            # Volatility analysis
            volatility = float(daily_data['Close'].rolling(window=20).std().iloc[-1])
            
            # Volume analysis
            avg_volume = float(daily_data['Volume'].rolling(window=20).mean().iloc[-1])
            current_volume = int(latest_daily['Volume'])
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            return {
                'symbol': self.current_stock,
                'company_name': self.stocks.get(self.current_stock, self.current_stock),
                'current_price': current_price,
                'volume': current_volume,
                'volume_ratio': volume_ratio,
                'price_change': price_change,
                'price_change_percent': price_change_percent,
                'rsi': rsi_current,
                'volatility': volatility,
                'sma_5': sma_5,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'daily_data': daily_data,
                'hourly_data': hourly_data,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            return None
    
    def perform_advanced_analysis(self, data: Dict) -> StockAnalysis:
        """Perform advanced technical and fundamental analysis"""
        
        # Extract key metrics
        rsi = data['rsi']
        volatility = data['volatility']
        price_change_percent = data['price_change_percent']
        volume_ratio = data['volume_ratio']
        sma_5 = data['sma_5']
        sma_20 = data['sma_20']
        sma_50 = data['sma_50']
        current_price = data['current_price']
        
        # Advanced trend analysis
        trend_signals = []
        confidence_factors = []
        
        # RSI Analysis
        if rsi > 70:
            trend_signals.append("BEARISH")
            confidence_factors.append(min((rsi - 70) * 2, 30))
            rsi_signal = "Overbought"
        elif rsi < 30:
            trend_signals.append("BULLISH") 
            confidence_factors.append(min((30 - rsi) * 2, 30))
            rsi_signal = "Oversold"
        else:
            rsi_signal = "Neutral"
            confidence_factors.append(10)
        
        # Moving Average Analysis
        if sma_5 > sma_20 > sma_50:
            trend_signals.append("BULLISH")
            confidence_factors.append(25)
            ma_trend = "Strong uptrend"
        elif sma_5 < sma_20 < sma_50:
            trend_signals.append("BEARISH")
            confidence_factors.append(25)
            ma_trend = "Strong downtrend"
        elif sma_5 > sma_20:
            trend_signals.append("BULLISH")
            confidence_factors.append(15)
            ma_trend = "Short-term bullish"
        else:
            trend_signals.append("BEARISH")
            confidence_factors.append(15)
            ma_trend = "Short-term bearish"
        
        # Volume Analysis
        if volume_ratio > 1.5:
            confidence_factors.append(20)
            volume_signal = "High volume confirms trend"
        elif volume_ratio < 0.7:
            confidence_factors.append(-10)
            volume_signal = "Low volume weakens signal"
        else:
            volume_signal = "Normal volume"
        
        # Price momentum
        if abs(price_change_percent) > 3:
            if price_change_percent > 0:
                trend_signals.append("BULLISH")
            else:
                trend_signals.append("BEARISH")
            confidence_factors.append(15)
        
        # Determine overall prediction
        bullish_count = trend_signals.count("BULLISH")
        bearish_count = trend_signals.count("BEARISH")
        
        if bullish_count > bearish_count:
            prediction = "BULLISH"
        elif bearish_count > bullish_count:
            prediction = "BEARISH"
        else:
            prediction = "NEUTRAL"
        
        # Calculate confidence
        base_confidence = sum(confidence_factors)
        
        # Adjust for volatility
        volatility_factor = min(volatility / (current_price * 0.02), 2.0)
        volatility_penalty = volatility_factor * 15
        
        final_confidence = max(min(base_confidence - volatility_penalty, 95), 20)
        
        # Generate detailed reasoning
        reasoning = self.generate_analysis_reasoning(
            rsi, rsi_signal, ma_trend, volume_signal, 
            price_change_percent, volatility_factor, prediction
        )
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(data, volatility_factor)
        
        # Market sentiment
        sentiment = self.determine_market_sentiment(prediction, final_confidence, volume_ratio)
        
        return StockAnalysis(
            symbol=data['symbol'],
            company_name=data['company_name'],
            current_price=current_price,
            volume=data['volume'],
            price_change=data['price_change'],
            price_change_percent=price_change_percent,
            rsi=rsi,
            volatility=volatility,
            sma_5=sma_5,
            sma_20=sma_20,
            sma_50=sma_50,
            prediction=prediction,
            confidence=final_confidence,
            reasoning=reasoning,
            risk_score=risk_score,
            market_sentiment=sentiment,
            timestamp=data['timestamp']
        )
    
    def generate_market_insights(self, analysis: StockAnalysis) -> MarketInsight:
        """Generate AI-powered market insights"""
        
        # Technical levels calculation
        recent_high = analysis.current_price * 1.05
        recent_low = analysis.current_price * 0.95
        
        # Support and resistance levels
        if analysis.prediction == "BULLISH":
            support_level = max(analysis.sma_20, recent_low)
            resistance_level = recent_high
            target_price = analysis.current_price * 1.08
            stop_loss = analysis.current_price * 0.96
        else:
            support_level = recent_low
            resistance_level = min(analysis.sma_20, recent_high)
            target_price = analysis.current_price * 0.92
            stop_loss = analysis.current_price * 1.04
        
        # Volume analysis interpretation
        if analysis.volume > 1000000:
            volume_analysis = "High institutional activity"
        elif analysis.volume > 500000:
            volume_analysis = "Moderate trading interest"
        else:
            volume_analysis = "Low retail participation"
        
        # Momentum indicator
        if analysis.rsi > 60:
            momentum = "Strong momentum"
        elif analysis.rsi > 40:
            momentum = "Moderate momentum"
        else:
            momentum = "Weak momentum"
        
        # Investment recommendation
        if analysis.confidence > 80 and analysis.prediction == "BULLISH":
            recommendation = "STRONG BUY"
            time_horizon = "3-5 days"
        elif analysis.confidence > 70 and analysis.prediction == "BULLISH":
            recommendation = "BUY"
            time_horizon = "1-3 days"
        elif analysis.confidence > 80 and analysis.prediction == "BEARISH":
            recommendation = "STRONG SELL"
            time_horizon = "3-5 days"
        elif analysis.confidence > 70 and analysis.prediction == "BEARISH":
            recommendation = "SELL"
            time_horizon = "1-3 days"
        else:
            recommendation = "HOLD"
            time_horizon = "Monitor closely"
        
        # Trend analysis summary
        if analysis.sma_5 > analysis.sma_20 > analysis.sma_50:
            trend_analysis = "Strong bullish alignment across all timeframes"
        elif analysis.sma_5 < analysis.sma_20 < analysis.sma_50:
            trend_analysis = "Strong bearish alignment across all timeframes"
        else:
            trend_analysis = "Mixed signals across different timeframes"
        
        return MarketInsight(
            symbol=analysis.symbol,
            trend_analysis=trend_analysis,
            support_level=support_level,
            resistance_level=resistance_level,
            volume_analysis=volume_analysis,
            momentum_indicator=momentum,
            recommendation=recommendation,
            target_price=target_price,
            stop_loss=stop_loss,
            time_horizon=time_horizon,
            confidence_score=analysis.confidence * 0.9  # Slightly conservative
        )
    
    def generate_analysis_reasoning(self, rsi, rsi_signal, ma_trend, volume_signal, 
                                  price_change_percent, volatility_factor, prediction):
        """Generate detailed analysis reasoning"""
        reasoning_parts = []
        
        # RSI component
        reasoning_parts.append(f"RSI at {rsi:.1f} indicates {rsi_signal.lower()} conditions")
        
        # Moving average component  
        reasoning_parts.append(f"Moving averages show {ma_trend.lower()}")
        
        # Volume component
        if volume_signal != "Normal volume":
            reasoning_parts.append(volume_signal.lower())
        
        # Price momentum
        if abs(price_change_percent) > 2:
            direction = "positive" if price_change_percent > 0 else "negative"
            reasoning_parts.append(f"Strong {direction} momentum ({price_change_percent:+.2f}%)")
        
        # Volatility warning
        if volatility_factor > 1.5:
            reasoning_parts.append(f"High volatility increases uncertainty")
        
        # Conclusion
        confidence_word = "high" if prediction != "NEUTRAL" else "moderate"
        reasoning_parts.append(f"Overall {prediction.lower()} outlook with {confidence_word} confidence")
        
        return ". ".join(reasoning_parts).capitalize() + "."
    
    def calculate_risk_score(self, data: Dict, volatility_factor: float) -> float:
        """Calculate comprehensive risk score (0-100, higher = riskier)"""
        risk_components = []
        
        # Volatility risk (0-40 points)
        volatility_risk = min(volatility_factor * 20, 40)
        risk_components.append(volatility_risk)
        
        # RSI risk (0-20 points)
        rsi = data['rsi']
        rsi_risk = max(abs(rsi - 50) - 20, 0) * 0.8
        risk_components.append(rsi_risk)
        
        # Volume risk (0-20 points)
        volume_ratio = data['volume_ratio']
        volume_risk = abs(volume_ratio - 1.0) * 10
        risk_components.append(volume_risk)
        
        # Price change risk (0-20 points)
        price_change_percent = abs(data['price_change_percent'])
        price_risk = min(price_change_percent * 3, 20)
        risk_components.append(price_risk)
        
        total_risk = sum(risk_components)
        return min(total_risk, 100)
    
    def determine_market_sentiment(self, prediction: str, confidence: float, volume_ratio: float) -> str:
        """Determine overall market sentiment"""
        if prediction == "BULLISH" and confidence > 70:
            return "OPTIMISTIC"
        elif prediction == "BEARISH" and confidence > 70:
            return "PESSIMISTIC"
        elif volume_ratio > 1.5:
            return "ACTIVE"
        elif volume_ratio < 0.7:
            return "CAUTIOUS"
        else:
            return "NEUTRAL"
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def update_performance_metrics(self, processing_time: float):
        """Update system performance metrics"""
        self.performance_metrics['total_predictions'] += 1
        
        # Update processing time average
        current_avg = self.performance_metrics['processing_time_avg']
        total = self.performance_metrics['total_predictions']
        self.performance_metrics['processing_time_avg'] = \
            ((current_avg * (total - 1)) + processing_time) / total
    
    def broadcast_live_update(self, analysis: StockAnalysis, insights: MarketInsight):
        """Broadcast live updates via WebSocket"""
        update_data = {
            'analysis': {
                'symbol': analysis.symbol,
                'company_name': analysis.company_name,
                'current_price': analysis.current_price,
                'price_change': analysis.price_change,
                'price_change_percent': analysis.price_change_percent,
                'prediction': analysis.prediction,
                'confidence': analysis.confidence,
                'reasoning': analysis.reasoning,
                'risk_score': analysis.risk_score,
                'market_sentiment': analysis.market_sentiment,
                'technical_indicators': {
                    'rsi': analysis.rsi,
                    'volatility': analysis.volatility,
                    'sma_5': analysis.sma_5,
                    'sma_20': analysis.sma_20,
                    'sma_50': analysis.sma_50
                },
                'timestamp': analysis.timestamp.isoformat()
            },
            'insights': {
                'trend_analysis': insights.trend_analysis,
                'support_level': insights.support_level,
                'resistance_level': insights.resistance_level,
                'volume_analysis': insights.volume_analysis,
                'momentum_indicator': insights.momentum_indicator,
                'recommendation': insights.recommendation,
                'target_price': insights.target_price,
                'stop_loss': insights.stop_loss,
                'time_horizon': insights.time_horizon,
                'confidence_score': insights.confidence_score
            },
            'system_status': {
                'processing_time_ms': self.performance_metrics['processing_time_avg'],
                'total_predictions': self.performance_metrics['total_predictions'],
                'last_update': datetime.now().isoformat()
            }
        }
        
        self.socketio.emit('live_market_update', update_data)
        self.last_update = datetime.now()
        
        logger.info(f"📊 Live update: {analysis.symbol} @ ₹{analysis.current_price:.2f} | {analysis.prediction}")
    
    def setup_routes(self):
        """Setup Flask routes for the web application"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard"""
            return render_template('advanced_fintech_dashboard.html',
                                 stocks=list(self.stocks.keys()),
                                 stock_names=self.stocks,
                                 current_stock=self.current_stock)
        
        @self.app.route('/change_stock', methods=['POST'])
        def change_stock():
            """Change current stock for analysis"""
            new_stock = request.json.get('stock')
            if new_stock in self.stocks:
                self.current_stock = new_stock
                logger.info(f"🔄 Stock changed to {new_stock}")
                return jsonify({'success': True, 'stock': new_stock})
            return jsonify({'success': False, 'message': 'Invalid stock symbol'})
        
        @self.app.route('/api/analysis/<symbol>', methods=['GET'])
        def get_analysis(symbol):
            """Get current analysis for a symbol"""
            if symbol in self.live_data:
                analysis = self.live_data[symbol]
                return jsonify({
                    'success': True,
                    'analysis': asdict(analysis)
                })
            return jsonify({'success': False, 'message': 'No analysis available'}), 404
        
        @self.app.route('/api/insights/<symbol>', methods=['GET'])
        def get_insights(symbol):
            """Get market insights for a symbol"""
            if symbol in self.market_insights:
                insights = self.market_insights[symbol]
                return jsonify({
                    'success': True,
                    'insights': asdict(insights)
                })
            return jsonify({'success': False, 'message': 'No insights available'}), 404
        
        @self.app.route('/api/performance', methods=['GET'])
        def get_performance():
            """Get system performance metrics"""
            return jsonify({
                'success': True,
                'performance': self.performance_metrics,
                'system_status': {
                    'active': self.system_active,
                    'last_update': self.last_update.isoformat() if self.last_update else None,
                    'supported_stocks': len(self.stocks),
                    'active_analyses': len(self.live_data)
                }
            })
        
        @self.app.route('/api/history', methods=['GET'])
        def get_history():
            """Get analysis history"""
            limit = request.args.get('limit', 50, type=int)
            history = self.analysis_history[-limit:] if self.analysis_history else []
            
            return jsonify({
                'success': True,
                'history': [
                    {
                        'timestamp': item['timestamp'].isoformat(),
                        'symbol': item['symbol'],
                        'price': item['price'],
                        'prediction': item['prediction'],
                        'confidence': item['confidence']
                    }
                    for item in history
                ]
            })
    
    def setup_websockets(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("🔌 Client connected to Advanced Stock Predictor")
            emit('connection_status', {
                'status': 'connected',
                'message': 'Connected to Advanced Fintech Stock Predictor',
                'system_active': self.system_active,
                'current_stock': self.current_stock,
                'supported_stocks': len(self.stocks)
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info("🔌 Client disconnected")
        
        @self.socketio.on('request_current_analysis')
        def handle_analysis_request():
            """Send current analysis data"""
            if self.current_stock in self.live_data:
                analysis = self.live_data[self.current_stock]
                insights = self.market_insights.get(self.current_stock)
                
                if insights:
                    self.broadcast_live_update(analysis, insights)
    
    def start_background_engine(self):
        """Start the background analysis engine"""
        if not self.system_active:
            analysis_thread = threading.Thread(target=self.start_real_time_analysis, daemon=True)
            analysis_thread.start()
            logger.info("🚀 Advanced analysis engine started")
    
    def run(self, host='localhost', port=5000, debug=False):
        """Run the advanced stock predictor application"""
        print("\n" + "="*70)
        print("🚀 ADVANCED FINTECH STOCK PREDICTOR")
        print("="*70)
        print("📊 PROFESSIONAL FEATURES:")
        print("   ✅ Real-time stock analysis with AI insights")
        print("   ✅ Advanced technical indicators (RSI, SMA, Volatility)")
        print("   ✅ Risk assessment and market sentiment analysis")
        print("   ✅ Professional fintech-grade dashboard")
        print("   ✅ RESTful API for programmatic access")
        print("   ✅ Live WebSocket updates")
        print("   ✅ Performance tracking and history")
        print(f"\n🌐 Dashboard: http://{host}:{port}")
        print("🔗 API Endpoints:")
        print(f"   • Analysis: http://{host}:{port}/api/analysis/<symbol>")
        print(f"   • Insights: http://{host}:{port}/api/insights/<symbol>")
        print(f"   • Performance: http://{host}:{port}/api/performance")
        print(f"   • History: http://{host}:{port}/api/history")
        print("="*70)
        
        # Start background engine
        self.start_background_engine()
        
        # Run Flask application
        self.socketio.run(self.app, host=host, port=port, debug=debug)

if __name__ == '__main__':
    # Launch Advanced Fintech Stock Predictor
    predictor = AdvancedStockPredictor()
    predictor.run()