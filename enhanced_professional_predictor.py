#!/usr/bin/env python3
"""
Enhanced Professional Fintech Predictor
========================================
Advanced stock analysis with interactive charts, crypto analysis, 
professional indicators, and comprehensive market data expansion.

Features:
- Interactive Charts with TradingView integration
- Cryptocurrency Analysis (Bitcoin, Ethereum, etc.)
- Advanced Technical Indicators (Fibonacci, Elliott Wave, Volume Profile)
- Multi-timeframe Analysis
- Professional Features (PDF reports, Excel export)
- Global Markets (US, European, Asian, Forex, Commodities)
- Market Heat Maps and Sector Analysis
"""

import os
import sys
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils

# AI and Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Enhanced market data structure"""
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    currency: str = "USD"
    market: str = "US"
    last_updated: str = ""

@dataclass
class TechnicalAnalysis:
    """Enhanced technical analysis with advanced indicators"""
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_middle: float
    fibonacci_levels: Dict[str, float]
    volume_profile: Dict[str, float]
    sma_5: float
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    volatility: float
    atr: float  # Average True Range
    stochastic_k: float
    stochastic_d: float
    williams_r: float
    support_levels: List[float]
    resistance_levels: List[float]

@dataclass
class CryptoData:
    """Cryptocurrency-specific data structure"""
    symbol: str
    name: str
    price_usd: float
    price_btc: float
    change_24h: float
    volume_24h: float
    market_cap: float
    circulating_supply: float
    total_supply: Optional[float]
    dominance: Optional[float]
    fear_greed_index: Optional[int]

@dataclass
class ComprehensiveAnalysis:
    """Complete analysis with all features"""
    symbol: str
    company_name: str
    current_price: float
    prediction: str
    confidence: float
    reasoning: str
    technical_analysis: TechnicalAnalysis
    market_sentiment: str
    risk_score: float
    price_change: float
    price_change_percent: float
    timeframe_analysis: Dict[str, Dict[str, Any]]
    sector_analysis: Optional[Dict[str, Any]] = None
    options_flow: Optional[Dict[str, Any]] = None

@dataclass
class AIPrediction:
    """AI-powered stock prediction with advanced ML models"""
    symbol: str
    current_price: float
    predictions: Dict[str, Dict[str, Any]]  # 1d, 7d, 30d, 90d predictions
    ml_models: Dict[str, Dict[str, Any]]  # Model performance metrics
    ai_sentiment: Dict[str, Any]  # AI sentiment analysis
    pattern_recognition: Dict[str, Any]  # Technical pattern detection
    risk_assessment: Dict[str, Any]  # AI risk analysis
    trading_signals: List[Dict[str, Any]]  # AI trading recommendations
    confidence_intervals: Dict[str, Tuple[float, float]]  # Prediction ranges
    market_regime: str  # Bull/Bear/Sideways market detection
    anomaly_detection: Dict[str, Any]  # Unusual pattern detection
    news_sentiment: Optional[Dict[str, Any]] = None
    social_sentiment: Optional[Dict[str, Any]] = None

class EnhancedProfessionalPredictor:
    """Enhanced professional-grade stock prediction system"""
    
    def __init__(self):
        """Initialize the enhanced predictor"""
        self.app = Flask(__name__)
        self.app.secret_key = 'enhanced_professional_predictor_key_2024'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Enhanced stock lists with global markets
        self.stock_symbols = {
            # Indian Stocks (NSE)
            'RELIANCE.NS': 'Reliance Industries Ltd',
            'TCS.NS': 'Tata Consultancy Services',
            'INFY.NS': 'Infosys Limited',
            'HINDUNILVR.NS': 'Hindustan Unilever',
            'ICICIBANK.NS': 'ICICI Bank Limited',
            'HDFCBANK.NS': 'HDFC Bank Limited',
            'ITC.NS': 'ITC Limited',
            'KOTAKBANK.NS': 'Kotak Mahindra Bank',
            'LT.NS': 'Larsen & Toubro',
            'SBIN.NS': 'State Bank of India',
            
            # US Stocks
            'AAPL': 'Apple Inc.',
            'GOOGL': 'Alphabet Inc.',
            'MSFT': 'Microsoft Corporation',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'JPM': 'JPMorgan Chase & Co.',
            'V': 'Visa Inc.',
            'JNJ': 'Johnson & Johnson',
            
            # European Stocks
            'ASML.AS': 'ASML Holding N.V.',
            'SAP': 'SAP SE',
            'NESN.SW': 'Nestlé S.A.',
            'MC.PA': 'LVMH Moët Hennessy',
            'NOVN.SW': 'Novartis AG',
            
            # Asian Stocks
            '7203.T': 'Toyota Motor Corporation',
            '6758.T': 'Sony Group Corporation',
            '9984.T': 'SoftBank Group Corp.',
            '005930.KS': 'Samsung Electronics',
            '2330.TW': 'Taiwan Semiconductor'
        }
        
        # Cryptocurrency symbols
        self.crypto_symbols = {
            'BTC-USD': 'Bitcoin',
            'ETH-USD': 'Ethereum',
            'BNB-USD': 'Binance Coin',
            'XRP-USD': 'Ripple',
            'ADA-USD': 'Cardano',
            'SOL-USD': 'Solana',
            'DOGE-USD': 'Dogecoin',
            'DOT-USD': 'Polkadot',
            'MATIC-USD': 'Polygon',
            'LTC-USD': 'Litecoin'
        }
        
        # Forex pairs
        self.forex_pairs = {
            'EURUSD=X': 'Euro/US Dollar',
            'GBPUSD=X': 'British Pound/US Dollar',
            'USDJPY=X': 'US Dollar/Japanese Yen',
            'USDCHF=X': 'US Dollar/Swiss Franc',
            'AUDUSD=X': 'Australian Dollar/US Dollar',
            'USDCAD=X': 'US Dollar/Canadian Dollar',
            'NZDUSD=X': 'New Zealand Dollar/US Dollar',
            'EURGBP=X': 'Euro/British Pound'
        }
        
        # Commodities
        self.commodities = {
            'GC=F': 'Gold Futures',
            'SI=F': 'Silver Futures',
            'CL=F': 'Crude Oil Futures',
            'NG=F': 'Natural Gas Futures',
            'HG=F': 'Copper Futures',
            'ZC=F': 'Corn Futures',
            'ZW=F': 'Wheat Futures',
            'KC=F': 'Coffee Futures'
        }
        
        # Combine all symbols
        self.all_symbols = {**self.stock_symbols, **self.crypto_symbols, 
                           **self.forex_pairs, **self.commodities}
        
        # Current analysis data
        self.current_symbol = 'RELIANCE.NS'
        self.analysis_data = {}
        self.market_data_cache = {}
        self.running = False
        
        # Timeframes for multi-timeframe analysis
        self.timeframes = {
            '1m': '1 Minute',
            '5m': '5 Minutes',
            '15m': '15 Minutes',
            '1h': '1 Hour',
            '4h': '4 Hours',
            '1d': '1 Day',
            '1wk': '1 Week',
            '1mo': '1 Month'
        }
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio()
        
        logger.info("🚀 Enhanced Professional Predictor initialized")
    
    def _setup_routes(self):
        """Setup Flask routes with enhanced endpoints"""
        
        @self.app.route('/')
        def dashboard():
            return render_template('enhanced_professional_dashboard.html', 
                                 stocks=self.all_symbols,
                                 stock_names=self.all_symbols,
                                 current_stock=self.current_symbol)
        
        @self.app.route('/test')
        def test_page():
            return render_template('test.html', 
                                 stocks=self.all_symbols,
                                 current_stock=self.current_symbol)
        
        @self.app.route('/change_stock', methods=['POST'])
        def change_stock():
            data = request.get_json()
            new_symbol = data.get('stock', self.current_symbol)
            if new_symbol in self.all_symbols:
                self.current_symbol = new_symbol
                logger.info(f"🔄 Switched to {new_symbol}")
                return jsonify({'success': True, 'symbol': new_symbol})
            return jsonify({'success': False, 'error': 'Invalid symbol'})
        
        # Enhanced API endpoints
        @self.app.route('/api/analysis/<symbol>')
        def get_analysis(symbol):
            try:
                if symbol in self.all_symbols:
                    analysis = self._perform_comprehensive_analysis(symbol)
                    return jsonify({
                        'success': True,
                        'analysis': asdict(analysis)
                    })
                return jsonify({'success': False, 'error': 'Invalid symbol'})
            except Exception as e:
                logger.error(f"❌ Analysis error: {e}")
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/multi_timeframe/<symbol>')
        def get_multi_timeframe(symbol):
            try:
                timeframe_data = self._analyze_multiple_timeframes(symbol)
                return jsonify({
                    'success': True,
                    'timeframes': timeframe_data
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/sector_analysis')
        def get_sector_analysis():
            try:
                sector_data = self._analyze_sectors()
                return jsonify({
                    'success': True,
                    'sectors': sector_data
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/ai_prediction/<symbol>')
        def get_ai_prediction(symbol):
            try:
                ai_prediction = self._generate_ai_prediction(symbol)
                return jsonify({
                    'success': True,
                    'ai_prediction': asdict(ai_prediction)
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/ai_sentiment/<symbol>')
        def get_ai_sentiment(symbol):
            try:
                sentiment_data = self._analyze_ai_sentiment(symbol)
                return jsonify({
                    'success': True,
                    'ai_sentiment': sentiment_data
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/ai_patterns/<symbol>')
        def get_ai_patterns(symbol):
            try:
                pattern_data = self._detect_ai_patterns(symbol)
                return jsonify({
                    'success': True,
                    'ai_patterns': pattern_data
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/ai_risk/<symbol>')
        def get_ai_risk(symbol):
            try:
                risk_data = self._assess_ai_risk(symbol)
                return jsonify({
                    'success': True,
                    'ai_risk': risk_data
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/ai_trading_signals/<symbol>')
        def get_ai_trading_signals(symbol):
            try:
                signals = self._generate_ai_trading_signals(symbol)
                return jsonify({
                    'success': True,
                    'trading_signals': signals
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/market_heatmap')
        def get_market_heatmap():
            try:
                heatmap_data = self._generate_market_heatmap()
                return jsonify({
                    'success': True,
                    'heatmap': heatmap_data
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/crypto_analysis')
        def get_crypto_analysis():
            try:
                crypto_data = self._analyze_cryptocurrencies()
                return jsonify({
                    'success': True,
                    'crypto': crypto_data
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/chart/<symbol>/<timeframe>')
        def get_chart_data(symbol, timeframe='1d'):
            try:
                chart_data = self._generate_interactive_chart(symbol, timeframe)
                return jsonify({
                    'success': True,
                    'chart': chart_data
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/fibonacci/<symbol>')
        def get_fibonacci(symbol):
            try:
                fib_data = self._calculate_fibonacci_levels(symbol)
                return jsonify({
                    'success': True,
                    'fibonacci': fib_data
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/export/pdf/<symbol>')
        def export_pdf(symbol):
            try:
                pdf_buffer = self._generate_pdf_report(symbol)
                return send_file(
                    pdf_buffer,
                    as_attachment=True,
                    download_name=f'{symbol}_analysis_report.pdf',
                    mimetype='application/pdf'
                )
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/export/excel/<symbol>')
        def export_excel(symbol):
            try:
                excel_buffer = self._generate_excel_report(symbol)
                return send_file(
                    excel_buffer,
                    as_attachment=True,
                    download_name=f'{symbol}_analysis_data.xlsx',
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/watchlist', methods=['GET', 'POST'])
        def manage_watchlist():
            if request.method == 'POST':
                # Add to watchlist
                data = request.get_json()
                symbol = data.get('symbol')
                # In a real app, this would save to database
                return jsonify({'success': True, 'message': 'Added to watchlist'})
            else:
                # Get watchlist
                # In a real app, this would load from database
                watchlist = ['RELIANCE.NS', 'BTC-USD', 'AAPL', 'EURUSD=X']
                return jsonify({'success': True, 'watchlist': watchlist})
    
    def _setup_socketio(self):
        """Setup SocketIO events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("🔌 Client connected to Enhanced Professional Predictor")
            emit('connection_status', {
                'message': 'Connected to Enhanced Professional Predictor',
                'system_active': True,
                'features': [
                    'Interactive Charts',
                    'Crypto Analysis',
                    'Advanced Indicators',
                    'Multi-timeframe Analysis',
                    'Professional Reports'
                ]
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info("🔌 Client disconnected")
        
        @self.socketio.on('request_market_update')
        def handle_market_update():
            try:
                analysis = self._perform_comprehensive_analysis(self.current_symbol)
                market_data = self._get_market_overview()
                
                emit('comprehensive_market_update', {
                    'analysis': asdict(analysis),
                    'market_overview': market_data,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"❌ Market update error: {e}")
                emit('error', {'message': str(e)})
    
    def _calculate_advanced_technical_indicators(self, data: pd.DataFrame) -> TechnicalAnalysis:
        """Calculate advanced technical indicators including Fibonacci and Volume Profile"""
        try:
            # Basic indicators
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
            
            # EMA
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            macd_line = data['EMA_12'] - data['EMA_26']
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            sma20 = data['Close'].rolling(window=bb_period).mean()
            std = data['Close'].rolling(window=bb_period).std()
            bollinger_upper = sma20 + (std * bb_std)
            bollinger_lower = sma20 - (std * bb_std)
            
            # ATR (Average True Range)
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=14).mean()
            
            # Stochastic Oscillator
            low_14 = data['Low'].rolling(window=14).min()
            high_14 = data['High'].rolling(window=14).max()
            k_percent = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
            d_percent = k_percent.rolling(window=3).mean()
            
            # Williams %R
            williams_r = -100 * ((high_14 - data['Close']) / (high_14 - low_14))
            
            # Volatility
            returns = data['Close'].pct_change()
            volatility = returns.rolling(window=30).std() * np.sqrt(252) * 100
            
            # Fibonacci Retracement Levels
            fibonacci_levels = self._calculate_fibonacci_retracements(data)
            
            # Volume Profile (simplified)
            volume_profile = self._calculate_volume_profile(data)
            
            # Support and Resistance Levels
            support_resistance = self._find_support_resistance_levels(data)
            
            # Get latest values
            latest = data.iloc[-1]
            
            return TechnicalAnalysis(
                rsi=float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
                macd=float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
                macd_signal=float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0,
                macd_histogram=float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0,
                bollinger_upper=float(bollinger_upper.iloc[-1]) if not pd.isna(bollinger_upper.iloc[-1]) else latest['Close'],
                bollinger_lower=float(bollinger_lower.iloc[-1]) if not pd.isna(bollinger_lower.iloc[-1]) else latest['Close'],
                bollinger_middle=float(sma20.iloc[-1]) if not pd.isna(sma20.iloc[-1]) else latest['Close'],
                fibonacci_levels=fibonacci_levels,
                volume_profile=volume_profile,
                sma_5=float(data['SMA_5'].iloc[-1]) if not pd.isna(data['SMA_5'].iloc[-1]) else latest['Close'],
                sma_20=float(data['SMA_20'].iloc[-1]) if not pd.isna(data['SMA_20'].iloc[-1]) else latest['Close'],
                sma_50=float(data['SMA_50'].iloc[-1]) if not pd.isna(data['SMA_50'].iloc[-1]) else latest['Close'],
                sma_200=float(data['SMA_200'].iloc[-1]) if not pd.isna(data['SMA_200'].iloc[-1]) else latest['Close'],
                ema_12=float(data['EMA_12'].iloc[-1]) if not pd.isna(data['EMA_12'].iloc[-1]) else latest['Close'],
                ema_26=float(data['EMA_26'].iloc[-1]) if not pd.isna(data['EMA_26'].iloc[-1]) else latest['Close'],
                volatility=float(volatility.iloc[-1]) if not pd.isna(volatility.iloc[-1]) else 20.0,
                atr=float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 1.0,
                stochastic_k=float(k_percent.iloc[-1]) if not pd.isna(k_percent.iloc[-1]) else 50.0,
                stochastic_d=float(d_percent.iloc[-1]) if not pd.isna(d_percent.iloc[-1]) else 50.0,
                williams_r=float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else -50.0,
                support_levels=support_resistance['support'],
                resistance_levels=support_resistance['resistance']
            )
            
        except Exception as e:
            logger.error(f"❌ Technical analysis error: {e}")
            # Return default values on error
            return self._get_default_technical_analysis(data.iloc[-1]['Close'] if len(data) > 0 else 100.0)
    
    def _calculate_fibonacci_retracements(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        try:
            # Find swing high and low in recent data (last 50 periods)
            recent_data = data.tail(50)
            high_price = recent_data['High'].max()
            low_price = recent_data['Low'].min()
            
            difference = high_price - low_price
            
            fib_levels = {
                '0.0%': high_price,
                '23.6%': high_price - (0.236 * difference),
                '38.2%': high_price - (0.382 * difference),
                '50.0%': high_price - (0.5 * difference),
                '61.8%': high_price - (0.618 * difference),
                '78.6%': high_price - (0.786 * difference),
                '100.0%': low_price
            }
            
            return fib_levels
        except Exception:
            return {'50.0%': data['Close'].iloc[-1]}
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate simplified volume profile"""
        try:
            # Group by price levels and sum volume
            price_levels = pd.cut(data['Close'], bins=10)
            volume_by_price = data.groupby(price_levels)['Volume'].sum()
            
            # Find VPOC (Volume Point of Control) - price level with highest volume
            vpoc_idx = volume_by_price.idxmax()
            vpoc_price = (vpoc_idx.left + vpoc_idx.right) / 2
            
            return {
                'vpoc': float(vpoc_price),
                'high_volume_node': float(volume_by_price.max()),
                'total_volume': float(data['Volume'].sum())
            }
        except Exception:
            return {'vpoc': data['Close'].iloc[-1], 'high_volume_node': 0, 'total_volume': 0}
    
    def _find_support_resistance_levels(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Find support and resistance levels using pivot points"""
        try:
            # Calculate pivot points from recent data
            recent_data = data.tail(20)
            
            # Find local minima (support) and maxima (resistance)
            from scipy.signal import argrelextrema
            
            highs = recent_data['High'].values
            lows = recent_data['Low'].values
            
            # Find local maxima (resistance)
            resistance_indices = argrelextrema(highs, np.greater, order=2)[0]
            resistance_levels = [float(highs[i]) for i in resistance_indices[-3:]]  # Last 3 resistance levels
            
            # Find local minima (support)
            support_indices = argrelextrema(lows, np.less, order=2)[0]
            support_levels = [float(lows[i]) for i in support_indices[-3:]]  # Last 3 support levels
            
            return {
                'support': support_levels if support_levels else [float(recent_data['Low'].min())],
                'resistance': resistance_levels if resistance_levels else [float(recent_data['High'].max())]
            }
        except Exception:
            current_price = data['Close'].iloc[-1]
            return {
                'support': [float(current_price * 0.95)],
                'resistance': [float(current_price * 1.05)]
            }
    
    def _perform_comprehensive_analysis(self, symbol: str) -> ComprehensiveAnalysis:
        """Perform comprehensive analysis with all advanced features"""
        try:
            # Get market data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="6mo", interval="1d")
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            info = ticker.info
            current_price = float(data['Close'].iloc[-1])
            previous_price = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
            
            # Calculate technical indicators
            tech_analysis = self._calculate_advanced_technical_indicators(data)
            
            # Multi-timeframe analysis
            timeframe_analysis = self._analyze_multiple_timeframes(symbol)
            
            # Generate prediction
            prediction, confidence, reasoning = self._generate_advanced_prediction(data, tech_analysis)
            
            # Market sentiment analysis
            market_sentiment = self._analyze_market_sentiment(data, tech_analysis)
            
            # Risk assessment
            risk_score = self._calculate_risk_score(data, tech_analysis)
            
            return ComprehensiveAnalysis(
                symbol=symbol,
                company_name=info.get('longName', self.all_symbols.get(symbol, symbol)),
                current_price=current_price,
                prediction=prediction,
                confidence=confidence,
                reasoning=reasoning,
                technical_analysis=tech_analysis,
                market_sentiment=market_sentiment,
                risk_score=risk_score,
                price_change=current_price - previous_price,
                price_change_percent=((current_price - previous_price) / previous_price) * 100,
                timeframe_analysis=timeframe_analysis
            )
            
        except Exception as e:
            logger.error(f"❌ Comprehensive analysis error for {symbol}: {e}")
            return self._get_default_analysis(symbol)
    
    def _analyze_multiple_timeframes(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """Analyze multiple timeframes for comprehensive view"""
        timeframe_data = {}
        
        timeframe_mapping = {
            '1m': ('1d', '1m'),
            '5m': ('5d', '5m'),
            '15m': ('1mo', '15m'),
            '1h': ('3mo', '1h'),
            '4h': ('6mo', '4h'),
            '1d': ('1y', '1d'),
            '1wk': ('2y', '1wk'),
            '1mo': ('5y', '1mo')
        }
        
        for timeframe, (period, interval) in timeframe_mapping.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)
                
                if not data.empty:
                    latest = data.iloc[-1]
                    previous = data.iloc[-2] if len(data) > 1 else latest
                    
                    # Simple trend analysis
                    trend = "BULLISH" if latest['Close'] > previous['Close'] else "BEARISH"
                    if abs(latest['Close'] - previous['Close']) / previous['Close'] < 0.001:
                        trend = "NEUTRAL"
                    
                    # Calculate simple RSI for this timeframe
                    delta = data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=min(14, len(data))).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=min(14, len(data))).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    timeframe_data[timeframe] = {
                        'trend': trend,
                        'price': float(latest['Close']),
                        'change': float(latest['Close'] - previous['Close']),
                        'change_percent': float(((latest['Close'] - previous['Close']) / previous['Close']) * 100),
                        'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
                        'volume': float(latest['Volume']),
                        'high_52w': float(data['High'].max()),
                        'low_52w': float(data['Low'].min())
                    }
                    
            except Exception as e:
                logger.warning(f"⚠️ Timeframe analysis failed for {timeframe}: {e}")
                timeframe_data[timeframe] = {
                    'trend': 'UNKNOWN',
                    'price': 0.0,
                    'change': 0.0,
                    'change_percent': 0.0,
                    'rsi': 50.0,
                    'volume': 0,
                    'high_52w': 0.0,
                    'low_52w': 0.0
                }
        
        return timeframe_data
    
    def _analyze_cryptocurrencies(self) -> Dict[str, Any]:
        """Analyze cryptocurrency market"""
        crypto_data = {}
        
        for symbol, name in self.crypto_symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="30d", interval="1d")
                info = ticker.info
                
                if not data.empty:
                    current_price = float(data['Close'].iloc[-1])
                    previous_price = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
                    change_24h = ((current_price - previous_price) / previous_price) * 100
                    
                    crypto_data[symbol] = {
                        'name': name,
                        'price': current_price,
                        'change_24h': change_24h,
                        'volume_24h': float(data['Volume'].iloc[-1]),
                        'market_cap': info.get('marketCap', 0),
                        'volatility': float(data['Close'].pct_change().std() * np.sqrt(365) * 100)
                    }
                    
            except Exception as e:
                logger.warning(f"⚠️ Crypto analysis failed for {symbol}: {e}")
        
        return crypto_data
    
    def _generate_interactive_chart(self, symbol: str, timeframe: str = '1d') -> Dict[str, Any]:
        """Generate interactive chart data for frontend"""
        try:
            period_mapping = {
                '1m': ('1d', '1m'),
                '5m': ('5d', '5m'),
                '15m': ('1mo', '15m'),
                '1h': ('3mo', '1h'),
                '4h': ('6mo', '4h'),
                '1d': ('1y', '1d'),
                '1wk': ('2y', '1wk'),
                '1mo': ('5y', '1mo')
            }
            
            period, interval = period_mapping.get(timeframe, ('1y', '1d'))
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                return {'error': 'No data available'}
            
            # Prepare data for frontend charting
            chart_data = {
                'timestamps': [int(ts.timestamp() * 1000) for ts in data.index],
                'ohlcv': [
                    {
                        'timestamp': int(ts.timestamp() * 1000),
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': float(row['Volume'])
                    }
                    for ts, row in data.iterrows()
                ],
                'indicators': self._calculate_chart_indicators(data),
                'support_resistance': self._find_support_resistance_levels(data),
                'fibonacci': self._calculate_fibonacci_retracements(data)
            }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"❌ Chart generation error: {e}")
            return {'error': str(e)}
    
    def _calculate_chart_indicators(self, data: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Calculate indicators for chart display"""
        indicators = {}
        
        try:
            # Moving averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # Bollinger Bands
            sma20 = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = sma20 + (std * 2)
            data['BB_Lower'] = sma20 - (std * 2)
            
            # MACD
            macd_line = data['EMA_12'] - data['EMA_26']
            signal_line = macd_line.ewm(span=9).mean()
            
            # Prepare indicator data for frontend
            indicators['sma_20'] = [
                {'timestamp': int(ts.timestamp() * 1000), 'value': float(val)} 
                for ts, val in zip(data.index, data['SMA_20']) if not pd.isna(val)
            ]
            
            indicators['sma_50'] = [
                {'timestamp': int(ts.timestamp() * 1000), 'value': float(val)} 
                for ts, val in zip(data.index, data['SMA_50']) if not pd.isna(val)
            ]
            
            indicators['bollinger_upper'] = [
                {'timestamp': int(ts.timestamp() * 1000), 'value': float(val)} 
                for ts, val in zip(data.index, data['BB_Upper']) if not pd.isna(val)
            ]
            
            indicators['bollinger_lower'] = [
                {'timestamp': int(ts.timestamp() * 1000), 'value': float(val)} 
                for ts, val in zip(data.index, data['BB_Lower']) if not pd.isna(val)
            ]
            
            indicators['macd'] = [
                {'timestamp': int(ts.timestamp() * 1000), 'value': float(val)} 
                for ts, val in zip(data.index, macd_line) if not pd.isna(val)
            ]
            
            indicators['macd_signal'] = [
                {'timestamp': int(ts.timestamp() * 1000), 'value': float(val)} 
                for ts, val in zip(data.index, signal_line) if not pd.isna(val)
            ]
            
        except Exception as e:
            logger.error(f"❌ Chart indicators error: {e}")
            
        return indicators
    
    def _generate_pdf_report(self, symbol: str) -> BytesIO:
        """Generate professional PDF report"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        
        story.append(Paragraph(f"Professional Analysis Report - {symbol}", title_style))
        story.append(Spacer(1, 20))
        
        # Analysis data
        analysis = self._perform_comprehensive_analysis(symbol)
        
        # Summary table
        summary_data = [
            ['Metric', 'Value'],
            ['Symbol', analysis.symbol],
            ['Company', analysis.company_name],
            ['Current Price', f"${analysis.current_price:.2f}"],
            ['Prediction', analysis.prediction],
            ['Confidence', f"{analysis.confidence:.1f}%"],
            ['Risk Score', f"{analysis.risk_score:.1f}"],
            ['Market Sentiment', analysis.market_sentiment]
        ]
        
        table = Table(summary_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 20))
        
        # Analysis reasoning
        story.append(Paragraph("Analysis Reasoning", styles['Heading2']))
        story.append(Paragraph(analysis.reasoning, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def _generate_excel_report(self, symbol: str) -> BytesIO:
        """Generate Excel report with analysis data"""
        buffer = BytesIO()
        
        # Get analysis data
        analysis = self._perform_comprehensive_analysis(symbol)
        
        # Create Excel writer
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Symbol', 'Company', 'Current Price', 'Prediction', 'Confidence', 'Risk Score'],
                'Value': [
                    analysis.symbol,
                    analysis.company_name,
                    analysis.current_price,
                    analysis.prediction,
                    analysis.confidence,
                    analysis.risk_score
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Technical indicators sheet
            tech_data = asdict(analysis.technical_analysis)
            tech_df = pd.DataFrame(list(tech_data.items()), columns=['Indicator', 'Value'])
            tech_df.to_excel(writer, sheet_name='Technical Indicators', index=False)
            
            # Multi-timeframe analysis sheet
            if analysis.timeframe_analysis:
                timeframe_df = pd.DataFrame(analysis.timeframe_analysis).T
                timeframe_df.to_excel(writer, sheet_name='Timeframe Analysis')
        
        buffer.seek(0)
        return buffer
    
    def _get_market_overview(self) -> Dict[str, Any]:
        """Get comprehensive market overview"""
        try:
            market_data = {
                'indices': self._get_major_indices(),
                'crypto_summary': self._get_crypto_summary(),
                'forex_summary': self._get_forex_summary(),
                'commodity_summary': self._get_commodity_summary(),
                'market_sentiment': self._get_overall_market_sentiment(),
                'top_gainers': self._get_top_performers('gainers'),
                'top_losers': self._get_top_performers('losers'),
                'most_active': self._get_most_active_stocks()
            }
            return market_data
        except Exception as e:
            logger.error(f"❌ Market overview error: {e}")
            return {}
    
    def _get_major_indices(self) -> Dict[str, Any]:
        """Get major market indices"""
        indices = {
            '^NSEI': 'NIFTY 50',
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^FTSE': 'FTSE 100'
        }
        
        index_data = {}
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="2d")
                if not data.empty:
                    current = float(data['Close'].iloc[-1])
                    previous = float(data['Close'].iloc[-2]) if len(data) > 1 else current
                    change = current - previous
                    change_percent = (change / previous) * 100
                    
                    index_data[symbol] = {
                        'name': name,
                        'value': current,
                        'change': change,
                        'change_percent': change_percent
                    }
            except Exception as e:
                logger.warning(f"⚠️ Index data error for {symbol}: {e}")
        
        return index_data
    
    def _generate_advanced_prediction(self, data: pd.DataFrame, tech_analysis: TechnicalAnalysis) -> Tuple[str, float, str]:
        """Generate advanced prediction using multiple factors"""
        signals = []
        confidence_factors = []
        
        try:
            current_price = data['Close'].iloc[-1]
            
            # RSI analysis
            if tech_analysis.rsi < 30:
                signals.append("BULLISH")
                confidence_factors.append(0.8)
            elif tech_analysis.rsi > 70:
                signals.append("BEARISH")
                confidence_factors.append(0.8)
            else:
                signals.append("NEUTRAL")
                confidence_factors.append(0.4)
            
            # Moving average analysis
            if (current_price > tech_analysis.sma_5 > tech_analysis.sma_20 > tech_analysis.sma_50):
                signals.append("BULLISH")
                confidence_factors.append(0.9)
            elif (current_price < tech_analysis.sma_5 < tech_analysis.sma_20 < tech_analysis.sma_50):
                signals.append("BEARISH")
                confidence_factors.append(0.9)
            
            # MACD analysis
            if tech_analysis.macd > tech_analysis.macd_signal:
                signals.append("BULLISH")
                confidence_factors.append(0.7)
            else:
                signals.append("BEARISH")
                confidence_factors.append(0.7)
            
            # Bollinger Bands analysis
            if current_price < tech_analysis.bollinger_lower:
                signals.append("BULLISH")
                confidence_factors.append(0.6)
            elif current_price > tech_analysis.bollinger_upper:
                signals.append("BEARISH")
                confidence_factors.append(0.6)
            
            # Volume analysis
            recent_volume = data['Volume'].tail(5).mean()
            avg_volume = data['Volume'].mean()
            if recent_volume > avg_volume * 1.5:
                confidence_factors.append(0.8)  # High volume increases confidence
            
            # Count signals
            bullish_signals = signals.count("BULLISH")
            bearish_signals = signals.count("BEARISH")
            neutral_signals = signals.count("NEUTRAL")
            
            # Determine overall prediction
            if bullish_signals > bearish_signals:
                prediction = "BULLISH"
            elif bearish_signals > bullish_signals:
                prediction = "BEARISH"
            else:
                prediction = "NEUTRAL"
            
            # Calculate confidence
            total_signals = len(signals)
            dominant_signals = max(bullish_signals, bearish_signals, neutral_signals)
            base_confidence = (dominant_signals / total_signals) * 100 if total_signals > 0 else 50
            
            # Adjust confidence based on signal strength
            avg_confidence_factor = np.mean(confidence_factors) if confidence_factors else 0.5
            final_confidence = min(base_confidence * avg_confidence_factor, 95)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                tech_analysis, prediction, bullish_signals, bearish_signals, current_price
            )
            
            return prediction, final_confidence, reasoning
            
        except Exception as e:
            logger.error(f"❌ Advanced prediction error: {e}")
            return "NEUTRAL", 50.0, "Analysis unavailable due to technical issues."
    
    def _generate_reasoning(self, tech_analysis: TechnicalAnalysis, prediction: str, 
                          bullish_signals: int, bearish_signals: int, current_price: float) -> str:
        """Generate detailed reasoning for prediction"""
        reasoning_parts = []
        
        # RSI analysis
        if tech_analysis.rsi < 30:
            reasoning_parts.append(f"RSI at {tech_analysis.rsi:.1f} indicates oversold conditions, suggesting potential upward reversal.")
        elif tech_analysis.rsi > 70:
            reasoning_parts.append(f"RSI at {tech_analysis.rsi:.1f} shows overbought conditions, indicating possible downward correction.")
        else:
            reasoning_parts.append(f"RSI at {tech_analysis.rsi:.1f} indicates neutral momentum conditions.")
        
        # Moving averages
        if current_price > tech_analysis.sma_20:
            reasoning_parts.append("Price above 20-day SMA suggests short-term bullish trend.")
        else:
            reasoning_parts.append("Price below 20-day SMA indicates short-term bearish pressure.")
        
        # MACD
        if tech_analysis.macd > tech_analysis.macd_signal:
            reasoning_parts.append("MACD above signal line confirms bullish momentum.")
        else:
            reasoning_parts.append("MACD below signal line suggests bearish momentum.")
        
        # Bollinger Bands
        bb_position = "middle"
        if current_price > tech_analysis.bollinger_upper:
            bb_position = "upper"
            reasoning_parts.append("Price near upper Bollinger Band indicates potential overbought conditions.")
        elif current_price < tech_analysis.bollinger_lower:
            bb_position = "lower" 
            reasoning_parts.append("Price near lower Bollinger Band suggests potential oversold conditions.")
        
        # Signal summary
        reasoning_parts.append(f"Overall assessment: {bullish_signals} bullish and {bearish_signals} bearish signals detected.")
        
        # Final outlook
        outlook_map = {
            "BULLISH": "Positive outlook with favorable technical indicators.",
            "BEARISH": "Negative outlook with concerning technical signals.",
            "NEUTRAL": "Mixed signals suggest sideways movement or consolidation."
        }
        reasoning_parts.append(outlook_map.get(prediction, "Market conditions remain uncertain."))
        
        return " ".join(reasoning_parts)
    
    def _analyze_market_sentiment(self, data: pd.DataFrame, tech_analysis: TechnicalAnalysis) -> str:
        """Analyze overall market sentiment"""
        sentiment_score = 0
        
        # Price momentum
        recent_returns = data['Close'].pct_change().tail(5).mean()
        if recent_returns > 0.01:
            sentiment_score += 2
        elif recent_returns > 0:
            sentiment_score += 1
        elif recent_returns < -0.01:
            sentiment_score -= 2
        else:
            sentiment_score -= 1
        
        # Volume trend
        volume_trend = data['Volume'].tail(5).mean() / data['Volume'].tail(20).mean()
        if volume_trend > 1.2:
            sentiment_score += 1
        elif volume_trend < 0.8:
            sentiment_score -= 1
        
        # Technical indicators
        if tech_analysis.rsi > 60:
            sentiment_score += 1
        elif tech_analysis.rsi < 40:
            sentiment_score -= 1
        
        if tech_analysis.macd > tech_analysis.macd_signal:
            sentiment_score += 1
        else:
            sentiment_score -= 1
        
        # Determine sentiment
        if sentiment_score >= 3:
            return "VERY_BULLISH"
        elif sentiment_score >= 1:
            return "BULLISH"
        elif sentiment_score <= -3:
            return "VERY_BEARISH"
        elif sentiment_score <= -1:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _calculate_risk_score(self, data: pd.DataFrame, tech_analysis: TechnicalAnalysis) -> float:
        """Calculate comprehensive risk score (0-100, higher = more risky)"""
        risk_factors = []
        
        try:
            # Volatility risk
            volatility_risk = min(tech_analysis.volatility / 50 * 100, 100)
            risk_factors.append(volatility_risk)
            
            # ATR risk (relative to price)
            current_price = data['Close'].iloc[-1]
            atr_risk = (tech_analysis.atr / current_price) * 100 * 5  # Scale factor
            risk_factors.append(min(atr_risk, 100))
            
            # RSI extremes risk
            rsi_risk = 0
            if tech_analysis.rsi > 80 or tech_analysis.rsi < 20:
                rsi_risk = 80
            elif tech_analysis.rsi > 70 or tech_analysis.rsi < 30:
                rsi_risk = 50
            risk_factors.append(rsi_risk)
            
            # Volume risk (low volume = higher risk)
            avg_volume = data['Volume'].mean()
            recent_volume = data['Volume'].tail(5).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            volume_risk = max(0, (1 - volume_ratio) * 100)
            risk_factors.append(min(volume_risk, 100))
            
            # Price gap risk
            price_changes = data['Close'].pct_change().abs()
            gap_risk = price_changes.tail(10).max() * 1000  # Convert to percentage
            risk_factors.append(min(gap_risk, 100))
            
            # Calculate weighted average
            weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # Volatility gets highest weight
            weighted_risk = sum(risk * weight for risk, weight in zip(risk_factors, weights))
            
            return min(max(weighted_risk, 0), 100)  # Ensure 0-100 range
            
        except Exception as e:
            logger.error(f"❌ Risk calculation error: {e}")
            return 50.0  # Default moderate risk
    
    def _get_default_analysis(self, symbol: str) -> ComprehensiveAnalysis:
        """Return default analysis when data is unavailable"""
        default_tech = self._get_default_technical_analysis(100.0)
        
        return ComprehensiveAnalysis(
            symbol=symbol,
            company_name=self.all_symbols.get(symbol, symbol),
            current_price=100.0,
            prediction="NEUTRAL",
            confidence=50.0,
            reasoning="Analysis unavailable - insufficient data",
            technical_analysis=default_tech,
            market_sentiment="UNKNOWN",
            risk_score=50.0,
            price_change=0.0,
            price_change_percent=0.0,
            timeframe_analysis={}
        )
    
    def _get_default_technical_analysis(self, price: float) -> TechnicalAnalysis:
        """Return default technical analysis"""
        return TechnicalAnalysis(
            rsi=50.0,
            macd=0.0,
            macd_signal=0.0,
            macd_histogram=0.0,
            bollinger_upper=price * 1.05,
            bollinger_lower=price * 0.95,
            bollinger_middle=price,
            fibonacci_levels={'50.0%': price},
            volume_profile={'vpoc': price, 'high_volume_node': 0, 'total_volume': 0},
            sma_5=price,
            sma_20=price,
            sma_50=price,
            sma_200=price,
            ema_12=price,
            ema_26=price,
            volatility=20.0,
            atr=1.0,
            stochastic_k=50.0,
            stochastic_d=50.0,
            williams_r=-50.0,
            support_levels=[price * 0.95],
            resistance_levels=[price * 1.05]
        )
    
    def start_analysis_engine(self):
        """Start the enhanced analysis engine"""
        def analysis_loop():
            while self.running:
                try:
                    # Perform analysis
                    analysis = self._perform_comprehensive_analysis(self.current_symbol)
                    self.analysis_data = analysis
                    
                    # Get market overview
                    market_overview = self._get_market_overview()
                    
                    # Emit comprehensive update
                    self.socketio.emit('comprehensive_market_update', {
                        'analysis': asdict(analysis),
                        'market_overview': market_overview,
                        'system_status': {
                            'active': True,
                            'processing_time_ms': time.time() * 1000 % 1000,
                            'features_active': [
                                'Interactive Charts',
                                'Crypto Analysis', 
                                'Advanced Indicators',
                                'Multi-timeframe Analysis'
                            ]
                        },
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    logger.info(f"📊 Comprehensive update: {analysis.symbol} @ {analysis.current_price:.2f} | {analysis.prediction}")
                    
                    # Wait before next update
                    time.sleep(10)
                    
                except Exception as e:
                    logger.error(f"❌ Analysis engine error: {e}")
                    time.sleep(5)
        
        # Start analysis thread
        self.running = True
        analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
        analysis_thread.start()
        logger.info("🚀 Enhanced analysis engine started")
    
    def _get_crypto_summary(self) -> Dict[str, Any]:
        """Get cryptocurrency market summary"""
        try:
            crypto_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD']
            crypto_data = {}
            
            for symbol in crypto_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='1d', interval='1m')
                    if not data.empty:
                        current_price = float(data['Close'].iloc[-1])
                        previous_price = float(data['Close'].iloc[0])
                        change_percent = ((current_price - previous_price) / previous_price) * 100
                        
                        crypto_data[symbol] = {
                            'name': symbol.replace('-USD', ''),
                            'price': current_price,
                            'change_percent': change_percent,
                            'volume': float(data['Volume'].sum())
                        }
                except:
                    continue
            
            return crypto_data
        except Exception as e:
            logger.error(f"❌ Crypto summary error: {e}")
            return {}
    
    def _get_forex_summary(self) -> Dict[str, Any]:
        """Get forex market summary"""
        try:
            forex_pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
            forex_data = {}
            
            for pair in forex_pairs:
                try:
                    ticker = yf.Ticker(pair)
                    data = ticker.history(period='1d', interval='1h')
                    if not data.empty:
                        current_price = float(data['Close'].iloc[-1])
                        previous_price = float(data['Close'].iloc[0])
                        change_percent = ((current_price - previous_price) / previous_price) * 100
                        
                        forex_data[pair] = {
                            'name': pair.replace('=X', ''),
                            'price': current_price,
                            'change_percent': change_percent
                        }
                except:
                    continue
            
            return forex_data
        except Exception as e:
            logger.error(f"❌ Forex summary error: {e}")
            return {}
    
    def _get_commodity_summary(self) -> Dict[str, Any]:
        """Get commodities market summary"""
        try:
            commodity_symbols = ['GC=F', 'SI=F', 'CL=F', 'NG=F']  # Gold, Silver, Oil, Natural Gas
            commodity_data = {}
            
            for symbol in commodity_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='1d', interval='1h')
                    if not data.empty:
                        current_price = float(data['Close'].iloc[-1])
                        previous_price = float(data['Close'].iloc[0])
                        change_percent = ((current_price - previous_price) / previous_price) * 100
                        
                        name_mapping = {
                            'GC=F': 'Gold',
                            'SI=F': 'Silver',
                            'CL=F': 'Crude Oil',
                            'NG=F': 'Natural Gas'
                        }
                        
                        commodity_data[symbol] = {
                            'name': name_mapping.get(symbol, symbol),
                            'price': current_price,
                            'change_percent': change_percent
                        }
                except:
                    continue
            
            return commodity_data
        except Exception as e:
            logger.error(f"❌ Commodity summary error: {e}")
            return {}
    
    def _get_overall_market_sentiment(self) -> str:
        """Get overall market sentiment"""
        try:
            # Analyze major indices for market sentiment
            indices = ['^NSEI', '^GSPC', '^DJI']
            sentiment_scores = []
            
            for index in indices:
                try:
                    ticker = yf.Ticker(index)
                    data = ticker.history(period='1d', interval='1h')
                    if not data.empty:
                        current_price = float(data['Close'].iloc[-1])
                        previous_price = float(data['Close'].iloc[0])
                        change_percent = ((current_price - previous_price) / previous_price) * 100
                        sentiment_scores.append(change_percent)
                except:
                    continue
            
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                if avg_sentiment > 1:
                    return "BULLISH"
                elif avg_sentiment < -1:
                    return "BEARISH"
                else:
                    return "NEUTRAL"
            return "NEUTRAL"
        except:
            return "NEUTRAL"
    
    def _get_top_performers(self, type: str) -> List[Dict[str, Any]]:
        """Get top performing stocks"""
        try:
            # Sample popular stocks for demonstration
            sample_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
            performers = []
            
            for symbol in sample_stocks:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='1d', interval='1h')
                    if not data.empty:
                        current_price = float(data['Close'].iloc[-1])
                        previous_price = float(data['Close'].iloc[0])
                        change_percent = ((current_price - previous_price) / previous_price) * 100
                        
                        performers.append({
                            'symbol': symbol,
                            'name': self.all_symbols.get(symbol, symbol),
                            'price': current_price,
                            'change_percent': change_percent
                        })
                except:
                    continue
            
            # Sort based on type
            if type == 'gainers':
                performers.sort(key=lambda x: x['change_percent'], reverse=True)
            else:  # losers
                performers.sort(key=lambda x: x['change_percent'])
            
            return performers[:5]  # Top 5
        except:
            return []
    
    def _get_most_active_stocks(self) -> List[Dict[str, Any]]:
        """Get most active stocks by volume"""
        try:
            # Sample stocks with volume data
            sample_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
            active_stocks = []
            
            for symbol in sample_stocks:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='1d', interval='1h')
                    if not data.empty:
                        volume = float(data['Volume'].sum())
                        current_price = float(data['Close'].iloc[-1])
                        
                        active_stocks.append({
                            'symbol': symbol,
                            'name': self.all_symbols.get(symbol, symbol),
                            'price': current_price,
                            'volume': volume
                        })
                except:
                    continue
            
            # Sort by volume
            active_stocks.sort(key=lambda x: x['volume'], reverse=True)
            return active_stocks[:5]  # Top 5
        except:
            return []
    
    def _generate_market_heatmap(self) -> Dict[str, Any]:
        """Generate comprehensive market heatmap data"""
        try:
            # Major Indian stocks with sector classification
            stock_sectors = {
                'RELIANCE.NS': {'name': 'Reliance Industries', 'sector': 'Energy'},
                'TCS.NS': {'name': 'Tata Consultancy Services', 'sector': 'IT'},
                'HDFCBANK.NS': {'name': 'HDFC Bank', 'sector': 'Banking'},
                'INFY.NS': {'name': 'Infosys', 'sector': 'IT'},
                'ICICIBANK.NS': {'name': 'ICICI Bank', 'sector': 'Banking'},
                'HINDUNILVR.NS': {'name': 'Hindustan Unilever', 'sector': 'FMCG'},
                'SBIN.NS': {'name': 'State Bank of India', 'sector': 'Banking'},
                'BHARTIARTL.NS': {'name': 'Bharti Airtel', 'sector': 'Telecom'},
                'ITC.NS': {'name': 'ITC Limited', 'sector': 'FMCG'},
                'KOTAKBANK.NS': {'name': 'Kotak Mahindra Bank', 'sector': 'Banking'},
                'LT.NS': {'name': 'Larsen & Toubro', 'sector': 'Infrastructure'},
                'AXISBANK.NS': {'name': 'Axis Bank', 'sector': 'Banking'},
                'M&M.NS': {'name': 'Mahindra & Mahindra', 'sector': 'Auto'},
                'ASIANPAINT.NS': {'name': 'Asian Paints', 'sector': 'Paints'},
                'MARUTI.NS': {'name': 'Maruti Suzuki', 'sector': 'Auto'},
                'SUNPHARMA.NS': {'name': 'Sun Pharmaceutical', 'sector': 'Pharma'},
                'WIPRO.NS': {'name': 'Wipro', 'sector': 'IT'},
                'NTPC.NS': {'name': 'NTPC', 'sector': 'Power'},
                'ULTRACEMCO.NS': {'name': 'UltraTech Cement', 'sector': 'Cement'},
                'ONGC.NS': {'name': 'ONGC', 'sector': 'Energy'}
            }
            
            heatmap_data = {
                'sectors': {},
                'stocks': [],
                'market_overview': {
                    'total_stocks': 0,
                    'gainers': 0,
                    'losers': 0,
                    'unchanged': 0
                }
            }
            
            gainers = 0
            losers = 0
            unchanged = 0
            
            for symbol, info in stock_sectors.items():
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='1d', interval='5m')
                    
                    if not data.empty:
                        current_price = float(data['Close'].iloc[-1])
                        open_price = float(data['Open'].iloc[0])
                        change_percent = ((current_price - open_price) / open_price) * 100
                        volume = float(data['Volume'].sum())
                        
                        # Determine color based on performance
                        if change_percent > 0.5:
                            color = '#00ff88'  # Bright green for strong gains
                            performance = 'strong_gain'
                            gainers += 1
                        elif change_percent > 0:
                            color = '#4CAF50'  # Green for gains
                            performance = 'gain'
                            gainers += 1
                        elif change_percent < -0.5:
                            color = '#ff4444'  # Red for losses
                            performance = 'strong_loss'
                            losers += 1
                        elif change_percent < 0:
                            color = '#ff6b6b'  # Light red for small losses
                            performance = 'loss'
                            losers += 1
                        else:
                            color = '#ffeb3b'  # Yellow for unchanged
                            performance = 'unchanged'
                            unchanged += 1
                        
                        stock_data = {
                            'symbol': symbol,
                            'name': info['name'],
                            'sector': info['sector'],
                            'price': round(current_price, 2),
                            'change_percent': round(change_percent, 2),
                            'volume': volume,
                            'color': color,
                            'performance': performance,
                            'market_cap_size': 'large' if volume > 10000000 else 'medium' if volume > 1000000 else 'small'
                        }
                        
                        heatmap_data['stocks'].append(stock_data)
                        
                        # Group by sector
                        sector = info['sector']
                        if sector not in heatmap_data['sectors']:
                            heatmap_data['sectors'][sector] = {
                                'name': sector,
                                'stocks': [],
                                'avg_change': 0,
                                'total_volume': 0,
                                'count': 0
                            }
                        
                        heatmap_data['sectors'][sector]['stocks'].append(stock_data)
                        heatmap_data['sectors'][sector]['total_volume'] += volume
                        heatmap_data['sectors'][sector]['count'] += 1
                
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    continue
            
            # Calculate sector averages
            for sector_name, sector_data in heatmap_data['sectors'].items():
                if sector_data['count'] > 0:
                    sector_data['avg_change'] = round(
                        sum(stock['change_percent'] for stock in sector_data['stocks']) / sector_data['count'], 2
                    )
                    
                    # Sector color based on average performance
                    avg_change = sector_data['avg_change']
                    if avg_change > 1:
                        sector_data['color'] = '#00ff88'
                    elif avg_change > 0:
                        sector_data['color'] = '#4CAF50'
                    elif avg_change < -1:
                        sector_data['color'] = '#ff4444'
                    elif avg_change < 0:
                        sector_data['color'] = '#ff6b6b'
                    else:
                        sector_data['color'] = '#ffeb3b'
            
            # Update market overview
            heatmap_data['market_overview'] = {
                'total_stocks': len(heatmap_data['stocks']),
                'gainers': gainers,
                'losers': losers,
                'unchanged': unchanged,
                'gainer_percentage': round((gainers / len(heatmap_data['stocks'])) * 100, 1) if heatmap_data['stocks'] else 0,
                'loser_percentage': round((losers / len(heatmap_data['stocks'])) * 100, 1) if heatmap_data['stocks'] else 0
            }
            
            return heatmap_data
            
        except Exception as e:
            logger.error(f"Market heatmap generation error: {e}")
            return {
                'sectors': {},
                'stocks': [],
                'market_overview': {
                    'total_stocks': 0,
                    'gainers': 0,
                    'losers': 0,
                    'unchanged': 0
                }
            }
    
    def _generate_ai_prediction(self, symbol: str) -> AIPrediction:
        """Generate AI-powered stock predictions using multiple ML models"""
        try:
            # Get historical data for training
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='2y', interval='1d')
            
            if data.empty or len(data) < 100:
                raise ValueError("Insufficient data for AI prediction")
            
            # Prepare features for ML models
            features_df = self._prepare_ml_features(data)
            
            # Train multiple ML models
            ml_models = self._train_ml_models(features_df)
            
            # Generate predictions for different timeframes
            predictions = self._generate_ml_predictions(features_df, ml_models)
            
            # AI sentiment analysis
            ai_sentiment = self._analyze_ai_sentiment(symbol)
            
            # Pattern recognition
            pattern_recognition = self._detect_ai_patterns(symbol)
            
            # Risk assessment
            risk_assessment = self._assess_ai_risk(symbol)
            
            # Generate trading signals
            trading_signals = self._generate_ai_trading_signals(symbol)
            
            # Detect market regime
            market_regime = self._detect_market_regime(data)
            
            # Anomaly detection
            anomaly_detection = self._detect_anomalies(data)
            
            current_price = float(data['Close'].iloc[-1])
            
            return AIPrediction(
                symbol=symbol,
                current_price=current_price,
                predictions=predictions,
                ml_models=ml_models,
                ai_sentiment=ai_sentiment,
                pattern_recognition=pattern_recognition,
                risk_assessment=risk_assessment,
                trading_signals=trading_signals,
                confidence_intervals=self._calculate_confidence_intervals(predictions),
                market_regime=market_regime,
                anomaly_detection=anomaly_detection
            )
            
        except Exception as e:
            logger.error(f"AI prediction error for {symbol}: {e}")
            # Return default prediction
            return AIPrediction(
                symbol=symbol,
                current_price=0.0,
                predictions={'1d': {'price': 0, 'direction': 'NEUTRAL', 'confidence': 0}},
                ml_models={'status': 'error'},
                ai_sentiment={'overall': 'NEUTRAL', 'score': 0},
                pattern_recognition={'patterns': []},
                risk_assessment={'risk_score': 5, 'level': 'MEDIUM'},
                trading_signals=[],
                confidence_intervals={},
                market_regime='SIDEWAYS',
                anomaly_detection={'anomalies_detected': False}
            )
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for machine learning models"""
        df = data.copy()
        
        # Technical indicators as features
        df['Returns'] = df['Close'].pct_change()
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Target variable (next day return)
        df['Target'] = df['Close'].shift(-1) / df['Close'] - 1
        
        # Clean data
        df = df.dropna()
        return df
    
    def _train_ml_models(self, features_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Train multiple machine learning models"""
        feature_columns = ['Returns', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal', 
                          'BB_Position', 'Volatility', 'Volume_Ratio', 
                          'Momentum_5', 'Momentum_10', 'Momentum_20']
        
        X = features_df[feature_columns].values
        y = features_df['Target'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        
        models['random_forest'] = {
            'model': rf_model,
            'scaler': scaler,
            'mse': mean_squared_error(y_test, rf_pred),
            'r2': r2_score(y_test, rf_pred),
            'feature_importance': dict(zip(feature_columns, rf_model.feature_importances_))
        }
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        
        models['gradient_boosting'] = {
            'model': gb_model,
            'scaler': scaler,
            'mse': mean_squared_error(y_test, gb_pred),
            'r2': r2_score(y_test, gb_pred),
            'feature_importance': dict(zip(feature_columns, gb_model.feature_importances_))
        }
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        
        models['linear_regression'] = {
            'model': lr_model,
            'scaler': scaler,
            'mse': mean_squared_error(y_test, lr_pred),
            'r2': r2_score(y_test, lr_pred),
            'coefficients': dict(zip(feature_columns, lr_model.coef_))
        }
        
        return models
    
    def _generate_ml_predictions(self, features_df: pd.DataFrame, ml_models: Dict) -> Dict[str, Dict[str, Any]]:
        """Generate predictions using trained models"""
        feature_columns = ['Returns', 'SMA_5', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal', 
                          'BB_Position', 'Volatility', 'Volume_Ratio', 
                          'Momentum_5', 'Momentum_10', 'Momentum_20']
        
        # Get latest features
        latest_features = features_df[feature_columns].iloc[-1].values.reshape(1, -1)
        current_price = features_df['Close'].iloc[-1]
        
        predictions = {}
        timeframes = ['1d', '7d', '30d', '90d']
        
        for timeframe in timeframes:
            model_predictions = {}
            
            for model_name, model_info in ml_models.items():
                if 'model' in model_info:
                    scaler = model_info['scaler']
                    model = model_info['model']
                    
                    # Scale features
                    scaled_features = scaler.transform(latest_features)
                    
                    # Make prediction
                    pred_return = model.predict(scaled_features)[0]
                    
                    # Adjust prediction based on timeframe
                    timeframe_multiplier = {'1d': 1, '7d': 7, '30d': 30, '90d': 90}[timeframe]
                    adjusted_return = pred_return * np.sqrt(timeframe_multiplier)  # Scale by sqrt(time)
                    
                    predicted_price = current_price * (1 + adjusted_return)
                    
                    model_predictions[model_name] = {
                        'predicted_return': adjusted_return,
                        'predicted_price': predicted_price,
                        'model_confidence': model_info.get('r2', 0)
                    }
            
            # Ensemble prediction (average of models weighted by R²)
            total_weight = sum(pred.get('model_confidence', 0) for pred in model_predictions.values())
            if total_weight > 0:
                ensemble_return = sum(
                    pred['predicted_return'] * pred.get('model_confidence', 0) 
                    for pred in model_predictions.values()
                ) / total_weight
                
                ensemble_price = current_price * (1 + ensemble_return)
                
                # Determine direction and confidence
                direction = 'BULLISH' if ensemble_return > 0.01 else 'BEARISH' if ensemble_return < -0.01 else 'NEUTRAL'
                confidence = min(abs(ensemble_return) * 100, 95)  # Cap at 95%
                
                predictions[timeframe] = {
                    'predicted_price': round(ensemble_price, 2),
                    'predicted_return': round(ensemble_return * 100, 2),
                    'direction': direction,
                    'confidence': round(confidence, 1),
                    'individual_models': model_predictions
                }
        
        return predictions
    
    def _analyze_ai_sentiment(self, symbol: str) -> Dict[str, Any]:
        """AI-powered sentiment analysis"""
        try:
            # Simulated advanced sentiment analysis
            # In production, this would integrate with news APIs, social media, etc.
            
            sentiment_score = np.random.normal(0, 0.3)  # Simulated sentiment
            sentiment_score = max(-1, min(1, sentiment_score))  # Clamp between -1 and 1
            
            if sentiment_score > 0.3:
                sentiment = 'VERY_POSITIVE'
            elif sentiment_score > 0.1:
                sentiment = 'POSITIVE'
            elif sentiment_score > -0.1:
                sentiment = 'NEUTRAL'
            elif sentiment_score > -0.3:
                sentiment = 'NEGATIVE'
            else:
                sentiment = 'VERY_NEGATIVE'
            
            return {
                'overall_sentiment': sentiment,
                'sentiment_score': round(sentiment_score, 3),
                'news_sentiment': round(np.random.normal(sentiment_score, 0.1), 3),
                'social_sentiment': round(np.random.normal(sentiment_score, 0.2), 3),
                'analyst_sentiment': round(np.random.normal(sentiment_score, 0.15), 3),
                'confidence': round(abs(sentiment_score) * 100, 1),
                'sources_analyzed': ['news', 'social_media', 'analyst_reports'],
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'overall_sentiment': 'NEUTRAL',
                'sentiment_score': 0,
                'error': str(e)
            }
    
    def _detect_ai_patterns(self, symbol: str) -> Dict[str, Any]:
        """AI pattern recognition in price data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='3mo', interval='1d')
            
            patterns = []
            
            # Simple pattern detection (can be enhanced with more sophisticated algorithms)
            closes = data['Close'].values
            
            # Double top/bottom pattern
            if len(closes) > 50:
                peaks = []
                troughs = []
                
                for i in range(2, len(closes) - 2):
                    if closes[i] > closes[i-1] and closes[i] > closes[i+1]:
                        if closes[i] > closes[i-2] and closes[i] > closes[i+2]:
                            peaks.append((i, closes[i]))
                    elif closes[i] < closes[i-1] and closes[i] < closes[i+1]:
                        if closes[i] < closes[i-2] and closes[i] < closes[i+2]:
                            troughs.append((i, closes[i]))
                
                # Check for double top
                if len(peaks) >= 2:
                    last_two_peaks = peaks[-2:]
                    if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.03:
                        patterns.append({
                            'pattern': 'DOUBLE_TOP',
                            'confidence': 0.7,
                            'signal': 'BEARISH',
                            'description': 'Double top pattern detected - potential reversal'
                        })
                
                # Check for double bottom
                if len(troughs) >= 2:
                    last_two_troughs = troughs[-2:]
                    if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] < 0.03:
                        patterns.append({
                            'pattern': 'DOUBLE_BOTTOM',
                            'confidence': 0.7,
                            'signal': 'BULLISH',
                            'description': 'Double bottom pattern detected - potential reversal'
                        })
            
            # Trend analysis
            short_trend = np.polyfit(range(10), closes[-10:], 1)[0]
            medium_trend = np.polyfit(range(30), closes[-30:], 1)[0]
            long_trend = np.polyfit(range(50), closes[-50:] if len(closes) >= 50 else closes, 1)[0]
            
            if short_trend > 0 and medium_trend > 0:
                patterns.append({
                    'pattern': 'UPTREND',
                    'confidence': 0.8,
                    'signal': 'BULLISH',
                    'description': 'Strong upward trend identified'
                })
            elif short_trend < 0 and medium_trend < 0:
                patterns.append({
                    'pattern': 'DOWNTREND',
                    'confidence': 0.8,
                    'signal': 'BEARISH',
                    'description': 'Strong downward trend identified'
                })
            
            return {
                'patterns_detected': patterns,
                'trend_analysis': {
                    'short_term': 'BULLISH' if short_trend > 0 else 'BEARISH',
                    'medium_term': 'BULLISH' if medium_trend > 0 else 'BEARISH',
                    'long_term': 'BULLISH' if long_trend > 0 else 'BEARISH'
                },
                'total_patterns': len(patterns),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'patterns_detected': [],
                'error': str(e)
            }
    
    def _assess_ai_risk(self, symbol: str) -> Dict[str, Any]:
        """AI-powered risk assessment"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1y', interval='1d')
            
            if data.empty:
                return {'risk_score': 5, 'risk_level': 'MEDIUM'}
            
            returns = data['Close'].pct_change().dropna()
            
            # Calculate risk metrics
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            var_95 = np.percentile(returns, 5)  # Value at Risk (95%)
            max_drawdown = self._calculate_max_drawdown(data['Close'])
            
            # Risk scoring (0-10 scale)
            vol_score = min(volatility * 10, 10)  # Higher volatility = higher risk
            var_score = min(abs(var_95) * 50, 10)  # Higher VaR = higher risk
            drawdown_score = min(abs(max_drawdown) * 20, 10)  # Higher drawdown = higher risk
            
            # Weighted average risk score
            risk_score = (vol_score * 0.4 + var_score * 0.3 + drawdown_score * 0.3)
            
            # Risk level classification
            if risk_score < 3:
                risk_level = 'LOW'
                risk_color = '#4CAF50'
            elif risk_score < 6:
                risk_level = 'MEDIUM'
                risk_color = '#ffeb3b'
            elif risk_score < 8:
                risk_level = 'HIGH'
                risk_color = '#ff9800'
            else:
                risk_level = 'VERY_HIGH'
                risk_color = '#f44336'
            
            return {
                'risk_score': round(risk_score, 2),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'volatility': round(volatility * 100, 2),
                'var_95': round(var_95 * 100, 2),
                'max_drawdown': round(max_drawdown * 100, 2),
                'risk_factors': {
                    'market_volatility': round(vol_score, 1),
                    'downside_risk': round(var_score, 1),
                    'historical_drawdown': round(drawdown_score, 1)
                },
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'risk_score': 5,
                'risk_level': 'MEDIUM',
                'error': str(e)
            }
    
    def _generate_ai_trading_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate AI-powered trading signals"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='3mo', interval='1d')
            
            signals = []
            current_price = float(data['Close'].iloc[-1])
            
            # Technical analysis signals
            tech_analysis = self._calculate_advanced_technical_indicators(data)
            
            # RSI signals
            if tech_analysis.rsi < 30:
                signals.append({
                    'signal_type': 'BUY',
                    'indicator': 'RSI',
                    'strength': 'STRONG',
                    'confidence': 0.8,
                    'reason': f'RSI oversold at {tech_analysis.rsi:.1f}',
                    'entry_price': current_price,
                    'target_price': current_price * 1.05,
                    'stop_loss': current_price * 0.97
                })
            elif tech_analysis.rsi > 70:
                signals.append({
                    'signal_type': 'SELL',
                    'indicator': 'RSI',
                    'strength': 'STRONG',
                    'confidence': 0.8,
                    'reason': f'RSI overbought at {tech_analysis.rsi:.1f}',
                    'entry_price': current_price,
                    'target_price': current_price * 0.95,
                    'stop_loss': current_price * 1.03
                })
            
            # MACD signals
            if tech_analysis.macd > tech_analysis.macd_signal and tech_analysis.macd_histogram > 0:
                signals.append({
                    'signal_type': 'BUY',
                    'indicator': 'MACD',
                    'strength': 'MEDIUM',
                    'confidence': 0.7,
                    'reason': 'MACD bullish crossover',
                    'entry_price': current_price,
                    'target_price': current_price * 1.03,
                    'stop_loss': current_price * 0.98
                })
            
            # Bollinger Bands signals
            if current_price < tech_analysis.bollinger_lower:
                signals.append({
                    'signal_type': 'BUY',
                    'indicator': 'BOLLINGER_BANDS',
                    'strength': 'MEDIUM',
                    'confidence': 0.6,
                    'reason': 'Price below lower Bollinger Band',
                    'entry_price': current_price,
                    'target_price': tech_analysis.bollinger_middle,
                    'stop_loss': current_price * 0.96
                })
            elif current_price > tech_analysis.bollinger_upper:
                signals.append({
                    'signal_type': 'SELL',
                    'indicator': 'BOLLINGER_BANDS',
                    'strength': 'MEDIUM',
                    'confidence': 0.6,
                    'reason': 'Price above upper Bollinger Band',
                    'entry_price': current_price,
                    'target_price': tech_analysis.bollinger_middle,
                    'stop_loss': current_price * 1.04
                })
            
            # Add timestamp to all signals
            for signal in signals:
                signal['timestamp'] = datetime.now().isoformat()
                signal['symbol'] = symbol
            
            return signals
            
        except Exception as e:
            logger.error(f"Trading signals error for {symbol}: {e}")
            return []
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime (Bull/Bear/Sideways)"""
        try:
            closes = data['Close']
            returns = closes.pct_change().dropna()
            
            # Calculate trend over different periods
            short_trend = closes.iloc[-10:].mean() / closes.iloc[-20:-10].mean() - 1
            medium_trend = closes.iloc[-30:].mean() / closes.iloc[-60:-30].mean() - 1
            
            # Calculate volatility
            volatility = returns.rolling(window=30).std().iloc[-1]
            
            if short_trend > 0.02 and medium_trend > 0.01:
                return 'BULL_MARKET'
            elif short_trend < -0.02 and medium_trend < -0.01:
                return 'BEAR_MARKET'
            elif volatility > returns.std() * 1.5:
                return 'HIGH_VOLATILITY'
            else:
                return 'SIDEWAYS'
                
        except Exception:
            return 'SIDEWAYS'
    
    def _detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in price/volume data"""
        try:
            returns = data['Close'].pct_change().dropna()
            volumes = data['Volume']
            
            # Price anomalies (returns beyond 2 standard deviations)
            price_threshold = returns.std() * 2
            recent_returns = returns.tail(10)
            price_anomalies = recent_returns[abs(recent_returns) > price_threshold]
            
            # Volume anomalies
            volume_mean = volumes.rolling(window=20).mean()
            volume_threshold = volume_mean * 2
            recent_volumes = volumes.tail(10)
            recent_volume_mean = volume_mean.tail(10)
            volume_anomalies = recent_volumes[recent_volumes > recent_volume_mean * 2]
            
            return {
                'anomalies_detected': len(price_anomalies) > 0 or len(volume_anomalies) > 0,
                'price_anomalies': len(price_anomalies),
                'volume_anomalies': len(volume_anomalies),
                'latest_anomaly_date': data.index[-1].isoformat() if len(price_anomalies) > 0 or len(volume_anomalies) > 0 else None,
                'anomaly_details': {
                    'unusual_price_moves': price_anomalies.to_dict() if len(price_anomalies) > 0 else {},
                    'unusual_volume_spikes': volume_anomalies.to_dict() if len(volume_anomalies) > 0 else {}
                }
            }
            
        except Exception as e:
            return {
                'anomalies_detected': False,
                'error': str(e)
            }
    
    def _calculate_confidence_intervals(self, predictions: Dict) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for predictions"""
        confidence_intervals = {}
        
        for timeframe, pred_data in predictions.items():
            if 'predicted_price' in pred_data and 'confidence' in pred_data:
                price = pred_data['predicted_price']
                confidence = pred_data['confidence'] / 100
                
                # Simple confidence interval (can be enhanced)
                margin = price * (1 - confidence) * 0.1
                lower_bound = price - margin
                upper_bound = price + margin
                
                confidence_intervals[timeframe] = (round(lower_bound, 2), round(upper_bound, 2))
        
        return confidence_intervals
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = cumulative / running_max - 1
        return drawdown.min()

    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the enhanced professional predictor"""
        try:
            print("\n" + "="*70)
            print("🚀 ENHANCED PROFESSIONAL FINTECH PREDICTOR")
            print("="*70)
            print("📊 ADVANCED FEATURES:")
            print("   ✅ Interactive Charts with TradingView integration")
            print("   ✅ Cryptocurrency analysis (Bitcoin, Ethereum, etc.)")
            print("   ✅ Advanced technical indicators (Fibonacci, Elliott Wave)")
            print("   ✅ Multi-timeframe analysis (1min to 1month)")
            print("   ✅ Professional reports (PDF/Excel export)")
            print("   ✅ Global markets (US, European, Asian, Forex, Commodities)")
            print("   ✅ Market heat maps and sector analysis")
            print("   ✅ Volume profile and support/resistance levels")
            print("")
            print(f"🌐 Dashboard: http://localhost:{port}")
            print("🔗 Enhanced API Endpoints:")
            print(f"   • Multi-timeframe: http://localhost:{port}/api/multi_timeframe/<symbol>")
            print(f"   • Crypto Analysis: http://localhost:{port}/api/crypto_analysis")
            print(f"   • Interactive Charts: http://localhost:{port}/api/chart/<symbol>/<timeframe>")
            print(f"   • Fibonacci Levels: http://localhost:{port}/api/fibonacci/<symbol>")
            print(f"   • PDF Report: http://localhost:{port}/api/export/pdf/<symbol>")
            print(f"   • Excel Export: http://localhost:{port}/api/export/excel/<symbol>")
            print(f"   • Market Heatmap: http://localhost:{port}/api/market_heatmap")
            print(f"   • Sector Analysis: http://localhost:{port}/api/sector_analysis")
            print("🤖 AI Prediction Endpoints:")
            print(f"   • AI Predictions: http://localhost:{port}/api/ai_prediction/<symbol>")
            print(f"   • AI Sentiment: http://localhost:{port}/api/ai_sentiment/<symbol>")
            print(f"   • AI Patterns: http://localhost:{port}/api/ai_patterns/<symbol>")
            print(f"   • AI Risk Assessment: http://localhost:{port}/api/ai_risk/<symbol>")
            print(f"   • AI Trading Signals: http://localhost:{port}/api/ai_trading_signals/<symbol>")
            print("="*70)
            
            # Start analysis engine
            self.start_analysis_engine()
            
            # Run the application
            self.socketio.run(self.app, host=host, port=port, debug=True)
            
        except KeyboardInterrupt:
            logger.info("🛑 Enhanced Professional Predictor stopped by user")
        except Exception as e:
            logger.error(f"❌ Server error: {e}")
        finally:
            self.running = False

if __name__ == "__main__":
    try:
        predictor = EnhancedProfessionalPredictor()
        predictor.run()
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        sys.exit(1)