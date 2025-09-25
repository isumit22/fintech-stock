# Stock Trend Prediction System - Quick Start Guide

## 🎯 What This System Does
Predicts 5-minute stock price trends (UP/DOWN/NEUTRAL) using:
- Real-time WebSocket data feeds
- Online machine learning (adapts continuously)
- 30+ technical indicators
- Risk management controls

## 🚀 How to Run It

### Option 1: Run the Complete Notebook
1. Open `stock_trend_analysis.ipynb` in VS Code
2. Run all cells from top to bottom (Ctrl+Shift+P → "Run All Cells")
3. The final cell will run a complete demo with 500 ticks

### Option 2: Quick Demo Script
Run the pre-built quick start script:
```bash
python quick_start.py
```

### Option 3: Enhanced Version
Run the full-featured version:
```bash
python enhanced_trend_predictor.py --mode simulation --sim-mode mock --max-ticks 500
```

## 📊 What You'll See

The system will show:
- **Real-time predictions**: `[PRED] 14:30:45 $150.25 → UP (0.75/0.15/0.10)`
- **Model learning**: `[LEARN] 14:25:45 → up (+0.15%) acc=0.6234`
- **Trading signals**: `[TRADE] UP signal: size=0.087, confidence=0.750`
- **Performance metrics**: Accuracy, Sharpe ratio, returns

## 🔧 Configuration Options

Edit the `CONFIG` dictionary in the notebook:

```python
CONFIG = {
    "SYMBOL": "AAPL",              # Stock symbol
    "PRED_HORIZON_MINUTES": 5,     # Prediction timeframe
    "LABEL_THRESHOLD": 0.001,      # 0.1% minimum move
    "MODEL_TYPE": "random_forest", # or "logistic"
    "FEATURE_WINDOW_MINUTES": 15,  # Historical data window
}
```

## 📈 Connect to Real Data

To use real WebSocket feeds instead of mock data:

1. **Get API credentials** from a provider:
   - Alpaca Markets (free tier available)
   - Polygon.io
   - IEX Cloud
   - Binance (for crypto)

2. **Update configuration**:
   ```python
   CONFIG["PROVIDER"] = "alpaca"
   CONFIG["API_KEY"] = "your_api_key"
   CONFIG["SECRET_KEY"] = "your_secret_key"
   ```

3. **Run with live data**:
   ```bash
   python enhanced_trend_predictor.py --mode live --provider alpaca --symbol AAPL
   ```

## 🛡️ Risk Management Features

- **Position sizing**: Automatic based on confidence and volatility
- **Daily loss limits**: Stop trading after 5% daily loss
- **Confidence threshold**: Only trade high-confidence signals (>60%)
- **Market hours**: Only trade during market hours

## 📋 System Requirements

- Python 3.8+
- 2GB RAM minimum
- Internet connection for live data
- Windows/Mac/Linux compatible

## 🎉 Quick Test

Just run this in the notebook:
```python
# This will process 100 mock ticks and show predictions
demo_system = await final_demo()
```

## ⚠️ Important Notes

- **Educational Purpose**: This is for learning and research
- **Backtest First**: Always test strategies before live trading
- **No Financial Advice**: Past performance ≠ future results
- **Risk Management**: Never risk more than you can afford to lose

## 🆘 Need Help?

1. **Notebook won't run?** Make sure all packages are installed
2. **Errors in code?** Check the Python environment is activated
3. **No predictions?** Wait for the system to collect enough data (15+ minutes)
4. **Poor accuracy?** Try adjusting `LABEL_THRESHOLD` or `MODEL_TYPE`

Start with the notebook and run cell by cell to understand how it works! 🚀