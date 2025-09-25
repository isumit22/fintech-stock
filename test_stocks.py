#!/usr/bin/env python3
"""
Fintech Stock Data Test - Verify API connectivity
"""
import yfinance as yf
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_stock_fetch(symbol):
    """Test fetching stock data for fintech system"""
    try:
        logger.info(f"🧪 Testing Fintech symbol: {symbol}")
        
        ticker = yf.Ticker(symbol)
        
        # Try multiple intervals like production system
        hist = None
        for interval in ["5m", "1m", "15m"]:
            try:
                hist = ticker.history(period="1d", interval=interval)
                if not hist.empty:
                    logger.info(f"   ✅ Got data with {interval} interval")
                    break
            except Exception as e:
                logger.info(f"   ❌ Failed with {interval}: {e}")
                continue
        
        # Fallback to daily
        if hist is None or hist.empty:
            hist = ticker.history(period="1d", interval="1d")
            if not hist.empty:
                logger.info(f"   ✅ Fallback to daily data worked")
        
        if hist.empty:
            logger.error(f"❌ NO DATA: {symbol}")
            return False
            
        latest = hist.iloc[-1]
        price = float(latest['Close'])
        volume = int(latest['Volume']) if not pd.isna(latest['Volume']) else 0
        high = float(latest['High'])
        low = float(latest['Low'])
        
        logger.info(f"✅ SUCCESS: {symbol}")
        logger.info(f"   💰 Price: ₹{price:.2f}")
        logger.info(f"   📊 Volume: {volume:,}")
        logger.info(f"   📈 High: ₹{high:.2f}")
        logger.info(f"   📉 Low: ₹{low:.2f}")
        logger.info(f"   📋 Data points: {len(hist)}")
        
        # Test company info
        try:
            info = ticker.info
            if info:
                sector = info.get('sector', 'N/A')
                market_cap = info.get('marketCap', 'N/A')
                logger.info(f"   🏢 Sector: {sector}")
                logger.info(f"   💎 Market Cap: {market_cap}")
        except:
            logger.info(f"   ℹ️ Company info not available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ERROR for {symbol}: {e}")
        return False

def test_prediction_features(hist):
    """Test prediction calculation features"""
    try:
        import numpy as np
        
        if len(hist) < 2:
            return False
            
        close_prices = hist['Close'].values
        volumes = hist['Volume'].values
        
        # Test moving averages
        ma_5 = np.mean(close_prices[-5:])
        
        # Test returns
        ret_1 = (close_prices[-1] / close_prices[-2] - 1) if len(close_prices) >= 2 else 0
        
        # Test volume ratio
        vol_ma = np.mean(volumes[-10:]) if len(volumes) >= 10 else volumes[-1]
        vol_ratio = volumes[-1] / vol_ma if vol_ma > 0 else 1
        
        logger.info(f"   🔧 MA5: {ma_5:.2f}")
        logger.info(f"   📊 Return: {ret_1*100:.2f}%")
        logger.info(f"   📈 Vol Ratio: {vol_ratio:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"   💥 Feature calculation error: {e}")
        return False

if __name__ == "__main__":
    import pandas as pd
    
    # Test symbols for fintech system
    symbols_to_test = [
        "KOTAKBANK.NS",
        "TCS.NS",
        "RELIANCE.NS", 
        "HDFCBANK.NS",
        "INFY.NS",
        "ITC.NS",
        "SBIN.NS",
        "BAJFINANCE.NS"
    ]
    
    logger.info("🏦 FINTECH STOCK DATA TEST")
    logger.info("=" * 60)
    
    successful = 0
    failed = 0
    
    for symbol in symbols_to_test:
        logger.info("")
        if test_stock_fetch(symbol):
            successful += 1
            
            # Test prediction features for successful stocks
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d", interval="5m")
                if hist.empty:
                    hist = ticker.history(period="1d", interval="1d")
                test_prediction_features(hist)
            except:
                pass
                
        else:
            failed += 1
        
        logger.info("-" * 60)
    
    logger.info("")
    logger.info(f"🎯 FINTECH TEST RESULTS:")
    logger.info(f"   ✅ Successful: {successful}/{len(symbols_to_test)}")
    logger.info(f"   ❌ Failed: {failed}/{len(symbols_to_test)}")
    logger.info(f"   📊 Success Rate: {successful/len(symbols_to_test)*100:.1f}%")
    
    if successful >= len(symbols_to_test) * 0.8:  # 80% success rate
        logger.info("🟢 FINTECH SYSTEM READY FOR DEPLOYMENT!")
    else:
        logger.info("🟡 FINTECH SYSTEM NEEDS ATTENTION")
    
    logger.info("=" * 60)