# simple_demo.py
"""
Simple stock trend prediction demo that works with current River version
"""

import asyncio
import json
import random
from collections import deque
from datetime import datetime, timedelta

import numpy as np
from river import linear_model, preprocessing, metrics

class SimplePredictor:
    """Simple trend predictor using logistic regression"""
    
    def __init__(self):
        self.model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        self.accuracy = metrics.Accuracy()
        self.predictions_made = 0
        self.samples_learned = 0
        self.trained = False
        
    def predict(self, features):
        try:
            if not self.trained:
                return "neutral", {"up": 0.33, "neutral": 0.34, "down": 0.33}
            prediction = self.model.predict_one(features)
            probabilities = self.model.predict_proba_one(features)
            self.predictions_made += 1
            return prediction, probabilities
        except:
            return "neutral", {"up": 0.33, "neutral": 0.34, "down": 0.33}
    
    def learn(self, features, label):
        try:
            if self.trained:
                pred, _ = self.predict(features)
                self.accuracy.update(label, pred)
            
            self.model.learn_one(features, label)
            self.samples_learned += 1
            self.trained = True
        except Exception as e:
            print(f"Learning error: {e}")

class MockDataGenerator:
    """Generate realistic mock stock data"""
    
    def __init__(self, symbol="AAPL", initial_price=150.0):
        self.symbol = symbol
        self.current_price = initial_price
        self.current_time = datetime.now()
        
    def generate_tick(self):
        # Random walk
        change = random.gauss(0, 0.5)
        self.current_price += change
        volume = int(random.expovariate(1.0/100))
        
        tick = {
            "symbol": self.symbol,
            "price": round(self.current_price, 2),
            "volume": volume,
            "timestamp": self.current_time
        }
        
        self.current_time += timedelta(seconds=random.randint(1, 10))
        return tick

def compute_simple_features(price_history):
    """Compute simple technical features"""
    if len(price_history) < 5:
        return {"last_price": price_history[-1] if price_history else 150.0}
    
    prices = np.array(price_history)
    features = {
        "last_price": float(prices[-1]),
        "ret_1": float((prices[-1] / prices[-2]) - 1) if len(prices) >= 2 else 0.0,
        "ret_5": float((prices[-1] / prices[-6]) - 1) if len(prices) >= 6 else 0.0,
        "ma_5": float(np.mean(prices[-5:])),
        "ma_10": float(np.mean(prices[-10:])) if len(prices) >= 10 else float(np.mean(prices)),
        "std_5": float(np.std(prices[-5:])),
    }
    
    features["ma_gap"] = features["ma_5"] - features["ma_10"]
    features["price_vs_ma"] = (features["last_price"] / features["ma_5"]) - 1
    
    return features

def make_label(current_price, future_price, threshold=0.001):
    """Create trend label"""
    ret = (future_price / current_price) - 1.0
    if ret > threshold:
        return "up"
    elif ret < -threshold:
        return "down"
    else:
        return "neutral"

async def run_simple_demo():
    """Run a simple demonstration"""
    print("🚀 Simple Stock Trend Prediction Demo")
    print("=====================================")
    
    # Setup
    predictor = SimplePredictor()
    data_generator = MockDataGenerator("AAPL", 150.0)
    
    # Storage
    price_history = deque(maxlen=20)
    pending_predictions = deque()
    
    print("Processing ticks and making predictions...")
    
    for i in range(200):
        # Generate tick
        tick = data_generator.generate_tick()
        price = tick["price"]
        timestamp = tick["timestamp"]
        
        # Store price
        price_history.append(price)
        
        # Generate features and predict
        if len(price_history) >= 10:  # Need some history
            features = compute_simple_features(price_history)
            prediction, probabilities = predictor.predict(features)
            
            # Store prediction for labeling
            pending_predictions.append({
                "timestamp": timestamp,
                "price": price,
                "features": features,
                "prediction": prediction,
                "target_time": timestamp + timedelta(minutes=5)
            })
            
            # Label old predictions (simulate 5-minute wait)
            if len(pending_predictions) > 10:  # Simulate time passage
                old_pred = pending_predictions.popleft()
                label = make_label(old_pred["price"], price)  # Use current price as "future"
                predictor.learn(old_pred["features"], label)
                
                # Log learning
                if predictor.samples_learned % 20 == 0:
                    acc = predictor.accuracy.get()
                    print(f"[{timestamp.strftime('%H:%M:%S')}] "
                          f"Learned: {label} | Accuracy: {acc:.3f} | "
                          f"Samples: {predictor.samples_learned}")
            
            # Log predictions
            if i % 30 == 0:
                prob_up = probabilities.get("up", 0)
                prob_down = probabilities.get("down", 0) 
                prob_neutral = probabilities.get("neutral", 0)
                print(f"[{timestamp.strftime('%H:%M:%S')}] "
                      f"Price: ${price:.2f} → {str(prediction).upper()} "
                      f"({prob_up:.2f}/{prob_neutral:.2f}/{prob_down:.2f})")
    
    # Final results
    print(f"\n📊 Final Results:")
    print(f"   Predictions Made: {predictor.predictions_made}")
    print(f"   Samples Learned: {predictor.samples_learned}")
    print(f"   Final Accuracy: {predictor.accuracy.get():.3f}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"💡 This shows the basic concept - real system uses:")
    print(f"   • WebSocket feeds for live data")
    print(f"   • More sophisticated features (30+)")
    print(f"   • Risk management and position sizing")
    print(f"   • Real-time performance monitoring")

if __name__ == "__main__":
    asyncio.run(run_simple_demo())