# working_demo.py
"""
Working stock trend prediction demo using scikit-learn
"""

import asyncio
import random
from collections import deque
from datetime import datetime, timedelta

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

class StockPredictor:
    """Stock trend predictor using SGD classifier"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = SGDClassifier(loss='log_loss', learning_rate='adaptive', random_state=42)
        self.label_map = {"up": 1, "neutral": 0, "down": -1}
        self.label_names = {1: "up", 0: "neutral", -1: "down"}
        self.predictions = []
        self.accuracies = []
        self.is_fitted = False
        
    def predict(self, features_dict):
        features = np.array(list(features_dict.values())).reshape(1, -1)
        
        if not self.is_fitted:
            return "neutral", {"up": 0.33, "neutral": 0.34, "down": 0.33}
        
        try:
            features_scaled = self.scaler.transform(features)
            prediction_num = self.model.predict(features_scaled)[0]
            prediction = self.label_names[prediction_num]
            
            # Get probabilities
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features_scaled)[0]
                probabilities = {
                    "down": proba[0] if len(proba) > 0 else 0.33,
                    "neutral": proba[1] if len(proba) > 1 else 0.33,
                    "up": proba[2] if len(proba) > 2 else 0.33
                }
            else:
                probabilities = {"up": 0.33, "neutral": 0.34, "down": 0.33}
            
            return prediction, probabilities
        except:
            return "neutral", {"up": 0.33, "neutral": 0.34, "down": 0.33}
    
    def learn(self, features_dict, label):
        try:
            features = np.array(list(features_dict.values())).reshape(1, -1)
            label_num = self.label_map[label]
            
            if not self.is_fitted:
                # First fit
                self.scaler.fit(features)
                features_scaled = self.scaler.transform(features)
                self.model.partial_fit(features_scaled, [label_num], classes=[-1, 0, 1])
                self.is_fitted = True
            else:
                # Get prediction first for accuracy
                pred, _ = self.predict(features_dict)
                is_correct = (pred == label)
                self.accuracies.append(is_correct)
                
                # Update model
                features_scaled = self.scaler.transform(features)
                self.model.partial_fit(features_scaled, [label_num])
                
        except Exception as e:
            print(f"Learning error: {e}")
    
    def get_accuracy(self):
        if not self.accuracies:
            return 0.0
        return sum(self.accuracies) / len(self.accuracies)

class MockDataGenerator:
    """Generate realistic mock stock data"""
    
    def __init__(self, symbol="AAPL", initial_price=150.0):
        self.symbol = symbol
        self.current_price = initial_price
        self.current_time = datetime.now()
        self.trend = random.choice([-1, 0, 1])  # Current trend
        self.trend_duration = 0
        
    def generate_tick(self):
        # Change trend occasionally
        if random.random() < 0.1:
            self.trend = random.choice([-1, 0, 1])
            self.trend_duration = 0
        
        # Generate price change with some trend
        base_change = random.gauss(0, 0.2)
        trend_change = self.trend * 0.1
        total_change = base_change + trend_change
        
        self.current_price += total_change
        self.current_price = max(self.current_price, 1.0)  # No negative prices
        self.trend_duration += 1
        
        volume = int(random.expovariate(1.0/100))
        
        tick = {
            "symbol": self.symbol,
            "price": round(self.current_price, 2),
            "volume": volume,
            "timestamp": self.current_time
        }
        
        self.current_time += timedelta(seconds=random.randint(1, 5))
        return tick

def compute_features(price_history, volume_history):
    """Compute technical features"""
    if len(price_history) < 5:
        return {"price": price_history[-1] if price_history else 150.0}
    
    prices = np.array(price_history)
    volumes = np.array(volume_history)
    
    features = {
        "price": float(prices[-1]),
        "return_1": float((prices[-1] / prices[-2]) - 1) if len(prices) >= 2 else 0.0,
        "return_5": float((prices[-1] / prices[-6]) - 1) if len(prices) >= 6 else 0.0,
        "ma_5": float(np.mean(prices[-5:])),
        "ma_10": float(np.mean(prices[-10:])) if len(prices) >= 10 else float(np.mean(prices)),
        "volatility": float(np.std(prices[-5:])),
        "volume": float(volumes[-1]) if len(volumes) > 0 else 100.0,
        "volume_ma": float(np.mean(volumes[-5:])) if len(volumes) >= 5 else 100.0,
    }
    
    # Derived features
    features["ma_ratio"] = features["ma_5"] / features["ma_10"] if features["ma_10"] != 0 else 1.0
    features["price_ma_ratio"] = features["price"] / features["ma_5"] if features["ma_5"] != 0 else 1.0
    features["volume_ratio"] = features["volume"] / features["volume_ma"] if features["volume_ma"] != 0 else 1.0
    
    return features

def make_label(current_price, future_price, threshold=0.002):
    """Create trend label"""
    ret = (future_price / current_price) - 1.0
    if ret > threshold:
        return "up"
    elif ret < -threshold:
        return "down"
    else:
        return "neutral"

async def run_demo():
    """Run the prediction demo"""
    print("🚀 Working Stock Trend Prediction Demo")
    print("======================================")
    
    # Setup
    predictor = StockPredictor()
    data_generator = MockDataGenerator("AAPL", 150.0)
    
    # Storage
    price_history = deque(maxlen=30)
    volume_history = deque(maxlen=30)
    predictions_queue = deque()
    
    print("Processing ticks and making predictions...")
    print("(Simulating 5-minute prediction horizon with accelerated time)\n")
    
    for i in range(300):
        # Generate tick
        tick = data_generator.generate_tick()
        price = tick["price"]
        volume = tick["volume"]
        timestamp = tick["timestamp"]
        
        # Store data
        price_history.append(price)
        volume_history.append(volume)
        
        # Make predictions after we have enough data
        if len(price_history) >= 15:
            features = compute_features(price_history, volume_history)
            prediction, probabilities = predictor.predict(features)
            
            # Store prediction for future labeling
            predictions_queue.append({
                "timestamp": timestamp,
                "price": price,
                "features": features,
                "prediction": prediction,
            })
            
            # Label and learn from old predictions (simulate 5-min delay)
            if len(predictions_queue) > 20:  # Simulate time passage
                old_prediction = predictions_queue.popleft()
                actual_label = make_label(old_prediction["price"], price)
                predictor.learn(old_prediction["features"], actual_label)
                
                # Log learning progress
                if len(predictor.accuracies) % 25 == 0 and len(predictor.accuracies) > 0:
                    accuracy = predictor.get_accuracy()
                    print(f"[{timestamp.strftime('%H:%M:%S')}] "
                          f"Learned from old prediction: {actual_label.upper()} | "
                          f"Running Accuracy: {accuracy:.3f} | "
                          f"Samples: {len(predictor.accuracies)}")
            
            # Log current predictions periodically
            if i % 40 == 0:
                prob_up = probabilities["up"]
                prob_neutral = probabilities["neutral"] 
                prob_down = probabilities["down"]
                print(f"[{timestamp.strftime('%H:%M:%S')}] "
                      f"Price: ${price:.2f} → Predict: {prediction.upper()} "
                      f"(↑{prob_up:.2f} ○{prob_neutral:.2f} ↓{prob_down:.2f})")
    
    # Final results
    print(f"\n📊 Final Results:")
    print(f"   • Total Predictions: {len(predictor.predictions)}")
    print(f"   • Samples Learned From: {len(predictor.accuracies)}")
    print(f"   • Final Accuracy: {predictor.get_accuracy():.3f}")
    print(f"   • Final Price: ${price_history[-1]:.2f}")
    
    # Show recent predictions
    if predictions_queue:
        print(f"\n🔮 Recent Predictions (unlabeled):")
        for pred in list(predictions_queue)[-3:]:
            print(f"   ${pred['price']:.2f} → {pred['prediction'].upper()}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"\n💡 Next steps to make this production-ready:")
    print(f"   • Connect to real WebSocket feeds (Alpaca, Polygon, etc.)")
    print(f"   • Add more sophisticated technical indicators") 
    print(f"   • Implement proper risk management")
    print(f"   • Add position sizing and portfolio management")
    print(f"   • Include fundamental data and news sentiment")
    print(f"   • Set up real-time monitoring and alerting")

if __name__ == "__main__":
    asyncio.run(run_demo())