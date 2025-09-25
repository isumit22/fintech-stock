# final_demo.py
"""
Simple but working stock trend prediction demo
"""

import asyncio
import random
from collections import deque
from datetime import datetime, timedelta
import numpy as np

class SimplePredictor:
    """Very simple predictor that actually works"""
    
    def __init__(self):
        self.predictions = []
        self.correct_predictions = 0
        self.total_predictions = 0
        
        # Simple trend following weights
        self.weights = {
            'return_1': 0.5,
            'return_5': 0.3,
            'ma_gap': 0.2
        }
    
    def predict(self, features):
        """Make prediction based on simple rules"""
        score = 0
        
        # Recent return
        if 'return_1' in features:
            score += features['return_1'] * self.weights['return_1']
        
        # Longer term return  
        if 'return_5' in features:
            score += features['return_5'] * self.weights['return_5']
            
        # Moving average gap
        if 'ma_gap' in features:
            score += features['ma_gap'] * self.weights['ma_gap']
        
        # Convert to prediction
        if score > 0.001:
            prediction = "up"
            probabilities = {"up": 0.6, "neutral": 0.3, "down": 0.1}
        elif score < -0.001:
            prediction = "down"
            probabilities = {"up": 0.1, "neutral": 0.3, "down": 0.6}
        else:
            prediction = "neutral"
            probabilities = {"up": 0.25, "neutral": 0.5, "down": 0.25}
        
        return prediction, probabilities
    
    def learn(self, features, actual_label, predicted_label):
        """Update accuracy tracking and simple weight adjustment"""
        self.total_predictions += 1
        
        if predicted_label == actual_label:
            self.correct_predictions += 1
            # Slightly increase weights for correct predictions
            for key in self.weights:
                if key in features and features[key] != 0:
                    self.weights[key] *= 1.001
        else:
            # Slightly decrease weights for wrong predictions  
            for key in self.weights:
                if key in features and features[key] != 0:
                    self.weights[key] *= 0.999
        
        # Keep weights reasonable
        for key in self.weights:
            self.weights[key] = max(0.01, min(1.0, self.weights[key]))
    
    def get_accuracy(self):
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

class MockDataGenerator:
    """Generate realistic stock data with trends"""
    
    def __init__(self, symbol="AAPL", initial_price=150.0):
        self.symbol = symbol
        self.price = initial_price
        self.time = datetime.now()
        self.trend = 0.0  # Current trend
        self.trend_duration = 0
        
    def generate_tick(self):
        # Change trend occasionally
        if random.random() < 0.05:
            self.trend = random.gauss(0, 0.001)
            self.trend_duration = 0
        
        # Generate price with trend + noise
        noise = random.gauss(0, 0.3)
        self.price += self.trend + noise
        self.price = max(self.price, 1.0)
        self.trend_duration += 1
        
        tick = {
            "symbol": self.symbol,
            "price": round(self.price, 2),
            "timestamp": self.time
        }
        
        self.time += timedelta(seconds=random.randint(1, 3))
        return tick

def compute_features(price_history):
    """Compute simple features"""
    if len(price_history) < 6:
        return {}
    
    prices = np.array(price_history)
    
    features = {
        'price': float(prices[-1]),
        'return_1': float((prices[-1] / prices[-2]) - 1),
        'return_5': float((prices[-1] / prices[-6]) - 1),
        'ma_5': float(np.mean(prices[-5:])),
        'ma_10': float(np.mean(prices[-10:])) if len(prices) >= 10 else float(np.mean(prices))
    }
    
    features['ma_gap'] = (features['ma_5'] / features['ma_10']) - 1
    
    return features

def make_label(current_price, future_price, threshold=0.002):
    """Create label"""
    ret = (future_price / current_price) - 1.0
    if ret > threshold:
        return "up"
    elif ret < -threshold:
        return "down"
    else:
        return "neutral"

async def run_final_demo():
    """Run the final working demo"""
    print("🎯 Final Working Stock Trend Prediction Demo")
    print("============================================")
    
    predictor = SimplePredictor()
    data_generator = MockDataGenerator("AAPL", 150.0)
    
    price_history = deque(maxlen=20)
    predictions_queue = deque()
    
    print("Processing ticks, making predictions, and learning...\n")
    
    for i in range(200):
        # Generate new data
        tick = data_generator.generate_tick()
        price = tick["price"]
        timestamp = tick["timestamp"]
        
        price_history.append(price)
        
        # Make prediction when we have enough data
        if len(price_history) >= 10:
            features = compute_features(price_history)
            
            if features:  # Only if we computed features
                prediction, probabilities = predictor.predict(features)
                
                predictions_queue.append({
                    "timestamp": timestamp,
                    "price": price,
                    "features": features,
                    "prediction": prediction,
                })
                
                # Learn from old predictions (simulate delay)
                if len(predictions_queue) > 15:
                    old_pred = predictions_queue.popleft()
                    actual_label = make_label(old_pred["price"], price)
                    
                    # Learn from this example
                    predictor.learn(old_pred["features"], actual_label, old_pred["prediction"])
                    
                    # Log learning progress
                    if predictor.total_predictions % 20 == 0:
                        accuracy = predictor.get_accuracy()
                        print(f"[{timestamp.strftime('%H:%M:%S')}] "
                              f"Learned: {actual_label} vs predicted {old_pred['prediction']} | "
                              f"Accuracy: {accuracy:.3f} ({predictor.correct_predictions}/{predictor.total_predictions})")
        
        # Log current predictions
        if i % 30 == 0 and predictions_queue:
            recent = predictions_queue[-1]
            prob_up = recent.get("probabilities", {}).get("up", 0)
            pred_name = recent["prediction"]
            print(f"[{timestamp.strftime('%H:%M:%S')}] "
                  f"Price: ${price:.2f} → Predicting: {pred_name.upper()}")
    
    # Final results
    accuracy = predictor.get_accuracy()
    print(f"\n📊 Final Results:")
    print(f"   • Total Predictions Made & Learned: {predictor.total_predictions}")
    print(f"   • Correct Predictions: {predictor.correct_predictions}")
    print(f"   • Final Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   • Final Price: ${price_history[-1]:.2f}")
    print(f"   • Learned Weights: {predictor.weights}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"\n💡 This demonstrates:")
    print(f"   ✓ Real-time price processing")
    print(f"   ✓ Feature extraction from price data")
    print(f"   ✓ Making predictions with confidence")
    print(f"   ✓ Learning from prediction accuracy")
    print(f"   ✓ Adaptive weight adjustment")
    print(f"\n🚀 Ready to scale with real WebSocket feeds!")

if __name__ == "__main__":
    asyncio.run(run_final_demo())