# simulation_mode.py
"""
Simulation mode for testing the 5-minute trend prediction system
Uses mock data or historical CSV data instead of live WebSocket feeds
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class MockDataGenerator:
    """Generate realistic mock stock data for testing"""
    
    def __init__(self, symbol="AAPL", initial_price=150.0, volatility=0.02):
        self.symbol = symbol
        self.current_price = initial_price
        self.volatility = volatility
        self.current_time = datetime.now()
        
    def generate_tick(self):
        """Generate a realistic stock tick"""
        # Random walk with some mean reversion
        change_pct = random.gauss(0, self.volatility / 100)
        self.current_price *= (1 + change_pct)
        
        # Add some noise to make it realistic
        self.current_price += random.gauss(0, 0.01)
        
        # Volume varies throughout the day
        base_volume = 100
        volume = int(random.expovariate(1.0/base_volume))
        
        tick = {
            "symbol": self.symbol,
            "price": round(self.current_price, 2),
            "size": volume,
            "ts": self.current_time
        }
        
        # Advance time by 100ms to 2 seconds
        self.current_time += timedelta(milliseconds=random.randint(100, 2000))
        
        return tick

class HistoricalDataPlayer:
    """Play back historical data from CSV file"""
    
    def __init__(self, csv_path, symbol_col="symbol", price_col="price", 
                 volume_col="volume", timestamp_col="timestamp", playback_speed=1.0):
        self.df = pd.read_csv(csv_path)
        self.symbol_col = symbol_col
        self.price_col = price_col
        self.volume_col = volume_col
        self.timestamp_col = timestamp_col
        self.playback_speed = playback_speed
        self.current_index = 0
        
        # Convert timestamp column to datetime
        self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
        self.df = self.df.sort_values(timestamp_col)
        
    def has_more_data(self):
        return self.current_index < len(self.df)
    
    def get_next_tick(self):
        if not self.has_more_data():
            return None
            
        row = self.df.iloc[self.current_index]
        tick = {
            "symbol": row[self.symbol_col],
            "price": float(row[self.price_col]),
            "size": float(row[self.volume_col]) if pd.notna(row[self.volume_col]) else 0.0,
            "ts": row[self.timestamp_col].to_pydatetime()
        }
        
        self.current_index += 1
        return tick
    
    def get_delay_until_next(self):
        """Calculate delay until next tick based on timestamp differences"""
        if self.current_index >= len(self.df) - 1:
            return 1.0
            
        current_time = self.df.iloc[self.current_index][self.timestamp_col]
        next_time = self.df.iloc[self.current_index + 1][self.timestamp_col]
        
        delay_seconds = (next_time - current_time).total_seconds()
        return max(0.01, delay_seconds / self.playback_speed)  # Minimum 10ms delay

async def simulate_websocket_feed(data_source, message_handler, max_ticks=None):
    """
    Simulate WebSocket feed using mock or historical data
    
    Args:
        data_source: MockDataGenerator or HistoricalDataPlayer
        message_handler: async function to handle each tick
        max_ticks: maximum number of ticks to generate (None for unlimited)
    """
    tick_count = 0
    
    print(f"Starting simulation with {type(data_source).__name__}")
    
    try:
        while True:
            if max_ticks and tick_count >= max_ticks:
                break
                
            if isinstance(data_source, MockDataGenerator):
                tick = data_source.generate_tick()
                delay = random.uniform(0.1, 2.0)  # Random delay between ticks
                
            elif isinstance(data_source, HistoricalDataPlayer):
                if not data_source.has_more_data():
                    print("Reached end of historical data")
                    break
                tick = data_source.get_next_tick()
                delay = data_source.get_delay_until_next()
            
            else:
                raise ValueError("Unknown data source type")
            
            # Convert tick to JSON message format
            msg_json = json.dumps({
                "type": "trade",
                "symbol": tick["symbol"],
                "price": tick["price"],
                "size": tick["size"],
                "t": tick["ts"].isoformat() + "Z"
            })
            
            # Send to message handler
            await message_handler(msg_json)
            
            # Wait before next tick
            await asyncio.sleep(delay)
            
            tick_count += 1
            
            if tick_count % 100 == 0:
                print(f"Processed {tick_count} ticks...")
                
    except KeyboardInterrupt:
        print(f"Simulation stopped after {tick_count} ticks")

# Example usage functions
def create_mock_data_source(symbol="AAPL", initial_price=150.0, volatility=0.02):
    """Create a mock data generator"""
    return MockDataGenerator(symbol, initial_price, volatility)

def create_historical_data_source(csv_path, playback_speed=1.0):
    """Create a historical data player from CSV file"""
    return HistoricalDataPlayer(csv_path, playback_speed=playback_speed)

def create_sample_csv(filename="sample_stock_data.csv", num_rows=10000):
    """Create a sample CSV file with stock data for testing"""
    print(f"Creating sample data file: {filename}")
    
    # Generate sample data
    start_time = datetime.now() - timedelta(hours=1)
    data = []
    current_price = 150.0
    
    for i in range(num_rows):
        # Random walk price
        change = random.gauss(0, 0.02)
        current_price *= (1 + change/100)
        current_price += random.gauss(0, 0.01)
        
        data.append({
            "timestamp": start_time + timedelta(seconds=i*0.5),
            "symbol": "AAPL",
            "price": round(current_price, 2),
            "volume": int(random.expovariate(1.0/100))
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created {filename} with {len(df)} rows")
    return filename