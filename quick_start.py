# quick_start.py
"""
Quick start script for the 5-minute trend prediction system
Run this to test the system with mock data
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_trend_predictor import run_simulation

async def main():
    print("=== 5-Minute Stock Trend Prediction System ===")
    print("Running simulation with mock data...")
    print("This will generate realistic stock ticks and predict 5-minute trends")
    print()
    
    # Run simulation with mock data for 500 ticks
    await run_simulation(
        mode="mock", 
        symbol="AAPL", 
        max_ticks=500
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have installed the required packages:")
        print("pip install -r requirements.txt")