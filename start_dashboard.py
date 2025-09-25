# start_dashboard.py
"""
Simple launcher for the real-time stock prediction dashboard
"""

import subprocess
import sys
import webbrowser
import time
import threading

def open_browser():
    """Open browser after a short delay"""
    time.sleep(3)
    print("🌐 Opening browser...")
    webbrowser.open('http://localhost:5000')

def main():
    print("🚀 Starting Real-time Stock Prediction Dashboard")
    print("=" * 50)
    print("📊 Features:")
    print("   ✅ Live market data from Yahoo Finance")
    print("   ✅ Real-time AI trend predictions")
    print("   ✅ Interactive charts and graphs")
    print("   ✅ Performance tracking")
    print("   ✅ Technical indicators")
    print("=" * 50)
    
    # Start browser opening in background
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    try:
        # Start the dashboard
        subprocess.run([sys.executable, "real_time_dashboard.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting dashboard: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()