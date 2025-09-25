#!/usr/bin/env python3
"""
🚀 Enhanced Professional Stock Predictor Launcher
Professional-grade fintech application with advanced analytics
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'flask', 'flask-socketio', 'yfinance', 'pandas', 
        'numpy', 'plotly', 'reportlab', 'openpyxl', 'scipy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"📦 Run: pip install {' '.join(missing)}")
        return False
    
    print("✅ All required packages installed")
    return True

def main():
    print("="*70)
    print("🚀 ENHANCED PROFESSIONAL STOCK PREDICTOR")
    print("="*70)
    print("📊 Features:")
    print("   • Interactive Charts with Plotly.js")
    print("   • Advanced Technical Indicators (Fibonacci, Elliott Wave)")
    print("   • Multi-Market Analysis (Stocks, Crypto, Forex)")
    print("   • Professional Export (PDF/Excel)")
    print("   • Real-time WebSocket Updates")
    print("   • Volume Profile & Market Heatmaps")
    print("="*70)
    
    if not check_requirements():
        return
    
    # Check if enhanced predictor exists
    enhanced_file = Path("enhanced_professional_predictor.py")
    if not enhanced_file.exists():
        print("❌ Enhanced predictor file not found!")
        print("📄 Looking for: enhanced_professional_predictor.py")
        return
    
    print("🎯 Starting Enhanced Professional Stock Predictor...")
    print("🌐 Dashboard will be available at: http://localhost:5000")
    print("📊 Features: Interactive charts, advanced indicators, multi-market analysis")
    print("-" * 70)
    
    try:
        # Run the enhanced predictor
        subprocess.run([sys.executable, "enhanced_professional_predictor.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Enhanced Stock Predictor stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running enhanced predictor: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()