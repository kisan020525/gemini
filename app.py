from flask import Flask, render_template, jsonify
from supabase import create_client
import os
import json
from datetime import datetime, timezone, timedelta

app = Flask(__name__)

# Supabase connections
SUPABASE_URL = os.getenv("SUPABASE_URL")  # For candles
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TRADES_SUPABASE_URL = os.getenv("TRADES_SUPABASE_URL")  # For trades
TRADES_SUPABASE_KEY = os.getenv("TRADES_SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)  # Candles
trades_supabase = create_client(TRADES_SUPABASE_URL, TRADES_SUPABASE_KEY) if TRADES_SUPABASE_URL else None

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/trades')
def get_trades():
    if trades_supabase:
        try:
            response = trades_supabase.table("paper_trades").select("*").order("timestamp", desc=True).limit(10).execute()
            return jsonify(response.data)
        except:
            pass
    
    # Fallback to local file
    try:
        with open('trades.json', 'r') as f:
            trades = json.load(f)
        return jsonify(trades[-10:])
    except:
        return jsonify([])

@app.route('/api/candles')
def get_candles():
    try:
        response = supabase.table("candles").select("*").order("timestamp", desc=True).limit(100).execute()
        return jsonify(response.data)
    except:
        return jsonify([])

@app.route('/api/status')
def get_status():
    try:
        # Get latest candle from main Supabase
        candles = supabase.table("candles").select("*").order("timestamp", desc=True).limit(1).execute()
        current_price = float(candles.data[0]['close']) if candles.data else 0
        
        # Get latest trade from trades Supabase
        latest_trade = None
        if trades_supabase:
            try:
                trades = trades_supabase.table("paper_trades").select("*").order("timestamp", desc=True).limit(1).execute()
                latest_trade = trades.data[0] if trades.data else None
            except:
                pass
        
        # Fallback to local file
        if not latest_trade:
            try:
                with open('trades.json', 'r') as f:
                    trades = json.load(f)
                latest_trade = trades[-1] if trades else None
            except:
                pass
        
        return jsonify({
            "current_price": current_price,
            "latest_trade": latest_trade,
            "timestamp": datetime.now().isoformat()
        })
    except:
        return jsonify({"current_price": 0, "latest_trade": None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
