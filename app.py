from flask import Flask, render_template, jsonify
from supabase import create_client
import os
import json
from datetime import datetime, timezone, timedelta

app = Flask(__name__)

# Supabase connections with error handling
try:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    TRADES_SUPABASE_URL = os.getenv("TRADES_SUPABASE_URL")
    TRADES_SUPABASE_KEY = os.getenv("TRADES_SUPABASE_KEY")

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None
    trades_supabase = create_client(TRADES_SUPABASE_URL, TRADES_SUPABASE_KEY) if TRADES_SUPABASE_URL else None
except Exception as e:
    print(f"Supabase connection error: {e}")
    supabase = None
    trades_supabase = None

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/trades')
def get_trades():
    if trades_supabase:
        try:
            # Get all trades ordered by timestamp
            response = trades_supabase.table("paper_trades").select("*").order("timestamp", desc=False).execute()
            return jsonify(response.data)
        except Exception as e:
            print(f"Trades DB error: {e}")
    
    # Fallback to local file
    try:
        with open('trades.json', 'r') as f:
            trades = json.load(f)
        return jsonify(trades)
    except:
        return jsonify([])

@app.route('/api/candles')
def get_candles():
    if supabase:
        try:
            response = supabase.table("candles").select("*").order("timestamp", desc=True).limit(100).execute()
            return jsonify(response.data)
        except Exception as e:
            print(f"Candles error: {e}")
    return jsonify([])

@app.route('/api/status')
def get_status():
    try:
        current_price = 0
        latest_trade = None
        
        # Get latest candle from main Supabase
        if supabase:
            try:
                candles = supabase.table("candles").select("*").order("timestamp", desc=True).limit(1).execute()
                if candles.data:
                    current_price = float(candles.data[0]['close'])
            except Exception as e:
                print(f"Price fetch error: {e}")
        
        # Get latest trade from trades Supabase
        if trades_supabase:
            try:
                trades = trades_supabase.table("paper_trades").select("*").order("timestamp", desc=True).limit(1).execute()
                if trades.data:
                    latest_trade = trades.data[0]
            except Exception as e:
                print(f"Latest trade error: {e}")
        
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
    except Exception as e:
        print(f"Status error: {e}")
        return jsonify({"current_price": 0, "latest_trade": None})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask app on port {port}")
    print(f"Supabase connected: {supabase is not None}")
    print(f"Trades Supabase connected: {trades_supabase is not None}")
    app.run(host='0.0.0.0', port=port, debug=False)
