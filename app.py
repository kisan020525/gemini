from flask import Flask, render_template, jsonify
from supabase import create_client
import os
from datetime import datetime, timezone, timedelta

app = Flask(__name__)

# Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/trades')
def get_trades():
    try:
        with open('trades.json', 'r') as f:
            trades = json.load(f)
        return jsonify(trades[-10:])  # Last 10 trades
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
        # Get latest candle (only read from candles table)
        candles = supabase.table("candles").select("*").order("timestamp", desc=True).limit(1).execute()
        current_price = float(candles.data[0]['close']) if candles.data else 0
        
        # Get latest trade from local file
        try:
            with open('trades.json', 'r') as f:
                trades = json.load(f)
            latest_trade = trades[-1] if trades else None
        except:
            latest_trade = None
        
        return jsonify({
            "current_price": current_price,
            "latest_trade": latest_trade,
            "timestamp": datetime.now().isoformat()
        })
    except:
        return jsonify({"current_price": 0, "latest_trade": None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
