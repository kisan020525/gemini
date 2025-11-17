#!/usr/bin/env python3
"""
Fetch historical Bitcoin data to give Pro 100 4H + 168 1H candles
"""

import requests
import json
from datetime import datetime, timedelta
import time

def fetch_binance_klines(symbol='BTCUSDT', interval='1m', limit=1000):
    """Fetch historical klines from Binance public API"""
    url = 'https://api.binance.com/api/v3/klines'
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error fetching from Binance: {e}")
        return []

def fetch_coingecko_data():
    """Fetch from CoinGecko (free, no API key needed)"""
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    
    # Get 30 days of data
    params = {
        'vs_currency': 'usd',
        'days': '30',
        'interval': 'hourly'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert to our format
        candles = []
        prices = data['prices']
        
        for i, price_data in enumerate(prices):
            timestamp = datetime.fromtimestamp(price_data[0] / 1000)
            price = price_data[1]
            
            # Create OHLCV candle (simplified - all same price)
            candle = {
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'open': price,
                'high': price * 1.001,  # Small variation
                'low': price * 0.999,
                'close': price,
                'volume': 100  # Dummy volume
            }
            candles.append(candle)
        
        return candles
        
    except Exception as e:
        print(f"❌ Error fetching from CoinGecko: {e}")
        return []

def create_synthetic_1min_data(hourly_candles):
    """Convert hourly candles to 1-minute candles (synthetic)"""
    minute_candles = []
    
    for candle in hourly_candles:
        base_price = candle['close']
        base_time = datetime.strptime(candle['timestamp'], '%Y-%m-%d %H:%M:%S')
        
        # Create 60 1-minute candles for each hour
        for minute in range(60):
            timestamp = base_time + timedelta(minutes=minute)
            
            # Small price variation within the hour
            variation = (minute - 30) * 0.0001  # ±0.01% variation
            price = base_price * (1 + variation)
            
            min_candle = {
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'open': price,
                'high': price * 1.0005,
                'low': price * 0.9995,
                'close': price,
                'volume': candle['volume'] / 60  # Distribute volume
            }
            minute_candles.append(min_candle)
    
    return minute_candles

def get_historical_data_for_pro():
    """Get enough historical data for Pro analysis"""
    print("📊 Fetching historical data for Pro analysis...")
    
    # Try CoinGecko first (free, reliable)
    hourly_data = fetch_coingecko_data()
    
    if hourly_data:
        print(f"✅ Got {len(hourly_data)} hourly candles from CoinGecko")
        
        # Convert to 1-minute synthetic data
        minute_data = create_synthetic_1min_data(hourly_data)
        print(f"✅ Created {len(minute_data)} synthetic 1-minute candles")
        
        return minute_data
    
    print("❌ Could not fetch historical data")
    return []

if __name__ == "__main__":
    # Test the historical data fetch
    data = get_historical_data_for_pro()
    print(f"📊 Total 1-minute candles: {len(data)}")
    
    if data:
        print(f"📅 From: {data[0]['timestamp']}")
        print(f"📅 To: {data[-1]['timestamp']}")
        
        # Test aggregation
        import sys
        sys.path.append('.')
        from test_aggregation import aggregate_candles_test
        
        candles_4h = aggregate_candles_test(data, '4h')
        candles_1h = aggregate_candles_test(data, '1h')
        
        print(f"🎯 Result: {len(candles_4h)} 4H candles, {len(candles_1h)} 1H candles")
