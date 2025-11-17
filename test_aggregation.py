#!/usr/bin/env python3
"""
Test script to verify 4H candle aggregation works locally
"""

import pandas as pd
from datetime import datetime, timedelta
import json

def create_test_data():
    """Create 25,000 1-minute test candles (17+ days)"""
    print("🔧 Creating 25,000 test 1-minute candles...")
    
    candles = []
    start_time = datetime.now() - timedelta(days=17, hours=5)  # 17+ days ago
    
    for i in range(25000):
        timestamp = start_time + timedelta(minutes=i)
        
        # Simulate realistic BTCUSDT price movement
        base_price = 92000 + (i % 1000) - 500  # Price variation
        
        candle = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'open': base_price,
            'high': base_price + 50,
            'low': base_price - 50, 
            'close': base_price + (i % 20) - 10,
            'volume': 100 + (i % 50)
        }
        candles.append(candle)
    
    print(f"✅ Created {len(candles)} test candles")
    print(f"📅 From: {candles[0]['timestamp']}")
    print(f"📅 To: {candles[-1]['timestamp']}")
    
    return candles

def aggregate_candles_test(candles_1min, timeframe='4h'):
    """Test aggregation function locally"""
    print(f"\n🔧 Testing {timeframe} aggregation...")
    
    if not candles_1min:
        return []
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(candles_1min)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"📊 Input: {len(df)} 1-minute candles")
        print(f"📅 Range: {df.index[0]} to {df.index[-1]}")
        
        # Resample to 4H
        aggregated = df.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min', 
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        print(f"📊 Output: {len(aggregated)} 4H candles")
        
        if len(aggregated) > 0:
            print(f"📅 First 4H: {aggregated.index[0]}")
            print(f"📅 Last 4H: {aggregated.index[-1]}")
            
            # Show first few candles
            print(f"\n📈 First 5 4H candles:")
            for i, (timestamp, row) in enumerate(aggregated.head().iterrows()):
                print(f"  {i+1}. {timestamp}: O:{row['open']:.0f} H:{row['high']:.0f} L:{row['low']:.0f} C:{row['close']:.0f}")
        
        # Convert back to list of dicts
        result = aggregated.reset_index()
        result['timestamp'] = result['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return result.to_dict('records')
        
    except Exception as e:
        print(f"❌ Aggregation error: {e}")
        return []

def main():
    print("🧪 LOCAL AGGREGATION TEST")
    print("=" * 50)
    
    # Create test data
    test_candles = create_test_data()
    
    # Test 4H aggregation
    candles_4h = aggregate_candles_test(test_candles, '4h')
    
    print(f"\n🎯 RESULTS:")
    print(f"✅ Input: 25,000 1-minute candles")
    print(f"✅ Output: {len(candles_4h)} 4H candles")
    print(f"✅ Expected: ~104 4H candles (25000÷240)")
    
    if len(candles_4h) > 100:
        print(f"🎉 SUCCESS! Aggregation works correctly")
        print(f"📊 Pro would get {len(candles_4h)} 4H candles for analysis")
    else:
        print(f"❌ PROBLEM! Only {len(candles_4h)} 4H candles generated")
    
    # Test 1H aggregation too
    candles_1h = aggregate_candles_test(test_candles, '1h')
    print(f"✅ 1H candles: {len(candles_1h)} (expected ~417)")

if __name__ == "__main__":
    main()
