"""
Simple Gemini Trading Bot - Daily API Key Rotation
Works within extreme free tier limits
"""

import os
import asyncio
import json
from datetime import datetime, timezone, timedelta
import google.generativeai as genai
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 6 Gemini API Keys - use 1 per day
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"), 
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
    os.getenv("GEMINI_API_KEY_6")
]

# Trading settings
DEMO_CAPITAL = 10000.0
ANALYSIS_INTERVAL = 21600  # 6 hours (4 analyses per day)

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

# Initialize
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Simple state
demo_balance = DEMO_CAPITAL
total_trades = 0

def get_daily_api_key() -> str:
    """Use different API key each day"""
    day_of_year = datetime.now(IST).timetuple().tm_yday
    key_index = day_of_year % 6  # Rotate through 6 keys
    
    api_key = GEMINI_API_KEYS[key_index]
    if api_key:
        print(f"🔑 Using daily API Key #{key_index + 1} (Day {day_of_year})")
        return api_key
    return None

async def get_simple_signal(current_price: float) -> dict:
    """Get simple trading signal"""
    try:
        api_key = get_daily_api_key()
        if not api_key:
            return {"signal": "HOLD", "confidence": 0}
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Ultra-minimal prompt
        prompt = f"Bitcoin ${current_price:.0f}. BUY/SELL/HOLD? JSON: {{\"signal\":\"HOLD\",\"confidence\":5}}"
        
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Extract JSON
        if '{' in text and '}' in text:
            json_part = text[text.find('{'):text.rfind('}')+1]
            return json.loads(json_part)
        
        return {"signal": "HOLD", "confidence": 5}
        
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return {"signal": "HOLD", "confidence": 0}

async def main():
    """Simple main loop"""
    print("🤖 Simple Gemini Bot - Daily Key Rotation")
    print("⏳ 6-hour analysis intervals")
    
    while True:
        try:
            # Get latest price
            response = supabase.table("candles").select("close").order("timestamp", desc=True).limit(1).execute()
            if not response.data:
                print("⏳ No candle data")
                await asyncio.sleep(300)
                continue
            
            current_price = float(response.data[0]['close'])
            
            # Get signal
            signal = await get_simple_signal(current_price)
            
            ist_time = datetime.now(IST).strftime("%H:%M IST")
            print(f"🧠 {ist_time}: {signal.get('signal')} | Confidence: {signal.get('confidence')}/10")
            
            # Simple trading logic (placeholder)
            if signal.get('confidence', 0) >= 7:
                print(f"🚀 Would trade: {signal['signal']} at ${current_price:.0f}")
            
            await asyncio.sleep(ANALYSIS_INTERVAL)  # 6 hours
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await asyncio.sleep(1800)  # 30 minutes on error

if __name__ == "__main__":
    asyncio.run(main())
