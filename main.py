"""
Gemini 2.5 Flash Trading Bot - Multi-API Key Version
6 API keys rotation for 1500 RPD (250 x 6) = Every minute analysis
"""

import os
import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import google.generativeai as genai
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 6 Gemini API Keys for rotation (250 RPD each = 1500 total)
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"), 
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
    os.getenv("GEMINI_API_KEY_6")
]

# Trading Configuration
DEMO_CAPITAL = 10000.0
RISK_PER_TRADE = 0.02
ANALYSIS_INTERVAL = 900  # 15 minutes for free tier (4 calls per hour per key = 24 total)

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

# Initialize Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# API Key rotation
current_api_key_index = 0
api_key_usage_count = [0] * 6  # Track usage per key
api_key_reset_time = [0] * 6   # Track reset times

# Trading state
current_position = None
demo_balance = DEMO_CAPITAL
total_trades = 0
winning_trades = 0

class Trade:
    def __init__(self, entry_price: float, stop_loss: float, take_profit: float, 
                 position_size: float, direction: str):
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.direction = direction
        self.entry_time = datetime.now(IST)
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.status = 'open'

def get_next_api_key() -> str:
    """Get next available API key with rate limit management"""
    global current_api_key_index, api_key_usage_count, api_key_reset_time
    
    current_time = time.time()
    
    # Check if current key needs reset (every hour)
    for i in range(6):
        if current_time - api_key_reset_time[i] > 3600:  # 1 hour
            api_key_usage_count[i] = 0
            api_key_reset_time[i] = current_time
    
    # Find available API key (under 240 requests to be safe)
    for attempt in range(6):
        key_index = (current_api_key_index + attempt) % 6
        
        if api_key_usage_count[key_index] < 240:  # Leave buffer
            current_api_key_index = key_index
            api_key_usage_count[key_index] += 1
            
            api_key = GEMINI_API_KEYS[key_index]
            if api_key:
                print(f"🔑 Using API Key #{key_index + 1} (Usage: {api_key_usage_count[key_index]}/250)")
                return api_key
    
    # All keys exhausted, use first one and wait
    print("⚠️  All API keys near limit, using key #1 with delay")
    return GEMINI_API_KEYS[0] if GEMINI_API_KEYS[0] else ""

async def fetch_latest_candles(limit: int = 6000) -> List[Dict]:
    """Fetch candles from candle collector server via Supabase"""
    try:
        response = supabase.table("candles").select("*").order("timestamp", desc=True).limit(limit).execute()
        return list(reversed(response.data))
    except Exception as e:
        print(f"❌ Error fetching candles: {e}")
        return []

def format_candles_for_gemini(candles: List[Dict]) -> str:
    """Format last 50 candles for Gemini (reduced for faster processing)"""
    if not candles:
        return "No candle data available"
    
    recent_candles = candles[-50:]  # Reduced to 50 for faster API calls
    formatted_data = "Bitcoin BTCUSDT - Last 50 Minutes:\n"
    formatted_data += "Time | OHLC | Volume\n"
    formatted_data += "-" * 40 + "\n"
    
    for candle in recent_candles:
        ts = candle.get('timestamp', '')[-8:-3]  # Just time part
        o = candle.get('open', 0)
        h = candle.get('high', 0) 
        l = candle.get('low', 0)
        c = candle.get('close', 0)
        v = candle.get('volume', 0)
        
        formatted_data += f"{ts} | {o:.0f}/{h:.0f}/{l:.0f}/{c:.0f} | {v:.2f}\n"
    
    return formatted_data

async def get_gemini_signal(candles_data: str, current_price: float) -> Dict:
    """Get trading signal using API key rotation"""
    try:
        # Get next available API key
        api_key = get_next_api_key()
        if not api_key:
            return {"signal": "HOLD", "confidence": 0, "reasoning": "No API key available"}
        
        # Configure Gemini with current API key
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Shorter, more efficient prompt
        prompt = f"""
        Bitcoin Analysis - Current: ${current_price:.0f}

        {candles_data}

        Quick trading decision in JSON:
        {{
            "signal": "BUY/SELL/HOLD",
            "entry": {current_price:.0f},
            "stop_loss": {current_price * 0.98:.0f},
            "take_profit": {current_price * 1.04:.0f},
            "confidence": 7,
            "reasoning": "Brief analysis"
        }}

        Rules: Min 6 confidence to trade, 1:2 risk/reward
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean JSON
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1]
            
        return json.loads(response_text)
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            print(f"🚫 Rate limit hit on API key #{current_api_key_index + 1}")
            # Mark this key as exhausted
            api_key_usage_count[current_api_key_index] = 250
        else:
            print(f"❌ Gemini error: {e}")
        
        return {"signal": "HOLD", "confidence": 0, "reasoning": f"Error: {e}"}

def calculate_position_size(entry: float, stop_loss: float, risk_amount: float) -> float:
    """Calculate position size for risk management"""
    risk_per_unit = abs(entry - stop_loss)
    return risk_amount / risk_per_unit if risk_per_unit > 0 else 0

async def execute_trade(signal: Dict, current_price: float) -> Optional[Trade]:
    """Execute paper trade"""
    global current_position, demo_balance, total_trades
    
    if signal['signal'] == 'HOLD' or signal['confidence'] < 6:
        return None
    
    # Close opposite position
    if current_position and current_position.status == 'open':
        if (current_position.direction == 'long' and signal['signal'] == 'SELL') or \
           (current_position.direction == 'short' and signal['signal'] == 'BUY'):
            await close_position(current_price, "Signal reversal")
    
    if current_position and current_position.status == 'open':
        return None
    
    # Calculate trade parameters
    risk_amount = demo_balance * RISK_PER_TRADE
    entry = signal.get('entry', current_price)
    stop_loss = signal.get('stop_loss', current_price * 0.98)
    take_profit = signal.get('take_profit', current_price * 1.04)
    direction = 'long' if signal['signal'] == 'BUY' else 'short'
    position_size = calculate_position_size(entry, stop_loss, risk_amount)
    
    if position_size <= 0:
        return None
    
    # Create trade
    trade = Trade(entry, stop_loss, take_profit, position_size, direction)
    current_position = trade
    total_trades += 1
    
    # Save to database
    trade_data = {
        "trade_id": total_trades,
        "timestamp": trade.entry_time.isoformat(),
        "direction": direction,
        "entry_price": entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "position_size": position_size,
        "risk_amount": risk_amount,
        "confidence": signal['confidence'],
        "reasoning": signal['reasoning'],
        "status": "open"
    }
    
    try:
        supabase.table("paper_trades").insert(trade_data).execute()
        print(f"🚀 TRADE #{total_trades}: {direction.upper()} @ ${entry:.0f} | SL: ${stop_loss:.0f} | TP: ${take_profit:.0f}")
    except Exception as e:
        print(f"❌ Save error: {e}")
    
    return trade

async def close_position(exit_price: float, reason: str):
    """Close current position"""
    global current_position, demo_balance, winning_trades
    
    if not current_position or current_position.status != 'open':
        return
    
    # Calculate PnL
    if current_position.direction == 'long':
        pnl = (exit_price - current_position.entry_price) * current_position.position_size
    else:
        pnl = (current_position.entry_price - exit_price) * current_position.position_size
    
    current_position.pnl = pnl
    current_position.exit_price = exit_price
    current_position.exit_time = datetime.now(IST)
    current_position.status = 'closed'
    
    demo_balance += pnl
    if pnl > 0:
        winning_trades += 1
    
    # Update database
    try:
        supabase.table("paper_trades").update({
            "exit_price": exit_price,
            "exit_time": current_position.exit_time.isoformat(),
            "pnl": pnl,
            "status": "closed",
            "close_reason": reason
        }).eq("trade_id", total_trades).execute()
        
        print(f"📊 CLOSED #{total_trades}: ${exit_price:.0f} | PnL: ${pnl:.2f} | {reason}")
    except Exception as e:
        print(f"❌ Update error: {e}")
    
    current_position = None

async def check_stop_loss_take_profit(current_price: float):
    """Check SL/TP levels"""
    if not current_position or current_position.status != 'open':
        return
    
    if current_position.direction == 'long':
        if current_price <= current_position.stop_loss:
            await close_position(current_price, "Stop Loss")
        elif current_price >= current_position.take_profit:
            await close_position(current_price, "Take Profit")
    else:  # short
        if current_price >= current_position.stop_loss:
            await close_position(current_price, "Stop Loss")
        elif current_price <= current_position.take_profit:
            await close_position(current_price, "Take Profit")

async def print_stats():
    """Print trading statistics"""
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_return = ((demo_balance - DEMO_CAPITAL) / DEMO_CAPITAL * 100)
    
    # Show API key usage
    total_usage = sum(api_key_usage_count)
    print(f"📈 Balance: ${demo_balance:.0f} | Return: {total_return:.1f}% | Trades: {total_trades} | Win: {win_rate:.0f}% | API: {total_usage}/1500")

async def main():
    """Main trading loop with 6 API keys"""
    print("🤖 Gemini Trading Bot - 6 API Keys (24 calls/hour)")
    print(f"💰 Demo Capital: ${DEMO_CAPITAL}")
    print("⚡ Every 15 minutes analysis (free tier optimized)")
    
    # Validate API keys
    valid_keys = [key for key in GEMINI_API_KEYS if key]
    print(f"🔑 API Keys loaded: {len(valid_keys)}/6")
    
    if len(valid_keys) == 0:
        print("❌ No API keys found! Add GEMINI_API_KEY_1 to GEMINI_API_KEY_6")
        return
    
    while True:
        try:
            # Get candles
            candles = await fetch_latest_candles(6000)
            if not candles:
                print("⏳ Waiting for candle data...")
                await asyncio.sleep(60)
                continue
            
            current_price = float(candles[-1].get('close', 0))
            if current_price <= 0:
                await asyncio.sleep(60)
                continue
            
            # Check position management
            await check_stop_loss_take_profit(current_price)
            
            # Get Gemini analysis with API rotation
            candles_data = format_candles_for_gemini(candles)
            signal = await get_gemini_signal(candles_data, current_price)
            
            print(f"🧠 Gemini: {signal.get('signal')} | Confidence: {signal.get('confidence')}/10")
            
            # Execute trade
            await execute_trade(signal, current_price)
            await print_stats()
            
            await asyncio.sleep(ANALYSIS_INTERVAL)  # 1 minute
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
