"""
Gemini 2.5 Flash Trading Bot - Standalone Version
Separate server for AI trading analysis and paper trading
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Trading Configuration
DEMO_CAPITAL = 10000.0
RISK_PER_TRADE = 0.02
ANALYSIS_INTERVAL = 300  # 5 minutes

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

# Initialize clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

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

async def fetch_latest_candles(limit: int = 6000) -> List[Dict]:
    """Fetch candles from candle collector server via Supabase"""
    try:
        response = supabase.table("candles").select("*").order("timestamp", desc=True).limit(limit).execute()
        return list(reversed(response.data))
    except Exception as e:
        print(f"❌ Error fetching candles: {e}")
        return []

def format_candles_for_gemini(candles: List[Dict]) -> str:
    """Format last 100 candles for Gemini analysis"""
    if not candles:
        return "No candle data available"
    
    recent_candles = candles[-100:]
    formatted_data = "Bitcoin BTCUSDT - Last 100 Minutes (IST):\n"
    formatted_data += "Time | Open | High | Low | Close | Volume | Trades\n"
    formatted_data += "-" * 70 + "\n"
    
    for candle in recent_candles:
        ts = candle.get('timestamp', '')
        o = candle.get('open', 0)
        h = candle.get('high', 0) 
        l = candle.get('low', 0)
        c = candle.get('close', 0)
        v = candle.get('volume', 0)
        t = candle.get('trades', 0)
        
        formatted_data += f"{ts} | {o:.2f} | {h:.2f} | {l:.2f} | {c:.2f} | {v:.4f} | {t}\n"
    
    return formatted_data

async def get_gemini_signal(candles_data: str, current_price: float) -> Dict:
    """Get trading signal from Gemini 2.5 Flash"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = f"""
        Analyze this Bitcoin market data and provide trading recommendation:

        {candles_data}

        Current Price: ${current_price:.2f}

        Provide analysis in JSON format:
        {{
            "signal": "BUY/SELL/HOLD",
            "entry": {current_price:.2f},
            "stop_loss": {current_price * 0.98:.2f},
            "take_profit": {current_price * 1.04:.2f},
            "confidence": 7,
            "reasoning": "Technical analysis summary"
        }}

        Requirements:
        - Minimum 1:2 risk/reward ratio
        - Confidence 1-10 (only trade if 6+)
        - Consider support/resistance, patterns, volume
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean JSON response
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1]
            
        return json.loads(response_text)
        
    except Exception as e:
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
        print(f"🚀 TRADE #{total_trades}: {direction.upper()} @ ${entry:.2f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}")
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
        
        print(f"📊 CLOSED #{total_trades}: ${exit_price:.2f} | PnL: ${pnl:.2f} | {reason}")
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
    
    print(f"\n📈 STATS: Balance: ${demo_balance:.2f} | Return: {total_return:.1f}% | Trades: {total_trades} | Win Rate: {win_rate:.1f}%")

async def main():
    """Main trading loop"""
    print("🤖 Gemini Trading Bot - Separate Server")
    print(f"💰 Demo Capital: ${DEMO_CAPITAL}")
    print("🔗 Connected to candle collector via Supabase")
    
    while True:
        try:
            # Get candles from collector server
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
            
            # Get Gemini analysis
            candles_data = format_candles_for_gemini(candles)
            signal = await get_gemini_signal(candles_data, current_price)
            
            print(f"🧠 Gemini: {signal.get('signal')} | Confidence: {signal.get('confidence')}/10")
            
            # Execute trade
            await execute_trade(signal, current_price)
            await print_stats()
            
            await asyncio.sleep(ANALYSIS_INTERVAL)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
