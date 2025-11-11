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
SUPABASE_URL = os.getenv("SUPABASE_URL")  # For candles (read-only)
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Separate Supabase for trades
TRADES_SUPABASE_URL = os.getenv("TRADES_SUPABASE_URL")
TRADES_SUPABASE_KEY = os.getenv("TRADES_SUPABASE_KEY")

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
ANALYSIS_INTERVAL = 3600  # 1 hour intervals for free tier safety

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

# Initialize Supabase connections
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)  # For candles
trades_supabase = create_client(TRADES_SUPABASE_URL, TRADES_SUPABASE_KEY) if TRADES_SUPABASE_URL else None

# API Key rotation with daily limits
current_api_key_index = 0
api_key_last_used = [0] * 6   # Track last usage time
api_key_daily_count = [0] * 6  # Track daily usage
api_key_daily_reset = [0] * 6  # Track daily reset time
api_key_usage_count = [0] * 6  # Legacy compatibility

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
    """Ultra-conservative API key management for free tier"""
    global current_api_key_index, api_key_last_used, api_key_daily_count, api_key_daily_reset
    
    current_time = time.time()
    
    # Reset daily counters (every 24 hours)
    for i in range(6):
        if current_time - api_key_daily_reset[i] > 86400:  # 24 hours
            api_key_daily_count[i] = 0
            api_key_daily_reset[i] = current_time
            print(f"🔄 Reset daily counter for API Key #{i + 1}")
    
    # Find available API key with strict limits
    for attempt in range(6):
        key_index = (current_api_key_index + attempt) % 6
        
        # Check daily limit (max 250 per day per key = 1500 total)
        if api_key_daily_count[key_index] >= 250:
            continue
            
        # Check minimum time between calls (1 minute for higher throughput)
        time_since_last = current_time - api_key_last_used[key_index]
        if time_since_last < 60:  # 1 minute
            continue
        
        # This key is available
        current_api_key_index = key_index
        api_key_last_used[key_index] = current_time
        api_key_daily_count[key_index] += 1
        
        api_key = GEMINI_API_KEYS[key_index]
        if api_key:
            print(f"🔑 Using API Key #{key_index + 1} (Daily: {api_key_daily_count[key_index]}/250)")
            return api_key
    
    # No keys available
    print("⏳ All API keys cooling down, waiting...")
    return None

async def fetch_latest_candles(limit: int = 6000) -> List[Dict]:
    """Fetch candles from candle collector server via Supabase"""
    try:
        response = supabase.table("candles").select("*").order("timestamp", desc=True).limit(limit).execute()
        return list(reversed(response.data))
    except Exception as e:
        print(f"❌ Error fetching candles: {e}")
        return []

def format_candles_for_gemini(candles: List[Dict]) -> str:
    """Format all 6k candles for comprehensive analysis"""
    if not candles:
        return "No candle data available"
    
    # Use all available candles (up to 6000)
    formatted_data = f"Bitcoin BTCUSDT - {len(candles)} minute candles:\n"
    formatted_data += "Recent 100 candles (Time|OHLC|Vol):\n"
    
    # Show last 100 for detailed view
    recent = candles[-100:]
    for candle in recent:
        ts = candle.get('timestamp', '')[-8:-3]
        o, h, l, c = candle.get('open', 0), candle.get('high', 0), candle.get('low', 0), candle.get('close', 0)
        v = candle.get('volume', 0)
        formatted_data += f"{ts}|{o:.0f}/{h:.0f}/{l:.0f}/{c:.0f}|{v:.1f}\n"
    
    # Add statistical summary of all candles
    closes = [float(c.get('close', 0)) for c in candles if c.get('close')]
    if closes:
        current = closes[-1]
        high_24h = max(closes[-1440:]) if len(closes) >= 1440 else max(closes)
        low_24h = min(closes[-1440:]) if len(closes) >= 1440 else min(closes)
        formatted_data += f"\nCurrent: ${current:.0f} | 24h High: ${high_24h:.0f} | 24h Low: ${low_24h:.0f}"
    
    return formatted_data

async def get_gemini_signal(candles_data: str, current_price: float) -> Dict:
    """Get trading signal with ultra-conservative rate limiting"""
    try:
        # Get next available API key
        api_key = get_next_api_key()
        if not api_key:
            print("🚫 No API keys available, skipping analysis")
            return {"signal": "HOLD", "confidence": 0, "reasoning": "No API keys available"}
        
        # Configure Gemini with current API key
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
        
        # Unleash full AI analytical power
        prompt = f"""
        You are an advanced AI with access to vast financial knowledge. Analyze {len(candles_data.split('\n'))} Bitcoin candles using your complete analytical capabilities.

        Current Bitcoin price: ${current_price:.0f}

        MARKET DATA:
        {candles_data}

        ANALYSIS DIRECTIVE:
        Use your FULL AI knowledge base. Apply every relevant concept from:
        - Mathematical models (statistics, probability, regression, neural patterns)
        - Financial theory (EMH, behavioral finance, market microstructure)
        - Technical analysis (all indicators, patterns, oscillators)
        - Quantitative methods (algorithmic signals, machine learning patterns)
        - Market psychology (sentiment analysis, crowd behavior)
        - Economic principles (supply/demand, liquidity, volatility)
        - Information theory (signal processing, noise filtering)
        - Game theory (market participant behavior)
        - Chaos theory (non-linear dynamics, fractals)
        - Any other relevant knowledge domains

        COMPUTATIONAL APPROACH:
        1. Process ALL candle data simultaneously
        2. Identify patterns across multiple timeframes and scales
        3. Calculate probabilities and statistical significance
        4. Cross-validate signals using multiple methodologies
        5. Optimize risk-adjusted returns
        6. Consider market regime and structural changes
        7. Apply ensemble methods for signal confirmation

        OUTPUT REQUIREMENTS:
        Provide comprehensive analysis with mathematical precision. Use your AI capabilities to find patterns humans might miss.

        JSON RESPONSE:
        {{
            "ai_analysis": "Your complete computational analysis",
            "pattern_recognition": "Patterns identified across all scales",
            "statistical_confidence": "Mathematical probability assessment",
            "signal": "BUY/SELL/HOLD",
            "confidence": 1-10,
            "entry": {current_price},
            "stop_loss": price,
            "take_profit": price,
            "risk_reward_ratio": "1:X",
            "probability_success": "X%",
            "market_regime": "trending/ranging/volatile/transitional",
            "key_factors": ["primary factors driving decision"],
            "reasoning": "Detailed analytical logic"
        }}

        MAXIMIZE YOUR AI ANALYTICAL POWER. FIND EVERY EDGE.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Enhanced JSON parsing for Gemini 2.5
        try:
            # Remove markdown formatting
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].strip()
            
            # Clean up common issues
            response_text = response_text.replace('\n', '').replace('  ', ' ')
            
            # Try to find JSON object
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                response_text = response_text[start:end]
            
            result = json.loads(response_text)
            
            # Validate and add missing fields
            result['signal'] = result.get('signal', 'HOLD').upper()
            result['confidence'] = int(result.get('confidence', 5))
            result['entry'] = float(result.get('entry', current_price))
            result['stop_loss'] = float(result.get('stop_loss', current_price * 0.98))
            result['take_profit'] = float(result.get('take_profit', current_price * 1.04))
            result['reasoning'] = str(result.get('reasoning', 'AI analysis'))
            result['ai_analysis'] = str(result.get('ai_analysis', 'Computational analysis'))
            result['key_factors'] = result.get('key_factors', ['Price Action'])
            
            return result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Enhanced fallback - try to extract signal from text
            text_upper = response_text.upper()
            signal = "HOLD"
            confidence = 5
            
            if "BUY" in text_upper and "SELL" not in text_upper:
                signal = "BUY"
                confidence = 7
            elif "SELL" in text_upper and "BUY" not in text_upper:
                signal = "SELL" 
                confidence = 7
            
            return {
                "signal": signal,
                "confidence": confidence,
                "reasoning": f"Text parse: {response_text[:50]}",
                "entry": current_price,
                "stop_loss": current_price * 0.98,
                "take_profit": current_price * 1.04
            }
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
            print(f"🚫 Rate limit on API key #{current_api_key_index + 1}")
            # Mark this key as exhausted for today
            api_key_daily_count[current_api_key_index] = 250
        else:
            print(f"❌ Gemini error: {e}")
        
        return {
            "signal": "HOLD", 
            "confidence": 0, 
            "reasoning": f"Error: {str(e)[:50]}",
            "entry": current_price,
            "stop_loss": current_price * 0.98,
            "take_profit": current_price * 1.04
        }

def calculate_position_size(entry: float, stop_loss: float, risk_amount: float) -> float:
    """Calculate position size for risk management"""
    risk_per_unit = abs(entry - stop_loss)
    return risk_amount / risk_per_unit if risk_per_unit > 0 else 0

async def execute_trade(signal: Dict, current_price: float) -> Optional[Trade]:
    """Execute paper trade only when highly confident"""
    global current_position, demo_balance, total_trades
    
    # Only trade with high confidence (8+ out of 10)
    if signal['signal'] == 'HOLD' or signal['confidence'] < 8:
        return None
    
    # Close opposite position only if new signal is very strong (9+)
    if current_position and current_position.status == 'open':
        if signal['confidence'] >= 9:
            if (current_position.direction == 'long' and signal['signal'] == 'SELL') or \
               (current_position.direction == 'short' and signal['signal'] == 'BUY'):
                await close_position(current_price, "Strong signal reversal")
        else:
            return None  # Don't reverse unless very confident
    
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
    
    # Save to separate trades database
    trade_data = {
        "trade_id": total_trades,
        "timestamp": trade.entry_time.isoformat(),
        "created_at": trade.entry_time.isoformat(),  # Add created_at
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
    
    if trades_supabase:
        try:
            trades_supabase.table("paper_trades").insert(trade_data).execute()
            print(f"🚀 TRADE #{total_trades}: {direction.upper()} @ ${entry:.0f} | SL: ${stop_loss:.0f} | TP: ${take_profit:.0f}")
        except Exception as e:
            print(f"🚀 TRADE #{total_trades}: {direction.upper()} @ ${entry:.0f} | SL: ${stop_loss:.0f} | TP: ${take_profit:.0f}")
            print(f"⚠️ Trades DB error: {str(e)}")
            # Fallback to local storage
            save_trade_locally(trade_data)
    else:
        # Fallback to local storage
        save_trade_locally(trade_data)
        print(f"🚀 TRADE #{total_trades}: {direction.upper()} @ ${entry:.0f} | SL: ${stop_loss:.0f} | TP: ${take_profit:.0f}")
        print("⚠️ Using local storage - no trades DB configured")
    
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
    
    # Update separate trades database
    if trades_supabase:
        try:
            trades_supabase.table("paper_trades").update({
                "exit_price": exit_price,
                "exit_time": current_position.exit_time.isoformat(),
                "pnl": pnl,
                "status": "closed",
                "close_reason": reason
            }).eq("trade_id", total_trades).execute()
            
            print(f"📊 CLOSED #{total_trades}: ${exit_price:.0f} | PnL: ${pnl:.2f} | {reason}")
        except Exception as e:
            print(f"📊 CLOSED #{total_trades}: ${exit_price:.0f} | PnL: ${pnl:.2f} | {reason}")
            print(f"⚠️ Trades DB update error: {str(e)}")
            # Fallback to local storage
            update_data = {
                "exit_price": exit_price,
                "exit_time": current_position.exit_time.isoformat(),
                "pnl": pnl,
                "status": "closed",
                "close_reason": reason
            }
            update_trade_locally(total_trades, update_data)
    else:
        # Fallback to local storage
        update_data = {
            "exit_price": exit_price,
            "exit_time": current_position.exit_time.isoformat(),
            "pnl": pnl,
            "status": "closed",
            "close_reason": reason
        }
        update_trade_locally(total_trades, update_data)
        print(f"📊 CLOSED #{total_trades}: ${exit_price:.0f} | PnL: ${pnl:.2f} | {reason}")
        print("⚠️ Using local storage - no trades DB configured")
    
    current_position = None

async def check_stop_loss_take_profit(current_price: float):
    """Check SL/TP levels with proper exit prices and correct logic"""
    if not current_position or current_position.status != 'open':
        return
    
    if current_position.direction == 'long':
        # LONG: SL below entry, TP above entry
        if current_price <= current_position.stop_loss:
            await close_position(current_position.stop_loss, "Stop Loss")
        elif current_price >= current_position.take_profit:
            await close_position(current_position.take_profit, "Take Profit")
    else:  # short
        # SHORT: SL above entry, TP below entry  
        if current_price >= current_position.stop_loss:
            await close_position(current_position.stop_loss, "Stop Loss")
        elif current_price <= current_position.take_profit:
            await close_position(current_position.take_profit, "Take Profit")

async def print_stats():
    """Print trading statistics"""
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_return = ((demo_balance - DEMO_CAPITAL) / DEMO_CAPITAL * 100)
    
    # Show API key usage
    total_daily_usage = sum(api_key_daily_count)
    print(f"📈 Balance: ${demo_balance:.0f} | Return: {total_return:.1f}% | Trades: {total_trades} | Win: {win_rate:.0f}% | API: {total_daily_usage}/1500")

async def wait_for_new_candle(last_candle_time: str) -> bool:
    """Wait for new candle to be added to Supabase"""
    while True:
        try:
            response = supabase.table("candles").select("timestamp").order("timestamp", desc=True).limit(1).execute()
            if response.data and response.data[0]['timestamp'] != last_candle_time:
                return True
            await asyncio.sleep(10)  # Check every 10 seconds
        except:
            await asyncio.sleep(30)

import json

# Local trade storage
TRADES_FILE = "trades.json"

def save_trade_locally(trade_data):
    """Save trade to local JSON file"""
    try:
        # Load existing trades
        try:
            with open(TRADES_FILE, 'r') as f:
                trades = json.load(f)
        except:
            trades = []
        
        # Add new trade
        trades.append(trade_data)
        
        # Save back
        with open(TRADES_FILE, 'w') as f:
            json.dump(trades, f)
        
        return True
    except:
        return False

def update_trade_locally(trade_id, update_data):
    """Update trade in local JSON file"""
    try:
        with open(TRADES_FILE, 'r') as f:
            trades = json.load(f)
        
        # Find and update trade
        for trade in trades:
            if trade.get('trade_id') == trade_id:
                trade.update(update_data)
                break
        
        # Save back
        with open(TRADES_FILE, 'w') as f:
            json.dump(trades, f)
        
        return True
    except:
        return False

async def main():
    """Main trading loop triggered by new candles"""
    print("🤖 Gemini Trading Bot - 6K Candle Analysis")
    print(f"💰 Demo Capital: ${DEMO_CAPITAL}")
    print("🕐 Triggered by new candle updates")
    
    valid_keys = [key for key in GEMINI_API_KEYS if key]
    print(f"🔑 API Keys loaded: {len(valid_keys)}/6")
    
    if len(valid_keys) == 0:
        print("❌ No API keys found!")
        return
    
    last_candle_time = ""
    
    while True:
        try:
            # Get all 6k candles
            candles = await fetch_latest_candles(6000)
            if not candles:
                print("⏳ Waiting for candle data...")
                await asyncio.sleep(60)
                continue
            
            current_candle_time = candles[-1].get('timestamp', '')
            current_price = float(candles[-1].get('close', 0))
            
            if current_price <= 0:
                await asyncio.sleep(30)
                continue
            
            # Only analyze when new candle arrives
            if current_candle_time != last_candle_time:
                print(f"🕐 New candle detected: {current_candle_time[-8:-3]}")
                
                # Check position management first
                await check_stop_loss_take_profit(current_price)
                
                # Analyze with full 6k dataset
                candles_data = format_candles_for_gemini(candles)
                signal = await get_gemini_signal(candles_data, current_price)
                
                print(f"🧠 Gemini: {signal.get('signal')} | Confidence: {signal.get('confidence')}/10")
                print(f"🤖 AI Analysis: {signal.get('ai_analysis', 'N/A')[:80]}...")
                print(f"📊 Success Prob: {signal.get('probability_success', 'N/A')} | R:R: {signal.get('risk_reward_ratio', 'N/A')}")
                print(f"🔍 Key Factors: {', '.join(signal.get('key_factors', [])[:3])}...")
                
                # Execute trade if signal is strong
                await execute_trade(signal, current_price)
                await print_stats()
                
                last_candle_time = current_candle_time
            
            # Wait for next candle update
            await wait_for_new_candle(current_candle_time)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
