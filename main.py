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

# 6 Gemini API Keys for analysis (250 RPD each = 1500 total)
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"), 
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
    os.getenv("GEMINI_API_KEY_6")
]

# 2 Gemini Lite API Keys for trade management (1000 RPD each = 2000 total)
GEMINI_LITE_API_KEYS = [
    os.getenv("GEMINI_LITE_API_KEY_1"),
    os.getenv("GEMINI_LITE_API_KEY_2")
]

# Trading Configuration - SCALPING SETUP
DEMO_CAPITAL = 10000.0
RISK_PER_TRADE = 0.005  # 0.5% risk per trade for scalping (was 2%)
ANALYSIS_INTERVAL = 60  # 1 minute analysis intervals

# IST timezone
IST = timezone(timedelta(hours=5, minutes=30))

# Initialize Supabase connections
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)  # For candles
trades_supabase = create_client(TRADES_SUPABASE_URL, TRADES_SUPABASE_KEY) if TRADES_SUPABASE_URL else None

# API Key rotation for both models
current_api_key_index = 0
api_key_last_used = [0] * 6   # Track last usage time
api_key_daily_count = [0] * 6  # Track daily usage
api_key_daily_reset = [0] * 6  # Track daily reset time
api_key_usage_count = [0] * 6  # Legacy compatibility

# Lite model rotation
current_lite_key_index = 0
lite_key_last_used = [0] * 2
lite_key_daily_count = [0] * 2
lite_key_daily_reset = [0] * 2

# Trading state - RESET TO START FRESH
current_position = None
demo_balance = 10000.0  # Starting capital
total_trades = 0        # Reset trade counter
winning_trades = 0      # Reset win counter

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

def get_next_lite_api_key() -> str:
    """Get next available Lite API key for trade management"""
    global current_lite_key_index, lite_key_last_used, lite_key_daily_count, lite_key_daily_reset
    
    current_time = time.time()
    
    # Reset daily counters (every 24 hours)
    for i in range(2):
        if current_time - lite_key_daily_reset[i] > 86400:  # 24 hours
            lite_key_daily_count[i] = 0
            lite_key_daily_reset[i] = current_time
            print(f"🔄 Reset daily counter for Lite API Key #{i + 1}")
    
    # Find available API key (1000 calls per day)
    for attempt in range(2):
        key_index = (current_lite_key_index + attempt) % 2
        
        # Check daily limit (max 1000 per day per key)
        if lite_key_daily_count[key_index] >= 1000:
            continue
            
        # Check minimum time between calls (30 seconds for lite)
        time_since_last = current_time - lite_key_last_used[key_index]
        if time_since_last < 30:  # 30 seconds
            continue
        
        # This key is available
        current_lite_key_index = key_index
        lite_key_last_used[key_index] = current_time
        lite_key_daily_count[key_index] += 1
        
        api_key = GEMINI_LITE_API_KEYS[key_index]
        if api_key:
            print(f"🔑 Using Lite API Key #{key_index + 1} (Daily: {lite_key_daily_count[key_index]}/1000)")
            return api_key
    
    # No keys available
    print("⏳ All Lite API keys cooling down, waiting...")
    return None

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
    """Fetch latest candles for comprehensive market analysis"""
    try:
        response = supabase.table("candles").select("*").order("timestamp", desc=True).limit(limit).execute()
        return list(reversed(response.data))
    except Exception as e:
        print(f"❌ Error fetching candles: {e}")
        return []

def format_candles_for_gemini(candles: List[Dict]) -> str:
    """Format ALL candles for comprehensive analysis - use full 1M token limit"""
    if not candles:
        return "No candle data available"
    
    formatted_data = f"Bitcoin BTCUSDT - COMPLETE DATASET ({len(candles)} minute candles):\n\n"
    
    # Send ALL candles - we have 1M+ token limit!
    formatted_data += "FULL CANDLE DATA (Time|Open|High|Low|Close|Volume):\n"
    
    # Send every single candle
    for candle in candles:
        ts = candle.get('timestamp', '')[-8:-3]
        o = float(candle.get('open', 0))
        h = float(candle.get('high', 0))
        l = float(candle.get('low', 0))
        c = float(candle.get('close', 0))
        v = float(candle.get('volume', 0))
        formatted_data += f"{ts}|{o:.0f}|{h:.0f}|{l:.0f}|{c:.0f}|{v:.1f}\n"
    
    # Add comprehensive market context
    all_closes = [float(c.get('close', 0)) for c in candles if c.get('close')]
    if all_closes:
        current = all_closes[-1]
        high_period = max(all_closes)
        low_period = min(all_closes)
        avg_volume = sum(float(c.get('volume', 0)) for c in candles) / len(candles)
        
        formatted_data += f"\nCOMPREHENSIVE MARKET CONTEXT:\n"
        formatted_data += f"Current Price: ${current:.0f}\n"
        formatted_data += f"Period High: ${high_period:.0f}\n" 
        formatted_data += f"Period Low: ${low_period:.0f}\n"
        formatted_data += f"Average Volume: {avg_volume:.2f}\n"
        formatted_data += f"Total Candles Analyzed: {len(candles)}\n"
        formatted_data += f"Price Range: ${high_period - low_period:.0f}\n"
        formatted_data += f"Current vs High: {((current - high_period) / high_period * 100):.2f}%\n"
        formatted_data += f"Current vs Low: {((current - low_period) / low_period * 100):.2f}%\n"
    
    return formatted_data

async def get_gemini_signal(candles_data: str, current_price: float) -> Dict:
    """Get trading signal with ultra-conservative rate limiting"""
    try:
        # Get next available API key
        api_key = get_next_api_key()
        if not api_key:
            print("🚫 No API keys available, skipping analysis")
            return {"signal": "HOLD", "confidence": 0, "reasoning": "No API keys available"}
        
        # Configure Gemini 2.5 Flash with structured output and thinking mode
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            'gemini-2.5-flash-preview-09-2025',
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "thinking": {"type": "string"},
                        "signal": {"type": "string"},
                        "confidence": {"type": "integer"},
                        "entry": {"type": "number"},
                        "stop_loss": {"type": "number"},
                        "take_profit": {"type": "number"},
                        "analysis": {"type": "string"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["thinking", "signal", "confidence", "entry", "stop_loss", "take_profit", "analysis", "reasoning"]
                }
            },
            system_instruction="Use the 'thinking' field to show your step-by-step reasoning process before making trading decisions. Think through market patterns, past performance, and risk analysis."
        )
        
        # Unleash full AI analytical power with PAST PERFORMANCE
        prompt = f"""
        You are an advanced AI SCALPING TRADER analyzing Bitcoin for EXCEPTIONAL opportunities.

        TRADING DISCIPLINE:
        - ONLY trade when 90%+ confident (confidence: 9-10) 
        - You DON'T need to trade every analysis - WAIT for perfect setups
        - Quality over quantity - Better to miss trades than take bad ones
        - Target meaningful profits ($80-200 per trade)
        - Use optimal risk/reward ratios (minimum 1:2, prefer 1:3)
        - Adjust stop loss and take profit based on your confidence level:
          * Confidence 9/10: Use 1:2 risk/reward (conservative)
          * Confidence 10/10: Use 1:3 risk/reward (aggressive)

        IMPORTANT: 
        - Set confidence to 9+ ONLY when you see CRYSTAL CLEAR signals
        - If you're not 90%+ sure, choose HOLD - there's always another opportunity
        - Don't force trades - patience is profitable

        Current Bitcoin price: ${current_price:.0f}

        {await get_past_trades_for_gemini()}

        CURRENT MARKET DATA:
        {candles_data}

        ANALYSIS DIRECTIVE:
        Learn from your past trades! Use your FULL AI knowledge base and PAST PERFORMANCE to make better decisions.

        Apply every relevant concept:
        - Mathematical models (statistics, probability, regression, neural patterns)
        - Technical analysis (indicators, patterns, oscillators)
        - Market psychology (sentiment, crowd behavior)
        - Your own trading history and mistakes
        - Risk management based on past performance
        - Avoid premature exits; follow your system strictly once a trade is placed

        ANALYSIS IMPROVEMENT DIRECTIVE:
        Use your internal knowledge base to identify which analytical concepts best apply to the current market.
        Evaluate multiple methods (statistical, technical, psychological) and select only the 2–3 most relevant for current conditions.
        Briefly explain why those were selected.
        Estimate probabilities for each possible direction (up, down, neutral) and choose the highest-probability action.

        COMPUTATIONAL APPROACH:
        1. Review your past performance and learn from mistakes
        2. Process ALL 6,000 candle data points simultaneously
        3. Identify patterns across multiple timeframes
        4. Select the best-fit analytical concepts for the current conditions
        5. Calculate probabilities based on historical success
        6. Optimize risk-adjusted returns from experience
        7. Only trade when you have HIGH CONFIDENCE (8+)
        8. Avoid premature exits — TP or SL must decide the outcome

        IMPORTANT: You can only have ONE TRADE AT A TIME. If you're not very confident (8+), choose HOLD.

        JSON RESPONSE:
        {{
            "thinking": "Step-by-step reasoning: 1) Market analysis... 2) Past performance review... 3) Risk assessment... 4) Final decision...",
            "performance_analysis": "What you learned from past trades",
            "market_analysis": "Current market assessment and key concepts used", 
            "signal": "BUY/SELL/HOLD",
            "confidence": 1-10,
            "entry": {current_price},
            "stop_loss": price,
            "take_profit": price,
            "risk_reward_ratio": "1:X",
            "reasoning": "Why this trade was chosen, which concepts were used, and how probability supports it"
        }}

        LEARN FROM YOUR MISTAKES. ONLY TRADE WITH HIGH CONFIDENCE. AVOID PREMATURE EXITS.
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
    if risk_per_unit > 0:
        position_size = risk_amount / risk_per_unit
        # Limit position size to reasonable amounts (max $50k notional)
        max_notional = 50000
        max_position = max_notional / entry
        return min(position_size, max_position)
    return 0

async def execute_trade(signal: Dict, current_price: float) -> Optional[Trade]:
    """Execute paper trade - STRICTLY ONE AT A TIME"""
    global current_position, demo_balance, total_trades
    
    # Only trade with VERY HIGH confidence (9+ out of 10 = 90%+)
    if signal['signal'] == 'HOLD' or signal['confidence'] < 9:
        if current_position and current_position.status == 'open':
            print(f"⏸️ Holding current position: {current_position.direction.upper()} @ ${current_position.entry_price:.0f}")
        else:
            print(f"⏳ WAITING for 90%+ confidence trade (Current: {signal['confidence']}/10)")
        return None
    
    # WAIT FOR CURRENT TRADE TO CLOSE - Only take new trade when 90%+ sure
    if current_position and current_position.status == 'open':
        print(f"🔒 WAITING: Position open - {current_position.direction.upper()} @ ${current_position.entry_price:.0f}")
        print(f"⏳ Will take new trade when current closes AND confidence ≥ 90%")
        return None
    
    # Validate trade parameters with LIVE CAPITAL
    risk_amount = demo_balance * RISK_PER_TRADE  # Use current balance, not fixed amount
    entry = signal.get('entry', current_price)
    stop_loss = signal.get('stop_loss', current_price * 0.98)
    take_profit = signal.get('take_profit', current_price * 1.04)
    direction = 'long' if signal['signal'] == 'BUY' else 'short'
    
    # Validate SL/TP logic
    if direction == 'long' and stop_loss >= entry:
        print(f"⚠️ Invalid LONG setup: SL ${stop_loss} should be < entry ${entry}")
        return None
    if direction == 'short' and stop_loss <= entry:
        print(f"⚠️ Invalid SHORT setup: SL ${stop_loss} should be > entry ${entry}")
        return None
    
    position_size = calculate_position_size(entry, stop_loss, risk_amount)
    
    if position_size <= 0:
        print(f"⚠️ Invalid position size: {position_size}")
        return None
    
    # Log trade setup with current capital
    print(f"📋 Trade Setup: {direction.upper()} ${entry} | SL: ${stop_loss} | TP: ${take_profit} | Size: {position_size:.4f} BTC")
    print(f"💰 Current Capital: ${demo_balance:.2f} | Risk: ${risk_amount:.2f}")
    
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
        "status": "open",
        "capital_before": demo_balance,  # Track capital before trade
        "capital_after": demo_balance,   # Will update when closed
        "total_pnl": demo_balance - 10000.0,  # Running total P&L
        "trade_result": "OPEN"  # Easy status indicator
    }
    
    if trades_supabase:
        try:
            result = trades_supabase.table("paper_trades").insert(trade_data).execute()
            print(f"🚀 TRADE #{total_trades}: {direction.upper()} @ ${entry:.0f} | SL: ${stop_loss:.0f} | TP: ${take_profit:.0f}")
            print(f"✅ Trade saved to database successfully")
        except Exception as e:
            print(f"🚀 TRADE #{total_trades}: {direction.upper()} @ ${entry:.0f} | SL: ${stop_loss:.0f} | TP: ${take_profit:.0f}")
            print(f"❌ TRADES DB ERROR: {str(e)}")
    else:
        print(f"🚀 TRADE #{total_trades}: {direction.upper()} @ ${entry:.0f} | SL: ${stop_loss:.0f} | TP: ${take_profit:.0f}")
        print(f"⚠️ NO TRADES DATABASE CONNECTION")
        print(f"🚀 TRADE #{total_trades}: {direction.upper()} @ ${entry:.0f} | SL: ${stop_loss:.0f} | TP: ${take_profit:.0f}")
        print("⚠️ Using local storage - no trades DB configured")
    
    return trade

async def close_position(exit_price: float, reason: str):
    """Close current position"""
    global current_position, demo_balance, winning_trades
    
    if not current_position or current_position.status != 'open':
        return
    
    # Calculate PnL with validation
    if current_position.direction == 'long':
        pnl = (exit_price - current_position.entry_price) * current_position.position_size
    else:  # short
        pnl = (current_position.entry_price - exit_price) * current_position.position_size
    
    # Validate PnL calculation with safe attribute access
    expected_pnl = getattr(current_position, 'risk_amount', 200.0) if reason == "Stop Loss" else None
    if reason == "Stop Loss" and expected_pnl and abs(abs(pnl) - expected_pnl) > 1:
        print(f"⚠️ PnL mismatch: Expected ~${expected_pnl}, Got ${pnl:.2f}")
    
    current_position.pnl = pnl
    current_position.exit_price = exit_price
    current_position.exit_time = datetime.now(IST)
    current_position.status = 'closed'
    
    demo_balance += pnl
    if pnl > 0:
        winning_trades += 1
    
    # Update separate trades database with LIVE CAPITAL
    if trades_supabase:
        try:
            # Determine trade result for easy viewing
            trade_result = "PROFIT" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"
            
            trades_supabase.table("paper_trades").update({
                "exit_price": exit_price,
                "exit_time": current_position.exit_time.isoformat(),
                "pnl": pnl,
                "status": "closed",
                "close_reason": reason,
                "capital_after": demo_balance,  # Update final capital
                "total_pnl": demo_balance - 10000.0,  # Running total P&L
                "trade_result": trade_result  # Easy profit/loss indicator
            }).eq("trade_id", total_trades).execute()
            
            print(f"📊 CLOSED #{total_trades}: ${exit_price:.0f} | PnL: ${pnl:.2f} | Capital: ${demo_balance:.2f} | {reason}")
        except Exception as e:
            print(f"📊 CLOSED #{total_trades}: ${exit_price:.0f} | PnL: ${pnl:.2f} | Capital: ${demo_balance:.2f} | {reason}")
            print(f"⚠️ Trades DB update error: {str(e)}")
            # Fallback to local storage
            update_data = {
                "exit_price": exit_price,
                "exit_time": current_position.exit_time.isoformat(),
                "pnl": pnl,
                "status": "closed",
                "close_reason": reason,
                "capital_after": demo_balance
            }
            update_trade_locally(total_trades, update_data)
    else:
        # Fallback to local storage
        update_data = {
            "exit_price": exit_price,
            "exit_time": current_position.exit_time.isoformat(),
            "pnl": pnl,
            "status": "closed",
            "close_reason": reason,
            "capital_after": demo_balance
        }
        update_trade_locally(total_trades, update_data)
        print(f"📊 CLOSED #{total_trades}: ${exit_price:.0f} | PnL: ${pnl:.2f} | Capital: ${demo_balance:.2f} | {reason}")
        print("⚠️ Using local storage - no trades DB configured")
    
    current_position = None

async def check_stop_loss_take_profit(current_candle: Dict):
    """Use Gemini 2.5 Lite to make trade management decisions"""
    if not current_position or current_position.status != 'open':
        return
    
    current_price = float(current_candle.get('close', 0))
    candle_high = float(current_candle.get('high', 0))
    candle_low = float(current_candle.get('low', 0))
    
    print(f"🔍 Trade Management Check: {current_position.direction.upper()} @ ${current_position.entry_price:.0f}")
    print(f"📊 Candle: H:${candle_high:.0f} L:${candle_low:.0f} C:${current_price:.0f} | SL:${current_position.stop_loss:.0f} TP:${current_position.take_profit:.0f}")
    
    # Prepare position data for Lite model
    position_data = {
        'direction': current_position.direction,
        'entry_price': current_position.entry_price,
        'stop_loss': current_position.stop_loss,
        'take_profit': current_position.take_profit,
        'capital': demo_balance,
        'candle_high': candle_high,
        'candle_low': candle_low,
        'current_price': current_price
    }
    
    # DISABLED: AI trade management - let trades reach natural TP/SL
    # decision = await get_trade_management_decision(current_price, position_data)
    # print(f"🤖 Lite AI Decision: {decision.get('action')} | Urgency: {decision.get('urgency')}/10")
    # print(f"💭 Reasoning: {decision.get('reasoning', 'N/A')[:60]}...")
    
    # Check only natural TP/SL levels
    decision = {'action': 'HOLD'}
    
    # ALWAYS check natural stop loss and take profit levels
    should_close = False
    close_price = current_price
    reason = "Hold"
    
    if current_position.direction == 'long':
        if candle_low <= current_position.stop_loss:
            should_close = True
            close_price = current_position.stop_loss
            reason = "Stop Loss"
        elif candle_high >= current_position.take_profit:
            should_close = True
            close_price = current_position.take_profit
            reason = "Take Profit"
    else:  # short
        if candle_high >= current_position.stop_loss:
            should_close = True
            close_price = current_position.stop_loss
            reason = "Stop Loss"
        elif candle_low <= current_position.take_profit:
            should_close = True
            close_price = current_position.take_profit
            reason = "Take Profit"
    
    if should_close:
        print(f"🚨 NATURAL {reason.upper()}: Closing at ${close_price:.0f}")
        await close_position(close_price, reason)
        return
    
    print(f"✅ Position held: {current_position.direction.upper()} @ ${current_position.entry_price:.0f} | Current: ${current_price:.0f}")

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

async def get_trade_management_decision(current_price: float, position_data: Dict) -> Dict:
    """Get trade management decision from Gemini 2.5 Flash Lite"""
    try:
        # Get next available Lite API key
        api_key = get_next_lite_api_key()
        if not api_key:
            print("🚫 No trade management API keys available")
            return {"action": "HOLD", "reasoning": "No API keys available"}
        
        # Configure Gemini 2.5 Flash Lite for trade management
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-09-2025')
        
        # Trade management prompt with full context
        prompt = f"""
        You are an AI SCALPING trade manager. Make QUICK closing decisions for small profits.
        
        SCALPING RULES:
        - Close positions quickly when in profit ($20-50 range)
        - Don't hold losing positions long - cut losses fast
        - Focus on momentum changes and immediate price action
        
        Current Bitcoin price: ${current_price:.0f}
        
        OPEN POSITION:
        Direction: {position_data.get('direction', 'UNKNOWN').upper()}
        Entry: ${position_data.get('entry_price', 0):.0f}
        Stop Loss: ${position_data.get('stop_loss', 0):.0f}
        Take Profit: ${position_data.get('take_profit', 0):.0f}
        Current Capital: ${position_data.get('capital', 10000):.0f}
        
        CURRENT CANDLE:
        High: ${position_data.get('candle_high', 0):.0f}
        Low: ${position_data.get('candle_low', 0):.0f}
        Close: ${position_data.get('current_price', 0):.0f}
        
        ANALYSIS NEEDED:
        Should we close this position NOW or continue holding?
        
        Consider:
        - Has SL/TP been hit by candle high/low?
        - Current profit/loss situation
        - Risk management and capital preservation
        - Market momentum and price action
        - Better exit opportunities
        
        DECISION RULES:
        - CLOSE if SL/TP levels hit
        - CLOSE if better exit opportunity
        - HOLD if position is still valid
        
        JSON RESPONSE:
        {{
            "action": "CLOSE/HOLD",
            "reasoning": "Why close or hold",
            "urgency": 1-10
        }}
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        try:
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            
            result = json.loads(response_text)
            return {
                "action": result.get("action", "HOLD"),
                "reasoning": result.get("reasoning", "Trade management analysis"),
                "urgency": int(result.get("urgency", 5))
            }
            
        except:
            return {
                "action": "HOLD",
                "reasoning": "Parse error - holding position",
                "urgency": 1
            }
            
    except Exception as e:
        print(f"❌ Trade management error: {e}")
        return {
            "action": "HOLD",
            "reasoning": f"Error: {str(e)[:30]}",
            "urgency": 1
        }

async def sync_position_from_database():
    """Sync current position from database on startup"""
    global current_position, demo_balance, total_trades, winning_trades
    
    try:
        if trades_supabase:
            # Get latest open trade
            response = trades_supabase.table("paper_trades").select("*").eq("status", "open").order("timestamp", desc=True).limit(1).execute()
            
            if response.data:
                trade_data = response.data[0]
                print(f"🔄 Syncing open position from database: Trade #{trade_data['trade_id']}")
                print(f"📊 DB Data: Entry ${trade_data['entry_price']} | SL ${trade_data['stop_loss']} | TP ${trade_data['take_profit']}")
                
                # Recreate position object with CORRECT database values
                entry_time = datetime.fromisoformat(trade_data['timestamp'].replace('Z', '+00:00'))
                current_position = Trade(
                    entry_price=float(trade_data['entry_price']),
                    stop_loss=float(trade_data['stop_loss']),
                    take_profit=float(trade_data['take_profit']),
                    position_size=float(trade_data['position_size']),
                    direction=trade_data['direction']
                )
                current_position.entry_time = entry_time
                current_position.status = 'open'
                current_position.risk_amount = float(trade_data.get('risk_amount', 200.0))  # Add missing attribute
                
                # Sync counters
                total_trades = int(trade_data['trade_id'])
                demo_balance = float(trade_data['capital_before'])
                
                print(f"✅ Position synced: {current_position.direction.upper()} @ ${current_position.entry_price:.0f} | SL: ${current_position.stop_loss:.0f}")
                print(f"✅ Capital synced: ${demo_balance:.2f}")
                
                # IMMEDIATE SL/TP CHECK after sync
                print("🔍 Checking if position should have been closed...")
                
                # Force close if entry/SL data looks wrong
                if current_position.entry_price > 104000 or current_position.stop_loss < 103000:
                    print("🚨 INVALID POSITION DATA - Force closing position")
                    await close_position(current_position.entry_price, "Invalid data - force close")
                
            else:
                print("✅ No open positions found in database")
                
    except Exception as e:
        print(f"⚠️ Error syncing position: {e}")

async def get_past_trades_for_gemini(limit: int = 30) -> str:
    """Get past trades to show Gemini its performance"""
    try:
        if trades_supabase:
            response = trades_supabase.table("paper_trades").select("*").order("timestamp", desc=True).limit(limit).execute()
            trades = response.data
        else:
            # Fallback to local file
            with open('trades.json', 'r') as f:
                all_trades = json.load(f)
            trades = all_trades[-limit:] if len(all_trades) > limit else all_trades
        
        if not trades:
            return "No previous trades available."
        
        # Calculate performance stats with null checks
        closed_trades = [t for t in trades if t.get('status') == 'closed']
        total_pnl = sum(float(t.get('pnl') or 0) for t in closed_trades)
        wins = len([t for t in closed_trades if float(t.get('pnl') or 0) > 0])
        win_rate = (wins / len(closed_trades) * 100) if closed_trades else 0
        
        performance_summary = f"""
YOUR PAST PERFORMANCE (Last {len(trades)} trades):
Total P&L: ${total_pnl:.2f}
Win Rate: {win_rate:.1f}% ({wins}/{len(closed_trades)} wins)
Capital: ${10000 + total_pnl:.2f}

RECENT TRADES:
"""
        
        for trade in reversed(trades[-10:]):  # Show last 10 trades
            pnl = float(trade.get('pnl') or 0)
            status = trade.get('status', 'unknown')
            direction = trade.get('direction', '').upper()
            entry = float(trade.get('entry_price') or 0)
            exit_price = trade.get('exit_price')
            confidence = trade.get('confidence') or 0
            
            if status == 'closed' and exit_price:
                result = "WIN" if pnl > 0 else "LOSS"
                performance_summary += f"#{trade.get('trade_id')}: {direction} @${entry:.0f} → ${float(exit_price):.0f} = ${pnl:.2f} ({result}) [Conf: {confidence}/10]\n"
            else:
                performance_summary += f"#{trade.get('trade_id')}: {direction} @${entry:.0f} → OPEN [Conf: {confidence}/10]\n"
        
        return performance_summary
        
    except Exception as e:
        print(f"Error getting past trades: {e}")
        return "Unable to retrieve past performance."

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
    
    # Sync position from database on startup
    await sync_position_from_database()
    
    last_candle_time = ""
    
    while True:
        try:
            # Get all 6k candles
            candles = await fetch_latest_candles(6000)
            if not candles:
                print("⏳ Waiting for candle data...")
                await asyncio.sleep(15)  # Check every 15 seconds for scalping
                continue
            
            current_candle_time = candles[-1].get('timestamp', '')
            current_price = float(candles[-1].get('close', 0))
            
            if current_price <= 0:
                await asyncio.sleep(30)
                continue
            
            # Only analyze when new candle arrives
            if current_candle_time != last_candle_time:
                print(f"🕐 New candle detected: {current_candle_time[-8:-3]} | Price: ${current_price:.0f}")
                
                # ALWAYS check position management first with full candle data
                await check_stop_loss_take_profit(candles[-1])  # Pass full candle data
                
                # Skip analysis if position was just closed
                if not current_position or current_position.status != 'open':
                    print("📊 No open position - analyzing for new trade")
                else:
                    print(f"📊 Open {current_position.direction.upper()} @ ${current_position.entry_price:.0f} | SL: ${current_position.stop_loss:.0f} | Current: ${current_price:.0f}")
                
                # Analyze with full 6k dataset
                candles_data = format_candles_for_gemini(candles)
                signal = await get_gemini_signal(candles_data, current_price)
                
                print(f"🧠 Gemini: {signal.get('signal')} | Confidence: {signal.get('confidence')}/10")
                print(f"💭 AI Thinking: {signal.get('thinking', 'N/A')[:100]}...")
                print(f"🤖 AI Analysis: {signal.get('analysis', 'N/A')[:80]}...")
                print(f"📊 Reasoning: {signal.get('reasoning', 'N/A')[:80]}...")
                
                # Execute trade if signal is strong
                await execute_trade(signal, current_price)
                await print_stats()
                
                last_candle_time = current_candle_time
            
            # Wait for next candle update
            await wait_for_new_candle(current_candle_time)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await asyncio.sleep(15)  # Faster retry for scalping

if __name__ == "__main__":
    asyncio.run(main())
