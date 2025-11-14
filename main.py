"""
Gemini 2.5 Flash Trading Bot - Multi-API Key Version
13 API keys rotation for 3250 RPD (250 x 13) = 48+ hours continuous operation
"""

import os
import asyncio
import json
import time
import aiohttp
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import google.generativeai as genai
from supabase import create_client
from dotenv import load_dotenv
from collections import deque

load_dotenv()

class GeminiRateLimiter:
    def __init__(self, rpm_limit=2, rpd_limit=50, min_interval=30):
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit  
        self.min_interval = min_interval
        self.request_times = deque()
        self.daily_count = 0
        self.last_reset = datetime.now().date()
        
    async def wait_for_slot(self):
        now = datetime.now()
        
        # Reset daily counter at midnight
        if now.date() > self.last_reset:
            self.daily_count = 0
            self.last_reset = now.date()
            
        # Check daily limit
        if self.daily_count >= self.rpd_limit:
            print(f"⏳ Daily limit reached ({self.daily_count}/{self.rpd_limit})")
            return False
            
        # Clean old requests (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()
            
        # Wait for RPM slot
        if len(self.request_times) >= self.rpm_limit:
            wait_time = 60 - (now - self.request_times[0]).total_seconds()
            if wait_time > 0:
                print(f"⏳ Rate limit window full. Waiting {wait_time:.1f}s for slot...")
                await asyncio.sleep(wait_time)
                
        # Wait for minimum interval
        if self.request_times:
            time_since_last = (now - self.request_times[-1]).total_seconds()
            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                print(f"⏳ Waiting {wait_time:.1f}s for minimum interval...")
                await asyncio.sleep(wait_time)
                
        # Record this request
        self.request_times.append(datetime.now())
        self.daily_count += 1
        return True

async def validate_pro_keys():
    """Test all Pro API keys to find working ones"""
    working_keys = []
    
    for i, api_key in enumerate(GEMINI_API_KEYS):
        if not api_key:
            continue
            
        try:
            print(f"🔍 Testing Pro Key #{i+1}...")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-pro')
            
            response = model.generate_content("Say 'OK'")
            if response.text.strip():
                working_keys.append((i+1, api_key))
                print(f"✅ Pro Key #{i+1}: Working")
            else:
                print(f"❌ Pro Key #{i+1}: Empty response")
                
        except Exception as e:
            if "429" in str(e):
                print(f"⚠️ Pro Key #{i+1}: Rate limited - excluded from Pro model")
            else:
                print(f"❌ Pro Key #{i+1}: {str(e)[:50]}...")
                
        await asyncio.sleep(2)  # Small delay between tests
    
    print(f"📊 Working Pro Keys: {len(working_keys)}/15")
    return working_keys

# Initialize rate limiter
pro_rate_limiter = GeminiRateLimiter()
working_pro_keys = []
blocked_pro_keys = set()  # Track keys that hit rate limits

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")  # For candles (read-only)
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Separate Supabase for trades
TRADES_SUPABASE_URL = os.getenv("TRADES_SUPABASE_URL")
TRADES_SUPABASE_KEY = os.getenv("TRADES_SUPABASE_KEY")

# 15 Gemini API Keys for dual model system
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"), 
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
    os.getenv("GEMINI_API_KEY_6"),
    os.getenv("GEMINI_API_KEY_7"),
    os.getenv("GEMINI_API_KEY_8"),
    os.getenv("GEMINI_API_KEY_9"),
    os.getenv("GEMINI_API_KEY_10"),
    os.getenv("GEMINI_API_KEY_11"),
    os.getenv("GEMINI_API_KEY_12"),
    os.getenv("GEMINI_API_KEY_13"),
    os.getenv("GEMINI_API_KEY_14"),
    os.getenv("GEMINI_API_KEY_15")
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

# API Key rotation for dual model system (15 keys)
current_api_key_index = 0
current_pro_key_index = 0  # Separate index for Pro model batching
pro_key_call_count = 0     # Count calls for current Pro key
api_key_last_used = [0] * 15   # Track last usage time for 15 keys
api_key_daily_count_pro = [0] * 15  # Track Pro model usage (50 RPD each)
api_key_daily_count_flash = [0] * 15  # Track Flash model usage (250 RPD each)
api_key_daily_reset = [0] * 15  # Track daily reset time for 15 keys
api_key_usage_count = [0] * 15  # Legacy compatibility for 15 keys
api_key_rate_limited = [0] * 15  # Track keys that hit rate limits

# Shared memory for dual AI system
pro_analysis_memory = []  # Store Pro's complete analysis
flash_analysis_memory = []  # Store Flash's analysis
last_pro_analysis = None  # Latest Pro analysis for Flash to use
analysis_counter = 0  # Track analysis cycles

# Lite model rotation
current_lite_key_index = 0
lite_key_last_used = [0] * 2
lite_key_daily_count = [0] * 2
lite_key_daily_reset = [0] * 2

# Trading state - RESET TO START FRESH
current_position = None
demo_balance = 10000.0  # Starting capital - RESET
total_trades = 0        # Reset trade counter to 0 - START FROM TRADE #1
analysis_memory = []    # Clear analysis memory for fresh start
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

def get_next_api_key(model_type="flash") -> str:
    """Get next available API key - Pro uses batching, Flash rotates normally"""
    global current_api_key_index, current_pro_key_index, pro_key_call_count
    global api_key_last_used, api_key_daily_count_pro, api_key_daily_count_flash, api_key_daily_reset
    
    current_time = time.time()
    
    # Reset daily counters at midnight (every 24 hours)
    for i in range(15):
        if current_time - api_key_daily_reset[i] > 86400:  # 24 hours
            api_key_daily_count_pro[i] = 0
            api_key_daily_count_flash[i] = 0
            api_key_daily_reset[i] = current_time
            print(f"🔄 Reset daily counter for API Key #{i + 1}")
    
    if model_type == "pro":
        # Pro model: Use batching (49 calls per key) with overload checking
        
        # Check if current Pro key needs rotation (49 calls used)
        if pro_key_call_count >= 49:
            current_pro_key_index = (current_pro_key_index + 1) % 15
            pro_key_call_count = 0
            print(f"🔄 Pro key rotation: Switching to Key #{current_pro_key_index + 1}")
        
        # Skip overloaded keys (check up to 15 keys)
        attempts = 0
        while attempts < 15:
            key_index = current_pro_key_index
            
            # Check if this key is overloaded (5 min cooldown)
            if current_time - api_key_rate_limited[key_index] < 300:  # 5 minutes
                print(f"⏭️ Skipping overloaded Pro Key #{key_index + 1}")
                current_pro_key_index = (current_pro_key_index + 1) % 15
                pro_key_call_count = 0
                attempts += 1
                continue
            
            # Use this key
            api_key = GEMINI_API_KEYS[key_index]
            if api_key:
                pro_key_call_count += 1
                api_key_daily_count_pro[key_index] += 1
                print(f"🔑 Using Pro API Key #{key_index + 1} (Batch: {pro_key_call_count}/49, Daily: {api_key_daily_count_pro[key_index]}/50)")
                return api_key
            
            attempts += 1
        
        print("❌ All Pro keys overloaded or unavailable")
        return None
            
    else:
        # Flash model: Normal rotation (unchanged)
        daily_limit = 250
        daily_count = api_key_daily_count_flash
        min_interval = 60
        
        # Try all 15 keys for Flash rotation
        for attempt in range(15):
            key_index = (current_api_key_index + attempt) % 15
            
            # Check daily limit
            if daily_count[key_index] >= daily_limit:
                continue
                
            # Check minimum time between calls
            time_since_last = current_time - api_key_last_used[key_index]
            if time_since_last < min_interval:
                continue
            
            # This key is available
            current_api_key_index = (key_index + 1) % 15
            api_key_last_used[key_index] = current_time
            daily_count[key_index] += 1
            
            api_key = GEMINI_API_KEYS[key_index]
            if api_key:
                print(f"🔑 Using Flash API Key #{key_index + 1} (Daily: {daily_count[key_index]}/{daily_limit})")
                return api_key
        
        # No Flash keys available
        print("⏳ All Flash API keys cooling down, waiting...")
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

async def handle_pro_analysis_completion(task_result):
    """Handle Pro analysis completion in background"""
    try:
        pro_signal = await task_result
        print(f"🎯 Pro Strategy: {pro_signal.get('signal')} | Confidence: {pro_signal.get('confidence')}/10")
        print(f"📊 Pro Direction: {pro_signal.get('trend_direction', 'Unknown')}")
    except Exception as e:
        print(f"❌ Background Pro analysis error: {e}")

async def get_gemini_pro_analysis(candles_data: str, current_price: float, retry_count: int = 0) -> Dict:
    """Get strategic analysis from Gemini 2.5 Pro with dynamic key filtering"""
    global pro_analysis_memory, last_pro_analysis, working_pro_keys, blocked_pro_keys
    
    # Prevent infinite retries - max 3 attempts
    if retry_count >= 3:
        print("❌ All Pro keys exhausted - using Flash only")
        return last_pro_analysis or {"signal": "HOLD", "confidence": 0, "reasoning": "All Pro keys exhausted"}
    
    try:
        # Check rate limiter first
        if not await pro_rate_limiter.wait_for_slot():
            print("🚫 Pro daily limit reached - using Flash only")
            return last_pro_analysis or {"signal": "HOLD", "confidence": 0, "reasoning": "Pro daily limit reached"}
        
        # Filter out blocked keys from working keys
        available_keys = [(idx, key) for idx, key in working_pro_keys if idx not in blocked_pro_keys]
        
        if not available_keys:
            print("🚫 No available Pro keys (all blocked) - using Flash only")
            return last_pro_analysis or {"signal": "HOLD", "confidence": 0, "reasoning": "All Pro keys blocked"}
        
        # Use next available key (rotate through them)
        key_index, api_key = available_keys[pro_rate_limiter.daily_count % len(available_keys)]
        
        prompt = f"""
        You are GEMINI 2.5 PRO - Strategic Master AI for Bitcoin Trading.
        
        Current Bitcoin Price: ${current_price}
        Market Data: {candles_data[-2000:]}
        
        Provide strategic analysis in JSON format:
        {{
            "signal": "LONG/SHORT/HOLD",
            "confidence": 1-10,
            "entry": {current_price},
            "stop_loss": price_level,
            "take_profit": price_level,
            "trend_direction": "Overall trend",
            "reasoning": "Strategic analysis"
        }}
        """
        
        # Configure Gemini Pro with Python SDK
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            'gemini-2.5-pro',
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "signal": {"type": "string"},
                        "confidence": {"type": "integer"},
                        "entry": {"type": "number"},
                        "stop_loss": {"type": "number"},
                        "take_profit": {"type": "number"},
                        "trend_direction": {"type": "string"},
                        "reasoning": {"type": "string"}
                    }
                }
            }
        )
        
        print(f"📤 Pro Request #{pro_rate_limiter.daily_count}/{pro_rate_limiter.rpd_limit} (Key #{key_index})")
        
        try:
            # Add timeout for Pro model request
            response = await asyncio.wait_for(
                asyncio.to_thread(model.generate_content, prompt),
                timeout=30.0  # 30 second timeout
            )
            print(f"📥 Pro Response received from Key #{key_index}")
            
        except asyncio.TimeoutError:
            print(f"⏰ Pro Request timeout (30s) - Key #{key_index}")
            # Block this key and try next one
            blocked_pro_keys.add(key_index)
            return await get_gemini_pro_analysis(candles_data, current_price, retry_count + 1)
        except Exception as api_error:
            print(f"❌ Pro API error: {api_error}")
            raise api_error
        
        try:
            pro_analysis = json.loads(response.text.strip())
            pro_analysis['timestamp'] = datetime.now(IST).isoformat()
            pro_analysis['model'] = 'Pro'
            
            # Store in memory
            pro_analysis_memory.append(pro_analysis)
            if len(pro_analysis_memory) > 10:
                pro_analysis_memory.pop(0)
            
            last_pro_analysis = pro_analysis
            print(f"✅ Pro Success! Key #{key_index} ({pro_rate_limiter.daily_count}/{pro_rate_limiter.rpd_limit} today)")
            return pro_analysis
            
        except json.JSONDecodeError as e:
            print(f"❌ Pro JSON decode error: {e}")
            return last_pro_analysis or {"signal": "HOLD", "confidence": 0, "reasoning": "Pro JSON error"}
            
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            # Block this key and try next one
            blocked_pro_keys.add(key_index)
            print(f"🚫 Pro Key #{key_index} rate limited - blocked from Pro model")
            print(f"📊 Available Pro keys: {len(available_keys)-1}/{len(working_pro_keys)}")
            
            # Try next available key
            return await get_gemini_pro_analysis(candles_data, current_price, retry_count + 1)
        
        print(f"❌ Gemini Pro error: {e}")
        return last_pro_analysis or {"signal": "HOLD", "confidence": 0, "reasoning": f"Pro error: {e}"}

async def get_gemini_flash_signal(candles_data: str, current_price: float) -> Dict:
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
        You are an advanced AI TRADER with COMPLETE CONTROL over all trading decisions.

        FULL TRADING CONTROL:
        - You control EVERYTHING: when to open, close, hold, or switch positions
        - Analyze current position (if any) and decide the optimal action
        - You can close profitable trades early or cut losses before stop loss
        - You can switch from LONG to SHORT instantly if market reverses
        - Use your FULL AI intelligence for all trading decisions

        DECISION OPTIONS:
        - HOLD: Keep current position if still valid
        - CLOSE: Close current position at market price
        - OPEN_LONG: Open new long position (only if no current position)
        - OPEN_SHORT: Open new short position (only if no current position)  
        - CLOSE_AND_LONG: Close current position AND immediately open long
        - CLOSE_AND_SHORT: Close current position AND immediately open short

        TRADING FREEDOM:
        - ONLY trade when 90%+ confident (confidence: 9-10)
        - You have COMPLETE FREEDOM to choose trade duration and management
        - Quality over quantity - Better to miss trades than take bad ones
        - Target profits based on market potential ($50-500+ per trade)
        - Use FLEXIBLE risk/reward ratios based on MARKET STRUCTURE ONLY
        - Let the MARKET decide your actions, not rigid rules

        RISK MANAGEMENT STRATEGY:
        - Consider partial profit taking in your decisions
        - Exit early if you see danger ahead
        - Hold longer if momentum continues
        - Switch positions if market structure changes
        - Use your intelligence, not fixed stop losses

        ANALYSIS APPROACH:
        - Study the COMPLETE candle structure (all 6,000 candles)
        - Analyze VOLUME and TRADE COUNT data for confirmation:
          * High volume + price move = Strong momentum
          * Low volume + price move = Weak momentum (likely reversal)
          * Volume spikes = Institutional activity or breakouts
          * Trade count = Market participation and interest
        - Identify if market is in: trending, ranging, breakout, or reversal phase
        - Choose trade type that FITS the current market behavior:
          * Strong trends = Swing trades with bigger targets
          * Tight ranges = Scalping trades with quick profits
          * Breakouts = Position trades with extended targets
          * Reversals = Counter-trend trades with logical exits
        - Analyze multiple timeframes within the 6,000 candle dataset
        - Find the BEST opportunity the market is offering right now

        PATTERN RECOGNITION & CONDITIONAL LOGIC:
        You do NOT predict the future. Instead, you detect repeating patterns and conditional behaviors in the market.
        When specific conditions appear (such as candle formations, volume spikes, liquidity sweeps, trend structure shifts, fake breakouts, or momentum changes), you compare them to thousands of similar historical patterns from your internal knowledge base.
        You identify which side the market usually moves toward when these patterns occur, and you estimate probabilities for upward, downward, and sideways movement.
        Base your decisions on conditional logic (IF X → THEN Y).
        Always select the highest-probability outcome based on market structure, pattern recognition, and statistical behavior.
        This approach improves accuracy, removes randomness, and aligns your trading decisions with how professional algorithmic systems operate.

        IMPORTANT: 
        - Set confidence to 9+ ONLY when you see CRYSTAL CLEAR signals
        - Choose trade duration based on MARKET STRUCTURE, not fixed rules
        - Sometimes scalp for $50, sometimes swing for $300+
        - Place stops and targets where they make TECHNICAL SENSE
        - If you're not 90%+ sure, choose HOLD - there's always another opportunity
        - Use your FULL ANALYTICAL POWER on all 6,000 candles

        Current Bitcoin price: ${current_price:.0f}

        CURRENT POSITION STATUS:
        {get_current_position_for_gemini()}

        {get_analysis_memory_for_gemini()}

        {await get_past_trades_for_gemini()}

        CURRENT MARKET DATA:
        {candles_data}

        ANALYSIS DIRECTIVE:
        Learn from your past trades! Use your FULL AI knowledge base and PAST PERFORMANCE to make better decisions.

        Apply every relevant concept:
        - Mathematical models (statistics, probability, regression, neural patterns)
        - Technical analysis (indicators, patterns, oscillators)
        - VOLUME ANALYSIS (volume spikes, volume trends, volume confirmation)
        - TRADE COUNT analysis (market participation, institutional activity)
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
        2. DEEPLY ANALYZE ALL 6,000 candle data points simultaneously
        3. Identify market phase: trending, ranging, breakout, reversal
        4. Determine optimal trade duration based on market structure
        5. Select the best-fit analytical concepts for current conditions
        6. Calculate probabilities for scalping vs swing opportunities
        7. Choose trade type that maximizes profit potential
        8. Plan PARTIAL PROFIT strategy: set TP where 50% close makes sense
        9. Place stops and targets at TECHNICAL LEVELS based on full analysis
        10. Only trade when you have HIGH CONFIDENCE (9+)
        11. Let the market structure decide everything: duration, targets, stops

        MARKET STRUCTURE ANALYSIS GUIDE:
        - Examine the complete 6,000 candle dataset for context
        - Identify major support/resistance levels across all timeframes
        - ANALYZE VOLUME PATTERNS:
          * Volume confirmation on breakouts (high volume = valid breakout)
          * Volume divergence (price up, volume down = weakness)
          * Volume spikes at key levels (institutional activity)
          * Trade count patterns (high trades = strong interest)
        - Determine current market phase and momentum
        - Find the highest probability opportunity available
        - Choose scalping if market is choppy/ranging
        - Choose swing trading if market is trending strongly
        - Adapt your strategy to what the market is offering
        - Don't force a trading style - let the market guide you

        IMPORTANT: You can only have ONE TRADE AT A TIME. If you're not very confident (8+), choose HOLD.

        JSON RESPONSE:
        {{
            "thinking": "Step-by-step reasoning: 1) Current position analysis... 2) Market analysis... 3) Decision rationale...",
            "position_analysis": "Analysis of current position (if any)",
            "market_analysis": "Current market assessment and key concepts used", 
            "action": "HOLD/CLOSE/OPEN_LONG/OPEN_SHORT/CLOSE_AND_LONG/CLOSE_AND_SHORT",
            "confidence": 1-10,
            "entry": {current_price},
            "stop_loss": price,
            "take_profit": price,
            "risk_reward_ratio": "1:X",
            "reasoning": "Why this action was chosen and how it optimizes the trading strategy"
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
            
            # Store this analysis in memory for future context
            add_to_analysis_memory(
                current_price, 
                result['signal'], 
                result['confidence'], 
                result['reasoning']
            )
            
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

def add_to_analysis_memory(price: float, signal: str, confidence: int, reasoning: str):
    """Store analysis decision in memory (last 30)"""
    global analysis_memory
    
    memory_entry = {
        "timestamp": datetime.now(IST).strftime("%H:%M:%S"),
        "price": price,
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning[:100]  # Truncate for memory efficiency
    }
    
    analysis_memory.append(memory_entry)
    
    # Keep only last 30 analyses
    if len(analysis_memory) > 30:
        analysis_memory = analysis_memory[-30:]

def get_current_position_for_gemini() -> str:
    """Format current position info for Gemini analysis"""
    if not current_position or current_position.status != 'open':
        return "No current position - You can open new trades"
    
    # Calculate unrealized P&L
    if current_position.direction == 'long':
        unrealized_pnl = (demo_balance - current_position.entry_price) * current_position.position_size
    else:
        unrealized_pnl = (current_position.entry_price - demo_balance) * current_position.position_size
    
    position_text = f"""CURRENT POSITION:
Direction: {current_position.direction.upper()}
Entry Price: ${current_position.entry_price:.0f}
Stop Loss: ${current_position.stop_loss:.0f}
Take Profit: ${current_position.take_profit:.0f}
Position Size: {current_position.position_size:.4f} BTC
Unrealized P&L: ${unrealized_pnl:.2f}
Status: OPEN - You can HOLD, CLOSE, or SWITCH positions"""
    
    return position_text

def get_analysis_memory_for_gemini() -> str:
    """Format recent analysis memory for Gemini context"""
    if not analysis_memory:
        return "No previous analysis memory available."
    
    memory_text = "RECENT ANALYSIS MEMORY (Your last decisions):\n"
    for entry in analysis_memory[-10:]:  # Show last 10 for context
        memory_text += f"{entry['timestamp']}: ${entry['price']} -> {entry['signal']} (Conf: {entry['confidence']}/10)\n"
    
    return memory_text
    """Calculate position size for risk management"""
    risk_per_unit = abs(entry - stop_loss)
    if risk_per_unit > 0:
        position_size = risk_amount / risk_per_unit
        # Limit position size to reasonable amounts (max $50k notional)
        max_notional = 50000
        max_position = max_notional / entry
        return min(position_size, max_position)
    return 0

def calculate_position_size(entry: float, stop_loss: float, risk_amount: float) -> float:
    """Calculate position size for risk management"""
    risk_per_unit = abs(entry - stop_loss)
    if risk_per_unit > 0:
        position_size = risk_amount / risk_per_unit
        # Limit position size to reasonable amounts (max $50k notional)
        max_notional = 50000
        max_position = max_notional / entry
        return min(position_size, max_position)
    return 0.0

async def execute_gemini_action(signal: Dict, current_price: float) -> Optional[Trade]:
    """Execute Gemini's trading decision with full control"""
    global current_position, demo_balance, total_trades
    
    action = signal.get('signal', 'HOLD')  # Fixed: use 'signal' not 'action'
    confidence = signal.get('confidence', 0)
    
    # Only execute high confidence actions
    if confidence < 9 and action not in ['HOLD', 'CLOSE']:
        print(f"⏳ WAITING for 90%+ confidence action (Current: {confidence}/10)")
        return None
    
    # Execute Gemini's decision
    if action == 'HOLD':
        if current_position:
            print(f"⏸️ HOLDING position: {current_position.direction.upper()} @ ${current_position.entry_price:.0f}")
        else:
            print(f"⏸️ HOLDING: No position, waiting for opportunity")
        return None
        
    elif action == 'CLOSE':
        if current_position:
            print(f"🚨 GEMINI CLOSE: Closing {current_position.direction.upper()} @ ${current_price:.0f}")
            await close_position(current_price, "Gemini Close")
        else:
            print(f"⚠️ No position to close")
        return None
        
    elif action in ['CLOSE_AND_LONG', 'CLOSE_AND_SHORT']:
        # Close current position first
        if current_position:
            print(f"🔄 GEMINI SWITCH: Closing {current_position.direction.upper()} @ ${current_price:.0f}")
            await close_position(current_price, "Gemini Switch")
        
        # Open new position
        new_direction = 'long' if action == 'CLOSE_AND_LONG' else 'short'
        return await open_new_position(signal, current_price, new_direction)
        
    elif action in ['OPEN_LONG', 'OPEN_SHORT']:
        if current_position:
            print(f"🚫 Cannot open new position: {current_position.direction.upper()} already open")
            return None
        
        new_direction = 'long' if action == 'OPEN_LONG' else 'short'
        return await open_new_position(signal, current_price, new_direction)
    
    return None

async def open_new_position(signal: Dict, current_price: float, direction: str) -> Optional[Trade]:
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
    
    # Fix direction logic for all signal types
    signal_type = signal.get('signal', 'HOLD')
    if signal_type in ['BUY', 'OPEN_LONG', 'CLOSE_AND_LONG']:
        direction = 'long'
    elif signal_type in ['SELL', 'OPEN_SHORT', 'CLOSE_AND_SHORT']:
        direction = 'short'
    else:
        direction = direction  # Use the passed direction parameter
    
    print(f"🔍 DEBUG: Signal={signal_type} | Direction={direction} | Entry=${entry} | SL=${stop_loss} | TP=${take_profit}")
    
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
        "trade_result": "OPEN",  # Easy status indicator
        "partial_profit_50": 0.0  # Track 50% TP partial profit
    }
    
async def save_trade_to_supabase(trade_data, max_retries=3):
    """Save trade to Supabase with retry logic"""
    for attempt in range(max_retries):
        try:
            if trades_supabase:
                print(f"💾 Saving trade to Supabase (attempt {attempt + 1}/{max_retries})")
                result = trades_supabase.table("paper_trades").insert(trade_data).execute()
                print(f"✅ Trade saved to database successfully")
                print(f"📊 DB Response: {len(result.data)} record(s) inserted")
                return True
            else:
                print(f"⚠️ NO TRADES DATABASE CONNECTION")
                return False
        except Exception as e:
            print(f"❌ TRADES DB ERROR (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)  # Wait 2 seconds before retry
            else:
                print(f"❌ Failed to save trade after {max_retries} attempts")
                return False
    return False

    # Save trade to database with retry logic
    print(f"🚀 TRADE #{total_trades}: {direction.upper()} @ ${entry:.0f} | SL: ${stop_loss:.0f} | TP: ${take_profit:.0f}")
    await save_trade_to_supabase(trade_data)
    
    return trade

async def close_partial_position(exit_price: float, reason: str):
    """Close 50% of position for partial profit taking"""
    global current_position, demo_balance, winning_trades
    
    if not current_position or current_position.status != 'open':
        return
    
    # Calculate 50% PnL
    if current_position.direction == 'long':
        partial_pnl = (exit_price - current_position.entry_price) * (current_position.position_size * 0.5)
    else:  # short
        partial_pnl = (current_position.entry_price - exit_price) * (current_position.position_size * 0.5)
    
    # Update balance with partial profit
    demo_balance += partial_pnl
    
    # Reduce position size by 50%
    current_position.position_size *= 0.5
    
    # Mark that partial profit was taken
    current_position.partial_taken = True
    
    # Update database with partial profit
    if trades_supabase:
        try:
            trades_supabase.table("paper_trades").update({
                "partial_profit_50": partial_pnl,
                "capital_after": demo_balance
            }).eq("trade_id", total_trades).execute()
        except Exception as e:
            print(f"❌ Partial profit DB error: {e}")
    
    print(f"🎯 PARTIAL PROFIT: 50% closed at ${exit_price:.0f} | Profit: ${partial_pnl:.2f} | Remaining: 50%")
    print(f"💰 Updated Capital: ${demo_balance:.2f}")

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
    partial_close = False
    close_price = current_price
    reason = "Hold"
    
    # Calculate 50% TP level for partial profit taking
    entry = current_position.entry_price
    tp = current_position.take_profit
    halfway_tp = entry + (tp - entry) * 0.5  # 50% of the way to TP
    
    if current_position.direction == 'long':
        # Check for partial profit at 50% TP
        if candle_high >= halfway_tp and not hasattr(current_position, 'partial_taken'):
            partial_close = True
            close_price = halfway_tp
            reason = "Partial Profit 50%"
        # Check full stop loss and take profit
        elif candle_low <= current_position.stop_loss:
            should_close = True
            close_price = current_position.stop_loss
            reason = "Stop Loss"
        elif candle_high >= current_position.take_profit:
            should_close = True
            close_price = current_position.take_profit
            reason = "Take Profit"
    else:  # short
        # Check for partial profit at 50% TP
        if candle_low <= halfway_tp and not hasattr(current_position, 'partial_taken'):
            partial_close = True
            close_price = halfway_tp
            reason = "Partial Profit 50%"
        # Check full stop loss and take profit
        elif candle_high >= current_position.stop_loss:
            should_close = True
            close_price = current_position.stop_loss
            reason = "Stop Loss"
        elif candle_low <= current_position.take_profit:
            should_close = True
            close_price = current_position.take_profit
            reason = "Take Profit"
    
    if partial_close:
        print(f"🎯 PARTIAL PROFIT: Closing 50% at ${close_price:.0f}")
        await close_partial_position(close_price, reason)
        return
    elif should_close:
        print(f"🚨 NATURAL {reason.upper()}: Closing at ${close_price:.0f}")
        await close_position(close_price, reason)
        return
    
    print(f"✅ Position held: {current_position.direction.upper()} @ ${current_position.entry_price:.0f} | Current: ${current_price:.0f}")

async def print_stats():
    """Print trading statistics"""
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_return = ((demo_balance - DEMO_CAPITAL) / DEMO_CAPITAL * 100)
    
    # Calculate total daily usage for both models
    total_pro_usage = sum(api_key_daily_count_pro)
    total_flash_usage = sum(api_key_daily_count_flash)
    
    print(f"📈 Balance: ${demo_balance:.0f} | Return: {total_return:.1f}% | Trades: {total_trades} | Win: {win_rate:.0f}%")
    print(f"🔑 API Usage: Pro {total_pro_usage}/750 (Key #{current_pro_key_index + 1}: {pro_key_call_count}/49) | Flash {total_flash_usage}/3750")

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
                
                # Position synced successfully - no validation needed
                print(f"✅ Position synced: {current_position.direction.upper()} @ ${current_position.entry_price:.0f} | SL: ${current_position.stop_loss:.0f}")
                print(f"✅ Capital synced: ${demo_balance:.2f}")
                
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
    global current_position, demo_balance, total_trades, winning_trades, analysis_counter, working_pro_keys
    
    print("🤖 Gemini Trading Bot - 6K Candle Analysis")
    print(f"💰 Demo Capital: ${DEMO_CAPITAL}")
    print("🕐 Triggered by new candle updates")
    
    valid_keys = [key for key in GEMINI_API_KEYS if key]
    print(f"🔑 API Keys loaded: {len(valid_keys)}/15")
    
    if len(valid_keys) == 0:
        print("❌ No API keys found!")
        return
    
    # Validate Pro keys on startup
    print("🔍 Validating Pro API keys...")
    working_pro_keys = await validate_pro_keys()
    
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
                
                # Gemini now controls all position management - no automatic TP/SL
                
                # Skip analysis if position was just closed
                if not current_position or current_position.status != 'open':
                    print("📊 No open position - analyzing for new trade")
                    current_position = None  # Ensure it's properly cleared
                else:
                    print(f"📊 Open {current_position.direction.upper()} @ ${current_position.entry_price:.0f} | SL: ${current_position.stop_loss:.0f} | Current: ${current_price:.0f}")
                
                # Dual AI Analysis System
                candles_data = format_candles_for_gemini(candles)
                
                global analysis_counter
                analysis_counter += 1
                
                # Run Pro analysis every 4 minutes (every 4th cycle) - NON-BLOCKING
                if analysis_counter % 4 == 0 and sum(api_key_daily_count_pro) < 50:
                    print("🧠 Running Gemini 2.5 Pro Strategic Analysis...")
                    # Start Pro analysis in background - don't wait for it
                    pro_task = asyncio.create_task(get_gemini_pro_analysis(candles_data, current_price))
                    # Handle completion in background
                    asyncio.create_task(handle_pro_analysis_completion(pro_task))
                
                # Run Flash analysis every minute
                print("⚡ Running Gemini 2.5 Flash Tactical Analysis...")
                signal = await get_gemini_flash_signal(candles_data, current_price)
                
                # Display analysis results
                model_used = "Flash + Pro" if last_pro_analysis else "Flash Only"
                print(f"🧠 {model_used}: {signal.get('signal')} | Confidence: {signal.get('confidence')}/10")
                print(f"💭 AI Thinking: {signal.get('thinking', 'N/A')[:100]}...")
                print(f"🤖 AI Analysis: {signal.get('analysis', 'N/A')[:80]}...")
                print(f"📊 Reasoning: {signal.get('reasoning', 'N/A')[:80]}...")
                
                # Execute Gemini's decision (full control)
                await execute_gemini_action(signal, current_price)
                await print_stats()
                
                last_candle_time = current_candle_time
            
            # Wait for next candle update
            await wait_for_new_candle(current_candle_time)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await asyncio.sleep(15)  # Faster retry for scalping

if __name__ == "__main__":
    asyncio.run(main())
