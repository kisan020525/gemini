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
import pandas as pd
import ccxt
from dotenv import load_dotenv
from collections import deque
import pandas as pd

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

# Initialize rate limiter and working keys
pro_rate_limiter = GeminiRateLimiter()
working_pro_keys = []

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
last_trade_time = None  # Track last trade time for 30min rule

# Pro model state tracking
last_pro_analysis = None
last_pro_call = datetime.now(IST) - timedelta(hours=2)  # Force first Pro call

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
    """Get next available API key - Always rotate to new key for each call"""
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
        # Pro model: Always rotate to next key for each call
        
        # Try all 15 keys to find available one
        attempts = 0
        while attempts < 15:
            # Always move to next key
            current_pro_key_index = (current_pro_key_index + 1) % 15
            key_index = current_pro_key_index
            
            # Check daily limit first (50 calls per day per key)
            if api_key_daily_count_pro[key_index] >= 50:
                print(f"⏳ Pro Key #{key_index + 1} daily limit reached ({api_key_daily_count_pro[key_index]}/50)")
                attempts += 1
                continue
            
            # Check if this key is overloaded (5 min cooldown)
            if current_time - api_key_rate_limited[key_index] < 300:  # 5 minutes
                print(f"⏭️ Skipping overloaded Pro Key #{key_index + 1}")
                attempts += 1
                continue
            
            # Use this key
            api_key = GEMINI_API_KEYS[key_index]
            if api_key:
                pro_key_call_count += 1
                api_key_daily_count_pro[key_index] += 1
                print(f"🔑 Using Pro API Key #{key_index + 1} (Batch: {pro_key_call_count}/49, Daily: {api_key_daily_count_pro[key_index]}/50)")
                return api_key
            else:
                print(f"❌ Pro Key #{key_index + 1} is None")
                current_pro_key_index = (current_pro_key_index + 1) % 15
                pro_key_call_count = 0
                attempts += 1
                continue
        
        print("❌ All Pro keys exhausted or unavailable")
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

async def fetch_latest_candles(limit: int = 25000) -> List[Dict]:
    """Fetch latest candles for comprehensive market analysis"""
    try:
        print(f"🔍 DEBUG: Requesting {limit} candles from Supabase...")
        response = supabase.table("candles").select("*").order("timestamp", desc=True).limit(limit).execute()
        candles = list(reversed(response.data))
        print(f"🔍 DEBUG: Actually received {len(candles)} candles from Supabase")
        return candles
    except Exception as e:
        print(f"❌ Error fetching candles: {e}")
        return []

async def aggregate_candles(candles_1min: List[Dict], timeframe: str, limit: int = None) -> List[Dict]:
    """
    Aggregate 1-minute candles to higher timeframes
    
    Args:
        candles_1min: List of 1-min candle dictionaries
        timeframe: '4h', '1h', '15m', '5m'
        limit: Number of aggregated candles to return (most recent)
    
    Returns:
        List of aggregated candles
    """
    if not candles_1min:
        return []
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(candles_1min)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Resample mapping - using new pandas time aliases
        resample_map = {
            '5m': '5min',
            '15m': '15min', 
            '1h': '1h',
            '4h': '4h'
        }
        
        rule = resample_map.get(timeframe, '1h')
        
        # Aggregate OHLCV
        aggregated = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min', 
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Return last N candles if limit specified
        if limit:
            aggregated = aggregated.tail(limit)
        
        # Convert back to list of dicts
        result = aggregated.reset_index()
        result['timestamp'] = result['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return result.to_dict('records')
        
    except Exception as e:
        print(f"❌ Aggregation error: {e}")
        return []

async def get_data_for_pro(candles_1min: List[Dict]) -> Dict:
    """
    Prepare data for Gemini Pro strategic analysis - Supabase aggregation
    
    Returns:
        Dict with 4H, 1H, 15m candles + current price
    """
    try:
        # Get current price
        current_price = float(candles_1min[-1]['close']) if candles_1min else 0
        
        print(f"🔍 DEBUG: Input 1min candles: {len(candles_1min)}")
        if candles_1min:
            print(f"🔍 DEBUG: First candle: {candles_1min[0]['timestamp']}")
            print(f"🔍 DEBUG: Last candle: {candles_1min[-1]['timestamp']}")
        
        # Aggregate to multiple timeframes from Supabase data - NO LIMITS
        candles_4h = await aggregate_candles(candles_1min, '4h')  # Get ALL 4H candles
        candles_1h = await aggregate_candles(candles_1min, '1h')  # Get ALL 1H candles
        candles_15m = await aggregate_candles(candles_1min, '15m', limit=96)  # Keep 15m limit
        
        print(f"📊 Pro Data: {len(candles_4h)} 4H candles, {len(candles_1h)} 1H candles, {len(candles_15m)} 15m candles")
        
        if candles_4h:
            print(f"🔍 DEBUG: First 4H candle: {candles_4h[0]['timestamp']}")
            print(f"🔍 DEBUG: Last 4H candle: {candles_4h[-1]['timestamp']}")
        
        return {
            '4h': candles_4h,
            '1h': candles_1h,
            '15m': candles_15m,
            'current_price': current_price
        }
    except Exception as e:
        print(f"❌ Pro data prep error: {e}")
        return {'4h': [], '1h': [], '15m': [], 'current_price': 0}

async def get_data_for_flash(candles_1min: List[Dict], pro_directive: Dict) -> Dict:
    """
    Prepare data for Gemini Flash tactical execution
    
    Returns:
        Dict with Pro's directive + limited recent data
    """
    try:
        # Get current price
        current_price = float(candles_1min[-1]['close']) if candles_1min else 0
        
        # Flash gets LESS data (last 24 hours max)
        recent_candles = candles_1min[-1440:]  # Last 1440 minutes = 24 hours
        
        # Aggregate recent data
        candles_1h = await aggregate_candles(recent_candles, '1h', limit=24)
        candles_15m = await aggregate_candles(recent_candles, '15m', limit=48)
        candles_1m = recent_candles[-100:]  # Last 100 1-min candles
        
        print(f"⚡ Flash Data: {len(candles_1h)} 1H, {len(candles_15m)} 15m, {len(candles_1m)} 1m candles")
        
        return {
            'pro_directive': pro_directive,
            '1h': candles_1h,
            '15m': candles_15m, 
            '1m': candles_1m,
            'current_price': current_price
        }
    except Exception as e:
        print(f"❌ Flash data prep error: {e}")
        return {
            'pro_directive': pro_directive or {},
            '1h': [], '15m': [], '1m': [],
            'current_price': 0
        }

def format_candles_summary(candles: List[Dict], timeframe: str) -> str:
    """Format candles as concise summary for Pro analysis"""
    
    if not candles:
        return f"No {timeframe} data"
    
    summary = f"{timeframe} Candles (Last {len(candles)}):\n"
    summary += "Time | Open | High | Low | Close | Volume\n"
    
    # Show first 3, middle 3, last 10
    if len(candles) > 20:
        shown = candles[:3] + candles[len(candles)//2-1:len(candles)//2+2] + candles[-10:]
    else:
        shown = candles
    
    for c in shown:
        ts = c['timestamp'][-8:-3] if len(c['timestamp']) > 8 else c['timestamp']
        summary += f"{ts}|{c['open']:.0f}|{c['high']:.0f}|{c['low']:.0f}|{c['close']:.0f}|{c['volume']:.0f}\n"
    
    # Add market context
    all_closes = [float(c['close']) for c in candles]
    current = all_closes[-1]
    high = max(all_closes)
    low = min(all_closes)
    
    summary += f"\n{timeframe} Summary: Current ${current:.0f} | High ${high:.0f} | Low ${low:.0f}\n"
    
    return summary

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
        
        # Wait 60 seconds after Pro analysis completes before allowing next call
        print("⏳ Pro analysis cooldown: 60 seconds...")
        await asyncio.sleep(60)
        print("✅ Pro analysis ready for next call")
        
    except Exception as e:
        print(f"❌ Background Pro analysis error: {e}")
        # Still wait 60s even on error to prevent rapid retries
        await asyncio.sleep(60)

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
                working_keys.append((i+1, api_key))  # Rate limited but working
                print(f"⚠️ Pro Key #{i+1}: Working (rate limited)")
            else:
                print(f"❌ Pro Key #{i+1}: {str(e)[:50]}...")
                
        await asyncio.sleep(2)  # Small delay between tests
    
    print(f"📊 Working Pro Keys: {len(working_keys)}/15")
    return working_keys

async def get_gemini_pro_analysis(pro_data: Dict) -> Dict:
    """
    MODIFIED: Now receives aggregated data, not raw candles
    """
    
    current_price = pro_data['current_price']
    candles_4h = pro_data['4h']
    candles_1h = pro_data['1h']
    candles_15m = pro_data['15m']
    
    try:
        # Get next available Pro API key using new rotation system
        api_key = get_next_api_key("pro")
        if not api_key:
            print("🚫 No Pro API keys available - using Flash only")
            return {"signal": "HOLD", "confidence": 0, "reasoning": "No Pro keys available"}
        
        # Format candles for Pro (show structure, not all data)
        formatted_4h = format_candles_summary(candles_4h, "4H")
        formatted_1h = format_candles_summary(candles_1h[-24:], "1H")  # Last 24 hours
        formatted_15m = format_candles_summary(candles_15m[-48:], "15m")  # Last 12 hours
        
        prompt = f"""
You are GEMINI 2.5 PRO - Strategic Commander for Bitcoin trading.

YOUR ROLE: Provide 1-HOUR strategic guidance for Flash tactical AI.

CURRENT MARKET DATA:
Price: ${current_price:.0f}

4H CANDLES (Last 100 - Strategic Structure):
{formatted_4h}

1H CANDLES (Last 24 - Momentum Context):
{formatted_1h}

15MIN CANDLES (Last 48 - Recent Action):
{formatted_15m}

INSTRUCTIONS:
1. Analyze 4H trend structure (Higher Highs/Higher Lows)
2. Identify major supply/demand zones
3. Determine strategic BIAS: LONG_BIAS, SHORT_BIAS, or HOLD
4. Define entry zones where Flash should look for tactical entries
5. Set invalidation criteria

Respond in JSON:
{{
    "signal": "LONG/SHORT/HOLD",
    "confidence": 1-10,
    "bias": "LONG_BIAS/SHORT_BIAS/NEUTRAL",
    "entry_zones": [{{"min": 94000, "max": 94500}}],
    "stop_loss": price_level,
    "take_profit": price_level,
    "trend_4h": "UPTREND/DOWNTREND/RANGE",
    "instructions_for_flash": "Specific guidance for Flash on what to look for",
    "invalidation": "What would make this plan wrong",
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
                        "bias": {"type": "string"},
                        "entry_zones": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "min": {"type": "number"},
                                    "max": {"type": "number"}
                                }
                            }
                        },
                        "stop_loss": {"type": "number"},
                        "take_profit": {"type": "number"},
                        "trend_4h": {"type": "string"},
                        "instructions_for_flash": {"type": "string"},
                        "invalidation": {"type": "string"},
                        "reasoning": {"type": "string"}
                    }
                }
            }
        )
        
        try:
            # Remove timeout - let Pro take as long as it needs
            response = await asyncio.to_thread(model.generate_content, prompt)
            
        except Exception as api_error:
            print(f"❌ Pro API error: {api_error}")
            return {"signal": "HOLD", "confidence": 0, "reasoning": "Pro API error"}
        
        try:
            pro_analysis = json.loads(response.text.strip())
            pro_analysis['timestamp'] = datetime.now(IST).isoformat()
            pro_analysis['model'] = 'Pro'
            
            print(f"✅ Pro Strategic Analysis Complete")
            return pro_analysis
            
        except json.JSONDecodeError as e:
            print(f"❌ Pro JSON decode error: {e}")
            return {"signal": "HOLD", "confidence": 0, "reasoning": "Pro JSON error"}
            
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            print(f"⏳ Pro quota exceeded - using Flash only")
            return {"signal": "HOLD", "confidence": 0, "reasoning": "Pro quota exceeded"}
        
        print(f"❌ Gemini Pro error: {e}")
        return {"signal": "HOLD", "confidence": 0, "reasoning": f"Pro error: {e}"}

async def get_gemini_flash_signal(flash_data: Dict, pro_analysis: Dict = None) -> Dict:
    """
    MODIFIED: Now receives structured data and Pro analysis
    """
    
    current_price = flash_data['current_price']
    candles_1h = flash_data['1h']
    candles_15m = flash_data['15m']
    candles_1m = flash_data['1m']
    
    try:
        # Get next available Flash API key
        api_key = get_next_api_key("flash")
        if not api_key:
            print("🚫 No Flash API keys available")
            return {"signal": "HOLD", "confidence": 0, "reasoning": "No Flash keys available"}
        
        # Format candles for Flash (recent tactical data)
        formatted_1h = format_candles_summary(candles_1h[-12:], "1H")  # Last 12 hours
        formatted_15m = format_candles_summary(candles_15m[-24:], "15m")  # Last 6 hours
        formatted_1m = format_candles_summary(candles_1m[-60:], "1m")  # Last 1 hour
        
        # Pro guidance context
        pro_context = ""
        if pro_analysis and pro_analysis.get('signal') != 'HOLD':
            pro_context = f"""
STRATEGIC GUIDANCE FROM PRO:
- Bias: {pro_analysis.get('bias', 'NEUTRAL')}
- Entry Zones: {pro_analysis.get('entry_zones', [])}
- Instructions: {pro_analysis.get('instructions_for_flash', 'No specific instructions')}
- Invalidation: {pro_analysis.get('invalidation', 'Not specified')}
"""
        
        prompt = f"""
FLASH - Tactical Bitcoin Trading AI

MARKET: ${current_price:.0f}

{pro_context}

RECENT DATA:
1H: {len(candles_1h)} candles
15m: {len(candles_15m)} candles  
1m: {len(candles_1m)} candles

TASK: Analyze current market for tactical entry opportunity.

RULES:
- Only trade with 8+ confidence
- Follow Pro's strategic bias if provided
- Use precise 1-minute timing

Respond JSON:
{{
    "signal": "LONG/SHORT/HOLD",
    "confidence": 1-10,
    "entry": {current_price},
    "stop_loss": price,
    "take_profit": price,
    "reasoning": "Brief analysis"
}}
"""
        
        # Configure Gemini Flash
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
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
                        "reasoning": {"type": "string"}
                    }
                }
            }
        )
        
        try:
            # Remove timeout - let Flash take as long as it needs
            response = await asyncio.to_thread(model.generate_content, prompt)
            
        except Exception as api_error:
            print(f"❌ Flash API error: {api_error}")
            return {"signal": "HOLD", "confidence": 0, "reasoning": "Flash API error"}
        
        try:
            flash_analysis = json.loads(response.text.strip())
            flash_analysis['timestamp'] = datetime.now(IST).isoformat()
            flash_analysis['model'] = 'Flash'
            
            print(f"✅ Flash Tactical Analysis Complete")
            return flash_analysis
            
        except json.JSONDecodeError as e:
            print(f"❌ Flash JSON decode error: {e}")
            return {"signal": "HOLD", "confidence": 0, "reasoning": "Flash JSON error"}
            
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            print(f"⏳ Flash quota exceeded")
            return {"signal": "HOLD", "confidence": 0, "reasoning": "Flash quota exceeded"}
        
        print(f"❌ Gemini Flash error: {e}")
        return {"signal": "HOLD", "confidence": 0, "reasoning": f"Flash error: {e}"}
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

def get_pro_analysis_for_flash() -> str:
    """Format Pro analysis for Flash model context"""
    if not last_pro_analysis:
        return "No recent Pro analysis available - Flash operating independently."
    
    pro_text = f"""LATEST PRO STRATEGIC ANALYSIS:
Signal: {last_pro_analysis.get('signal', 'HOLD')}
Confidence: {last_pro_analysis.get('confidence', 0)}/10
Trend Direction: {last_pro_analysis.get('trend_direction', 'Unknown')}
Strategic Reasoning: {last_pro_analysis.get('reasoning', 'No reasoning available')[:200]}...
Timestamp: {last_pro_analysis.get('timestamp', 'Unknown')}

Use this strategic context to inform your tactical decisions."""
    
    return pro_text
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
        
    elif action in ['LONG', 'SHORT']:
        # If position exists, close and switch
        if current_position:
            new_direction = 'long' if action == 'LONG' else 'short'
            print(f"🔄 GEMINI SWITCH: Closing {current_position.direction.upper()} @ ${current_price:.0f}")
            await close_position(current_price, "Gemini Switch")
            return await open_new_position(signal, current_price, new_direction)
        else:
            # No position, open new one
            new_direction = 'long' if action == 'LONG' else 'short'
            return await open_new_position(signal, current_price, new_direction)
    
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

async def open_new_position(signal: Dict, current_price: float, direction: str) -> Optional[Trade]:
    """Execute paper trade - STRICTLY ONE AT A TIME"""
    global current_position, demo_balance, total_trades, last_trade_time
    
    # Check 30-minute rule (but allow VERY strong signals)
    if last_trade_time:
        time_since_last = (datetime.now() - last_trade_time).total_seconds() / 60
        if time_since_last < 30:
            # Allow trade if confidence is MAXIMUM (10/10) for very strong moves
            if signal['confidence'] < 10:
                print(f"⏳ Must wait {30 - time_since_last:.1f} more minutes (or confidence 10/10 for strong moves)")
                return None
            else:
                print(f"🚀 STRONG MOVE DETECTED: Confidence 10/10 - bypassing 30min rule")
    
    # Only trade with VERY HIGH confidence (9+ out of 10 = 90%+)
    if signal['signal'] == 'HOLD' or signal['confidence'] < 9:
        return None
    
    # Calculate trade parameters
    entry = float(signal.get('entry', current_price))
    stop_loss = float(signal.get('stop_loss', current_price * 0.98))
    take_profit = float(signal.get('take_profit', current_price * 1.04))
    risk_amount = 50.0  # Fixed $50 risk per trade
    
    # Validate trade setup
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
    last_trade_time = datetime.now()  # Update last trade time for 30min rule
    
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
        "status": "open",
        "pnl": 0.0,
        "exit_price": None,
        "exit_time": None,
        "close_reason": None,
        "capital_before": demo_balance,
        "capital_after": demo_balance,   # Will update when closed
        "total_pnl": demo_balance - 10000.0,  # Running total P&L
        "trade_result": "OPEN",  # Easy status indicator
        "partial_profit_50": 0.0  # Track 50% TP partial profit
    }
    
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

async def check_position_exit(current_price: float):
    """Check if current position should be closed (SL/TP hit)"""
    global current_position
    
    if not current_position or current_position.status != 'open':
        return
    
    should_close = False
    close_reason = ""
    
    # Check stop loss
    if current_position.direction == 'long':
        if current_price <= current_position.stop_loss:
            should_close = True
            close_reason = "Stop Loss Hit"
        elif current_price >= current_position.take_profit:
            should_close = True
            close_reason = "Take Profit Hit"
    else:  # short
        if current_price >= current_position.stop_loss:
            should_close = True
            close_reason = "Stop Loss Hit"
        elif current_price <= current_position.take_profit:
            should_close = True
            close_reason = "Take Profit Hit"
    
    if should_close:
        print(f"🎯 {close_reason} @ ${current_price:.0f}")
        await close_position(current_price, close_reason)

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
    """Main trading loop with Pro + Flash architecture - v2.1"""
    global current_position, demo_balance, total_trades, winning_trades, last_pro_analysis, last_pro_call
    
    print("🤖 Gemini Trading Bot - Strategic Pro + Tactical Flash v2.1")
    print(f"💰 Demo Capital: ${DEMO_CAPITAL}")
    print("🕐 Pro every 1H, Flash every 1min")
    print("⚡ Flash timeout: REMOVED (unlimited)")  # No timeout
    
    valid_keys = [key for key in GEMINI_API_KEYS if key]
    print(f"🔑 API Keys loaded: {len(valid_keys)}/15")
    
    if len(valid_keys) == 0:
        print("❌ No API keys found!")
        return
    
    # Sync position from database on startup
    await sync_position_from_database()
    
    last_candle_time = ""
    
    while True:
        try:
            # Get all 25k candles for comprehensive analysis
            candles_1min = await fetch_latest_candles(25000)
            if not candles_1min:
                print("⏳ Waiting for candle data...")
                await asyncio.sleep(15)
                continue
            
            current_candle_time = candles_1min[-1].get('timestamp', '')
            current_price = float(candles_1min[-1].get('close', 0))
            
            if current_price <= 0:
                await asyncio.sleep(30)
                continue
            
            # Only analyze when new candle arrives
            if current_candle_time != last_candle_time:
                print(f"🕐 New candle: {datetime.now(IST).strftime('%H:%M:%S')} | Price: ${current_price:.0f}")
                
                # Check for automatic position exits (SL/TP)
                await check_position_exit(current_price)
                
                # Position status
                if current_position and current_position.status == 'open':
                    print(f"📊 Open {current_position.direction.upper()} @ ${current_position.entry_price:.0f} | SL: ${current_position.stop_loss:.0f}")
                else:
                    print("📊 No open position - analyzing for new trade")
                
                # PRO ANALYSIS (Every 1 Hour)
                current_time = datetime.now(IST)
                should_run_pro = False
                
                if last_pro_call is None:
                    should_run_pro = True
                    print("🧠 First Pro analysis")
                else:
                    time_since_pro = (current_time - last_pro_call).total_seconds() / 3600  # hours
                    if time_since_pro >= 1.0:  # 1 hour
                        should_run_pro = True
                        print(f"🧠 Pro analysis due (last: {time_since_pro:.1f}h ago)")
                
                if should_run_pro:
                    print("🧠 Running Gemini 2.5 Pro Strategic Analysis...")
                    pro_data = await get_data_for_pro(candles_1min)
                    pro_analysis = await get_gemini_pro_analysis(pro_data)
                    
                    # Update global Pro state
                    last_pro_analysis = pro_analysis
                    last_pro_call = current_time
                    
                    print(f"✅ Pro Strategy: {pro_analysis.get('signal')} | Confidence: {pro_analysis.get('confidence', 0)}/10")
                    print(f"📊 Pro Bias: {pro_analysis.get('bias', 'NEUTRAL')}")
                    print(f"🎯 Entry Zones: {pro_analysis.get('entry_zones', [])}")
                
                # FLASH ANALYSIS (Every 1 Minute)
                print("⚡ Running Gemini 2.5 Flash Tactical Analysis...")
                flash_data = await get_data_for_flash(candles_1min, last_pro_analysis or {})
                flash_analysis = await get_gemini_flash_signal(flash_data, last_pro_analysis)
                
                # Display Flash results
                model_used = "Flash + Pro" if last_pro_analysis else "Flash Only"
                print(f"⚡ {model_used}: {flash_analysis.get('signal')} | Confidence: {flash_analysis.get('confidence', 0)}/10")
                print(f"📊 Flash Reasoning: {flash_analysis.get('reasoning', 'N/A')[:80]}...")
                
                # Execute Flash's tactical decision
                await execute_flash_decision(flash_analysis, current_price)
                await print_stats()
                
                last_candle_time = current_candle_time
            
            # Wait for next candle update
            await wait_for_new_candle(current_candle_time)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await asyncio.sleep(15)

async def execute_flash_decision(flash_analysis: Dict, current_price: float):
    """Execute Flash's tactical trading decision"""
    global current_position
    
    signal = flash_analysis.get('signal', 'HOLD')
    confidence = flash_analysis.get('confidence', 0)
    
    # Only execute high-confidence signals
    if confidence < 8:
        print(f"⏸️ Low confidence ({confidence}/10) - holding")
        return
    
    # Handle different signals
    if signal == 'LONG' and not current_position:
        await open_new_position(flash_analysis, current_price, 'long')
    elif signal == 'SHORT' and not current_position:
        await open_new_position(flash_analysis, current_price, 'short')
    elif signal == 'HOLD':
        print("⏸️ Flash decision: HOLD")
    else:
        print(f"⏸️ Flash signal {signal} ignored (position exists or invalid)")

if __name__ == "__main__":
    asyncio.run(main())
