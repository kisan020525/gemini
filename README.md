# Gemini Trading Bot - Strategic Pro + Tactical Flash

Advanced two-AI Bitcoin trading system with strategic and tactical analysis layers.

## Architecture

- **Gemini 2.5 Pro**: Strategic analysis every 1 hour (4H/1H/15m timeframes)
- **Gemini 2.5 Flash**: Tactical execution every 1 minute (1H/15m/1m timeframes)
- **Historical Context**: 25,000 candles (17+ days) for comprehensive analysis
- **Multi-timeframe Aggregation**: Pandas-based OHLCV resampling

## Railway Deployment

1. **Fork/Clone this repository**
2. **Set Environment Variables in Railway:**
   - Copy from `.env.example`
   - Add your Gemini API keys (15 keys)
   - Add Supabase URLs and keys
3. **Deploy**: Railway will auto-deploy from GitHub

## Environment Variables Required

```
GEMINI_API_KEY_1 through GEMINI_API_KEY_15
SUPABASE_URL
SUPABASE_KEY  
TRADES_SUPABASE_URL
TRADES_SUPABASE_KEY
```

## Features

- ✅ Two-AI coordinated trading system
- ✅ Strategic Pro + Tactical Flash architecture  
- ✅ Multi-timeframe analysis (4H to 1min)
- ✅ Automatic position management
- ✅ Historical data integration
- ✅ IST timezone tracking
- ✅ Pandas-based aggregation

## Local Development

```bash
pip install -r requirements.txt
python3 main.py
```
