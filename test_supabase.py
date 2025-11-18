#!/usr/bin/env python3
"""
Test script to check if we can get more than 1000 candles from Supabase
"""

import os
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing Supabase credentials")
    exit(1)

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def test_supabase_limits():
    print("🧪 TESTING SUPABASE CANDLE LIMITS")
    print("=" * 50)
    
    # Test 1: Default query (no limit)
    print("\n📊 Test 1: No limit specified")
    try:
        response = supabase.table("candles").select("*").order("timestamp", desc=True).execute()
        print(f"✅ Got {len(response.data)} candles (no limit)")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Limit 1000
    print("\n📊 Test 2: Limit 1000")
    try:
        response = supabase.table("candles").select("*").order("timestamp", desc=True).limit(1000).execute()
        print(f"✅ Got {len(response.data)} candles (limit 1000)")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Limit 5000
    print("\n📊 Test 3: Limit 5000")
    try:
        response = supabase.table("candles").select("*").order("timestamp", desc=True).limit(5000).execute()
        print(f"✅ Got {len(response.data)} candles (limit 5000)")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Limit 10000
    print("\n📊 Test 4: Limit 10000")
    try:
        response = supabase.table("candles").select("*").order("timestamp", desc=True).limit(10000).execute()
        print(f"✅ Got {len(response.data)} candles (limit 10000)")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: Limit 25000
    print("\n📊 Test 5: Limit 25000")
    try:
        response = supabase.table("candles").select("*").order("timestamp", desc=True).limit(25000).execute()
        print(f"✅ Got {len(response.data)} candles (limit 25000)")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 6: Count total rows
    print("\n📊 Test 6: Count total rows in table")
    try:
        response = supabase.table("candles").select("*", count="exact").execute()
        print(f"✅ Total rows in table: {response.count}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 7: Get date range
    print("\n📊 Test 7: Check date range")
    try:
        # Get oldest candle
        oldest = supabase.table("candles").select("timestamp").order("timestamp", desc=False).limit(1).execute()
        # Get newest candle  
        newest = supabase.table("candles").select("timestamp").order("timestamp", desc=True).limit(1).execute()
        
        if oldest.data and newest.data:
            print(f"✅ Date range: {oldest.data[0]['timestamp']} to {newest.data[0]['timestamp']}")
        else:
            print("❌ Could not get date range")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_supabase_limits()
