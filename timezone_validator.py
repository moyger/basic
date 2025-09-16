#!/usr/bin/env python3
"""
Timezone Validation Script
Tests if our current EUR/USD data is properly aligned with London market hours
"""

import pandas as pd
import pytz
from datetime import datetime

def validate_current_data():
    """
    Validate timezone handling in our current EURUSD data
    """
    print("🔍 TIMEZONE VALIDATION ANALYSIS")
    print("=" * 50)
    
    # Load current data
    df = pd.read_csv('data/EURUSD_M15.csv')
    
    print(f"📊 Loaded {len(df):,} records from EURUSD_M15.csv")
    print(f"📅 Date range: {df['Datetime'].iloc[0]} to {df['Datetime'].iloc[-1]}")
    
    # Parse datetime
    df['Datetime_parsed'] = pd.to_datetime(df['Datetime'])
    df['Hour'] = df['Datetime_parsed'].dt.hour
    df['Date'] = df['Datetime_parsed'].dt.date
    df['Month'] = df['Datetime_parsed'].dt.month
    
    print(f"\n🕐 HOURLY DISTRIBUTION:")
    hourly_dist = df['Hour'].value_counts().sort_index()
    for hour, count in hourly_dist.items():
        print(f"   {hour:02d}:XX - {count:,} records")
    
    # Check specific London market hours
    london_open_hours = df[(df['Hour'] >= 7) & (df['Hour'] <= 9)]
    print(f"\n🌅 LONDON OPEN HOURS (7-9 AM):")
    print(f"   Total records: {len(london_open_hours):,}")
    print(f"   Percentage: {len(london_open_hours)/len(df)*100:.1f}%")
    
    # Test timezone conversion scenarios
    print(f"\n🌍 TIMEZONE CONVERSION TEST:")
    
    # Test winter date (GMT+0)
    winter_date = "2024-01-15 08:00:00+00:00"
    winter_dt = pd.to_datetime(winter_date)
    winter_london = winter_dt.tz_convert('Europe/London')
    
    print(f"   Winter (Jan): {winter_date}")
    print(f"   → London:     {winter_london}")
    print(f"   → Should be:  8 AM GMT (same as UTC)")
    
    # Test summer date (GMT+1)  
    summer_date = "2024-07-15 08:00:00+00:00"
    summer_dt = pd.to_datetime(summer_date)
    summer_london = summer_dt.tz_convert('Europe/London')
    
    print(f"   Summer (Jul): {summer_date}")
    print(f"   → London:     {summer_london}")
    print(f"   → Should be:  9 AM BST (UTC+1)")
    
    # Check our data for DST patterns
    print(f"\n📈 SEASONAL ANALYSIS:")
    
    winter_months = df[df['Month'].isin([1, 2, 3, 11, 12])]  # GMT months
    summer_months = df[df['Month'].isin([4, 5, 6, 7, 8, 9, 10])]  # BST months
    
    winter_8am = winter_months[winter_months['Hour'] == 8]
    summer_8am = summer_months[summer_months['Hour'] == 8]
    
    print(f"   Winter 8 AM records: {len(winter_8am):,}")
    print(f"   Summer 8 AM records: {len(summer_8am):,}")
    
    if len(summer_8am) > 0:
        print(f"\n⚠️  POTENTIAL ISSUE DETECTED:")
        print(f"   Summer data shows 8 AM entries, but during BST period")
        print(f"   8 AM UTC = 9 AM London time (should be outside our 7-8:59 range)")
        print(f"   This suggests timezone conversion may be incorrect!")
    
    # Check for proper London market timing
    print(f"\n🎯 LONDON MARKET TIMING VALIDATION:")
    
    # Sample some trades and check if they align with London open
    sample_dates = [
        "2024-01-15",  # Winter
        "2024-07-15",  # Summer  
        "2024-04-01",  # BST starts
        "2024-10-27"   # GMT returns
    ]
    
    for date_str in sample_dates:
        day_data = df[df['Datetime_parsed'].dt.date == pd.to_datetime(date_str).date()]
        morning_hours = day_data[(day_data['Hour'] >= 6) & (day_data['Hour'] <= 10)]
        
        if len(morning_hours) > 0:
            print(f"\n   {date_str} morning hours:")
            for _, row in morning_hours.head(8).iterrows():
                dt = pd.to_datetime(row['Datetime'])
                print(f"     {dt.strftime('%H:%M')} - {row['Open']:.5f}")
    
    # Final assessment
    print(f"\n📋 ASSESSMENT:")
    total_8am = len(df[df['Hour'] == 8])
    total_7am = len(df[df['Hour'] == 7])
    
    print(f"   7 AM entries: {total_7am:,}")
    print(f"   8 AM entries: {total_8am:,}")
    
    if total_8am > total_7am * 2:
        print(f"   ❌ LIKELY TIMEZONE ISSUE: Too many 8 AM vs 7 AM entries")
        print(f"   💡 Recommendation: Use OANDA API for proper timezone handling")
    else:
        print(f"   ✅ Timezone distribution looks reasonable")

def compare_timezones():
    """
    Show what proper timezone conversion should look like
    """
    print(f"\n🕰️  PROPER LONDON TIMEZONE EXAMPLES:")
    print("=" * 50)
    
    # Show key timezone transitions
    test_times = [
        "2024-01-15 07:00:00+00:00",  # Winter morning
        "2024-01-15 08:00:00+00:00",  # Winter morning
        "2024-07-15 06:00:00+00:00",  # Summer - should be 7 AM London
        "2024-07-15 07:00:00+00:00",  # Summer - should be 8 AM London
        "2024-07-15 08:00:00+00:00",  # Summer - should be 9 AM London
    ]
    
    for time_str in test_times:
        dt_utc = pd.to_datetime(time_str)
        dt_london = dt_utc.tz_convert('Europe/London')
        
        is_target_range = 7 <= dt_london.hour <= 8
        status = "✅ TARGET RANGE" if is_target_range else "❌ Outside range"
        
        print(f"   {time_str}")
        print(f"   → {dt_london.strftime('%Y-%m-%d %H:%M:%S %Z')} - {status}")
        print()

if __name__ == "__main__":
    validate_current_data()
    compare_timezones()
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Get OANDA API key from: https://developer.oanda.com/")
    print(f"2. Update oanda_data_fetcher.py with your credentials")
    print(f"3. Run: python oanda_data_fetcher.py")
    print(f"4. Use EURUSD_M15_OANDA.csv for proper London timezone data")