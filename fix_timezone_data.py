#!/usr/bin/env python3
"""
Fix timezone issues in current EUR/USD data
Adjusts summer data to proper London timezone
"""

import pandas as pd
from datetime import datetime

def fix_timezone_data():
    """Fix timezone issues in EURUSD_M15.csv"""
    
    print("🔄 Loading current EUR/USD data...")
    df = pd.read_csv('data/EURUSD_M15.csv')
    
    print(f"📊 Loaded {len(df):,} records")
    
    # Parse datetime
    df['Datetime_parsed'] = pd.to_datetime(df['Datetime'])
    df['Month'] = df['Datetime_parsed'].dt.month
    df['Hour'] = df['Datetime_parsed'].dt.hour
    
    # Identify summer months (BST period - roughly April to October)
    summer_months = [4, 5, 6, 7, 8, 9, 10]
    summer_mask = df['Month'].isin(summer_months)
    
    print(f"📅 Summer records (BST): {summer_mask.sum():,}")
    print(f"📅 Winter records (GMT): {(~summer_mask).sum():,}")
    
    # Shift summer data back 1 hour
    df.loc[summer_mask, 'Datetime_parsed'] = df.loc[summer_mask, 'Datetime_parsed'] - pd.Timedelta(hours=1)
    
    # Update the string datetime column
    df['Datetime'] = df['Datetime_parsed'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Remove the helper columns
    df = df.drop(['Datetime_parsed', 'Month', 'Hour'], axis=1)
    
    # Save fixed data
    df.to_csv('data/EURUSD_M15_timezone_fixed.csv', index=False)
    
    print("✅ Timezone-fixed data saved to: data/EURUSD_M15_timezone_fixed.csv")
    
    # Validate fix
    df_check = pd.read_csv('data/EURUSD_M15_timezone_fixed.csv')
    df_check['Datetime_parsed'] = pd.to_datetime(df_check['Datetime'])
    df_check['Hour'] = df_check['Datetime_parsed'].dt.hour
    
    hourly_dist = df_check['Hour'].value_counts().sort_index()
    print("\n🕐 FIXED HOURLY DISTRIBUTION:")
    for hour, count in hourly_dist.items():
        if 6 <= hour <= 10:  # Focus on morning hours
            print(f"   {hour:02d}:XX - {count:,} records")
    
    london_hours = df_check[(df_check['Hour'] >= 7) & (df_check['Hour'] <= 9)]
    print(f"\n🌅 London Open Hours (7-9 AM): {len(london_hours):,} records")

if __name__ == "__main__":
    fix_timezone_data()