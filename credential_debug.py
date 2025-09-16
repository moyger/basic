#!/usr/bin/env python3
"""
OANDA Credential Debug and Alternative Data Solutions
"""

def debug_credentials():
    """Debug OANDA credential issues"""
    
    print("🔍 OANDA CREDENTIAL DEBUGGING")
    print("=" * 50)
    
    print("\n❌ CURRENT ISSUE:")
    print("   401 Unauthorized - Insufficient authorization")
    
    print("\n🔧 POSSIBLE CAUSES:")
    print("   1. API Key might be invalid or expired")
    print("   2. Account ID format might be incorrect") 
    print("   3. API Key might not have proper permissions")
    print("   4. Using wrong environment (practice vs live)")
    
    print("\n✅ HOW TO FIX:")
    print("   1. Go to: https://developer.oanda.com/")
    print("   2. Login to your account")
    print("   3. Go to 'Manage API Access' or 'My Account' → 'API Access'")
    print("   4. Generate a NEW Personal Access Token")
    print("   5. Make sure it has 'Read' permissions for market data")
    print("   6. Copy the EXACT token (including any dashes)")
    print("   7. Find your Account ID in the dashboard")
    
    print("\n📝 CREDENTIAL FORMAT CHECK:")
    print("   API Key should look like: 'abc123def456-ghi789jkl012'")
    print("   Account ID should look like: '001-011-12345678-001'")
    
    print("\n🌐 ENVIRONMENT CHECK:")
    print("   Practice URL: https://api-fxpractice.oanda.com")
    print("   Live URL: https://api-fxtrade.oanda.com")
    print("   Make sure you're using the right one for your account type")

def alternative_data_solutions():
    """Show alternative data solutions while fixing OANDA"""
    
    print("\n\n🔄 ALTERNATIVE DATA SOLUTIONS")
    print("=" * 50)
    
    print("\n📊 OPTION 1: Use our current timezone-adjusted data")
    print("   • We can manually fix timezone issues in EURUSD_M15.csv")
    print("   • Apply proper London timezone conversion")
    print("   • Less accurate but immediately available")
    
    print("\n📈 OPTION 2: Download from other sources")
    print("   • MetaTrader 5 (MT5) - free historical data")
    print("   • Alpha Vantage API - free tier available")
    print("   • FXCM API - some historical data available")
    
    print("\n⚡ OPTION 3: Fix current data timezone issues")
    print("   • Shift summer data back 1 hour to correct DST")
    print("   • Maintain consistency with London market hours")
    print("   • Quick solution while working on OANDA access")

def create_timezone_fix():
    """Create a script to fix timezone issues in current data"""
    
    print("\n\n🛠️  TIMEZONE FIX SOLUTION")
    print("=" * 50)
    
    fix_script = '''#!/usr/bin/env python3
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
    print("\\n🕐 FIXED HOURLY DISTRIBUTION:")
    for hour, count in hourly_dist.items():
        if 6 <= hour <= 10:  # Focus on morning hours
            print(f"   {hour:02d}:XX - {count:,} records")
    
    london_hours = df_check[(df_check['Hour'] >= 7) & (df_check['Hour'] <= 9)]
    print(f"\\n🌅 London Open Hours (7-9 AM): {len(london_hours):,} records")
    
if __name__ == "__main__":
    fix_timezone_data()
'''
    
    with open('fix_timezone_data.py', 'w') as f:
        f.write(fix_script)
    
    print("   📄 Created: fix_timezone_data.py")
    print("   🚀 Run: python fix_timezone_data.py")

if __name__ == "__main__":
    debug_credentials()
    alternative_data_solutions()
    create_timezone_fix()
    
    print("\n\n🎯 RECOMMENDED NEXT STEPS:")
    print("1. Fix OANDA credentials following the debug guide above")
    print("2. OR run the timezone fix script: python fix_timezone_data.py")
    print("3. Continue with strategy testing using corrected data")
    print("\nWhich approach would you like to take?")