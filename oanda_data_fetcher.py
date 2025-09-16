#!/usr/bin/env python3
"""
OANDA API Data Fetcher for EUR/USD M15 with proper London timezone handling
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import pytz

# OANDA API Configuration
# Get your free API key from: https://developer.oanda.com/
OANDA_API_KEY = "d3907a1ef932e55a8a93f5d4f2c262a2-1dddadccb133719789a2aa4d561329be"  # Replace with your OANDA API key
OANDA_ACCOUNT_ID = "001-011-10255302-001"  # Replace with your account ID
OANDA_BASE_URL = "https://api-fxtrade.oanda.com"  # Live URL

# For live trading, use: "https://api-fxtrade.oanda.com"

class OandaDataFetcher:
    def __init__(self, api_key, account_id, base_url=OANDA_BASE_URL):
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def fetch_historical_data(self, instrument="EUR_USD", granularity="M15", 
                            start_date=None, end_date=None, count=5000):
        """
        Fetch historical data from OANDA API
        
        Parameters:
        - instrument: Currency pair (e.g., "EUR_USD")
        - granularity: Timeframe (M15 for 15-minute)
        - start_date: Start date (YYYY-MM-DD format)
        - end_date: End date (YYYY-MM-DD format)  
        - count: Number of candles (max 5000 per request)
        """
        
        url = f"{self.base_url}/v3/instruments/{instrument}/candles"
        
        params = {
            "granularity": granularity,
            "price": "M",  # Mid prices (average of bid/ask)
        }
        
        if start_date and end_date:
            params["from"] = f"{start_date}T00:00:00Z"
            params["to"] = f"{end_date}T23:59:59Z"
        else:
            params["count"] = count
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            return self.process_candles(data["candles"])
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return None
    
    def process_candles(self, candles):
        """
        Convert OANDA candle data to pandas DataFrame with proper timezone handling
        """
        data_list = []
        
        for candle in candles:
            if candle["complete"]:  # Only use complete candles
                # Parse timestamp (UTC)
                dt_utc = pd.to_datetime(candle["time"])
                
                # Convert to London timezone (handles GMT/BST automatically)
                london_tz = pytz.timezone('Europe/London')
                dt_london = dt_utc.tz_convert(london_tz)
                
                data_list.append({
                    "Datetime": dt_london.strftime('%Y-%m-%d %H:%M:%S%z'),
                    "Open": float(candle["mid"]["o"]),
                    "High": float(candle["mid"]["h"]),
                    "Low": float(candle["mid"]["l"]),
                    "Close": float(candle["mid"]["c"]),
                    "Volume": candle["volume"]
                })
        
        df = pd.DataFrame(data_list)
        return df
    
    def fetch_extended_history(self, instrument="EUR_USD", granularity="M15", 
                             years_back=5):
        """
        Fetch extended historical data by making multiple API calls
        OANDA limits to 5000 candles per request
        """
        
        all_data = []
        end_date = datetime.now()
        
        # Calculate how many requests needed
        # M15 = 96 candles per day, ~35,000 per year
        # So we need multiple requests for multi-year data
        
        current_end = end_date
        
        for i in range(years_back * 8):  # ~8 requests per year for M15 data
            start_date = current_end - timedelta(days=45)  # ~45 days per request
            
            print(f"Fetching data from {start_date.date()} to {current_end.date()}...")
            
            batch_data = self.fetch_historical_data(
                instrument=instrument,
                granularity=granularity,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=current_end.strftime('%Y-%m-%d')
            )
            
            if batch_data is not None and len(batch_data) > 0:
                all_data.append(batch_data)
            
            current_end = start_date
            
            # Stop if we've gone back far enough
            if current_end < datetime.now() - timedelta(days=years_back * 365):
                break
        
        if all_data:
            # Combine all batches
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates and sort
            combined_df = combined_df.drop_duplicates(subset=['Datetime'])
            combined_df = combined_df.sort_values('Datetime')
            combined_df = combined_df.reset_index(drop=True)
            
            return combined_df
        
        return None

def main():
    """
    Main function to fetch EUR/USD data and save to CSV
    """
    
    # Check if API key is set
    if OANDA_API_KEY == "your-api-key-here":
        print("❌ Please set your OANDA API key in the script!")
        print("\n📋 Steps to get OANDA API access:")
        print("1. Go to: https://developer.oanda.com/")
        print("2. Create free account")
        print("3. Generate API key")
        print("4. Replace OANDA_API_KEY in this script")
        print("5. Replace OANDA_ACCOUNT_ID in this script")
        return
    
    print("🔄 Setting up OANDA data fetcher...")
    fetcher = OandaDataFetcher(OANDA_API_KEY, OANDA_ACCOUNT_ID)
    
    print("📊 Fetching EUR/USD M15 data with London timezone...")
    
    # Fetch 5 years of data
    df = fetcher.fetch_extended_history(
        instrument="EUR_USD",
        granularity="M15", 
        years_back=5
    )
    
    if df is not None:
        # Save to CSV
        output_file = "data/EURUSD_M15_OANDA.csv"
        df.to_csv(output_file, index=False)
        
        print(f"✅ Data saved to {output_file}")
        print(f"📈 Total records: {len(df):,}")
        print(f"📅 Date range: {df['Datetime'].iloc[0]} to {df['Datetime'].iloc[-1]}")
        
        # Show sample of London morning data
        print(f"\n🌅 Sample London morning data (7-9 AM):")
        # Parse datetime with UTC flag to handle mixed timezones
        df['Datetime_parsed'] = pd.to_datetime(df['Datetime'], utc=True)
        # Convert to London timezone for hour extraction
        london_tz = pytz.timezone('Europe/London')
        df['Datetime_london'] = df['Datetime_parsed'].dt.tz_convert(london_tz)
        df['Hour'] = df['Datetime_london'].dt.hour
        morning_data = df[(df['Hour'] >= 7) & (df['Hour'] <= 9)]
        print(morning_data[['Datetime', 'Open', 'High', 'Low', 'Close']].head(10))
        
    else:
        print("❌ Failed to fetch data. Please check your API credentials.")

if __name__ == "__main__":
    main()