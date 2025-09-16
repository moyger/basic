#!/usr/bin/env python3
"""
Simple OANDA API authentication test
"""

import requests

# Your credentials
OANDA_API_KEY = "dda71753e3c8c227be288694878bf98a-3bfebd9b75052ec2e4f609314dea6fe5"
OANDA_ACCOUNT_ID = "001-011-10255302-001"
OANDA_BASE_URL = "https://api-fxpractice.oanda.com"

def test_auth():
    """Test basic authentication with OANDA API"""
    
    print("🔐 Testing OANDA API Authentication...")
    print(f"API Key: {OANDA_API_KEY[:20]}...{OANDA_API_KEY[-10:]}")
    print(f"Account ID: {OANDA_ACCOUNT_ID}")
    print(f"Base URL: {OANDA_BASE_URL}")
    
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Test 1: Get account details
    print("\n📋 Test 1: Get Account Details")
    url = f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}"
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Account access successful!")
            print(f"Account Currency: {data['account']['currency']}")
            print(f"Account Balance: {data['account']['balance']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 2: Get instruments (simpler endpoint)
    print("\n📊 Test 2: Get Available Instruments")
    url = f"{OANDA_BASE_URL}/v3/accounts/{OANDA_ACCOUNT_ID}/instruments"
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Instruments access successful!")
            # Find EUR_USD
            eurusd = next((inst for inst in data['instruments'] if inst['name'] == 'EUR_USD'), None)
            if eurusd:
                print(f"EUR_USD found: {eurusd['displayName']}")
            else:
                print("EUR_USD not found in instruments")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 3: Simple candle request (last 10 candles)
    print("\n🕯️  Test 3: Get Recent Candles")
    url = f"{OANDA_BASE_URL}/v3/instruments/EUR_USD/candles"
    params = {
        "granularity": "M15",
        "count": 10,
        "price": "M"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Candle data access successful!")
            print(f"Received {len(data['candles'])} candles")
            if data['candles']:
                latest = data['candles'][-1]
                print(f"Latest candle: {latest['time']} - Close: {latest['mid']['c']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_auth()