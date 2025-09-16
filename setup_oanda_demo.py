#!/usr/bin/env python3
"""
OANDA API Setup Demo
Quick setup guide for getting proper forex data
"""

def show_setup_instructions():
    """
    Display step-by-step OANDA setup instructions
    """
    
    print("🔧 OANDA API SETUP GUIDE")
    print("=" * 50)
    
    print("\n📋 Step 1: Create OANDA Account")
    print("   • Go to: https://developer.oanda.com/")
    print("   • Click 'Get Started' or 'Sign Up'")
    print("   • Create free developer account")
    print("   • No deposit required for practice account")
    
    print("\n🔑 Step 2: Get API Credentials")
    print("   • Login to OANDA developer portal")
    print("   • Go to 'Manage API Access'")
    print("   • Generate new Personal Access Token")
    print("   • Copy your API Key (long string)")
    print("   • Copy your Account ID (numbers)")
    
    print("\n⚙️  Step 3: Update Credentials")
    print("   • Open: oanda_data_fetcher.py")
    print("   • Replace 'your-api-key-here' with your actual API key")
    print("   • Replace 'your-account-id' with your actual account ID")
    
    print("\n🚀 Step 4: Install Requirements")
    print("   pip install requests pandas pytz")
    
    print("\n📊 Step 5: Fetch Data")
    print("   python oanda_data_fetcher.py")
    
    print("\n✅ Result: EURUSD_M15_OANDA.csv with proper London timezone")

def create_requirements_file():
    """
    Create requirements.txt for OANDA setup
    """
    requirements = """# OANDA Data Fetcher Requirements
requests>=2.28.0
pandas>=1.5.0
pytz>=2022.1
numpy>=1.21.0"""
    
    with open("requirements_oanda.txt", "w") as f:
        f.write(requirements)
    
    print("📄 Created requirements_oanda.txt")

def test_timezone_fix():
    """
    Show what the timezone fix will accomplish
    """
    print("\n🎯 WHAT THIS FIXES:")
    print("=" * 30)
    
    print("\n❌ CURRENT PROBLEM:")
    print("   • Winter: 8 AM UTC = 8 AM GMT ✅ (correct)")
    print("   • Summer: 8 AM UTC = 9 AM BST ❌ (1 hour late!)")
    
    print("\n✅ OANDA SOLUTION:")
    print("   • Winter: 7-8 AM London = 7-8 AM GMT")
    print("   • Summer: 7-8 AM London = 6-7 AM UTC (auto-adjusted)")
    print("   • Consistent London market timing year-round")
    
    print("\n📈 EXPECTED IMPROVEMENTS:")
    print("   • Consistent entry timing across seasons")
    print("   • Better alignment with London Open momentum")
    print("   • More accurate backtesting results")
    print("   • Elimination of DST-related performance dips")

if __name__ == "__main__":
    show_setup_instructions()
    create_requirements_file()
    test_timezone_fix()
    
    print(f"\n🔄 NEXT ACTION:")
    print(f"   Follow the setup guide above to get OANDA API access")
    print(f"   This will give you proper London timezone data!")
    print(f"   Expected improvement: More consistent trading performance")