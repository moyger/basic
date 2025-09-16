#!/opt/homebrew/bin/python3
"""
Forex ORB (Opening Range Breakout) Strategy - London Open
EUR/USD pair with London session (7:00-7:30 range)
"""

# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import time
import pandas_ta as ta

# Configuration parameters
timeframe = "M15"
symbols = ["EURUSD"]
systems = ["Strat"]

# Basic Trading Configuration
starting_balance = 100000  # Starting with $100K for analysis
risk_per_trade = 0.02  # 2% risk per trade (standard)
results_dir = "results"  # Directory for saving results

# London Open range (7:00-7:30 London time)
range_start_time = time(7, 0)
range_end_time = time(7, 30)
latest_entry_time = time(8, 59)  # Restrict entries to 8:00-8:59 AM only (optimization)
market_close_time = time(16, 45)

trade_direction = "long"
timezone = "Europe/London"

exit_eod = False
trailing_stop = True
take_partial = False
trailing_multiplier = 1.5


# Load CSV File
def get_price_data(symbol):
    # Try relative path first (when run from strategies folder), then absolute path (when run from project root)
    import os
    import pytz
    
    # Use OANDA data with proper timezone handling
    data_paths = [
        f"../data/{symbol}_{timeframe}_OANDA.csv",  # OANDA data from strategies folder
        f"data/{symbol}_{timeframe}_OANDA.csv",      # OANDA data from project root
        f"../data/{symbol}_{timeframe}.csv",  # Fallback to old data
        f"data/{symbol}_{timeframe}.csv"      # Fallback from project root
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            # Check if it's OANDA data (has timezone info)
            if 'OANDA' in path:
                # Parse with timezone awareness
                df = pd.read_csv(path)
                df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
                # Convert to London timezone
                london_tz = pytz.timezone('Europe/London')
                df['Datetime'] = df['Datetime'].dt.tz_convert(london_tz)
                df.set_index('Datetime', inplace=True)
                
                # Filter for 2024 data only
                df = df[df.index.year == 2024]
                print(f"✅ Using OANDA data with proper London timezone from: {path}")
                print(f"📅 Filtered to 2024 only: {len(df):,} records from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            else:
                # Old data without timezone
                df = pd.read_csv(path, parse_dates=['Datetime'], index_col='Datetime')
                df.index = pd.to_datetime(df.index, utc=True).tz_convert(timezone)
                
                # Filter for 2024 data only
                df = df[df.index.year == 2024]
                print(f"⚠️ Using old data (timezone may be inaccurate) from: {path}")
                print(f"📅 Filtered to 2024 only: {len(df):,} records")
            return df
    
    # If neither path works, raise error with helpful message
    raise FileNotFoundError(f"Could not find {symbol}_{timeframe}.csv in ../data/ or data/ directories")
    return df


def calculate_inputs(df):
    # Ensure the Datetime column is a datetime object
    df.index = pd.to_datetime(df.index)
    df['Date'] = df.index.date
    df['Time'] = df.index.time
    df['Last_Candle'] = df['Date'] != df['Date'].shift(-1)
    
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=(14))

    # Filter the DataFrame for the opening range period (7:00-7:30)
    df_open_range = df[(df['Time'] >= range_start_time) & (df['Time'] <= range_end_time)]

    # Group by each trading day and calculate the high and low for that range
    opening_range = df_open_range.groupby('Date').agg(
        Open_Range_High=('High', 'max'),
        Open_Range_Low=('Low', 'min')
    )

    # Merge the opening range back to the original DataFrame without resetting the index
    df = df.join(opening_range, on='Date')
   
    df = df.drop(['Time', 'Date'], axis=1)

    return df


def generate_signals(df, s, tp_ratio):
    # Get hour values
    df["Hour"] = df.index.hour
    
    df['Breakout_Above'] = df['Close'] - df['Open_Range_High']
    df['Open_Range_Width'] = df['Open_Range_High'] - df['Open_Range_Low']

    # Various entry conditions
    c1 = (df['Open'] <= df['Open_Range_High']) & (df['Close'] > df['Open_Range_High'])  # Breakout above opening range (long signal)
    c2 = (df.index.time >= range_end_time) & (df.index.time < market_close_time)
    c3 = df['Open_Range_High'].notna()  # Check that the current candle actually has a range
    c4 = df.index.time <= latest_entry_time
    c5 = df['ATR'].notna()
    
    # Generate entries and exits
    # Entry is taken if the conditions were met at yesterday's close
    # Entries depend on the strategy
    if s == "Strat":
        # Default entry rules
        df[f"{s}_Signal"] = c1.shift(1) & c2 & c3 & c4 & c5.shift(1)
        # Generate exits
        df['SL'] = df['Open_Range_Low']
        stop_dist = df['Open'] - df['SL']
        df['TP'] = df['Open'] + stop_dist * tp_ratio
    
    return df


def generate_trades(df, s, mult):
    # Create empty list for trades
    trades_list = []
    trade_open = False
    open_change = {}
    balance = starting_balance
    equity = starting_balance
    balance_history = []
    equity_history = []
    trailing = False
    
    # Basic tracking
    violations = []

    # Extract numpy arrays for relevant columns
    open_prices = df['Open'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    close_prices = df['Close'].values
    prev_atr_values = df['ATR'].shift(1).values if 'ATR' in df.columns else None
    sl_values = df['SL'].values
    tp_values = df['TP'].values
    signal_values = df[f"{s}_Signal"].values
    last_candle_values = df['Last_Candle'].values
    index_values = df.index
    
    # Iterate through rows to work out entries and exits
    for i in range(len(df)):
        current_date = index_values[i].date()
        
        # Basic checks - none for now
        
        # If there is currently no trade
        if not trade_open and signal_values[i]:
            entry_date = index_values[i]
            entry_price = open_prices[i]
            sl = sl_values[i]
            tp = tp_values[i]
            
            # Simple position sizing based on current balance
            risk_amount = balance * risk_per_trade
            
            # Calculate position size based on risk amount
            position_size = 0.01 if entry_price == sl else risk_amount / abs(entry_price - sl)
            trade_open = True
            trailing = False
        # Check if a trade is already open
        if trade_open:
            # Get price values
            low = low_prices[i]
            high = high_prices[i]
            open = open_prices[i]
            close = close_prices[i]
            last_candle = last_candle_values[i]

            # Calculate unrealized PnL
            floating_pnl = (high - entry_price) * position_size
            equity = balance + floating_pnl  # Update equity dynamically

            # Check if stop is hit
            if low <= sl:
                exit_price = open if open <= sl else sl
                trade_open = False

            # Now do the same check for take profit
            elif high >= tp:
                if trailing_stop:
                    trailing = True
                    if take_partial:
                        partial_exit_price = open if open >= tp else tp
                        position_size *= 0.5
                        pnl = (partial_exit_price - entry_price) * position_size  # PnL in currency terms
                        balance += pnl  # Update balance with PnL
                    tp = 100000000000
                else:
                    exit_price = open if open >= tp else tp
                    trade_open = False

            elif exit_eod:
                if (index_values[i].time() == market_close_time) or last_candle:
                    exit_price = close  # Close at the market close price
                    trade_open = False

            # Update trailing stop
            elif trailing:
                new_stop = open - (prev_atr_values[i] * mult)
                if new_stop > sl:
                    sl = new_stop

            if not trade_open:  # If trade has been closed
                exit_date = index_values[i]
                trade_open = False

                if trade_direction == "long":   
                    pnl = (exit_price - entry_price) * position_size  # PnL in pips for forex
                elif trade_direction == "short":
                    pnl = -1 * (exit_price - entry_price) * position_size  # PnL in pips for forex
                balance += pnl  # Update balance with PnL             

                # Basic tracking - no complex rules

                # Store trade data in a list
                trade = [entry_date, entry_price, exit_date, exit_price, position_size, pnl, balance, True]
                # Append trade to overall trade list
                trades_list.append(trade)

        # Store balance and equity
        balance_history.append(balance)
        equity_history.append(equity)

    trades = pd.DataFrame(trades_list, columns=["Entry_Date", "Entry_Price", "Exit_Date", "Exit_Price", "Position_Size", "PnL", "Balance", "Sys_Trade"])
    
    # Calculate return of each trade as well as the trade duration
    trades[f"{s}_Return"] = trades.Balance / trades.Balance.shift(1)
    dur = []
    for i, row in trades.iterrows():
        d1 = row.Entry_Date
        d2 = row.Exit_Date
        dur.append(np.busday_count(d1.date(), d2.date()) + 1)  # Add 1 because formula doesn't include the end date otherwise
    
    trades[f"{s}_Duration"] = dur

    # Create a new dataframe with an index of exit dfs
    returns = pd.DataFrame(index=trades.Exit_Date)
    # Create a new dataframe with an index of entries to track entry price
    entries = pd.DataFrame(index=trades.Entry_Date)

    entries[f"{s}_Entry_Price"] = pd.Series(trades.Entry_Price).values
    # Add the Return column to this new data frame
    returns[f"{s}_Ret"] = pd.Series(trades[f"{s}_Return"]).values
    returns[f"{s}_Trade"] = pd.Series(trades.Sys_Trade).values
    returns[f"{s}_Duration"] = pd.Series(trades[f"{s}_Duration"]).values
    returns[f"{s}_PnL"] = pd.Series(trades.PnL).values
    returns[f"{s}_Balance"] = pd.Series(trades.Balance).values
    change_ser = pd.Series(open_change, name=f"{s}_Change")

    # Add the returns from the trades to the main data frame
    df = pd.concat([df, returns, entries, change_ser], axis=1)
    # Fill all the NaN return values with 1 as there was no profit or loss on those days
    df[f"{s}_Ret"] = df[f"{s}_Ret"].fillna(1)
    # Fill all the NaN trade values with False as there was no trade on those days
    df[f"{s}_Trade"] = df[f"{s}_Trade"].infer_objects(copy=False)
    # Fill all the NaN return values with 1 as there was no loss on those days
    df[f"{s}_Change"] = df[f"{s}_Change"].astype(float).fillna(1)
    
    # Use the updated balance and equity variables (handle length mismatch)
    # Ensure balance_history matches df length
    while len(balance_history) < len(df):
        balance_history.append(balance_history[-1] if balance_history else starting_balance)
    while len(equity_history) < len(df):
        equity_history.append(equity_history[-1] if equity_history else starting_balance)
    
    df[f"{s}_Bal"] = pd.Series(balance_history[:len(df)], index=df.index).ffill()
    df[f"{s}_Equity"] = pd.Series(equity_history[:len(df)], index=df.index).ffill()

    active_trades = np.where(df[f"{s}_Trade"] == True, True, False)
    df[f"{s}_In_Market"] = df[f"{s}_Trade"].copy()
    # Populate trades column based on duration
    for count, t in enumerate(active_trades):
        if t == True:
            dur = df[f"{s}_Duration"].iat[count]
            for i in range(int(dur)):
                # Starting from the exit date, move backwards and mark each trading day
                df[f"{s}_In_Market"].iat[count - i] = True
    
    return df, trades, violations


def backtest(price, tp_ratio, mult):
    # Calculate strategy inputs
    price = calculate_inputs(price)
    violations = []

    for s in systems:
        # Generate signals
        price = generate_signals(price, s, tp_ratio)

        # Generate trades
        price, trades, violations = generate_trades(price, s, mult)

    for s in systems:
        # Calculate drawdown
        price[f"{s}_Peak"] = price[f"{s}_Bal"].cummax()
        price[f"{s}_DD"] = price[f"{s}_Bal"] - price[f"{s}_Peak"]

    return price, trades, violations


def get_metrics(system, data):
    metrics = {}
    years = (data.index[-1] - data.index[0]).days / 365.25
    sys_cagr = round(((((data[f"{system}_Bal"].iloc[-1]/data[f"{system}_Bal"].iloc[0])**(1/years))-1)*100), 2)
    sys_dd = round(((data[f"{system}_DD"] / data[f"{system}_Peak"]).min()) * 100, 2)
 
    win = data[f"{system}_Ret"] > 1
    loss = data[f"{system}_Ret"] < 1
    trades_triggered = data[f"{system}_Trade"].sum()
    wins = win.sum()
    losses = loss.sum()
    winrate = round(wins / (wins + losses) * 100, 2)
    
    # Calculate the size of the move from the entry data to the close
    avg_up_move = (data[f"{system}_Ret"][data[f"{system}_Ret"] > 1].mean() - 1) * 100
    avg_down_move = (data[f"{system}_Ret"][data[f"{system}_Ret"] < 1].mean() - 1) * 100
    avg_rr = round(abs(avg_up_move / avg_down_move), 2)

    # Save data
    metrics["Start_Balance"] = round(data[f"{system}_Bal"].iat[0], 2)
    metrics["Final_Balance"] = round(data[f"{system}_Bal"].iat[-1], 2)
    metrics["Annual_Return"] = round(sys_cagr, 2)
    metrics["Max_Drawdown"] = round(sys_dd, 2)
    metrics["Trades"] = round(trades_triggered, 2)
    
    metrics["Winrate"] = round(winrate, 2)
    metrics["Avg_RR"] = avg_rr

    return metrics


def plot_performance(results, symbols, systems, tp_range, multipliers):
    """Plot performance charts for the backtest results"""
    plt.style.use("dark_background")
    plt.rcParams["figure.figsize"] = (16, 8)
    plt.rcParams.update({"font.size": 18})

    colours = ["tab:olive", "tab:blue", "tab:purple", "tab:orange", "tab:green", "tab:cyan", "tab:red", "tab:gray", "tab:pink"]

    for count, sym in enumerate(symbols):
        plt.figure()
        plt.title(f"Performance of {sym} - London Open ORB")

        legend_entries = []  # Store legend labels
        colour_idx = 0  # To cycle through colors if needed

        for tp_ratio in tp_range:
            for mult in multipliers:
                for s in systems:
                    # Generate a unique index for each (TP, Mult) combination
                    result_idx = (
                        count * len(tp_range) * len(multipliers)
                        + list(tp_range).index(tp_ratio) * len(multipliers)
                        + list(multipliers).index(mult)
                    )

                    if result_idx >= len(results):  
                        continue  # Prevent out-of-bounds error

                    label = f"{s} (TP={tp_ratio}, Mult={mult})"
                    color = colours[colour_idx % len(colours)]
                    
                    plt.plot(results[result_idx][f"{s}_Bal"], color=color, label=label)
                    legend_entries.append(label)

                    colour_idx += 1  # Move to next color

        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.xlabel("Time")
        plt.ylabel("Balance")
        plt.savefig(os.path.join(results_dir, "plot.png"), format="png", dpi=300, bbox_inches="tight")
        plt.show()


def analyze_trades(trades, systems):
    """Analyze trades by entry hour"""
    # Split index into 'Date' and 'Time' columns
    trades['Entry_Date'] = pd.to_datetime(trades["Entry_Date"])
    trades['Entry_Hour'] = trades['Entry_Date'].dt.hour

    # Convert "Strat_Breakout_Return" to percentage
    trades["Return_Percentage"] = (trades[f"{systems[-1]}_Return"] - 1) * 100

    # Group by Hour and Calculate Average PnL
    hourly_return = trades.groupby("Entry_Hour")["Return_Percentage"].mean()

    # Group by Hour and Count Number of Trades
    hourly_trades_count = trades.groupby("Entry_Hour")["Return_Percentage"].count()

    # Combine the two results into a single DataFrame
    hourly_stats = pd.DataFrame({
        "Average_Return_Percentage": hourly_return,
        "Trades_Count": hourly_trades_count
    })

    return hourly_stats


def main():
    """Main execution function"""
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    prog = 0
    tp_range = np.arange(1.5, 2.0, 0.5)
    multipliers = np.arange(1.0, 1.5, 0.5)
    max_prog = len(symbols) * len(tp_range) * len(multipliers)
    
    print("Starting basic London Open ORB backtest for EUR/USD - 2024 ONLY...")
    print(f"Starting Balance: ${starting_balance:,}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    
    violations = []
    
    for sym in symbols:
        price = get_price_data(sym)
        for tp_ratio in tp_range:
            for mult in multipliers:
                result, trades, trade_violations = backtest(price, tp_ratio, mult)
                results.append(result)
                violations.extend(trade_violations)
                prog += 1
                print(f"Progress: {round((prog / max_prog) * 100)} %")
    
    # Get metrics for the last result
    sys_metrics = {}
    for s in systems:
        sys_metrics.update({s: get_metrics(s, result)})
    sys_metrics_df = pd.DataFrame.from_dict(sys_metrics)
    
    print("\nBacktest Results:")
    print(sys_metrics_df)
    
    
    # Analyze trades by hour
    hourly_stats = analyze_trades(trades, systems)
    print("\nHourly Statistics:")
    print(hourly_stats)
    
    # Save results to CSV
    trades.to_csv(os.path.join(results_dir, "trades.csv"))
    result.to_csv(os.path.join(results_dir, "result.csv"))
    print(f"\nResults saved to {results_dir}/trades.csv and {results_dir}/result.csv")
    
    # Optional: Plot performance (comment out if not needed)
    plot_performance(results, symbols, systems, tp_range, multipliers)
    
    return results, trades, sys_metrics_df


if __name__ == "__main__":
    results, trades, metrics = main()