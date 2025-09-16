#!/opt/homebrew/bin/python3
"""
SPX500 ORB (Opening Range Breakout) Strategy with FTMO Compliance
Based on the successful SPX500 strategy with FTMO rules integration
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import time, datetime, timedelta
import pandas_ta as ta

# FTMO Configuration parameters
timeframe = "M15"
symbols = ["SPX500"]
systems = ["Strat"]
starting_balance = 100000  # FTMO account size
risk_per_trade = 0.01  # Risking 1% per trade for FTMO compliance
results_dir = "results"

# Trading Cost Parameters (SPX500 CFD/Futures)
spread_points = 0.6  # Average spread in points (0.6 points = $0.60 per contract)
commission_per_side = 2.5  # Commission per side in USD ($2.50 * 2 = $5 round trip)
slippage_points = 0.2  # Average slippage in points (0.2 points = $0.20 per contract)
points_cost_per_contract = spread_points + slippage_points  # Cost in points per contract
commission_cost_per_contract = commission_per_side * 2  # Round trip commission cost

# SPX500 ORB Settings (US Market Hours)
range_start_time = time(9, 30)   # 9:30 AM EST
range_end_time = time(9, 45)     # 9:45 AM EST  
latest_entry_time = time(11, 59) # 11:59 AM EST
market_close_time = time(16, 45) # 4:45 PM EST

trade_direction = "long"
timezone = "America/New_York"

exit_eod = False
trailing_stop = True
take_partial = False
trailing_multiplier = 1.5

# FTMO Swing Challenge Rules
class FTMORules:
    def __init__(self):
        # Challenge Phase Rules
        self.challenge_profit_target = 0.10  # 10%
        self.verification_profit_target = 0.05  # 5%
        self.challenge_max_loss = 0.10  # 10% max drawdown during challenge
        self.challenge_daily_loss_limit = 0.05  # 5% daily loss limit
        self.challenge_min_days = 10
        self.challenge_max_days = 60  # Swing challenge
        
        # Funded Phase Rules (Modified per user request)
        self.funded_monthly_target = 0.05  # 5%
        self.funded_max_loss = None  # DISABLED after passing challenge
        self.funded_daily_loss_limit = 0.05  # 5% daily loss limit (keep active)
        
        # Weekend holding allowed for swing challenge
        self.weekend_holding_allowed = True

# Initialize FTMO rules
ftmo = FTMORules()

# Trading phases
CHALLENGE = "challenge"
VERIFICATION = "verification"
FUNDED = "funded"


def get_price_data(symbol):
    df = pd.read_csv(f"data/{symbol}_{timeframe}.csv", parse_dates=['Datetime'], index_col='Datetime')
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(timezone)
    return df


def calculate_inputs(df):
    # Ensure the Datetime column is a datetime object
    df.index = pd.to_datetime(df.index)
    df['Date'] = df.index.date
    df['Time'] = df.index.time
    df['Last_Candle'] = df['Date'] != df['Date'].shift(-1)
    
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=(14))

    # Filter the DataFrame for the opening range period
    df_open_range = df[df['Time'] == range_start_time]

    # Group by each trading day and calculate the high and low for that range
    opening_range = df_open_range.groupby('Date').agg(
        Open_Range_High=('High', 'max'),
        Open_Range_Low=('Low', 'min')
    )

    # Determine if the opening candle is bullish: Close > Open
    df_open_range = df[df['Time'] == range_start_time].copy()
    
    # Merge the opening range back to the original DataFrame without resetting the index
    df = df.join(opening_range, on='Date')
   
    df = df.drop(['Time', 'Date'], axis=1)

    return df


def get_trading_phase(balance, starting_balance, days_traded):
    """Determine current FTMO trading phase"""
    profit_pct = (balance - starting_balance) / starting_balance
    
    if days_traded <= ftmo.challenge_max_days:
        if profit_pct >= ftmo.challenge_profit_target and days_traded >= ftmo.challenge_min_days:
            return VERIFICATION
        else:
            return CHALLENGE
    else:
        return FUNDED


def check_ftmo_violations(balance, starting_balance, daily_start_balance, phase, days_traded):
    """Check for FTMO rule violations with phase-specific rules"""
    total_drawdown = (starting_balance - balance) / starting_balance
    daily_loss = (daily_start_balance - balance) / starting_balance if daily_start_balance > 0 else 0
    
    violations = []
    account_closed = False
    
    # Daily loss limit applies to ALL phases
    if daily_loss >= ftmo.challenge_daily_loss_limit:
        violations.append(f"Daily loss limit exceeded: {daily_loss:.2%}")
        account_closed = True
    
    if phase == CHALLENGE:
        # Challenge phase: both daily and total drawdown limits
        if total_drawdown >= ftmo.challenge_max_loss:
            violations.append(f"Challenge max drawdown exceeded: {total_drawdown:.2%}")
            account_closed = True
            
        if days_traded > ftmo.challenge_max_days:
            violations.append(f"Challenge time limit exceeded: {days_traded} days")
            account_closed = True
    
    elif phase == VERIFICATION:
        # Verification phase: same rules as challenge
        if total_drawdown >= ftmo.challenge_max_loss:
            violations.append(f"Verification max drawdown exceeded: {total_drawdown:.2%}")
            account_closed = True
    
    elif phase == FUNDED:
        # Funded phase: only daily loss limit (max drawdown disabled per user request)
        # Max drawdown check is disabled - only daily loss limit remains active
        pass
    
    return violations, account_closed


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
    if s == "Strat":
        # Default entry rules
        df[f"{s}_Signal"] = c1.shift(1) & c2 & c3 & c4 & c5.shift(1)
        # Generate exits
        df['SL'] = df['Open_Range_Low']
        stop_dist = df['Open'] - df['SL']
        df['TP'] = df['Open'] + stop_dist * tp_ratio
    
    return df


def generate_ftmo_trades(df, s, mult):
    # Create empty list for trades
    trades_list = []
    trade_open = False
    open_change = {}
    balance = starting_balance
    equity = starting_balance
    balance_history = []
    equity_history = []
    trailing = False
    
    # FTMO tracking
    days_traded = 0
    current_date = None
    daily_start_balance = starting_balance
    ftmo_violations = []
    account_closed = False
    phase_history = []

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
        current_timestamp = index_values[i]
        trade_date = current_timestamp.date()
        
        # Track new trading days
        if current_date != trade_date:
            current_date = trade_date
            days_traded += 1
            daily_start_balance = balance
        
        # Determine current FTMO phase
        current_phase = get_trading_phase(balance, starting_balance, days_traded)
        phase_history.append(current_phase)
        
        # Check FTMO violations
        if not account_closed:
            violations, closed = check_ftmo_violations(
                balance, starting_balance, daily_start_balance, current_phase, days_traded
            )
            if violations:
                ftmo_violations.extend(violations)
            if closed:
                account_closed = True
                print(f"FTMO Account Closed on {current_timestamp}: {violations}")
                break
        
        # If there is currently no trade and account is not closed
        if not account_closed and not trade_open and signal_values[i]:
            # Calculate daily loss to prevent breaching 5% limit
            daily_loss_pct = (daily_start_balance - balance) / starting_balance if daily_start_balance > 0 else 0
            max_additional_risk = ftmo.challenge_daily_loss_limit - daily_loss_pct - 0.005  # 0.5% buffer
            
            # Only take trade if we have enough daily loss buffer (at least 1.5% remaining)
            if max_additional_risk >= 0.015:
                entry_date = index_values[i]
                entry_price = open_prices[i]
                sl = sl_values[i]
                tp = tp_values[i]
                # Calculate position size based on risk percentage, adjusted for daily loss protection and trading costs
                base_risk = min(risk_per_trade, max_additional_risk)
                risk_amount = balance * base_risk
                
                # Calculate stop loss distance and adjust for trading costs
                stop_distance = abs(entry_price - sl)
                if stop_distance > 0:
                    # Net risk per point including trading costs
                    net_risk_per_point = stop_distance + points_cost_per_contract
                    # Calculate position size considering commission as fixed cost
                    preliminary_position_size = (risk_amount - commission_cost_per_contract) / net_risk_per_point
                    position_size = max(0.01, preliminary_position_size)  # Minimum position size
                else:
                    position_size = 0.01
                trade_open = True
                trailing = False
            
        # Check if a trade is already open and account not closed
        if trade_open and not account_closed:
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

            elif exit_eod and not ftmo.weekend_holding_allowed:
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
                    raw_pnl = (exit_price - entry_price) * position_size  # Raw P&L in currency terms
                elif trade_direction == "short":
                    raw_pnl = -1 * (exit_price - entry_price) * position_size  # Raw P&L in currency terms
                
                # Deduct trading costs (spread + slippage + commission)
                points_cost = position_size * points_cost_per_contract  # Cost in points
                commission_cost = commission_cost_per_contract  # Fixed commission per trade
                total_trading_costs = points_cost + commission_cost
                pnl = raw_pnl - total_trading_costs  # Net P&L after costs
                balance += pnl  # Update balance with net P&L             

                # Store trade data in a list
                trade = [entry_date, entry_price, exit_date, exit_price, position_size, pnl, balance, True, current_phase]
                # Append trade to overall trade list
                trades_list.append(trade)

        # Store balance and equity
        balance_history.append(balance)
        equity_history.append(equity)

    trades = pd.DataFrame(trades_list, columns=["Entry_Date", "Entry_Price", "Exit_Date", "Exit_Price", "Position_Size", "PnL", "Balance", "Sys_Trade", "FTMO_Phase"])
    
    # Calculate return of each trade as well as the trade duration
    if len(trades) > 0:
        trades[f"{s}_Return"] = trades.Balance / trades.Balance.shift(1)
        dur = []
        for i, row in trades.iterrows():
            d1 = row.Entry_Date
            d2 = row.Exit_Date
            dur.append(np.busday_count(d1.date(), d2.date()) + 1)  # Add 1 because formula doesn't include the end date otherwise
        
        trades[f"{s}_Duration"] = dur

        # Create a new dataframe with an index of exit dates
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
    df[f"{s}_Ret"] = df[f"{s}_Ret"].fillna(1) if f"{s}_Ret" in df.columns else 1
    # Fill all the NaN trade values with False as there was no trade on those days
    df[f"{s}_Trade"] = df[f"{s}_Trade"].fillna(False) if f"{s}_Trade" in df.columns else False
    # Fill all the NaN return values with 1 as there was no loss on those days
    df[f"{s}_Change"] = df[f"{s}_Change"].astype(float).fillna(1) if f"{s}_Change" in df.columns else 1
    
    # Use the updated balance and equity variables
    if len(balance_history) == len(df):
        df[f"{s}_Bal"] = pd.Series(balance_history, index=df.index).ffill()
        df[f"{s}_Equity"] = pd.Series(equity_history, index=df.index).ffill()
        df[f"{s}_Phase"] = pd.Series(phase_history, index=df.index).ffill()
    else:
        df[f"{s}_Bal"] = starting_balance
        df[f"{s}_Equity"] = starting_balance
        df[f"{s}_Phase"] = CHALLENGE

    # Calculate in-market periods
    if len(trades) > 0:
        active_trades = np.where(df[f"{s}_Trade"] == True, True, False)
        df[f"{s}_In_Market"] = df[f"{s}_Trade"].copy()
        # Populate trades column based on duration
        for count, t in enumerate(active_trades):
            if t == True:
                dur = df[f"{s}_Duration"].iat[count]
                for i in range(int(dur)):
                    if count - i >= 0:
                        # Starting from the exit date, move backwards and mark each trading day
                        df.loc[df.index[count - i], f"{s}_In_Market"] = True
    
    return df, trades, ftmo_violations, account_closed, days_traded


def backtest(price, tp_ratio, mult):
    # Calculate strategy inputs
    price = calculate_inputs(price)

    for s in systems:
        # Generate signals
        price = generate_signals(price, s, tp_ratio)

        # Generate trades with FTMO compliance
        price, trades, violations, closed, days = generate_ftmo_trades(price, s, mult)

    for s in systems:
        # Calculate drawdown
        if f"{s}_Bal" in price.columns:
            price[f"{s}_Peak"] = price[f"{s}_Bal"].cummax()
            price[f"{s}_DD"] = price[f"{s}_Bal"] - price[f"{s}_Peak"]

    return price, trades, violations, closed, days


def get_ftmo_metrics(system, data, violations, closed, days_traded):
    if f"{system}_Bal" not in data.columns or data[f"{system}_Bal"].isna().all():
        return {"Error": "No trades generated"}
        
    metrics = {}
    years = (data.index[-1] - data.index[0]).days / 365.25
    
    start_bal = data[f"{system}_Bal"].dropna().iloc[0] if not data[f"{system}_Bal"].dropna().empty else starting_balance
    final_bal = data[f"{system}_Bal"].dropna().iloc[-1] if not data[f"{system}_Bal"].dropna().empty else starting_balance
    
    # Calculate returns
    total_return = (final_bal - start_bal) / start_bal * 100
    sys_cagr = round(((((final_bal/start_bal)**(1/years))-1)*100), 2) if years > 0 and start_bal > 0 else 0
    
    # Calculate drawdown
    if f"{system}_DD" in data.columns and f"{system}_Peak" in data.columns:
        dd_series = data[f"{system}_DD"] / data[f"{system}_Peak"]
        sys_dd = round((dd_series.min()) * 100, 2) if not dd_series.isna().all() else 0
        max_dd_value = abs(dd_series.min() * start_bal) if not dd_series.isna().all() else 0
    else:
        sys_dd = 0
        max_dd_value = 0

    # FTMO specific metrics
    challenge_target_reached = total_return >= ftmo.challenge_profit_target * 100
    verification_target_reached = total_return >= ftmo.verification_profit_target * 100
    
    # Trading stats
    if f"{system}_Ret" in data.columns:
        win = data[f"{system}_Ret"] > 1
        loss = data[f"{system}_Ret"] < 1
        trades_triggered = data[f"{system}_Trade"].sum() if f"{system}_Trade" in data.columns else 0
        wins = win.sum()
        losses = loss.sum()
        winrate = round(wins / (wins + losses) * 100, 2) if (wins + losses) > 0 else 0
        
        avg_up_move = (data[f"{system}_Ret"][data[f"{system}_Ret"] > 1].mean() - 1) * 100 if wins > 0 else 0
        avg_down_move = (data[f"{system}_Ret"][data[f"{system}_Ret"] < 1].mean() - 1) * 100 if losses > 0 else 0
        avg_rr = round(abs(avg_up_move / avg_down_move), 2) if avg_down_move != 0 else 0
    else:
        trades_triggered = winrate = avg_rr = 0

    metrics["Start_Balance"] = round(start_bal, 2)
    metrics["Final_Balance"] = round(final_bal, 2)
    metrics["Total_Return"] = round(total_return, 2)
    metrics["Annual_Return"] = sys_cagr
    metrics["Max_Drawdown"] = sys_dd
    metrics["Max_DD_Value"] = round(max_dd_value, 2)
    metrics["Trades"] = trades_triggered
    metrics["Winrate"] = winrate
    metrics["Avg_RR"] = avg_rr
    metrics["Days_Traded"] = days_traded
    metrics["Challenge_Target_Reached"] = challenge_target_reached
    metrics["Verification_Target_Reached"] = verification_target_reached
    metrics["FTMO_Violations"] = len(violations)
    metrics["Account_Closed"] = closed

    return metrics


def analyze_ftmo_phases(trades):
    """Analyze performance by FTMO phase"""
    if len(trades) == 0:
        return pd.DataFrame()
    
    phase_analysis = trades.groupby('FTMO_Phase').agg({
        'PnL': ['count', 'sum', 'mean'],
        'Balance': 'last'
    }).round(2)
    
    return phase_analysis


def analyze_trades(trades, systems):
    """Analyze trades by entry hour"""
    if len(trades) == 0:
        return pd.DataFrame()
        
    # Split index into 'Date' and 'Time' columns
    trades['Entry_Date'] = pd.to_datetime(trades["Entry_Date"])
    trades['Entry_Hour'] = trades['Entry_Date'].dt.hour

    # Convert return to percentage
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


def plot_performance(results, symbols, systems, tp_range, multipliers):
    """Plot performance charts for the backtest results"""
    plt.style.use("dark_background")
    plt.rcParams["figure.figsize"] = (16, 8)
    plt.rcParams.update({"font.size": 18})

    colours = ["tab:olive", "tab:blue", "tab:purple", "tab:orange", "tab:green", "tab:cyan", "tab:red", "tab:gray", "tab:pink"]

    for count, sym in enumerate(symbols):
        plt.figure()
        plt.title(f"Performance of {sym}")

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
        plt.savefig(os.path.join(results_dir, "ftmo_spx500_plot.png"), format="png", dpi=300, bbox_inches="tight")
        plt.show()


def main():
    """Main execution function"""
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    results = []
    prog = 0
    tp_range = np.arange(1.5, 2.0, 0.5)
    multipliers = np.arange(1.0, 1.5, 0.5)
    max_prog = len(symbols) * len(tp_range) * len(multipliers)
    
    print("Starting SPX500 ORB FTMO Swing Challenge backtest...")
    print(f"Challenge Target: {ftmo.challenge_profit_target:.1%}")
    print(f"Max Drawdown (Challenge): {ftmo.challenge_max_loss:.1%}")
    print(f"Daily Loss Limit: {ftmo.challenge_daily_loss_limit:.1%}")
    print(f"Funded Max Loss: {'Disabled' if ftmo.funded_max_loss is None else f'{ftmo.funded_max_loss:.1%}'}")
    print(f"Weekend Holding: {'Yes' if ftmo.weekend_holding_allowed else 'No'}")
    print("-" * 60)
    
    for sym in symbols:
        price = get_price_data(sym)
        for tp_ratio in tp_range:
            for mult in multipliers:
                result, trades, violations, closed, days = backtest(price, tp_ratio, mult)
                results.append(result)
                prog += 1
                print(f"Progress: {round((prog / max_prog) * 100)} %")
    
    # Get metrics for the last result
    if len(trades) > 0:
        sys_metrics = get_ftmo_metrics("Strat", result, violations, closed, days)
        
        print(f"\n=== SPX500 ORB FTMO RESULTS ===")
        print(f"Account Status: {'CLOSED' if closed else 'ACTIVE'}")
        print(f"Days Traded: {days}")
        print(f"Total Return: {sys_metrics['Total_Return']}%")
        print(f"Annual Return: {sys_metrics['Annual_Return']}%")
        print(f"Final Balance: ${sys_metrics['Final_Balance']:,.2f}")
        print(f"Max Drawdown: {sys_metrics['Max_Drawdown']}% (${sys_metrics['Max_DD_Value']:,.2f})")
        print(f"Total Trades: {sys_metrics['Trades']}")
        print(f"Win Rate: {sys_metrics['Winrate']}%")
        print(f"Avg Risk/Reward: {sys_metrics['Avg_RR']}")
        print(f"Challenge Target (10%): {'✅ PASSED' if sys_metrics['Challenge_Target_Reached'] else '❌ FAILED'}")
        print(f"Verification Target (5%): {'✅ PASSED' if sys_metrics['Verification_Target_Reached'] else '❌ FAILED'}")
        print(f"FTMO Violations: {sys_metrics['FTMO_Violations']}")
        
        if violations:
            print(f"\nViolations:")
            for violation in violations:
                print(f"  - {violation}")
        
        # Phase analysis
        phase_analysis = analyze_ftmo_phases(trades)
        if not phase_analysis.empty:
            print(f"\n=== PHASE ANALYSIS ===")
            print(phase_analysis)
        
        # Analyze trades by hour
        hourly_stats = analyze_trades(trades, systems)
        if not hourly_stats.empty:
            print("\nHourly Statistics:")
            print(hourly_stats)
        
        # Save results to CSV
        trades.to_csv(os.path.join(results_dir, "ftmo_spx500_trades.csv"))
        result.to_csv(os.path.join(results_dir, "ftmo_spx500_result.csv"))
        print(f"\nResults saved to {results_dir}/ftmo_spx500_trades.csv and {results_dir}/ftmo_spx500_result.csv")
        
        # Optional: Plot performance (comment out if not needed)
        # plot_performance(results, symbols, systems, tp_range, multipliers)
        
        return results, trades, sys_metrics
    else:
        print("No trades generated")
        return None, None, None


if __name__ == "__main__":
    results, trades, metrics = main()