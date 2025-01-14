import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams['font.family'] = 'serif'
from sklearn.linear_model import LinearRegression

def BL_statistics(bt_data, ptf_weights):
    """
    Compute portfolio statistics including Variance, Standard Deviation, Sharpe Ratio, 
    Information Ratio, Alpha, and Beta for each strategy.
    
    Args:
    bt_data (DataFrame): Back-test data of asset returns.
    ptf_weights (DataFrame): Portfolio weights for each strategy (percentage values).

    Returns:
    DataFrame: A DataFrame containing the statistics for each strategy.
    """
    # Align weights with the index
    ptf_weights = ptf_weights.set_index('Asset')
    
    # Calculate back-tested returns for each strategy
    bt_returns = {}
    for strategy in ptf_weights.columns:
        bt_returns[strategy] = bt_data.apply(
            lambda row: np.dot(row, ptf_weights[strategy] / 100), axis=1
        )
    bt_returns = pd.DataFrame(bt_returns, index=bt_data.index)

    # Set Strategy2 as the benchmark
    benchmark = bt_returns['M_Cap_weights']

    # Initialize list to store statistics
    stats_list = []

    # Compute statistics for each strategy
    for strategy in bt_returns.columns:
        strategy_returns = bt_returns[strategy]
        
        # Annualize metrics assuming 252 trading days
        avg_return = strategy_returns.mean() * 252
        variance = strategy_returns.var() * 252
        std_dev = strategy_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio (assuming zero risk-free rate)
        sharpe_ratio = avg_return / std_dev

        # Information Ratio
        tracking_error = (strategy_returns - benchmark).std() * np.sqrt(252)
        information_ratio = (avg_return - benchmark.mean() * 252) / tracking_error

        # Alpha and Beta (using Strategy2 as benchmark)
        regression = LinearRegression().fit(
            benchmark.values.reshape(-1, 1), strategy_returns.values
        )
        beta = regression.coef_[0]
        alpha = regression.intercept_ * 252

        # Append metrics to the stats list
        stats_list.append({
            'Strategy': strategy,
            'Average Return (%)': avg_return * 100,
            'Variance (%)': variance * 100,
            'Standard Deviation (%)': std_dev * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Tracking Error (%)': tracking_error * 100,
            'Information Ratio': information_ratio,
            'Alpha (%)': alpha * 100,
            'Beta': beta
        })
    
    # Convert stats list to DataFrame
    statistics_df = pd.DataFrame(stats_list)
    return statistics_df.T
    
def equity_curve(bt_data,ptf_weights,benchmark):
    ptf_weights = ptf_weights.set_index('Securities')
    bt_returns = {}
    cumulative_returns = {} 
    colors = sns.color_palette("tab10", n_colors=9)
    for strategy in ptf_weights.columns:
        bt_returns[strategy] = bt_data.apply(lambda row: np.dot(row, list(ptf_weights[strategy]/100)), axis=1)
        cumulative_returns[strategy] = (1 + bt_returns[strategy]).cumprod() - 1

    cumulative_returns['Benchmark'] = (1+benchmark).cumprod()-1
    # Plot the cumulative returns
    plt.figure(figsize=(12, 6))
    for i,x in enumerate(cumulative_returns):
        rts = cumulative_returns[x]*100
        rts.plot(label=x,color=colors[i])

    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns (%)')
    plt.title('Cumulative Returns of Strategies vs Benchmark')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=8, fontsize=10, frameon=True)
    plt.grid(which='both', linestyle=':', axis='both')  # Dotted symmetrical grid lines
    #plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Symmetry at y=0
    plt.tight_layout()
    plt.tight_layout()  # Adjust layout to fit legend properly
    plt.show()