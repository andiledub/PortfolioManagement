import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn as sns

mpl.rcParams['font.family'] = 'serif'

def equity_curve(bt_data,ptf_weights,benchmark):
    ptf_weights = ptf_weights.set_index('Securities')
    bt_returns = {}
    cumulative_returns = {} 
    for strategy in ptf_weights.columns:
        bt_returns[strategy] = bt_data.apply(lambda row: np.dot(row, list(ptf_weights[strategy]/100)), axis=1)
        cumulative_returns[strategy] = (1 + bt_returns[strategy]).cumprod() - 1

    cumulative_returns['Benchmark'] = (1+benchmark).cumprod()-1
    # Plot the cumulative returns
    plt.figure(figsize=(12, 6))
    for x in cumulative_returns:
        cumulative_returns[x].plot(label=x)

    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title('Cumulative Returns of Strategies vs Benchmark')
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=8, fontsize=10, frameon=True)
    plt.grid(which='both', linestyle=':', axis='both')  # Dotted symmetrical grid lines
    #plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Symmetry at y=0
    plt.tight_layout()
    plt.tight_layout()  # Adjust layout to fit legend properly
    plt.show()