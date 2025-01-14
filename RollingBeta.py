import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'

# Define the factor ETFs and corresponding stocks
tickers = {
    "VTV": "BRK-B",
    "SPHQ": "AVGO",
    "VB": "AXON",
    "SPHD": "WMB",
    "SPLV": "TMUS",
    "MTUM": "ORCL",
    "VV": "GOOGL",
    "DIHP": "TTFNF"
}

title = ['Value','Quality','Small-Cap','High-Dividend','Low-Vol','Momentum','Large-Cap','Profitability']

# Define date range for the analysis
start_date = "2021-01-01"
end_date = "2023-12-31"

# Download historical price data
all_tickers = list(tickers.keys()) + list(tickers.values())
data = yf.download(all_tickers, start=start_date, end=end_date)["Close"]

# Calculate daily returns and handle missing data
returns = data.pct_change().dropna()

# Initialize a dictionary to store rolling betas
rolling_betas = {}
average_betas = {}
window_size = 5
epsilon = 1e-6  # Small constant to prevent division by zero

# Compute rolling beta for each stock-factor pair
for factor, stock in tickers.items():
    # Ensure proper alignment and drop NaN values
    combined_returns = returns[[factor, stock]].dropna()

    # Calculate rolling covariance and variance
    cov = combined_returns[factor].rolling(window_size).cov(combined_returns[stock])
    var = combined_returns[factor].rolling(window_size).var()

    # Calculate rolling beta and handle small variances
    beta = cov / (var + epsilon)
    
    # Filter out large outliers
    lower_bound = beta.quantile(0.30)
    upper_bound = beta.quantile(0.70)
    beta_filtered = beta[(beta >= lower_bound) & (beta <= upper_bound)]
    
    rolling_betas[stock] = beta_filtered
    average_betas[stock] = beta_filtered.mean()

# Plot rolling betas in a 4x2 layout
fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True, sharey=True)
axes = axes.flatten()

for i, (stock, beta) in enumerate(rolling_betas.items()):
    avg_beta = average_betas[stock]
    axes[i].plot(beta, label=f"{stock} Beta", color='b')
    axes[i].axhline(1, color='r', linestyle='--', linewidth=1, label="Beta=1")
    axes[i].axhline(avg_beta, color='g', linestyle='-.', linewidth=1, label=f"Avg Beta={avg_beta:.2f}")
    
    # Reduce the font size of the title
    axes[i].set_title(f"{title[i]}", fontsize=10)  # Reduced font size for titles
    
    # Set the font size for the y-axis label
    axes[i].set_ylabel("Rolling Beta", fontsize=10)  # Reduced font size for y-axis label
    
    # Reduce the font size of the legend
    axes[i].legend(fontsize=8)  # Reduced font size for legend
    axes[i].grid()


# Adjust layout and add global labels
fig.suptitle("5-day Rolling Beta", fontsize=12)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.xlabel("Date")
plt.show()

# # Justification for stock selection
# print("Justification for Stock Selection:")
# for factor, stock in tickers.items():
#     print(f"- {stock} is chosen as a proxy for the {factor} factor because of its historical alignment with the factor's performance, as evidenced by consistent rolling beta values near 1 over the analyzed period.")
