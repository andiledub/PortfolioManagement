import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Download price data
tickers = ['BRK-B','AVGO','AXON','WMB','TMUS','WMT','META','RHHVF']

data = yf.download(tickers, start='2021-01-01', end='2023-12-31')['Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Covariance matrix
cov_matrix = returns.cov()

# Function to calculate risk contributions
def calculate_risk_contributions(weights, cov_matrix):
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    marginal_contributions = np.dot(cov_matrix, weights) / portfolio_std
    risk_contributions = weights * marginal_contributions
    return risk_contributions

# Define risk parity objective function
def risk_parity_obj(weights, cov_matrix):
    risk_contributions = calculate_risk_contributions(weights, cov_matrix)
    target_contribution = risk_contributions.mean()
    relative_diff = (risk_contributions - target_contribution) / target_contribution
    return np.sum(relative_diff ** 2)

# Initial weights
init_weights = np.array([1 / len(tickers)] * len(tickers))

# Constraints: sum of weights = 1
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

# Bounds: weights between 0 and 1
bounds = [(0, 1) for _ in range(len(tickers))]

# Optimize weights
result = minimize(
    risk_parity_obj, init_weights, args=(cov_matrix), method='SLSQP',
    bounds=bounds, constraints=constraints,
    options={'maxiter': 1000, 'ftol': 1e-9}
)

# Check optimization success
if result.success:
    risk_parity_weights = result.x
else:
    print("Optimization did not converge. Try adjusting parameters.")
    risk_parity_weights = init_weights

# Calculate risk contributions after optimization
optimized_risk_contributions = calculate_risk_contributions(risk_parity_weights, cov_matrix)

# Verify risk parity
print("Optimized Risk Contributions (should be equal):")
print(optimized_risk_contributions)

# Plotting optimized risk contributions
plt.figure(figsize=(12, 6))
plt.bar(tickers, optimized_risk_contributions, color="skyblue")
plt.axhline(y=optimized_risk_contributions.mean(), color='red', linestyle='--', label='Target Risk Contribution')
plt.title("Risk Contributions After Optimization (Risk Parity)")
plt.xlabel("Assets")
plt.ylabel("Risk Contribution")
plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# Display weights and risk contributions
portfolio = pd.DataFrame({
    "Ticker": tickers,
    "Optimized Weight": risk_parity_weights,
    "Optimized Risk Contribution": optimized_risk_contributions
})

print("Portfolio Weights and Risk Contributions:")
print(portfolio)