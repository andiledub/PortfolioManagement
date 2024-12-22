import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Define assets grouped by factors
assets = {
    "Growth": ["AAPL", "NVDA", "MSFT"],
    "Value": ["BRK-B", "JPM", "UNH"],
    "Quality": ["AVGO", "MA", "V"],
    "Small Cap": ["AXON", "SW", "DECK"],
    "Dividend": ["MO", "KMI", "T"],
    "Commodity": ["DBC"],
    "Bonds": ["SCHO", "TLT"],
    "Momentum": ["MTUM"],
    "VIX": ["^VIX"]
}

# Flatten the asset list for all factors
all_assets = [asset for factor_assets in assets.values() for asset in factor_assets]

# Generate a mock correlation matrix (replace with real data)
np.random.seed(42)  # For reproducibility
n_assets = len(all_assets)
correlation_matrix = pd.DataFrame(
    np.random.uniform(-1, 1, (n_assets, n_assets)),  # Random correlations
    index=all_assets,
    columns=all_assets
)

# Ensure the matrix is symmetric with ones on the diagonal
correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
np.fill_diagonal(correlation_matrix.values, 1)

# Function to calculate average correlation for an asset
def average_correlation(asset, selected_assets, corr_matrix):
    if not selected_assets:  # If no assets are selected yet, return 0
        return 0
    return corr_matrix.loc[asset, selected_assets].mean()

# Asset selection: One per factor with minimal correlation
selected_assets = []
for factor, factor_assets in assets.items():
    if len(factor_assets) == 1:
        # Automatically select the single asset for the factor
        selected_assets.append(factor_assets[0])
    else:
        # Calculate average correlation for each asset in the factor
        correlations = {
            asset: average_correlation(asset, selected_assets, correlation_matrix)
            for asset in factor_assets
        }
        # Select the asset with the lowest average correlation
        best_asset = min(correlations, key=correlations.get)
        selected_assets.append(best_asset)

# Verify the selected assets
print("Selected Assets:", selected_assets)

# Correlation matrix for the selected assets
selected_corr_matrix = correlation_matrix.loc[selected_assets, selected_assets]

# Visualize the reduced correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(selected_corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Reduced Correlation Matrix")
plt.show()

# Total correlation for selected assets (optional for validation)
def total_correlation(selected, corr_matrix):
    return sum(
        corr_matrix.loc[asset1, asset2]
        for asset1, asset2 in combinations(selected, 2)
    ) / len(list(combinations(selected, 2)))

print("Total Average Correlation:", total_correlation(selected_assets, correlation_matrix))