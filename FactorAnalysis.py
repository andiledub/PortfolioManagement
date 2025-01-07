# preprocessor = DataPreprocessing(tickers=tickers
#     , start_date='2021-01-01', end_date='2024-12-31')
# print(preprocessor.reduced_cum_returns())


# class FactorAnalysis:
#     def __init__(self,log_returns):
#         self.log_returns = log_returns

#     def exploratory_data_analysis(self):
#         #visualize log return data
#         plt.figure(figsize=(12,6))
#         for ticker in self.log_returns.columns:
#             plt.plot(self.log_returns.index,self.log_returns[ticker],label=ticker)
#         plt.title('Log Returns of Assets')
#         plt.xlabel('Date')
#         plt.ylabel('Log Returns')
#         plt.legend()
#         plt.show()

#     def correlation_matrix(self):
#         return self.log_returns.corr()

#     def average_correlation(self,asset,selected_assets,corr_matrix):
#         if not selected_assets:
#             return 0
#         return corr_matrix.loc[asset, selected_assets].mean()


#     def average_correlation(self,asset,selected_assets,corr_matrix):
#         if not selected_assets:
#             return 0
#         return corr_matrix.loc[asset, selected_assets].mean()

#     def reduce_correlation(self):
#         assets = {
#                 "Emerging": ['TSM','BABA'],
#                 "Value": ["BRK-B", "JPM", "UNH"],
#                 "Quality": ["AVGO", "MA", "V"],
#                 "Small Cap": ["AXON", "DECK"],
#                 "Dividend": ["MO", "KMI", "T"],
#                 "Energy": ["XOM","CVX","COP"],
#                 "Tech": ["PLTR"],
#                 "Momentum": ['WMT','COST','ABBV'],
#                 "Real Estate": ["MAA","UMH","FPI"],
#                 "Material": ["RIO","NUE","APD"]
#                 }

#         #Flatten the asset list for all factors
#         all_assets = [asset for factor_assets in assets.values() for asset in factor_assets]
#         correlation_matrix = self.correlation_matrix()
#         correlation_matrix = (correlation_matrix + correlation_matrix.T)/2
#         np.fill_diagonal(correlation_matrix.values,1)
#         # Asset selection: One per factor with minimal correlation
#         selected_assets = []
#         for factor, factor_assets in assets.items():
#             if len(factor_assets) == 1:
#                 # Automatically select the single asset for the factor
#                 selected_assets.append(factor_assets[0])
#             else:
#                 # Calculate average correlation for each asset in the factor
#                 correlations = {
#                     asset: self.average_correlation(asset, selected_assets, correlation_matrix)
#                     for asset in factor_assets
#                 }
#                 # Select the asset with the lowest average correlation
#                 best_asset = min(correlations, key=correlations.get)
#                 selected_assets.append(best_asset)

#         # Verify the selected assets
#         # print("Selected Assets:", selected_assets)

#         # # Correlation matrix for the selected assets
#         # selected_corr_matrix = correlation_matrix.loc[selected_assets, selected_assets]

#         # # Visualize the reduced correlation matrix
#         # plt.figure(figsize=(8, 6))
#         # #sns.heatmap(selected_corr_matrix, annot=True, cmap="grey", vmin=-1, vmax=1)
#         # sns.heatmap(selected_corr_matrix, annot=True, linewidth=.5,fmt=".2f")
#         # plt.title("Reduced Correlation Matrix")
#         # plt.show()
#         return selected_assets

#     def factor_identification(self):
#         correlation_matrix = self.correlation_matrix()

#         #Display correlation matrix heatmap
#         plt.figure(figsize=(8,6))
#         # plt.imshow(correlation_matrix,cmap='coolwarm', interpolation='nearest')
#         sns.heatmap(correlation_matrix, cmap='coolwarm',annot=True, linewidth=.5,fmt=".2f")
#         plt.show()

# #Perform Factor Analysis on log returns data
# factor_analysis = FactorAnalysis(log_returns)
# reduced_tickers = factor_analysis.reduce_correlation()
# df = preprocessor.download_market_cap(reduced_tickers)
# total_market_cap = df['MarketCap'].sum()
# # Calculate the Market Cap Weight
# df['M_cap_weight'] = round((df['MarketCap'] / total_market_cap)*100,2)
# print(df)

# class FactorModeling:
#     def __init__(self, log_returns, factors):
#         self.log_returns = log_returns
#         self.factors = factors

#     def build_factor_model(self):
#         # Implement factor modeling using regression analysis
#         # Assume a linear regression model: log_returns = alpha + beta1*f1 + beta2*f2 + ... + betaN*fN + error
#         # Where f1, f2, ..., fN are the identified factors

#         # Perform regression analysis for each asset
#         factors_matrix = np.column_stack(
#             (np.ones(len(self.log_returns)), self.factors))
#         results = {}

#         for ticker in self.log_returns.columns:
#             y = self.log_returns[ticker].values
#             betas = np.linalg.lstsq(factors_matrix, y, rcond=None)[0]
#             results[ticker] = betas[1:]

#         return results

#     def validate_model(self):
#         # Implement model validation using statistical tests or out-of-sample testing

#         # Split data into training and testing sets
#         train_data = self.log_returns.iloc[:int(0.8 * len(self.log_returns))]
#         test_data = self.log_returns.iloc[int(0.8 * len(self.log_returns)):]

#         # Train the factor model on training data
#         train_factors = train_data[self.factors]
#         train_model = self.build_factor_model().copy()

#         # Predict log returns on test data
#         test_factors = test_data[self.factors]
#         test_results = {}

#         for ticker in test_data.columns:
#             predicted_returns = np.dot(np.column_stack(
#                 (np.ones(len(test_factors)), test_factors), train_model[ticker]))
#             test_results[ticker] = predicted_returns

#         return test_results