import yfinance as yf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
mpl.rcParams['font.family'] = 'serif'



class DataPreprocessing:
    def __init__(self,tickers,start_date,end_date,benchmark = "SPY"):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.priceData = self.download_data()

    def download_data(self):
        ticker_list = copy.deepcopy(self.tickers)
        if self.benchmark:
            ticker_list.append(self.benchmark)
        data = yf.download(ticker_list, start=self.start_date,end=self.end_date)['Close']
        return data

    def clean_data(self,data):
        clean_data = data.dropna()
        return clean_data

    def calculate_returns(self):
        returns = self.priceData.pct_change()
        return returns
    
    def benchmark_returns(self):
        return self.calculate_returns()[self.benchmark]

    def backTest_benchmark(self):
        rts = self.benchmark_returns()
        rts = rts.loc[rts.index >= '2023-12-31']
        return rts

    def get_asset_returns(self):
       return self.calculate_returns()[self.tickers]

    def plot_returns(self):
        #visualize log return data
        plt.figure(figsize=(12,6))
        for ticker in self.tickers:
            plt.plot(self.calculate_returns().index,self.calculate_returns()[ticker],label=ticker)
        plt.title('Log Returns of Assets')
        plt.xlabel('Date')
        plt.ylabel('Log Returns')
        plt.legend()
        plt.show()
    
    def training_data(self):
        full_data = self.calculate_returns()
        training_data = full_data.loc[full_data.index <= '2023-12-31']
        return training_data

    def backTest_data(self):
        full_data = self.calculate_returns()
        bt_data = full_data.loc[full_data.index >= '2023-12-31']
        bt_data = bt_data[self.reduce_correlation()].dropna()
        return bt_data
    
    def reduced_training_data(self):
        return self.training_data()[self.reduce_correlation()].dropna()
    
    def reduced_cum_returns(self):
        cumulative_returns = (1 + self.reduced_training_data()).cumprod() - 1
        plt.figure(figsize=(12, 6))
        for ticker in cumulative_returns.columns:
            plt.plot(cumulative_returns.index, cumulative_returns[ticker]*100, label=ticker)
        plt.title('Cumulative Returns of Each Factor')
        plt.xlabel('Date')
        plt.ylabel('Returns (%)')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=8, fontsize=10, frameon=True)
        plt.grid(which='both', linestyle=':', axis='both')  # Dotted symmetrical grid lines
        #plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Symmetry at y=0
        plt.tight_layout()
        plt.tight_layout()  # Adjust layout to fit legend properly
        plt.show()

    def correlation_matrix(self):
        return self.training_data().corr()

    def plot_full_correlation_matrix(self):
        correlation_matrix = self.correlation_matrix()
        #Display correlation matrix heatmap
        plt.figure(figsize=(8,6))
        # plt.imshow(correlation_matrix,cmap='coolwarm', interpolation='nearest')
        sns.heatmap(correlation_matrix, cmap='coolwarm',annot=True, linewidth=.5,fmt=".2f")
        plt.show()

    def average_correlation(self,asset,selected_assets,corr_matrix):
        if not selected_assets:
            return 0
        return corr_matrix.loc[asset, selected_assets].mean()

    def reduce_correlation(self):
        assets = {
                "Value": ["BRK-B", "JPM", "UNH"],
                "Quality": ["AVGO", "MA", "V"],
                "Small-Cap": ["AXON", "APP","DECK"],
                "Dividend": ['WMB','VICI','OKE'],
                "Low-Vol": ['L','TMUS','ADP'],
                "Momentum": ['WMT','NVDA','COST'],
                "Large-Cap": ['AAPL','MSFT','META'],
                "Profitability":['ASML','RHHVF','LVMHF']
                }

        #Flatten the asset list for all factors
        all_assets = [asset for factor_assets in assets.values() for asset in factor_assets]
        correlation_matrix = self.correlation_matrix()
        correlation_matrix = (correlation_matrix + correlation_matrix.T)/2
        np.fill_diagonal(correlation_matrix.values,1)
        # Asset selection: One per factor with minimal correlation
        selected_assets = []
        for factor, factor_assets in assets.items():
            if len(factor_assets) == 1:
                # Automatically select the single asset for the factor
                selected_assets.append(factor_assets[0])
            else:
                #Calculate average correlation for each asset in the factor
                correlations = {
                    asset: self.average_correlation(asset, selected_assets, correlation_matrix)
                    for asset in factor_assets
                }
                #Select the asset with the lowest average correlation
                best_asset = min(correlations, key=correlations.get)
                selected_assets.append(best_asset)

        # Correlation matrix for the selected assets
        selected_corr_matrix = correlation_matrix.loc[selected_assets, selected_assets]

        # # Visualize the reduced correlation matrix
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(selected_corr_matrix,cmap='coolwarm', annot=True, linewidth=.5,fmt=".2f")
        # plt.title("Reduced Correlation Matrix")
        # plt.show()
        return selected_assets

    def reduced_cov_matrix(self):
        selected_assets = self.reduce_correlation()
        rts = self.correlation_matrix().loc[selected_assets,selected_assets]
        return rts

    def plot_reduced_correlation(self):
        selected_corr_matrix = self.reduced_cov_matrix()#correlation_matrix.loc[selected_assets, selected_assets]
        # Visualize the reduced correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(selected_corr_matrix,cmap='coolwarm', annot=True, linewidth=.5,fmt=".2f")
        plt.title("Reduced Correlation Matrix")
        plt.show()

    def market_cap_weights(self):
        market_caps = {}
        for ticker_symbol in self.reduce_correlation():
            try:
                ticker = yf.Ticker(ticker_symbol)
                market_cap = ticker.info.get('marketCap')
                market_caps[ticker_symbol] = market_cap
            except Exception as e:
                market_caps[ticker_symbol] = None
        market_caps_df = pd.DataFrame(
            list(market_caps.items()), columns=["Ticker", "MarketCap"]
        )
        total_market_cap = market_caps_df['MarketCap'].sum()
        market_caps_df['M_Cap_weights'] = round(market_caps_df['MarketCap']/total_market_cap,4)*100
        return market_caps_df

    def calculate_beta(self):
        asset_returns = self.get_asset_returns()[self.reduce_correlation()]
        asset_returns['Benchmark'] = self.benchmark_returns()
        # Remove rows with NaN
        asset_returns.dropna(inplace=True)

        # Calculate Beta for each ticker
        betas = {}
        for ticker in asset_returns.columns[:-1]:  # Exclude the benchmark column
            covariance = np.cov(asset_returns[ticker], asset_returns['Benchmark'])[0, 1]
            variance = np.var(asset_returns['Benchmark'])
            betas[ticker] = covariance / variance
        return betas

    def capm_implied_returns(self):
        rf = 0.03
        benchmark  = self.benchmark_returns().mean()*252
        capm_returns = self.calculate_beta()
        for tk in capm_returns:
            capm_returns[tk] = rf + capm_returns[tk]*(benchmark-rf)

        return np.array(list(capm_returns.values()))

    def calculate_weight_vector(self,delta,cov_matrix,expected_returns):
        cov_matrix = np.array(cov_matrix)
        expected_returns = np.array(expected_returns)
        scaled_cov = delta*cov_matrix
        inv_scaled_cov = np.linalg.inv(scaled_cov)
        weights = np.dot(inv_scaled_cov,expected_returns)
        # Normalize the weights to ensure they sum to 1
        normalized_weights = weights / np.sum(weights)
        return normalized_weights

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