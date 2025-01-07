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
            list(market_caps.items()), columns=["Asset", "MarketCap"]
        )
        total_market_cap = market_caps_df['MarketCap'].sum()
        market_caps_df['M_Cap_weights'] = market_caps_df['MarketCap']/total_market_cap
        return market_caps_df

    def training_data_cov_matrix(self,annualised=True):
        returns = self.reduced_training_data()
        cov_matrix = returns.cov()
        if annualised:
            return cov_matrix*252
        else:
            return cov_matrix

    # def calculate_beta(self):
    #     asset_returns = self.get_asset_returns()[self.reduce_correlation()]
    #     asset_returns['Benchmark'] = self.benchmark_returns()
    #     # Remove rows with NaN
    #     asset_returns.dropna(inplace=True)

    #     # Calculate Beta for each ticker
    #     betas = {}
    #     for ticker in asset_returns.columns[:-1]:  # Exclude the benchmark column
    #         covariance = np.cov(asset_returns[ticker], asset_returns['Benchmark'])[0, 1]
    #         variance = np.var(asset_returns['Benchmark'])
    #         betas[ticker] = covariance / variance
    #     return betas

    # def capm_implied_returns(self):
    #     rf = 0.03
    #     benchmark  = self.benchmark_returns().mean()*252
    #     capm_returns = self.calculate_beta()
    #     for tk in capm_returns:
    #         capm_returns[tk] = rf + capm_returns[tk]*(benchmark-rf)

    #     return np.array(list(capm_returns.values()))

    # def calculate_weight_vector(self,delta,cov_matrix,expected_returns):
    #     cov_matrix = np.array(cov_matrix)
    #     expected_returns = np.array(expected_returns)
    #     scaled_cov = delta*cov_matrix
    #     inv_scaled_cov = np.linalg.inv(scaled_cov)
    #     weights = np.dot(inv_scaled_cov,expected_returns)
    #     # Normalize the weights to ensure they sum to 1
    #     normalized_weights = weights / np.sum(weights)
    #     return normalized_weights

