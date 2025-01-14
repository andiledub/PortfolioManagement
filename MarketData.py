import yfinance as yf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
mpl.rcParams['font.family'] = 'serif'
sns.set(style="darkgrid")
from sklearn.covariance import LedoitWolf


stock_factor_dict = {
        "BRK-B": "Value",
        "AVGO": "Quality",
        "AXON": "Small-Cap",
        "WMB": "High-Dividend",
        "TMUS": "Low-Vol",
        "ORCL": "Momentum",
        "GOOGL": "Large-Cap",
        "TTFNF": "Profitability"
        }

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
    
    def training_benchmark(self):
        rts = self.benchmark_returns()
        rts = rts.loc[rts.index <= '2023-12-31']
        return rts

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
        rawData = self.training_data()[self.reduce_correlation()].dropna()
        rawData = rawData.rename(columns=stock_factor_dict)
        return rawData#self.training_data()[self.reduce_correlation()].dropna()
    
    def reduced_cum_returns(self):
        data = self.reduced_training_data()
        data['Benchmark'] = self.training_benchmark()

        cumulative_returns = (1 + data).cumprod() - 1
        plt.figure(figsize=(12, 6))

        # Define 9 distinct colors
        colors = sns.color_palette("tab10", n_colors=9)  # Using Seaborn's "tab10" palette for 9 unique colors

        # Plot each line with a unique color
        for i, ticker in enumerate(cumulative_returns.columns):
            plt.plot(cumulative_returns.index, cumulative_returns[ticker]*100, label=ticker, color=colors[i])

        # Plot customization
        plt.title('Cumulative Returns of Each Factor')
        plt.xlabel('Date')
        plt.ylabel('Returns (%)')
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=8, fontsize=10, frameon=True)
        plt.grid(linestyle=':', color='white')  # Dotted grid lines

        plt.tight_layout()  # Adjust layout to fit legend properly
        plt.show()

    def correlation_matrix(self):
        return self.training_data().corr()

    def plot_full_correlation_matrix(self):
        correlation_matrix = self.correlation_matrix()
        #Display correlation matrix heatmap
        plt.figure(figsize=(15, 12))
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
                "Momentum": ['ORCL','NVDA','COST'],
                "Large-Cap": ['AAPL','MSFT','GOOGL'],
                "Profitability":['BP','UL','TTFNF']
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

    def risk_free_rate(self):
        #data = yf.download(['^IRX'], start='2023-12-31',end='2023-12-31')['Close']
        return 0.04208 

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
        market_caps_df['Factor'] = market_caps_df['Asset'].map(stock_factor_dict)
        market_caps_df['M_Cap_weights'] = market_caps_df['MarketCap']/total_market_cap
        return market_caps_df

    def training_data_cov_matrix(self,annualised=True):
        returns = self.reduced_training_data()
        lw = LedoitWolf()
        assets = returns.columns
        shrunk_cov_matrix = lw.fit(returns.values).covariance_
        cov_matrix = pd.DataFrame(shrunk_cov_matrix,index=assets,columns=assets)
        if annualised:
            return cov_matrix*252
        else:
            return cov_matrix

    def market_aversion(self):
        bench_returns = self.training_benchmark()
        mean_rts = bench_returns.mean()*252 - self.risk_free_rate()
        variance = (bench_returns.var())*252
        return mean_rts/variance
    
    def plot_reduced_covariance(self):
        selected_cov_matrix = self.training_data_cov_matrix()#correlation_matrix.loc[selected_assets, selected_assets]
        # Visualize the reduced correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(selected_cov_matrix, annot=True, linewidth=.5,fmt=".3f")
        plt.title("Covariance Matrix")
        plt.show()

