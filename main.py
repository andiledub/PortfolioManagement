from MarketData import DataPreprocessing
from Optimisation import Optimisation
from pprint import pprint
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
from PerformanceAnalysis import (equity_curve)

tickers=['BRK-B','JPM','UNH','AVGO','MA','V','AXON','APP','DECK',
        'WMB','VICI','OKE','L','TMUS','ADP','WMT','NVDA','COST',
        'AAPL','MSFT','META','ASML','RHHVF','LVMHF']

if __name__ == "__main__":
    preprocessor = DataPreprocessing(tickers=tickers, start_date='2021-01-01', end_date='2024-12-31')
    training_data = preprocessor.reduced_training_data()
    optimisation = Optimisation(returns=training_data)
    optimisation_weights = optimisation.optimisation_summary()
    ###Back test and Performance Analysis
    backTest_data = preprocessor.backTest_data()
    bt_benchmark = preprocessor.backTest_benchmark()
    print(equity_curve(backTest_data,optimisation_weights,bt_benchmark))
    
    