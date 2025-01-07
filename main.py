from MarketData import DataPreprocessing
from Optimisation import Optimisation
from pprint import pprint
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
from PerformanceAnalysis import (equity_curve)
from BL import BlackLitterman

tickers=['BRK-B','JPM','UNH','AVGO','MA','V','AXON','APP','DECK',
        'WMB','VICI','OKE','L','TMUS','ADP','WMT','NVDA','COST',
        'AAPL','MSFT','META','ASML','RHHVF','LVMHF']

if __name__ == "__main__":
    #Data pre-processing
    preprocessor = DataPreprocessing(tickers=tickers, start_date='2021-01-01', end_date='2024-12-31')
    training_data = preprocessor.reduced_training_data()
    training_data_cov_matrix = np.array(preprocessor.training_data_cov_matrix())
    market_weight = np.array(preprocessor.market_cap_weights()['M_Cap_weights'])

    risk_aversion = 3.065

    BL = BlackLitterman(weight=market_weight,
                     asset_covariance=training_data_cov_matrix,
                     risk_aversion=risk_aversion)
    print(BL.get_expected_return())
    #Optimisation methods
    # optimisation = Optimisation(returns=training_data,cov_matrix=training_data_cov_matrix)
    # optimisation_weights = optimisation.optimisation_summary()
    ##Black-Litterman

    ##Back test and Performance Analysis
    # backTest_data = preprocessor.backTest_data()
    # bt_benchmark = preprocessor.backTest_benchmark()
    # equity_curve(backTest_data,optimisation_weights,bt_benchmark)

    
    