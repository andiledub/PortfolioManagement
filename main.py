from MarketData import DataPreprocessing
from Optimisation import Optimisation
from pprint import pprint
import numpy as np
# import matplotlib as mpl
# mpl.rcParams['font.family'] = 'serif'
from PerformanceAnalysis import (equity_curve,BL_statistics)
from BL import BlackLitterman
import pandas as pd


tickers=['BRK-B','JPM','UNH','AVGO','MA','V','AXON','APP','DECK',
        'WMB','VICI','OKE','L','TMUS','ADP','ORCL','NVDA','COST',
        'AAPL','MSFT','GOOGL','BP','UL','TTFNF']

if __name__ == "__main__":
    #Data pre-processing
    preprocessor = DataPreprocessing(tickers=tickers, start_date='2021-01-01', end_date='2024-12-31')
    training_data = preprocessor.reduced_training_data()
    training_data_cov_matrix = np.array(preprocessor.training_data_cov_matrix())
    
    market_weight = preprocessor.market_cap_weights()
    # market_weight["MarketCap"] = market_weight["MarketCap"].apply(lambda x: f"{x:,}")
    # market_weight["M_Cap_weights"] = market_weight["M_Cap_weights"].apply(lambda x: f"{x:.2%}")
    
    risk_aversion = 2.37

    BL = BlackLitterman(weight=np.array(preprocessor.market_cap_weights()['M_Cap_weights']),
                     asset_covariance=training_data_cov_matrix,
                     risk_aversion=risk_aversion)
    equilibrium_return = BL.get_expected_return()

    market_weight['Implied Return'] = equilibrium_return
    market_weight["Implied Return"] = market_weight["Implied Return"].apply(lambda x: f"{x:.2%}")
    #preprocessor.plot_reduced_covariance()

    # view_portfolios = np.array([[0,  0,  0,   0,  0,   0, 1, 0],
    #                         [-1, 1,  0,   0,  0,   0, 0, 0],
    #                         [0,  0, .9, -.9, .1, -.1, 0, 0]])

    view_portfolios = np.array([[0,  0,  0,   0,  0,   0, 1, 0],
                        [-1, 1,  0,   0,  0,   0, 0, 0],
                        [0,  0, 1, -1, 1, -1, 0, 0]])

    # The change from equilibrium that we expect, Q
    view_change = np.array([0.18, 0.015, 0.02])

    # The confidence level we have in each view, C
    view_confidence = np.array([0.25, 0.50, 0.654])

    he_return = BL.he_litterman_return(equilibrium_return=equilibrium_return,view_portfolios=view_portfolios,view_change=view_change)
    # #idz_return = BL.idzorek_return(equilibrium_return = equilibrium_return,risk_free_rate=0,view_portfolios=view_portfolios,view_change=view_change,view_confidence=view_confidence)
    he_litterman_weight = BL.get_weight(he_return)
    # #idzorek_weight = BL.get_weight(idz_return)
    table6 = market_weight[['Asset','Factor','M_Cap_weights']]
    table6['M_Returns'] = equilibrium_return
    table6['New returns'] = he_return
    table6['New weights'] = he_litterman_weight
    table6['Difference E[R] − Π'] = table6['New returns'] - table6['M_Returns']
    table6['Difference wˆ − wmkt'] = table6['New weights'] - table6['M_Cap_weights']
    table6 = table6[['Asset','Factor','New returns','M_Returns','Difference E[R] − Π','New weights','M_Cap_weights','Difference wˆ − wmkt']]
    #table6.to_csv('table6.csv')
    #print(table6)

    # Optimisation methods
    optimisation = Optimisation(returns=training_data,cov_matrix=training_data_cov_matrix)
    optimisation_weights = optimisation.optimisation_summary()
    optimisation_weights['Black-litterman'] = he_litterman_weight*100
    optimisation_weights.to_csv('Optimisation_Weights.csv')
    ##Back test and Performance Analysis
    backTest_data = preprocessor.backTest_data()
    bt_benchmark = preprocessor.backTest_benchmark()
    #equity_curve(backTest_data,optimisation_weights,bt_benchmark)
    ##BT only Black-Litterman
    weights = table6[['Asset','New weights', 'M_Cap_weights']]
    BL_statistics(bt_data=backTest_data,ptf_weights=weights).to_csv('BL_Stats.csv')

    
    