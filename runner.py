from DataPreprocessing import DataProcessing
from BL import BlackLitterman
import numpy as np
import matplotlib.pyplot as plt

# obj = DataProcessing(tickers =['TSM','BABA','BRK-B','JPM','UNH','AVGO','MA','V','AXON','DECK',
#         'MO','KMI','T','WMT','COST','ABBV','PLTR','XOM','CVX','COP','MAA','UMH','FPI',
#         'RIO','NUE','APD'],
#                      start_date='2021-01-30',end_date='2024-11-30')
# print(obj.calculate_market_cap_weights())

# The names of the assets
assets = np.array([
    "US Bonds",
    "International Bonds",
    "US Large Growth",
    "US Large Value",
    "US Small Growth",
    "US Small Value",
    "International Developed Equity",
    "International Emerging Equity"
])

market_weight = np.array([
    19.34,
    26.13,
    12.09,
    12.09,
    1.34,
    1.34,
    24.18,
    3.49
])/100

# The covariance matrix for the assets
asset_covariance = np.array([[0.001005, 0.001328, -0.000579, -0.000675, 0.000121, 0.000128, -0.000445, -0.000437],
                             [0.001328, 0.007277, -0.001307, -0.000610, -0.002237, -0.000989, 0.001442, -0.001535],
                             [-0.000579, -0.001307, 0.059852, 0.027588, 0.063497, 0.023036, 0.032967, 0.048039],
                             [-0.000675, -0.000610, 0.027588, 0.029609, 0.026572, 0.021465, 0.020697, 0.029854],
                             [0.000121, -0.002237, 0.063497, 0.026572, 0.102488, 0.042744, 0.039943, 0.065994],
                             [0.000128, -0.000989, 0.023036, 0.021465, 0.042744, 0.032056, 0.019881, 0.032235],
                             [-0.000445, 0.001442, 0.032967, 0.020697, 0.039943, 0.019881, 0.028355, 0.035064],
                             [-0.000437, -0.001535, 0.048039, 0.029854, 0.065994, 0.032235, 0.035064, 0.079958]])

risk_aversion = 3.065

BL = BlackLitterman(weight=market_weight,
                     asset_covariance=asset_covariance,
                     risk_aversion=risk_aversion)
equilibrium_return = BL.get_expected_return()

view_portfolios = np.array([[0,  0,  0,   0,  0,   0, 1, 0],
                            [-1, 1,  0,   0,  0,   0, 0, 0],
                            [0,  0, .9, -.9, .1, -.1, 0, 0]])

# The change from equilibrium that we expect, Q
view_change = np.array([0.0525, 0.0025, 0.02])

# The confidence level we have in each view, C
view_confidence = np.array([0.25, 0.50, 0.65])

he_return = BL.he_litterman_return(equilibrium_return=equilibrium_return,view_portfolios=view_portfolios,view_change=view_change)
idz_return = BL.idzorek_return(equilibrium_return = equilibrium_return,risk_free_rate=0,view_portfolios=view_portfolios,view_change=view_change,view_confidence=view_confidence)

he_litterman_weight = BL.get_weight(he_return)
idzorek_weight = BL.get_weight(idz_return)

x = np.arange(len(assets))
bar_width = 0.1

fig, ax = plt.subplots()
market_b = ax.bar(x + bar_width * -1.5, market_weight*100, bar_width, label="Market Capitalisation Weight")
he_lit_b = ax.bar(x + bar_width * -0.5, he_litterman_weight * 100, bar_width, label="He-Litterman Weight")
idz_b = ax.bar(x + bar_width * 0.5, idzorek_weight * 100, bar_width, label="Idzorek Weight")

ax.set_ylabel("Portfolio Weight (%)")
# ax.set_xticks(x)
# ax.set_xticklabels(assets)
ax.legend()

plt.show()