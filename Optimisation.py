import statsmodels.api as sm
import warnings
import numpy as np
from pylab import mpl,plt
import pandas as pd
import scipy.optimize as sco
from MarketData import DataPreprocessing

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'Sarif'

class Optimisation:
    def __init__(self,returns,cov_matrix):
        self.returns_data = returns
        self.tickers = returns.columns
        self.cov_matrix = cov_matrix
        self.CONSTRAINTS = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        self.BOUNDARY = tuple((0,1) for x in range(len(self.tickers)))
        self.INTIAL_GUESS = np.array(len(self.tickers)*[1./len(self.tickers),])

    def mean_variance_func(self, weights, lambda_risk=2):

        #cov_matrix = self.returns_data.cov()*252
        # Expected portfolio return (calculated using the portfolio_returns method)
        portfolio_return = self.portfolio_returns(weights)
        # Portfolio variance (risk)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        # Objective: Maximize return - lambda * risk
        objective = -(portfolio_return - lambda_risk * portfolio_variance)
        return objective

    def plot_distribution_historgram(self):
        self.returns_data.hist(bins=40,figsize=(12,20))
        plt.show()
    
    def portfolio_returns(self,weights):
        return np.sum(self.returns_data.mean()*weights)*252
    
    def portfolio_volatility(self,weights):
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix,weights)))

    def simulate_portfolios(self):
        NUMBER_OF_SIMULATIONS = 5000
        port_returns = []
        portfolio_volatility = []
        for sample_portfolio in range(NUMBER_OF_SIMULATIONS):
            weights = np.random.random(len(self.tickers))
            weights /= np.sum(weights)
            port_returns.append(self.portfolio_returns(weights))
            portfolio_volatility.append(self.portfolio_volatility(weights))

        port_returns = np.array(port_returns)
        port_vol = np.array(portfolio_volatility)
        return {
                "Returns":port_returns,
                "Volatility":port_vol
        }

    def max_sharpe_func(self,weights):
        return -self.portfolio_returns(weights)/self.portfolio_volatility(weights)

    @classmethod
    def calculate_risk_contributions(cls,weights, cov_matrix):
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contributions = np.dot(cov_matrix, weights) / portfolio_std
        risk_contributions = weights * marginal_contributions
        return risk_contributions

    # Define risk parity objective function
    def risk_parity_func(self,weights,cov_matrix):
        risk_contributions = self.calculate_risk_contributions(weights, cov_matrix)
        target_contribution = risk_contributions.mean()
        relative_diff = (risk_contributions - target_contribution) / target_contribution
        return np.sum(relative_diff ** 2)

    def risk_parity_optimisation(self):
        # Optimize weights
        result = sco.minimize(
            self.risk_parity_func, self.INTIAL_GUESS, args=(self.cov_matrix), method='SLSQP',
            bounds=self.BOUNDARY, constraints=self.CONSTRAINTS,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        return result

    def sharpe_optimisation(self):
        return sco.minimize(self.max_sharpe_func,self.INTIAL_GUESS,method='SLSQP',constraints=self.CONSTRAINTS,bounds=self.BOUNDARY)
    
    def variance_optimisation(self):
        return sco.minimize(self.portfolio_volatility,self.INTIAL_GUESS,method='SLSQP',bounds=self.BOUNDARY,constraints=self.CONSTRAINTS)

    def markowitz_optimisation(self):
        return sco.minimize(self.mean_variance_func,self.INTIAL_GUESS,method='SLSQP',constraints=self.CONSTRAINTS,bounds=self.BOUNDARY)

    def optimisation_summary(self):
        summary = {}
        eweights = np.array(len(self.tickers)*[1./len(self.tickers),])#equal weights
        weights = pd.DataFrame()
        weights['Securities'] = self.tickers
        weights['Equal Allocation'] = np.round(eweights,4)*100
        weights['Max Sharpe'] = np.round(self.sharpe_optimisation().x,4)*100
        weights['Minimum Variance'] = np.round(self.variance_optimisation().x,4)*100
        weights['Risk Parity'] = np.round(self.risk_parity_optimisation().x,4)*100
        weights['Mean Variance'] = np.round(self.markowitz_optimisation().x,4)*100

        performance = pd.DataFrame()
        performance['Portfolio'] = ['Equal Weight','Max Sharpe','Minimum Variance']
        performance['Returns'] = [self.portfolio_returns(eweights),
                                            self.portfolio_returns(self.sharpe_optimisation().x),
                                            self.portfolio_returns(self.variance_optimisation().x)
                                            ]
        performance['Volatility'] = [self.portfolio_volatility(eweights),
                                            self.portfolio_volatility(self.sharpe_optimisation().x),
                                            self.portfolio_volatility(self.variance_optimisation().x)
                                            ]
        performance.Returns = np.round(performance.Returns,4)*100
        performance.Volatility = np.round(performance.Volatility,4)*100

        performance['Sharpe'] = performance['Returns']/performance['Volatility']
        performance = performance.set_index('Portfolio')
        summary = {'Allocation':weights,
                'Performance':performance}
        return weights

    def plot_efficient_frontier(self):
        port_simulation_results = self.simulate_portfolios()
        eweights = np.array(len(self.tickers)*[1./len(self.tickers),])#equal weights
        plt.figure(figsize=(10,6))
        plt.scatter(np.round(port_simulation_results['Volatility']*100,4),
                    np.round(port_simulation_results['Returns']*100,4),
                    c=port_simulation_results['Returns']/port_simulation_results['Volatility'],
                    marker='o',
                    cmap='coolwarm')
        plt.plot(self.portfolio_volatility(self.sharpe_optimisation().x)*100,
                self.portfolio_returns(self.sharpe_optimisation().x)*100,'r*',markersize=5.0,label='Max Sharpe')

        plt.plot(self.portfolio_volatility(eweights)*100,
                self.portfolio_returns(eweights)*100,'^',color='orange',markersize=5.0,label='Equally weighted')

        plt.plot(self.portfolio_volatility(self.variance_optimisation().x)*100,
                self.portfolio_returns(self.variance_optimisation().x)*100,'bo',markersize=5.0,label='Minimum Variance')

        plt.title('Portfolio Optimization')
        plt.legend()  # Displays the legend
        plt.xlabel('Expected Risk(Volatility) (%)')
        plt.ylabel('Expected Return (%)')
        plt.colorbar(label='Sharpe ratio')
        plt.show()

    # def equity_curve(self):
    #     portfolio_returns = (
    #         self.price_data
    #         .pct_change()
    #         .dropna()
    #     )
    #     eweights = np.array(len(self.tickers)*[1./len(self.tickers),])#equal weights
    #     equally_weighted = portfolio_returns.apply(lambda row: np.dot(row, eweights), axis=1)
    #     max_sharpe = portfolio_returns.apply(lambda row: np.dot(row, self.sharpe_optimisation().x), axis=1)
    #     min_variance = portfolio_returns.apply(lambda row: np.dot(row, self.variance_optimisation().x), axis=1)

    #     # Calculate cumulative returns for both benchmark and portfolio
    #     cumulative_portfolio_returns = (1 + max_sharpe).cumprod() - 1
    #     cumulative_equalWeighted_returns = (1 + equally_weighted).cumprod() - 1
    #     cumulative_minimumVariance_returns = (1 + min_variance).cumprod() - 1

    #     # Plot the cumulative returns
    #     plt.figure(figsize=(12, 6))
    #     cumulative_portfolio_returns.plot(label='Optimized')
    #     cumulative_equalWeighted_returns.plot(label='Equally Weighted')
    #     cumulative_minimumVariance_returns.plot(label='Minimum Variance')

    #     plt.xlabel('Date')
    #     plt.ylabel('Cumulative Returns')
    #     plt.title('Cumulative Returns of Portfolio vs Benchmark')
    #     plt.legend()
    #     plt.show()