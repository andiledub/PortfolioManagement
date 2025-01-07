import numpy as np
import matplotlib.pyplot as plt

class BlackLitterman:
    def __init__(self,weight,asset_covariance,risk_aversion):
        self.asset_weight = weight
        self.asset_covariance = asset_covariance
        self.risk_aversion = risk_aversion

    def black_litterman_return(self,equlibrium_return,tau,view_portfolios,view_change,view_uncertainty):
        ER = equlibrium_return
        S = self.asset_covariance
        t = tau
        P = view_portfolios
        Q = view_change
        O = view_uncertainty

        tS_inv = np.linalg.inv(t*S)
        O_inv = np.linalg.inv(O)

        return np.linalg.inv(tS_inv + P.T @ O_inv @ P) @ (tS_inv @ ER + P.T @ O_inv @ Q)
    
    
    def get_expected_return(self):
        return self.risk_aversion * self.asset_covariance @ self.asset_weight


    def get_weight(self,expected_return):
        return np.linalg.inv(self.risk_aversion*self.asset_covariance) @ expected_return
    
    def he_litterman_omega(self,tau,view_portfolios):
        S = self.asset_covariance
        t = tau
        P = view_portfolios
        K,N = P.shape
        O = np.zeros((K,K))

        for i in range(K):
            O[i,i] = (P[i,:] @ S @ P[i,:].T) * t
        
        return O
    
    def he_litterman_return(self,equilibrium_return,view_portfolios,view_change):
        tau = 1
        omega = self.he_litterman_omega(tau,view_portfolios)
        return self.black_litterman_return(equilibrium_return,tau,view_portfolios,view_change,omega)
    
    def idzorek_omega(self,equilibrium_return,tau,risk_free_rate,view_portfolios,view_change,view_confidence):
        ER = equilibrium_return
        S = self.asset_covariance
        t = tau
        rr = risk_free_rate
        L = self.risk_aversion
        P = view_portfolios
        Q = view_change
        C = view_confidence
        
        w_mkt = np.linalg.inv(L * S) @ ER
        
        K, N = P.shape
        O = np.zeros((K, K))
        
        # We treat each view separately
        for i in range(K):
            p = P[i,:]
            q = Q[i] - ((p.sum() != 0) * rr)
            c = C[i]
            
            D = (t/L) * (q - p @ ER) * p / (p @ S @ p.T)
            Tilt = D * ((p != 0) * c)
            
            w_target = w_mkt + Tilt
            
            def w(o):
                return np.linalg.inv(np.identity(N)/t + np.outer(p, p) @ S / o) @ (w_mkt / t + p * q / o / L)
            
            def sum_squared_difference(o):
                diff = w_target - w(o)
                return np.dot(diff, diff)
            
            O[i,i] = min(np.linspace(0,1,1000)[1:], key=sum_squared_difference)
            
        return O
    
    # A wrapper for the Black-Litterman model that uses Idzorek's new method for calculating omega
    def idzorek_return(self,equilibrium_return,risk_free_rate,view_portfolios,view_change,view_confidence):
        tau = 1
        return self.black_litterman_return(
            equilibrium_return,
            tau,
            view_portfolios,
            view_change,
            self.idzorek_omega(
                equilibrium_return,
                tau,
                risk_free_rate,
                view_portfolios,
                view_change,
                view_confidence
            )
        )
    
    # The return based on 100% confidence in the views
    def certain_return(self,equilibrium_return,asset_covariance,view_portfolios,view_change):
        ER = equilibrium_return
        S = asset_covariance
        P = view_portfolios
        Q = view_change
        return ER + S @ P.T @ np.linalg.inv(P @ S @ P.T) @ (Q - P @ ER)
