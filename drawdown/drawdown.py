# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import linprog


class PortfolioSeries(object):
    """
    Portfolio optimization given drawdown restriction
    
    Attributes
    ----------
    series : numpy.ndarray
        The portfolio series
    ntimes : int
        The number of measured times
    dim : int
        The number of portfolio dimensions
    choices : None or numpy.ndarray
        The optimal portfolio alocation (after solving)
    """
    def __init__(self, series):
        """
        Initializer

        Parameters
        ----------
        series : numpy.ndarray
            (N, D) array, where each column d in D is a 
            different asset in the portfolio, 
            and each row n in N is the relative price at t
            (relative to buy price)
        """
        self.series = series
        self.ntimes, self.dim = self.series.shape
        self.choices = None
        
    def solve_portfolio(self, gamma=0.1, lambd1=1.0, lambd2=1.0):
        """
        Solve the portfolio optimization problem, 
        given drawdown restriction

        Parameters
        ----------
        gamma : float, optional
            The drawdown tolerance. The default is 0.1.
        lambd1 : float, optional
            The scaling factor for original problem. The default is 1.0.
        lambd2 : float, optional
            The scaling factor for auxiliary problem. The default is 1.0.

        Returns
        -------
        None. The results will be found in self.choices

        """
        # Objective: maximize x_T^T \alpha,
        # subject to:
        #     \alpha in the simplex
        #     (1-gamma) \max_{s \in {0,\ldots,t}}{x_s^T \alpha} - x_t^T \alpha <= 0, all t
        # The problem is reformulated as:
        #     minimize (-lambd1*x_T,lambd2*\1)^T (\alpha,m)
        #     s.t.
        #         -\alpha \leq 0
        #         (1-\gamma) m_t - x_t^T^\alpha \leq 0, t=1,...,T
        #         m_t - m_{t+1} \leq 0 t=1,...,T-1
        #         x_t^T \alpha -m_t \leq 0, t=2,...,T
        #         \alpha^T \1 - 1 = 0
        #         m_1 - \alpha^T x_1 = 0
        
        # Make A_ub
        D = self.dim
        N = self.ntimes
        neg_id1 = -1*np.eye(D)
        A_ub1 = np.block([neg_id1, np.zeros((D, N))])
        gamma_id2 = (1-gamma)*np.eye(N)
        A_ub2 = np.block([-self.series, gamma_id2])
        block2 = np.eye(N-1, N) - np.eye(N-1, N, k=1)
        A_ub3 = np.block([np.zeros((N-1, D)), block2])
        neg_id4 = -1.0*np.eye(N)
        A_ub4 = np.block([self.series, neg_id4])[1:, :]
        A_ub = np.block([[A_ub1], [A_ub2], [A_ub3], [A_ub4]])
        b_ub = np.zeros(A_ub.shape[0])
        # Make A_eq
        A_eq1 = np.block([np.ones((1, D)), np.zeros((1, N))])
        m_term_eq2 = np.zeros((1, N))
        m_term_eq2[0, 0] = -1.0
        A_eq2 = np.block([self.series[:1, :], m_term_eq2])
        A_eq = np.block([[A_eq1], [A_eq2]])
        b_eq = np.array([1.0, 0.0])

        # Objective array
        c = np.hstack([-lambd1*self.series[-1, :], lambd2*np.ones(N)])
        optres = linprog(c, A_ub, b_ub, A_eq, b_eq)
        self.choices = optres.x[:D]
        return optres

    def return_series(self):
        """
        Returns the optimized portfolio value
        """
        assert hasattr(self, 'choices')
        return (self.choices*self.series).sum(axis=-1)
