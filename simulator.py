#!/usr/bin/env python
# -*- coding: latin-1 -*-

"""A Lotka-Volterra simulator
"""

__author__  = "Florent Leclercq"
__version__ = "1.0"
__date__    = "2022"
__license__ = "GPLv3"

class LVsimulator():
    """This class contains a basic Lotka-Volterra simulator
    and the observational simulator.
    """

    def __init__(self, R0, F0, alpha, beta, gamma, delta):
        self.R0=R0
        self.F0=F0
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.delta=delta


    def EEuler(self, t):
        """Solves Lotka-Volterra equations for one prey and one predator species using
        the explicit Euler method.

        R0 and F0 are inputs and are the initial populations of each species
        alpha, beta, gamma, delta are inputs and problem parameters
        t is an input and 1D NumPy array of t values where we approximate y values.
        Time step at each iteration is given by t[n+1] - t[n].
        """
        import numpy as np

        R = np.zeros(len(t)) # Pre-allocate the memory for R
        F = np.zeros(len(t)) # Pre-allocate the memory for F

        R[0] = self.R0
        F[0] = self.F0

        for n in range(len(t)-1):
            dt = t[n+1] - t[n]
            R[n+1] = R[n]*(1 + self.alpha*dt - self.gamma*dt*F[n])
            F[n+1] = F[n]*(1 - self.beta*dt + self.delta*dt*R[n])
        return R, F


class LVobserver():
    """This class contains observational simulators
    for a prey and predator populations.
    """

    def __init__(self, Rtrue, Ftrue, R0, F0):
        self.Rtrue=Rtrue
        self.Ftrue=Ftrue
        self.R0=R0
        self.F0=F0

    @property
    def tmax(self):
        return len(self.Rtrue)

    @property
    def t(self):
        import numpy as np
        return np.arange(self.tmax)

    def signal(self, Rtrue, Ftrue, t, **metaparams):
        """Simulate the signal. The signal is a retarded and non-linear
        observation of the true underlying functions.
        """
        import numpy as np
        P=metaparams["P"]
        Q=metaparams["Q"]
        R0=self.R0
        F0=self.F0

        Rsignal = np.zeros(len(t)) # Pre-allocate the memory for Rsignal
        Fsignal = np.zeros(len(t)) # Pre-allocate the memory for Fsignal

        Rsignal[0] = R0
        Fsignal[0] = F0

        for n in range(len(t)-1):
            dt = t[n+1] - t[n]
            Rsignal[n+1] = Rtrue[n] - P*Rtrue[n]*Ftrue[n]*dt
            Fsignal[n+1] = Ftrue[n] - Q*Rtrue[n]*Ftrue[n]*dt

        return Rsignal, Fsignal

    def Dnoise_cov(self, **metaparams):
        """Covariance matrix for the demographic noise.
        Demographic noise depends only on the observed population.
        """
        import numpy as np
        Rtrue=self.Rtrue
        Ftrue=self.Ftrue
        R=metaparams["R"]
        tmax=self.tmax

        D00=np.diag(Rtrue)
        D01=np.zeros((tmax,tmax))
        D10=np.zeros((tmax,tmax))
        D11=np.diag(Ftrue)
        D=np.block([ [R*D00, R*D01], [R*D10, R*D11] ])

        return D

    def Onoise_cov(self, **metaparams):
        """Covariance matrix for the observational noise.
        Prey and predator populations introduce a noise to the other
        population, and there is also a non-diagonal term
        proportional to the geometric mean of both populations.
        """
        import numpy as np
        Rtrue=self.Rtrue
        Ftrue=self.Ftrue
        S=metaparams["S"]
        T=metaparams["T"]

        O00=np.diag(Ftrue)
        O01=T*np.diag(np.sqrt(Ftrue*Rtrue))
        O10=T*np.diag(np.sqrt(Ftrue*Rtrue))
        O11=np.diag(Rtrue)
        O=np.block([ [S*O00, S*O01], [S*O10, S*O11] ])

        return O

    def noise_cov(self, **metaparams):
        """Full noise covariance matrix.
        """
        Rtrue=self.Rtrue
        Ftrue=self.Ftrue

        # demographic noise
        D=self.Dnoise_cov(**metaparams)

        # observational noise
        O=self.Onoise_cov(**metaparams)

        # total noise covariance matrix
        N=D+O

        return N

    def simulate_obs(self, s, N, **metaparams):
        """Simulate the observational process, assuming that
        data = signal + noise
        """
        import numpy as np
        import scipy.stats as ss
        tmax=self.tmax

        # draw noise
        n = ss.multivariate_normal(mean=np.zeros(2*tmax), cov=N).rvs()

        # data=signal+noise
        data = s+n
        self.Robs = np.arange(tmax)
        self.Fobs = np.arange(tmax)
        self.Rdata = data[0:tmax]
        self.Fdata = data[tmax:2*tmax]

    def censor(self, **metaparams):
        """Censor part of the data during periods
        when preys are not observable."""
        import numpy as np
        mask=metaparams["mask"]

        self.Robs = self.Robs[np.where(mask==0)]
        self.Rdata = self.Rdata[np.where(mask==0)]

    def threshold(self, **metaparams):
        """Threshold the data, excluding negative measurements
        and measurements above a detection maximum.
        """
        import numpy as np
        threshold=metaparams["threshold"]

        # threshold data
        self.Rdata[np.where(self.Rdata>threshold)]=threshold
        self.Fdata[np.where(self.Fdata>threshold)]=threshold
        self.Rdata[np.where(self.Rdata<0)]=0
        self.Fdata[np.where(self.Fdata<0)]=0

    def observe(self, **metaparams):
        """Simulate the observation process using the full data model.
        """
        import numpy as np
        # signal
        Rsignal, Fsignal = self.signal(self.Rtrue, self.Ftrue, self.t, **metaparams)
        s=np.concatenate((Rsignal, Fsignal))

        # noise covariance
        N = self.noise_cov(**metaparams)

        # simulate observation
        self.simulate_obs(s, N, **metaparams)

        # censor and threshold data
        self.censor(**metaparams)
        self.threshold(**metaparams)

    def misspecified_observe(self, **metaparams):
        """Simulate the observation process using a misspecified data model.
        """
        import numpy as np
        Rtrue=self.Rtrue
        Ftrue=self.Ftrue

        # signal: consider that the measurement is a direct observation of the function
        Rsignal, Fsignal = Rtrue, Ftrue
        s=np.concatenate((Rsignal, Fsignal))

        # noise covariance: only consider demographic noise
        N = self.Dnoise_cov(**metaparams)

        # simulate observation
        self.simulate_obs(s, N, **metaparams)

        # censor, but do not threshold data
        self.censor(**metaparams)
