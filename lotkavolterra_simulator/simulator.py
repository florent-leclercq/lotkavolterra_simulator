#!/usr/bin/env python

"""A Lotka-Volterra Bayesian hierarchical model simulator
"""

__author__  = "Florent Leclercq"
__version__ = "1.0"
__date__    = "2022"
__license__ = "GPLv3"

class LVsimulator(object):
    """This class contains a basic Lotka-Volterra simulator
    and the observational simulator.

    X0 and Y0 are inputs and are the initial populations of each species
    alpha, beta, gamma, delta are inputs and problem parameters

    Attributes
    ----------
    X0 : double
        initial condition: number of preys
    Y0 : double
        initial condition: number of predators
    alpha : double
        intrinsic reproduction rate of preys (independent of the number of predators)
    beta : double
        prey mortality rate due to predators encountered
    gamma : double
        predator reproduction rate according to preys encountered and eaten
    delta : double
        intrinsic predator mortality rate (independent of the number of prey)

    """

    def __init__(self, X0, Y0, alpha, beta, gamma, delta):
        self.X0=X0
        self.Y0=Y0
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.delta=delta

    def EEuler(self, t):
        """Solves Lotka-Volterra equations for one prey and one predator species using
        the explicit Euler method.

        Parameters
        ----------
        t : array, double, dimension=n
            array of time values where we approximate X and Y values
            timestep at each iteration is given by t[n+1] - t[n].

        Returns
        -------
        X : array, double, dimension=n
            number of preys
        Y : array, double, dimension=n
            number of predators

        """
        import numpy as np

        X = np.zeros(len(t)) # Pre-allocate the memory for X
        Y = np.zeros(len(t)) # Pre-allocate the memory for Y

        X[0] = self.X0
        Y[0] = self.Y0

        for n in range(len(t)-1):
            dt = t[n+1] - t[n]
            X[n+1] = X[n]*(1 + self.alpha*dt - self.beta*dt*Y[n])
            Y[n+1] = Y[n]*(1 - self.delta*dt + self.gamma*dt*X[n])
        return X, Y
#end class(LVsimulator)

class LVobserver(object):
    """This class contains observational simulators
    for a prey and predator populations.

    Attributes
    ----------
    Xtrue : array, double, dimension=n
        true number of preys to be observed
    Ytrue : array, double, dimension=n
        true number of predators to be observed
    X0 : double
        initial condition: number of preys
    Y0 : double
        initial condition: number of predators

    """

    def __init__(self, Xtrue, Ytrue, X0, Y0):
        self.Xtrue=Xtrue
        self.Ytrue=Ytrue
        self.X0=X0
        self.Y0=Y0

    @property
    def tmax(self):
        """double: number of timesteps (each of length 1).
        """
        return len(self.Xtrue)

    @property
    def t(self):
        """array, double: timestepping of the problem.
        """
        import numpy as np
        return np.arange(self.tmax)

    def make_signal(self, Xtrue, Ytrue, t, **metaparams):
        """Simulate the signal. The (unobserved) signal is a retarded and non-linear
        observation of the true underlying functions.

        Parameters
        ----------
        Xtrue : array, double, dimension=n
            true number of preys to be observed
        Ytrue : array, double, dimension=n
            true number of predators to be observed
        t : array, double, dimension=n
            array of time values where we compute Xsignal and Ysignal values
            timestep at each iteration is given by t[n+1] - t[n].
        P : double
            rate of prey misses due to correlation between preys and predators
        Q : double
            rate of predator misses due to correlation between preys and predators

        Returns
        -------
        Xsignal : array, double, dimension=n
            unobserved signal for the number of preys
        Ysignal : array, double, dimension=n
            unobserved signal for the number of predators

        """
        import numpy as np
        P=metaparams["P"]
        Q=metaparams["Q"]
        X0=self.X0
        Y0=self.Y0

        Xsignal = np.zeros(len(t)) # Pre-allocate the memory for Xsignal
        Ysignal = np.zeros(len(t)) # Pre-allocate the memory for Ysignal

        Xsignal[0] = X0
        Ysignal[0] = Y0

        for n in range(len(t)-1):
            dt = t[n+1] - t[n]
            Xsignal[n+1] = Xtrue[n] - P*Xtrue[n]*Ytrue[n]*dt
            Ysignal[n+1] = Ytrue[n] - Q*Xtrue[n]*Ytrue[n]*dt

        return Xsignal, Ysignal

    def Dnoise_cov(self, **metaparams):
        """Covariance matrix for the demographic noise.
        Demographic noise depends only on the observed population.

        Parameters
        ----------
        R : double
            strength of demographic noise

        Returns
        -------
        D : array, double, dimension=(2*n,2*n)
            demographic noise covariance matrix

        """
        import numpy as np
        Xtrue=self.Xtrue
        Ytrue=self.Ytrue
        R=metaparams["R"]
        tmax=self.tmax

        D00=np.diag(Xtrue)
        D01=np.zeros((tmax,tmax))
        D10=np.zeros((tmax,tmax))
        D11=np.diag(Ytrue)
        D=np.block([ [R*D00, R*D01], [R*D10, R*D11] ])

        return D

    def Onoise_cov(self, **metaparams):
        """Covariance matrix for the observational noise.
        Prey and predator populations introduce a noise to the other
        population, and there is also a non-diagonal term
        proportional to the geometric mean of both populations.

        Parameters
        ----------
        S : double
            overall strength of observational noise
        T : double
            strength of non-diagonal term in observational noise

        Returns
        -------
        O : array, double, dimension=(2*n,2*n)
            observational noise covariance matrix

        Note
        ----
        Getting a positive semidefinite matrix should be enforced by user parameter choice.

        """
        import numpy as np
        Xtrue=self.Xtrue
        Ytrue=self.Ytrue
        S=metaparams["S"]
        T=metaparams["T"]

        O00=np.diag(Ytrue)
        O01=T*np.diag(np.sqrt(Ytrue*Xtrue))
        O10=T*np.diag(np.sqrt(Ytrue*Xtrue))
        O11=np.diag(Xtrue)
        O=np.block([ [S*O00, S*O01], [S*O10, S*O11] ])

        return O

    def noise_cov(self, **metaparams):
        """Full noise covariance matrix.

        Parameters
        ----------
        R : double
            strength of demographic noise
        S : double
            overall strength of observational noise
        T : double
            strength of non-diagonal term in observational noise

        Returns
        -------
        N : array, double, dimension=(2*n,2*n)
            full noise covariance matrix, sum of the demographic
            and observational noise covariance matrices, N=D+O.

        """

        # demographic noise
        D=self.Dnoise_cov(**metaparams)

        # observational noise
        O=self.Onoise_cov(**metaparams)

        # total noise covariance matrix
        N=D+O

        return N

    def simulate_obs(self, s, N):
        """Simulate the observational process, assuming that
        data = signal + noise

        Parameters
        ----------
        s : array, double, dimension=2*n
            unobserved intrinsic signal
        N : array, double, dimension=(2*n,2*n)
            noise covariance matrix

        """
        import numpy as np
        import scipy.stats as ss
        tmax=self.tmax

        # draw noise
        n = ss.multivariate_normal(mean=np.zeros(2*tmax), cov=N).rvs()

        # data=signal+noise
        data = s+n
        self.Xobs = np.arange(tmax)
        self.Yobs = np.arange(tmax)
        self.Xdata = data[0:tmax]
        self.Ydata = data[tmax:2*tmax]

    def censor(self, **metaparams):
        """Censor part of the data during periods
        when preys are not observable.

        Parameters
        ----------
        mask : array, double, dimension=n
            a mask containing zeros and ones

        """
        import numpy as np
        mask=metaparams["mask"]

        self.Xobs = self.Xobs[np.where(mask==1)]
        self.Xdata = self.Xdata[np.where(mask==1)]

    def threshold(self, **metaparams):
        """Threshold the data, excluding negative measurements
        and measurements above a detection maximum.

        Parameters
        ----------
        threshold : double
            maximum number of individuals (preys or predators) that can be observed

        """
        import numpy as np
        threshold=metaparams["threshold"]

        # threshold data
        self.Xdata[np.where(self.Xdata>threshold)]=threshold
        self.Ydata[np.where(self.Ydata>threshold)]=threshold
        self.Xdata[np.where(self.Xdata<0)]=0
        self.Ydata[np.where(self.Ydata<0)]=0

    def observe(self, **metaparams):
        """Simulate the observation process using a given data model.

        Parameters
        ----------
        model : int, optional, default=0
            0= correct data model; 1=misspecified data model
        threshold : double
            maximum number of individuals (preys or predators) that can be observed
        mask : array, double, dimension=n
            a mask containing zeros and ones
        P : double
            rate of prey misses due to correlation between preys and predators
        Q : double
            rate of predator misses due to correlation between preys and predators
        R : double
            strength of demographic noise
        S : double
            overall strength of observational noise
        T : double
            strength of non-diagonal term in observational noise

        """
        import numpy as np

        if not "model" in metaparams or metaparams["model"] == 0:
            # Simulate the observation process using the full data model

            # signal
            Xsignal, Ysignal = self.make_signal(self.Xtrue, self.Ytrue, self.t, **metaparams)
            s=np.concatenate((Xsignal, Ysignal))

            # noise covariance
            N = self.noise_cov(**metaparams)

            # simulate observation
            self.simulate_obs(s, N)

            # censor and threshold data
            self.censor(**metaparams)
            self.threshold(**metaparams)

        elif metaparams["model"] == 1:
            # Simulate the observation process using a misspecified data model.

            Xtrue=self.Xtrue
            Ytrue=self.Ytrue

            # signal: consider that the measurement is a direct observation of the function
            Xsignal, Ysignal = Xtrue, Ytrue
            s=np.concatenate((Xsignal, Ysignal))

            # noise covariance: only consider demographic noise
            N = self.Dnoise_cov(**metaparams)

            # simulate observation
            self.simulate_obs(s, N)

            # censor, but do not threshold data
            self.censor(**metaparams)

#end class(LVobserver)
