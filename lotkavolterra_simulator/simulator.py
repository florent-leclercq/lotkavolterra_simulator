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

    def make_signal(self, **metaparams):
        """Simulate the signal.

        Parameters
        ----------
        model : int, optional, default=0
            0= correct data model; 1=misspecified data model
        Xefficiency : array, double, dimension=len(t)
            detection efficiency of preys as a function of time
        Yefficiency : array, double, dimension=len(t)
            detection efficiency of preys as a function of time
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
        t=self.t
        X0=self.X0
        Y0=self.Y0
        Xtrue=self.Xtrue
        Ytrue=self.Ytrue

        Xsignal, Ysignal = self.Xtrue, self.Ytrue
        if not "model" in metaparams or metaparams["model"] == 0:
            # correct data model: The (unobserved) signal is a retarded and non-linear
            # observation of the true underlying functions.
            P=metaparams["obsP"]
            Q=metaparams["obsQ"]
            Xefficiency=metaparams["Xefficiency"]
            Yefficiency=metaparams["Yefficiency"]

            Xretard = np.roll(Xtrue,1)
            Yretard = np.roll(Ytrue,1)

            Xsignal = Xretard - P*Xretard*Yretard + Q*Xretard**2
            Xsignal *= Xefficiency
            Ysignal = Yretard + P*Xretard*Yretard - Q*Yretard**2
            Ysignal *= Yefficiency

            Xsignal[0] = X0
            Ysignal[0] = Y0
        elif "model" == 1:
            # misspecified data model : consider that the measurement is a direct
            # observation of the function.
            pass

        return Xsignal, Ysignal

    def Dnoise_cov(self, **metaparams):
        """Covariance matrix for the demographic noise.
        Demographic noise depends only on the observed population.

        Parameters
        ----------
        obsR : double
            strength of demographic noise

        Returns
        -------
        D : array, double, dimension=(2*n,2*n)
            demographic noise covariance matrix

        """
        import numpy as np
        Xtrue=self.Xtrue
        Ytrue=self.Ytrue
        obsR=metaparams["obsR"]

        D=obsR*np.diag(np.concatenate((Xtrue, Ytrue)))

        return D

    def make_demographic_noise(self, **metaparams):
        """Simulate demographic noise.
        Demographic noise depends only on the observed population.

        Parameters
        ----------
        obsR : double
            strength of demographic noise

        Returns
        -------
        XDnoise : array, double, dimension=n
            demographic noise for preys
        YDnoise : array, double, dimension=n
            demographic noise for predators

        """
        import numpy as np
        import scipy.stats as ss
        Xtrue=self.Xtrue
        Ytrue=self.Ytrue
        obsR=metaparams["obsR"]

        XDnoise = np.sqrt(obsR*Xtrue) * ss.multivariate_normal(mean=np.zeros_like(Xtrue)).rvs()
        YDnoise = np.sqrt(obsR*Ytrue) * ss.multivariate_normal(mean=np.zeros_like(Xtrue)).rvs()

        return XDnoise, YDnoise

    def Onoise_cov(self, X, Y, **metaparams):
        """Covariance matrix for the observational noise.
        Prey and predator populations introduce a noise to the other
        population, and there is also a non-diagonal term
        proportional to the geometric mean of both populations.

        Parameters
        ----------
        obsS : double
            overall strength of observational noise
        obsT : double
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
        obsS=metaparams["obsS"]
        obsT=metaparams["obsT"]

        O00=np.diag(Y)
        O01=obsT*np.diag(np.sqrt(X*Y))
        O10=obsT*np.diag(np.sqrt(X*Y))
        O11=np.diag(X)
        O=np.block([ [obsS*O00, obsS*O01], [obsS*O10, obsS*O11] ])

        return O

    def make_observational_noise(self, Xsignal, Ysignal, **metaparams):
        """Simulate observational noise.

        Parameters
        ----------
        obsS : double
            overall strength of observational noise
        obsT : double
            strength of non-diagonal term in observational noise

        Returns
        -------
        XOnoise : array, double, dimension=n
            observational noise for preys
        YOnoise : array, double, dimension=n
            observational noise for predators

        """
        import numpy as np
        import scipy.stats as ss
        from scipy.linalg import sqrtm
        tmax=self.tmax

        # Writing this in a single call to multivariate_normal causes numerical instabilities
        XOnoise = np.zeros(tmax) # Pre-allocate the memory for XOnoise
        YOnoise = np.zeros(tmax) # Pre-allocate the memory for YOnoise
        for n in range(tmax-1):
            O = self.Onoise_cov(np.array([Xsignal[n]]), np.array([Ysignal[n]]), **metaparams)
            XOnoise[n], YOnoise[n] = ss.multivariate_normal(mean=[0,0], cov=O).rvs()

        return XOnoise, YOnoise

    def simulate_obs(self, Xsignal, Ysignal, **metaparams):
        """Simulate the observational process, assuming additive noise, i.e.
        data = signal + noise.

        Parameters
        ----------
        Xsignal : array, double, dimension=n
            unobserved signal for the number of preys
        Ysignal : array, double, dimension=n
            unobserved signal for the number of predators
        model : int, optional, default=0
            0= correct data model; 1=misspecified data model
        obsR : double
            strength of demographic noise
        obsS : double
            overall strength of observational noise
        obsT : double
            strength of non-diagonal term in observational noise

        """
        import numpy as np
        import scipy.stats as ss
        tmax=self.tmax

        # draw demographic noise
        XDnoise, YDnoise = self.make_demographic_noise(**metaparams)

        # observational noise
        XOnoise, YOnoise = np.zeros_like(XDnoise), np.zeros_like(YDnoise)
        if not "model" in metaparams or metaparams["model"] == 0:
            # correct data model: draw observational noise
            XOnoise, YOnoise = self.make_observational_noise(Xsignal, Ysignal, **metaparams)
        elif "model" == 1:
            # misspecified data model : only consider demographic noise
            pass

        # data = signal+noise
        self.Xdata = Xsignal+XDnoise+XOnoise
        self.Ydata = Ysignal+YDnoise+YOnoise

        self.Xobs = np.arange(tmax)
        self.Yobs = np.arange(tmax)

        return Xsignal, XDnoise, XOnoise, self.Xdata, Ysignal, YDnoise, YOnoise, self.Ydata

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

        if not "model" in metaparams or metaparams["model"] == 0:
            # correct data model: threshold data
            threshold=metaparams["threshold"]
            self.Xdata[np.where(self.Xdata>threshold)]=threshold
            self.Ydata[np.where(self.Ydata>threshold)]=threshold
            self.Xdata[np.where(self.Xdata<0)]=0
            self.Ydata[np.where(self.Ydata<0)]=0
        elif "model" == 1:
            # misspecified data model : do not threshold data
            pass

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

        # Simulate the observation process using a given data model

        # signal
        Xsignal, Ysignal = self.make_signal(**metaparams)

        # simulate observation using correct data model
        self.simulate_obs(Xsignal, Ysignal, **metaparams)

        # censor and threshold data
        self.censor(**metaparams)
        self.threshold(**metaparams)

#end class(LVobserver)
