"""
Running Hamiltonian Monte Carlo on an oscillator with coupled degrees
of freedom.
"""
import isdhic
import numpy as np
import pylab as plt

from isdhic import utils
from isdhic.core import take_time
from isdhic.params import Array 

from scipy import optimize

from csb.numeric import log_sum_exp
from csb.statistics import autocorrelation

class HamiltonianMonteCarlo(isdhic.HamiltonianMonteCarlo):

    stepsizes = []

    @property
    def stepsize(self):
        return self.leapfrog.stepsize

    @stepsize.setter
    def stepsize(self, value):

        self.leapfrog.stepsize = float(value)
        HamiltonianMonteCarlo.stepsizes.append(self.leapfrog.stepsize)

class Oscillator(isdhic.Probability):

    @property
    def x(self):
        return self.params['coordinates'].get()

    @x.setter
    def x(self, value):
        self.params['coordinates'].set(value)

    def __init__(self, K=np.eye(2)):

        super(Oscillator, self).__init__('{}d-oscillator'.format(len(K)))

        self.K = (K + K.T) / 2.

        self.params.add(Array('coordinates',len(K)))
        self.params.add(Array('forces',len(K)))

    def log_prob(self):
        return - 0.5 * np.dot(self.x, np.dot(self.K, self.x))

    def update_forces(self):
        self.params['forces'].set(-np.dot(self.K, self.x))

if __name__ == '__main__':

    osci = Oscillator(np.diag([10.,1.]))
    osci = Oscillator(np.array([[10.,-2.5],
                                [-2.5,1.]]))

    hmc = HamiltonianMonteCarlo(osci,stepsize=1e-3)
    hmc.leapfrog.n_steps = 100
    hmc.adapt_until      = int(1e6)
    hmc.activate()

    with take_time('running HMC'):
        samples = []
        while len(samples) < 1e4:
            samples.append(hmc.next())

    q = np.array([state.positions for state in samples])
    p = np.array([state.momenta for state in samples])
    E = np.array([state.potential_energy for state in samples])

    sigma  = np.diagonal(np.linalg.inv(osci.K))
    limits = (-4*sigma.max(),4*sigma.max())

    x = 5 * sigma.max() * np.linspace(-1.,1.,1000)
    kw_hist = dict(normed=True, histtype='stepfilled', bins=30, color='k', alpha=0.3)

    fig, ax = plt.subplots(1,4,figsize=(16,4))
    ax[0].scatter(*q.T,color='k',s=60,alpha=0.5)
    ax[0].set_xlim(*limits)
    ax[0].set_ylim(*limits)

    p = -0.5 * x**2 / sigma[0]
    p-= log_sum_exp(p)
    p = np.exp(p - np.log(x[1]-x[0]))

    ax[1].hist(q[:,0], label='MCMC', **kw_hist)
    ax[1].plot(x,p, lw=3, color='r', label='target')
    ax[1].yaxis.set_visible(False)
    ax[1].legend(fontsize=10)

    p = -0.5 * x**2 / sigma[1]
    p-= log_sum_exp(p)
    p = np.exp(p - np.log(x[1]-x[0]))

    ax[2].hist(q[:,1], label='MCMC', **kw_hist)
    ax[2].plot(x,p, lw=3, color='r', label='target')
    ax[2].yaxis.set_visible(False)
    ax[2].legend(fontsize=10)

    ax[3].plot(hmc.stepsizes[:1000],color='k',alpha=0.3,lw=3)
    ax[3].axhline(hmc.stepsize,ls='--',color='r')
    ax[3].set_xlabel('Monte Carlo iteration')
    ax[3].set_ylabel('stepsize')
    ax[3].semilogy()

    fig.tight_layout()

