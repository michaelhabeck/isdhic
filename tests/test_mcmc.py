"""
Test random walk Monte Carlo on a Gaussian model.
"""
import isdhic
import numpy as np
import pylab as plt

from isdhic import mcmc

from test_model import Gaussian

from csb.numeric import log_sum_exp
from csb.statistics import autocorrelation

class AdaptiveWalk(isdhic.AdaptiveWalk):
    """Adaptive Walk

    keeps track of the stepsizes...
    """
    stepsizes = []

    @property
    def stepsize(self):
        return self._stepsize

    @stepsize.setter
    def stepsize(self, value):

        self._stepsize = float(value)
        AdaptiveWalk.stepsizes.append(self._stepsize)

if __name__ == '__main__':
        
    n_steps = 1e3
    rate    = 0.2
    history = mcmc.History()

    while len(history) < n_steps:
        history.update(np.random.random() < rate)

    print 'target rate = {0:.1f}%'.format(rate*100)

    print history

    isdhic.Probability.set_params(isdhic.Parameters())

    gaussian = Gaussian()

    ## start at improbable initial sample

    gaussian.x = 10.

    ## run random walk Metropolis/Hastings with adaptive stepsize

    stepsize = 0.01
    n_steps  = 1e4
    burnin   = int(0.05*n_steps)

    sampler = AdaptiveWalk(gaussian, gaussian._location, stepsize, adapt_until=2*burnin)
    sampler.activate()
    sampler.run(1e4)

    print sampler.history

    ## evaluate true model

    y = np.array([state.value for state in sampler.samples])

    x = gaussian.mu + 5 * gaussian.sigma * np.linspace(-1.,1.,1000)
    p = -0.5 * gaussian.tau * (x-gaussian.mu)**2
    p-= log_sum_exp(p)
    p = np.exp(p - np.log(x[1]-x[0]))

    ## plot results

    kw_hist = dict(normed=True, histtype='stepfilled', bins=50, color='k', alpha=0.3)

    fig, ax = plt.subplots(1,4,figsize=(12,3))

    ax[0].plot(y,color='k',alpha=0.3,lw=3)
    ax[0].set_xlabel('Monte Carlo iteration')
    ax[0].set_xlim(0, burnin)

    ax[1].hist(y[burnin:], label='MCMC', **kw_hist)
    ax[1].plot(x,p, lw=3, color='r', label='target')
    ax[1].yaxis.set_visible(False)
    ax[1].legend(fontsize=10)

    ax[2].plot(autocorrelation(y[burnin:],100),color='k',alpha=0.3,lw=3)
    ax[2].axhline(0.,ls='--',color='r')
    ax[2].set_xlabel('Monte Carlo iteration')
    ax[2].set_ylabel('autocorrelation')

    ax[3].plot(AdaptiveWalk.stepsizes,color='k',alpha=0.3,lw=3)
    ax[3].axhline(sampler.stepsize,ls='--',color='r')
    ax[3].set_xlabel('Monte Carlo iteration')
    ax[3].set_ylabel('stepsize')
    ax[3].semilogy()

    fig.tight_layout()

