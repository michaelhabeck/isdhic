import isdhic
import random
import numpy as np
import pylab as plt

from isdhic.rex import Swaps
from csb.numeric import log_sum_exp
from test_mcmc import Gaussian
from collections import OrderedDict

def print_rex(states, swaps=[]):

    s = []
    for i, j in zip(states,states[1:]):
        s.append(str(i))
        s.append('<-->' if (i,j) in swaps else '    ')
    s.append(str(j))
    
    return ''.join(s)

class ReplicaExchange(isdhic.ReplicaExchange):

    def sample(self, current=None):

        candidate, accept = super(ReplicaExchange, self).sample(current)

        if len(self.samples) % 500:
            print '-' * 50
            print self.history

        return candidate, accept
            
class bcolors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == '__main__':

    n_rex   = 8
    states  = range(n_rex)
    swaps   = Swaps(n_rex)    
    margin  = 3
    output  = '{{0:0{0}d}} {{1}}'.format(margin)

    print ' ' * margin + ' ' + print_rex(states)
    print '-' * (margin + len(states) + 4 * (len(states)-1))

    counter = 0

    while counter < 10:
        print bcolors.OKBLUE + \
              output.format(counter, print_rex(states, swaps.next())) + \
              bcolors.ENDC
        counter += 1

    ## test replica history

    history = isdhic.ReplicaHistory()
    counter = 0

    while counter < 1000:
        accepted = OrderedDict()
        for pair in swaps.next():
            accepted[pair] = random.sample([True,False],1)[0]
        history.update(accepted)
        counter +=1

    print history

    ## setup replicas

    isdhic.Probability.set_params(isdhic.Parameters())
    
    schedule = np.linspace(0.01, 1., 7)
    samplers = []
    
    for beta in schedule:

        gaussian     = Gaussian()
        gaussian.tau = beta
        gaussian.x   = 10.

        sampler = isdhic.AdaptiveWalk(
            gaussian, gaussian._location, 1/beta**0.5, adapt_until=1e2)
        sampler.activate()

        samplers.append(sampler)

    ## run replica exchange

    rex = ReplicaExchange(samplers)
    rex.run(1e3)

    x = np.array([[state.value for state in samples]
                  for samples in rex.samples])

    ## show sampled replicas

    burnin = 200
    rates  = np.array([rex.history[pair].acceptance_rate()
                       for pair in rex.history.pairs])
    beta   = np.array([sampler.model.tau for sampler in rex])

    y = np.linspace(-1.,1.,10000) * 3.5 * np.max(1/beta**0.5)

    limits = (y.min(), y.max())
    n_cols = 5
    n_rows = len(rex) / n_cols
    
    kw_hist = dict(normed=True,bins=30,histtype='stepfilled',color='k',alpha=0.3)
    
    fig, axes = plt.subplots(n_rows,n_cols,figsize=(20,4),
                             subplot_kw=dict(xlim=limits,ylim=(0.,0.45)))
    
    for k, ax in enumerate(axes.flat):
        p = -0.5 * rex[k].model.tau * y**2
        p-= log_sum_exp(p)
        p = np.exp(p - np.log(y[1]-y[0]))
        ax.hist(x[burnin:,k],**kw_hist)
        ax.plot(y,p,ls='--',color='r',lw=3)
        ax.yaxis.set_visible(False)

    fig.tight_layout()

    fig, ax = plt.subplots(1,1)
    ax.plot(0.5*(beta[1:]+beta[:-1]), rates)
    