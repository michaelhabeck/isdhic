"""
Testing functionality of the conformational prior
"""
import utils
import isdhic
import numpy as np

from isdhic.core import take_time

from scipy import optimize

def log_prob(x, prior):

    params = prior.params
    params['coordinates'].set(x)

    return prior.log_prob()

if __name__ == '__main__':

    n_particles  = int(1e3)
    diameter     = 4.

    params       = isdhic.Parameters()
    isdhic.Probability.set_params(params)

    universe     = isdhic.Universe(n_particles)
    coords       = isdhic.Coordinates(universe)
    forcefield   = isdhic.ForcefieldFactory.create_forcefield('rosetta',universe)

    for param in (coords,): params.add(param)

    coords.set(utils.randomwalk(n_particles) * diameter)

    boltzmann = isdhic.BoltzmannEnsemble('boltzmann',forcefield)
    tsallis   = isdhic.TsallisEnsemble('tsallis',forcefield)
    
    print forcefield.energy(coords.get())
    print boltzmann.log_prob()

    boltzmann.beta = 0.1
    tsallis.E_min  = np.floor(tsallis.log_prob()) - 500.
    tsallis.q      = 1.06
    tsallis.beta   = 0.5

    eps = 1e-7
    out = '  max discrepancy={0:.1e}, corr={1:.1f}'
    
    for prior in (boltzmann, tsallis):

        with take_time('\ncalculate forces {}'.format(prior)):
            a = prior.gradient()

        f = lambda x, prior=prior: log_prob(x, prior)
        x = params['coordinates'].get().copy()
        b = optimize.approx_fprime(x, f, eps)

        cc = np.corrcoef(a,b)[0,1] * 100
        d_max = np.max(np.fabs(a-b))# / (np.fabs(a) + 1e-300))
        
        print out.format(d_max, cc)
