"""
Testing functionality of the conformational prior. Checking gradient etc.
"""
import isdhic
import numpy as np

from isdhic import utils
from isdhic.core import take_time

from scipy import optimize

def log_prob(x, prior):

    params = prior.params
    params['coordinates'].set(x)

    return prior.log_prob()

if __name__ == '__main__':

    params     = isdhic.Parameters()
    universe   = utils.create_universe(n_particles=1e3, diameter=4.)
    coords     = isdhic.Coordinates(universe)
    forces     = isdhic.Forces(universe)
    forcefield = isdhic.ForcefieldFactory.create_forcefield('rosetta',universe)

    for param in (coords,forces): params.add(param)

    boltzmann  = isdhic.BoltzmannEnsemble('boltzmann',forcefield,params)
    tsallis    = isdhic.TsallisEnsemble('tsallis',forcefield,params)

    print forcefield.energy(coords.get())
    print boltzmann.log_prob()

    boltzmann.beta = 0.1
    tsallis.E_min  = np.floor(tsallis.log_prob()) - 500.
    tsallis.q      = 1.06
    tsallis.beta   = 0.5

    eps = 1e-7
    out = '  max discrepancy={0:.1e}, corr={1:.1f}'
    
    for prior in (boltzmann, tsallis):

        forces.set(0.)        
        with take_time('\ncalculate forces {}'.format(prior)):
            prior.update_forces()

        a = forces.get().copy()
        f = lambda x, prior=prior: log_prob(x, prior)
        x = params['coordinates'].get().copy()
        b = optimize.approx_fprime(x, f, eps)

        cc = np.corrcoef(a,b)[0,1] * 100
        d_max = np.max(np.fabs(a-b))
        
        print out.format(d_max, cc)
