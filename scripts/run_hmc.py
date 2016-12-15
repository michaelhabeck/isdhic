"""
Inferential structure determination of the X chromosome at 500 kb resolution
using Hamiltonian Monte Carlo.
"""
import sys
import isdhic
import numpy as np

from isdhic.core import take_time

from scipy import optimize

class HamiltonianMonteCarlo(isdhic.HamiltonianMonteCarlo):

    def sample(self):

        result = super(HamiltonianMonteCarlo, self).sample()

        if len(self.history) and not len(self.history) % 20:
            print '{0}, stepsize = {1:.3e}, log_prob = {2:.3e}'.format(
                self.history, self.stepsize, self.samples[-1].potential_energy)

        return result

if __name__ == '__main__':

    ## set up X chromosome simulation at 500 kb / 50 kb resolution

    resolution = 500 ## Kb
    filename   = './chrX_cell1_{0}kb.py'.format(resolution)

    with open(filename) as script:
        exec script

    ## start from stretched out chromosome structure

    extended = np.multiply.outer(np.arange(n_particles), np.eye(3)[0]) * diameter
    coords.set(extended)

    ## use Hamiltonian Monte Carlo generate X chromosome structures
    ## from the posterior distribution

    n_steps  = 1e3                                    ## number of HMC iterations
    n_leaps  = 1e2                                    ## number of leapfrog integration steps
    stepsize = 1e-3                                   ## initial integration stepsize
    
    hmc = HamiltonianMonteCarlo(posterior,stepsize=stepsize)
    hmc.leapfrog.n_steps = int(n_leaps)
    hmc.adapt_until      = int(0.5 * n_steps)
    hmc.activate()

    with take_time('running HMC'):
        hmc.run(n_steps)

