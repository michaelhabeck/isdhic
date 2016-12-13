import utils
import isdhic
import numpy as np

from isdhic.core import take_time
from isdhic.model import Likelihood

from scipy import optimize

class Logistic(isdhic.Logistic):
    """Logistic

    Python implementation of Logistic likelihood.
    """
    def log_prob(self):

        y, x = self.data, self.mock.get()

        return - np.logaddexp(np.zeros(len(x)), self.alpha * (x-y)).sum()

    def update_derivatives(self):

        y, x = self.data, self.mock.get()

        self.grad[...] = - self.alpha / (1 + np.exp(-self.alpha * (x-y)))

def log_prob(x, coords, likelihood):

    coords.set(x)

    likelihood.mock.update()

    return likelihood.log_prob()

if __name__ == '__main__':

    from test_params import random_pairs

    universe  = utils.create_universe(n_particles=1000, diameter=4.)
    coords    = isdhic.Coordinates(universe)
    n_data    = 100
    pairs     = random_pairs(universe.n_particles, n_data)
    data      = np.random.random(n_data) * 10.
    mock      = isdhic.ModelDistances(coords, pairs, 'contacts')
    logistic  = Logistic('contacts', data, mock)
    logistic2 = isdhic.Logistic('contacts', data, mock)
    params    = isdhic.Parameters()

    for param in (coords, mock, logistic.steepness):
        params.add(param)

    mock.update()

    with take_time('evaluating python version of logistic likelihood'):
        lgp = logistic.log_prob()

    print 'log_prob={0:.3e}'.format(lgp)

    with take_time('evaluating cython version of logistic likelihood'):
        lgp = logistic2.log_prob()

    print 'log_prob={0:.3e}'.format(lgp)

    with take_time('evaluating derivatives of python version'):
        logistic.update_derivatives()

    with take_time('evaluating derivatives of cython version'):
        logistic2.update_derivatives()

    forces = np.ascontiguousarray(universe.forces.reshape(-1,))
    forces[...] = 0.

    logistic.update_forces(forces)

    ## numerical gradient

    f = lambda x, coords=coords, likelihood=logistic: \
        log_prob(x, coords, likelihood)

    x = coords.get().copy()

    forces_num = optimize.approx_fprime(x, f, 1e-5)

    print 'max discrepancy={0:.5e}, corr={1:.1f}'.format(
        np.fabs(forces-forces_num).max(), np.corrcoef(forces,forces_num)[0,1]*100)
