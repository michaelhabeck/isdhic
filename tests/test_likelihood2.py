import isdhic
import numpy as np

from isdhic import utils
from isdhic.core import take_time
from isdhic.model import Likelihood

from scipy import optimize

from test_params import random_pairs

class Relu(isdhic.Relu):
    """Relu

    Python implementation of Relu likelihood.
    """
    def log_prob(self):

        x = (self.mock.get() - self.data) * self.alpha

        return - np.clip(x, 0., 1e308).sum()

    def update_derivatives(self):

        x = (self.mock.get() - self.data)

        self.grad[...] = - self.alpha * (x > 0.).astype('i')

def log_prob(x, params, likelihood):

    params['coordinates'].set(x)

    likelihood.update()

    return likelihood.log_prob()

if __name__ == '__main__':

    ## create universe

    universe  = utils.create_universe(n_particles=1000, diameter=4.)
    coords    = isdhic.Coordinates(universe)
    forces    = isdhic.Forces(universe)

    ## create parameters
    
    params    = isdhic.Parameters()

    ## create contact data
    
    n_data = 100
    pairs  = random_pairs(universe.n_particles, n_data)
    data   = np.random.random(n_data) * 10.
    mock   = isdhic.ModelDistances( pairs, 'contacts')
    relu   = Relu('contacts', data, mock, params=params)
    relu2  = isdhic.Relu('contacts2', data, mock, params=params)

    for param in (coords, forces, mock, relu.steepness):
        params.add(param)

    mock.update(params)

    with take_time('evaluating python version of relu likelihood'):
        lgp = relu.log_prob()

    print 'log_prob={0:.3e}'.format(lgp)

    with take_time('evaluating cython version of relu likelihood'):
        lgp = relu2.log_prob()

    print 'log_prob={0:.3e}'.format(lgp)

    with take_time('evaluating derivatives of python version'):
        relu.update_derivatives()

    with take_time('evaluating derivatives of cython version'):
        relu2.update_derivatives()

    forces.set(0.)
    relu.update_forces()

    ## numerical gradient

    f = lambda x, params=params, likelihood=relu: \
        log_prob(x, params, likelihood)

    x = coords.get().copy()

    forces_num = optimize.approx_fprime(x, f, 1e-5)

    print 'max discrepancy={0:.5e}, corr={1:.1f}'.format(
        np.fabs(forces.get()-forces_num).max(),
        np.corrcoef(forces.get(),forces_num)[0,1]*100)
