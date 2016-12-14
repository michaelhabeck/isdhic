import utils
import isdhic
import numpy as np

from isdhic.core import take_time
from isdhic.params import Parameter

from scipy import optimize

class Array(Parameter):

    def set(self, value):

        self._value = np.array(value)

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

        self.params.add(Array('coordinates'))
        self.params.add(Array('forces'))

        self.params['forces'].set(np.zeros(len(K)))
        self.params['coordinates'].set(np.zeros(len(K)))

    def log_prob(self):

        return - 0.5 * np.dot(self.x, np.dot(self.K, self.x))

    def update_forces(self):

        self.params['forces'].set(-np.dot(self.K, self.x))

if __name__ == '__main__':

    isdhic.Probability.set_params(isdhic.Parameters())
    osci = Oscillator(np.diag([10.,1.]))

    hmc = HamiltonianMonteCarlo(osci,stepsize=1e-3)
    hmc.leapfrog.n_steps = 100
    hmc.adapt_until      = int(1e6)
    hmc.activate()

    with take_time('running HMC'):
        hmc.run(1e3)

    q = np.array([state.positions for state in hmc.samples])
    p = np.array([state.momenta for state in hmc.samples])
    E = np.array([state.potential_energy for state in hmc.samples])

    limits = (-2.5,2.5)
    scatter(*q.T)
    xlim(*limits)
    ylim(*limits)
