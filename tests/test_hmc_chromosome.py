import utils
import isdhic
import numpy as np

from isdhic.core import take_time

from scipy import optimize

class HamiltonianMonteCarlo(isdhic.HamiltonianMonteCarlo):

    stepsizes = []

    @property
    def stepsize(self):
        return self.leapfrog.stepsize

    @stepsize.setter
    def stepsize(self, value):

        self.leapfrog.stepsize = float(value)
        HamiltonianMonteCarlo.stepsizes.append(self.leapfrog.stepsize)

    def sample(self):

        result = super(HamiltonianMonteCarlo, self).sample()

        if len(self.history) and not len(self.history) % 20:
            print '{0}, stepsize = {1:.3e}, log_prob = {2:.3e}'.format(
                self.history, self.stepsize, self.samples[-1].potential_energy)

        return result

if __name__ == '__main__':

    resolution = 500 ## Kb
    filename   = './chrX_cell1_{0}kb.py'.format(resolution)

    with open(filename) as script:
        exec script

    if not False:

        extended = np.multiply.outer(np.arange(n_particles), np.eye(3)[0]) * diameter
        coords.set(extended)
    
    pymol = utils.ChainViewer()

    hmc = HamiltonianMonteCarlo(posterior,stepsize=1e-3)
    hmc.leapfrog.n_steps = 100
    hmc.adapt_until      = int(1e6)
    hmc.activate()

    with take_time('running HMC'):
        hmc.run(1e3)

    X = np.array([state.positions for state in hmc.samples]).reshape(len(hmc.samples),-1,3)
    E = np.array([state.potential_energy for state in hmc.samples])
    K = np.array([state.kinetic_energy for state in hmc.samples])
    E2 = np.array(map(hmc.leapfrog.hamiltonian.potential_energy, X))

    print hmc.history, hmc.stepsize

if False:

    from csb.bio.utils import distance_matrix

    burnin = 500

    D = 0.
    for x in X[burnin:]:
        D += distance_matrix(x)
    D/= len(X)

if False:

    from isd import utils

    prior = posterior.priors[0]
    E_isd = utils.Load('/tmp/E')

    X = E_isd.torsion_angles.copy()

    backbone, contacts, rog = posterior.likelihoods

    E_prior = []
    E_bbone = []
    E_intra = []
    E_rog   = []
    
    for x in X:

        coords.set(x)
        posterior.update()
        
        E_prior.append(-prior.log_prob())
        E_bbone.append(-backbone.log_prob())
        E_intra.append(-contacts.log_prob())
        E_rog.append(-rog.log_prob())
        
    E_prior = np.array(E_prior)
    E_bbone = np.array(E_bbone)
    E_intra = np.array(E_intra)
    E_rog   = np.array(E_rog)
    
if False:

    x = coords.get().copy()

    ## check gradient

    energy   = hmc.leapfrog.hamiltonian.potential_energy
    gradient = hmc.leapfrog.hamiltonian.gradient_positions
    
    with take_time('calculating energy'):
        E = energy(x)
    print E

    a = gradient(x)
    b = optimize.approx_fprime(x, energy, 1e-6)

    print np.fabs(a-b).max(), round(100*np.corrcoef(a,b)[0,1])

