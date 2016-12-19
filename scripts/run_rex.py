"""
Running replica-exchange Monte Carlo on chromosome.
"""
import isdhic

from isdhic import utils
from run_hmc import HamiltonianMonteCarlo

class Replica(HamiltonianMonteCarlo):

    def set_replica_params(self):

        self.model['tsallis'].q     = self.q
        self.model['contacts'].beta = self.beta
        self.model['rog'].beta      = self.beta

    def __init__(self, posterior, n_steps=10, stepsize=1e-3, n_leaps=100, q=1.0, beta=1.0):

        self.beta = float(beta)
        self.q = float(q)
        
        super(Replica, self).__init__(posterior, stepsize)

        self.leapfrog.n_steps = int(n_leaps)
        self.adapt_until = 1e6
        self.n_steps = int(n_steps)
        self.activate()

    def __str__(self):

        s = 'Replica: q={0:.3f}, beta={1:.3f}'.format(self.q, self.beta)
        
        return s

    def next(self):

        self.set_replica_params()

        for i in xrange(self.n_steps):
            state = super(Replica, self).next()

        return state

        if len(self.history) and not len(self.history) % 20:
            print '{0}, stepsize = {1:.3e}, -log_prob = {2:.3e}'.format(
                self.history, self.stepsize, self.samples[-1].potential_energy)

        return result

    def create_state(self):

        self.set_replica_params()

        state = super(Replica, self).create_state()
        if len(self.history):
            state.momenta, state.kinetic_energy = \
                           self.state.momenta, self.state.kinetic_energy

        return state
    
class ReplicaExchange(isdhic.ReplicaExchange):

    def next(self):

        state = super(ReplicaExchange, self).next()

        if not len(self.history) % 1:
            print '-' * 50
            print self.history

        return state
            
if __name__ == '__main__':

    n_replicas = 50
    schedule   = np.transpose([
        np.logspace(0., np.log10(1.06), n_replicas),
        np.linspace(1., 0.1, n_replicas)])
    
    replicas = []
    
    ## set up X chromosome simulation at 500 kb / 50 kb resolution

    resolution = 500  
    filename   = './chrX_cell1_{0}kb.py'.format(resolution)

    with open(filename) as script:
        exec script

    extended = np.multiply.outer(np.arange(n_particles), np.eye(3)[0]) * diameter
    coords.set(extended)

    replicas = [Replica(posterior, n_steps=1, q=q, beta=beta) for q, beta in schedule]

    rex = ReplicaExchange(replicas)

if False:
    
    samples = []
    while len(samples) < 1000:
        samples.append(rex.next())

if False:

    state = rex.state[10]
    states = []
    for replica in replicas:
        replica.parameter.set(state.value)
        states.append(replica.create_state())

    
