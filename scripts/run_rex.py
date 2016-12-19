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
    
    ## set up X chromosome simulation at 500 kb / 50 kb resolution

    resolution = 500  
    filename   = './chrX_cell1_{0}kb.py'.format(resolution)

    with open(filename) as script:
        exec script

    extended = np.multiply.outer(np.arange(n_particles), np.eye(3)[0]) * diameter
    coords.set(extended)

    replicas = [Replica(posterior, n_steps=10, q=q, beta=beta) for q, beta in schedule]

    rex = ReplicaExchange(replicas)

    samples = []
    while len(samples) < 1000:
        samples.append(rex.next())

if False:

    posterior['backbone'].beta = 0.
    posterior['tsallis'].beta = 0.

    p = Replica(posterior, n_steps=10, q=1.06, beta=1.0)
    q = Replica(posterior, n_steps=10, q=1.00, beta=1.0)

    p = Replica(posterior, n_steps=10, q=1.00, beta=1.0)
    q = Replica(posterior, n_steps=10, q=1.00, beta=0.1)

    ## to enable correct copying

    ## p.history.update(True)
    ## q.history.update(True)

    a = p.next()
    b = q.next()

    p.parameter.set(b.value)
    B = p.create_state()

    q.parameter.set(a.value)
    A = q.create_state()

    assert np.all(a.positions==A.positions)
    assert np.all(b.positions==B.positions)

    assert np.all(b.momenta==A.momenta)
    assert np.all(a.momenta==B.momenta)

    print a.kinetic_energy, B.kinetic_energy
    print b.kinetic_energy, A.kinetic_energy

    print A.log_prob + B.log_prob - a.log_prob - b.log_prob

    print 0.9 * (a.potential_energy-B.potential_energy)
