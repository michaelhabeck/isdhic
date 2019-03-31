import numpy as np

from .mcmc import AdaptiveWalk

class Hamiltonian(object):

    def __init__(self, model):
        self.model = model

    def set_coords(self, q):
        self.model.params['coordinates'].set(q)

    def potential_energy(self, q):
        self.set_coords(q)
        return -self.model.log_prob()

    def kinetic_energy(self, p):
        return 0.5 * np.dot(p,p)

    def gradient_momenta(self, p):
        return p

    def gradient_positions(self, q):

        self.model.params['forces'].set(0.)        
        self.set_coords(q)
        self.model.update_forces()
        
        return -self.model.params['forces'].get().copy()

    def sample_momenta(self, q):
        
        return np.random.standard_normal(np.shape(q))

class State(object):
    """State

    A state in phase space.
    """
    def __init__(self, state, energies):

        positions, momenta = state
        potential_energy, kinetic_energy = energies

        self.positions = positions
        self.momenta   = momenta

        self.potential_energy = potential_energy
        self.kinetic_energy   = kinetic_energy

    @property
    def value(self):
        return self.positions #, self.momenta

    @property
    def log_prob(self):
        return - self.potential_energy - self.kinetic_energy

class Leapfrog(object):
    """Leapfrog

    Velocity Verlet algorithm.
    """
    def __init__(self, hamiltonian, stepsize=1e-3, n_steps=100):

        self.hamiltonian = hamiltonian        
        self.stepsize    = float(stepsize)
        self.n_steps     = int(n_steps)
        
    def run(self, positions, momenta):

        q, p   = positions, momenta

        grad_q = self.hamiltonian.gradient_positions
        grad_p = self.hamiltonian.gradient_momenta
        eps    = self.stepsize
        
        ## half step for the momenta

        p -= 0.5 * eps * grad_q(q)

        for _ in xrange(self.n_steps-1):

            q += eps * grad_p(p)
            p -= eps * grad_q(q)

        ## last (half) step

        q += eps * grad_p(p)
        p -= 0.5 * eps * grad_q(q)
        
        return q, p

class HamiltonianMonteCarlo(AdaptiveWalk):

    @property
    def stepsize(self):
        """
        Stepsize of Leapfrog integrator.
        """
        return self.leapfrog.stepsize

    @stepsize.setter
    def stepsize(self, value):
        self.leapfrog.stepsize = float(value)

    def __init__(self, model, stepsize=1e-3):

        self.leapfrog = Leapfrog(Hamiltonian(model))

        super(HamiltonianMonteCarlo, self).__init__(
            model, model.params['coordinates'], stepsize=stepsize)

        self.uprate   = 1.05
        self.downrate = 0.96
        
    def create_state(self):
        """
        Creates a state from current configuration and generates
        random momenta.
        """
        hamiltonian = self.leapfrog.hamiltonian        
        positions   = self.parameter.get().copy()
        momenta     = hamiltonian.sample_momenta(positions)        

        ## TODO: there is space for speed up by using previously
        ## calculated log prob value

        potential_energy = hamiltonian.potential_energy(positions)
        kinetic_energy   = hamiltonian.kinetic_energy(momenta)

        return State((positions,momenta), (potential_energy,kinetic_energy))

    def propose(self, state):

        hamiltonian = self.leapfrog.hamiltonian

        ## resample momenta
        
        q = state.positions
        p = hamiltonian.sample_momenta(q)

        state.kinetic_energy = hamiltonian.kinetic_energy(p)
        state.momenta = p

        Q, P = self.leapfrog.run(q.copy(), p.copy())

        V = hamiltonian.potential_energy(Q)
        K = hamiltonian.kinetic_energy(P)

        return State((Q,P), (V,K))
        
