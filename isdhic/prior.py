"""
Collection of priors.
"""
import numpy as np

from .model import Probability
from .params import Scale, Location

class BoltzmannEnsemble(Probability):

    @property
    def beta(self):
        """
        Inverse temperature
        """
        return self._beta.get()

    @beta.setter
    def beta(self, value):
        self._beta.set(value)

    def __init__(self, name, forcefield):

        super(BoltzmannEnsemble, self).__init__(name)

        self.forcefield = forcefield

        self._beta = Scale(self.name + '.beta')
        self.params.add(self._beta)

        ## local copy of Cartesian gradient

        self._forces = self.params['coordinates'].get() * 0.
        
    def log_prob(self):

        coords = self.params['coordinates'].get()

        return - self.beta * self.forcefield.energy(coords)

    def update_forces(self):

        coords = self.params['coordinates'].get()

        self._forces[...] = 0.

        self.forcefield.ctype.update_gradient(\
            coords, self._forces, self.forcefield.types, 1)

        self._forces *= -self.beta
        
        self.params['forces']._value += self._forces

class TsallisEnsemble(BoltzmannEnsemble):

    @property
    def q(self):
        """
        Tsallis parameter
        """
        return self._q.get()

    @q.setter
    def q(self, value):
        value = float(value)
        if value < 1.:
            msg = 'Tsallis q must be greater or equal to one'
            raise ValueError(msg)
        self._q.set(value)

    @property
    def E_min(self):
        """
        Minimum energy
        """
        return self._E_min.get()

    @E_min.setter
    def E_min(self, value):
        self._E_min.set(value)

    def __init__(self, name, forcefield):

        super(TsallisEnsemble, self).__init__(name, forcefield)

        self._q = Scale(self.name + '.q')
        self.params.add(self._q)

        self._E_min = Location(self.name + '.E_min')
        self.params.add(self._E_min)

    def log_prob(self):

        if self.q == 1.:
            return super(TsallisEnsemble, self).log_prob()

        else:
            coords = self.params['coordinates'].get()
            
            E = self.beta * self.forcefield.energy(coords)
            q = self.q
            E_min = self.beta * self.E_min
            
            return - q / (q-1) * np.log(1 + (q-1) * (E-E_min)) - E_min

    def update_forces(self):

        if self.q == 1.:
            super(TsallisEnsemble, self).update_forces()

        else:

            coords = self.params['coordinates'].get()

            self._forces[...] = 0.

            E  = self.forcefield.ctype.update_gradient(
                coords, self._forces, self.forcefield.types, 1)
            q  = self.q
            E *= self.beta
            E_min = self.beta * self.E_min
            f = - self.beta * q / (1 + (q-1) * (E-E_min))
        
            self._forces *= f

            self.params['forces']._value += self._forces

        
