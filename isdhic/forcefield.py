"""
Non-bonded force fields: PROLSQ uses a quartic repulsion term to penalize
particle-particle clashses. ROSETTA is a ramped Lennard-Jones potential.
"""
import numpy as np

from ._isdhic import prolsq, rosetta
from .core import ctypeproperty, CWrapper, Nominable

class PROLSQ(Nominable, CWrapper):

    @ctypeproperty(np.array)
    def k():
        pass

    @ctypeproperty(np.array)
    def d():
        pass

    @ctypeproperty(int)
    def n_types():
        pass

    @property
    def nblist(self):
        return self._nblist

    @nblist.setter
    def nblist(self, value):
        
        self._nblist = value
        if value is not None:
            self.ctype.nblist = value.ctype

    def __init__(self, name='PROLSQ'):

        self.init_ctype()
        self.set_default_values()

        self.name = name

    def init_ctype(self):
        self.ctype = prolsq()

    def set_default_values(self):

        self.nblist = None
        
        self.enable()

    def is_enabled(self):
        return self.ctype.enabled == 1

    def enable(self, enabled = 1):
        self.ctype.enabled = int(enabled)

    def disable(self):
        self.enable(0)

    def link_parameters(self, universe):
        self.types = np.zeros(universe.n_particles,'i')

    def energy(self, universe, update=True):

        ctype = self.ctype
        c_univ = universe.ctype

        if update: ctype.nblist.update(universe.coords,1)

        return ctype.energy(c_univ, self.types)

    def update_gradient(self, universe, update=True):

        ctype = self.ctype
        c_univ = universe.ctype
        
        if update: ctype.nblist.update(universe.coords,1)

        return c_univ.forces

    def __str__(self):

        s = '{0}(k={1:.2f}, n_types={2:.2f}, 14_decrement={3:.2f})'

        return s.format(self.__class__.__name__, self.K, self.n_types,
                        self.decrement_14)

    __repr__ = __str__

class ROSETTA(PROLSQ):

    @ctypeproperty(float)
    def r_max():
        pass

    @ctypeproperty(float)
    def r_lin():
        pass

    @ctypeproperty(float)
    def r_sw():
        pass

    def init_ctype(self):
        self.ctype = rosetta()

