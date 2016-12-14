"""
Non-bonded force fields.

PROLSQ uses a quartic repulsion term to penalize particle-particle
clashses.

ROSETTA is a ramped Lennard-Jones potential.
"""
import numpy as np

from ._isdhic import prolsq, rosetta
from .nblist import NBList
from .core import ctypeproperty, CWrapper, Nominable

class Forcefield(Nominable, CWrapper):
    """Forcefield

    Non-bonded force field enforcing volume exclusion. 
    """
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

    def __init__(self, name):

        self.init_ctype()
        self.set_default_values()

        self.name = name

    def set_default_values(self):

        self.nblist = None
        
        self.enable()

    def is_enabled(self):
        return self.ctype.enabled == 1

    def enable(self, enabled = 1):
        self.ctype.enabled = int(enabled)

    def disable(self):
        self.enable(0)

    def update_list(self, coords):
        """
        Update neighbor list.
        """
        self.ctype.nblist.update(coords.reshape(-1,3),1)
        
    def energy(self, coords, update=True):

        if update: self.update_list(coords)

        return self.ctype.energy(coords.reshape(-1,3), self.types)

    def update_gradient(self, coords, forces):

        self.ctype.update_gradient(coords, forces, self.types, 1)

    def __str__(self):

        s = '{0}(n_types={1:.2f})'
        
        return s.format(self.__class__.__name__, self.n_types)

    __repr__ = __str__

class PROLSQ(Forcefield):

    def __init__(self, name='PROLSQ'):
        super(PROLSQ, self).__init__(name)

    def init_ctype(self):
        self.ctype = prolsq()

class ROSETTA(Forcefield):

    @ctypeproperty(float)
    def r_max():
        pass

    @ctypeproperty(float)
    def r_lin():
        pass

    @ctypeproperty(float)
    def r_sw():
        pass

    def __init__(self, name='ROSETTA'):
        super(ROSETTA, self).__init__(name)

    def init_ctype(self):
        self.ctype = rosetta()

class ForcefieldFactory(object):

    @classmethod
    def create_forcefield(cls, name, universe):

        name = name.lower()

        if name == 'rosetta':
            forcefield = ROSETTA()
            cellsize = 5.51
            
        elif name == 'prolsq':
            forcefield = PROLSQ()
            cellsize = 3.71
            
        else:
            msg = 'Forcefield "{}" not supported'
            raise ValueError(msg.format(name))

        nblist   = NBList(cellsize, 100, 500, universe.n_particles)

        forcefield.nblist = nblist
        forcefield.n_types = 1
        forcefield.k = np.array([[1.]])
        forcefield.d = np.array([[1.]])
        forcefield.types = np.zeros(universe.n_particles,'i')

        return forcefield    
