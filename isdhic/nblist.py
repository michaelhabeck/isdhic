"""
Wrapper class for neighbor list.
"""
from ._isdhic import nblist
from .core import ctypeproperty, CWrapper

class NBList(CWrapper):

    @ctypeproperty(float)
    def cellsize():
        pass

    @ctypeproperty(int)
    def n_cells():
        pass

    @ctypeproperty(int)
    def n_per_cell():
        pass

    @ctypeproperty(int)
    def n_atoms():
        pass
    
    def __init__(self, cellsize, n_cells, n_per_cell, n_atoms):
        """
        'cellsize':   length of a single cell,
        'n_cells':    number of cells in one dimension,
        'n_per_cell': max. no. of atoms assignable to one cell.
        """
        self.init_ctype()

        self.cellsize   = cellsize
        self.n_per_cell = n_per_cell
        self.n_atoms    = n_atoms

        ## do this at last        
        self.n_cells    = n_cells
        
        self.enable()

    def init_ctype(self):
        self.ctype = nblist()              

    def enable(self, enable = 1):
        self.ctype.enabled = int(enable)

    def disable(self):
        self.enable(0)

    def is_enabled(self):
        return self.ctype.enabled == 1

    def update(self, universe, update_box=True):
        self.ctype.update(universe.coords, int(update_box))

    def __str__(self):

        s = '{0}({1},{2},{3},{4})'

        return s.format(self.__class__.__name__,
                        self.cellsize, self.n_cells,
                        self.n_per_cell, self.n_atoms)

    __repr__ = __str__

