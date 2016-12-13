import os
import time
import isdhic
import tempfile
import numpy as np

from csb.bio import structure

def randomwalk(n_steps, dim=3):
    """
    Generate a random walk in n-dimensional space by making steps of
    fixed size (unit length) and uniformly chosen direction.

    Parameters
    ----------

    n_steps :
      length of random walk, i.e. number of steps

    dim:
      dimension of embedding space (default: dim=3)
    """
    ## generate isotropically distributed bond vectors of
    ## unit length
    
    bonds = np.random.standard_normal((int(n_steps),int(dim)))
    norms = np.sum(bonds**2,1)**0.5
    bonds = (bonds.T / norms).T

    return np.add.accumulate(bonds,0)

def create_universe(n_particles=1, diameter=1):
    """
    Create a universe containing 'n_particles' Particles of
    given diameter. The coordinates of the particles follow
    a random walk in 3d space.

    Parameters
    ----------

    n_particles : non-negative number
      number of particles contained in universe

    diameter : non-negative float
      particle diameter
    """
    universe = isdhic.Universe(int(n_particles))
    universe.coords[...] = randomwalk(n_particles) * diameter

    return universe

def make_chain(coordinates, sequence=None, chainid='A'):
    """
    Creates a Chain instance from a coordinate array assuming
    that these are the positions of CA atoms
    """
    if sequence is None: sequence = ['ALA'] * len(coordinates)

    residues = []

    for i in range(len(sequence)):
        residue = structure.ProteinResidue(i+1, sequence[i], sequence_number=i+1)
        atom = structure.Atom(i+1, 'CA', 'C', coordinates[i])
        atom.occupancy = 1.0
        residue.atoms.append(atom)
        residues.append(residue)
        
    chain = structure.Chain(chainid, residues=residues)

    return chain

class Viewer(object):
    """Viewer
    
    A low-level viewer that allows one to visualize 3d arrays as molecular
    structures using programs such as pymol or rasmol.
    """
    def __init__(self, cmd, **options):

        import distutils.spawn

        exe = distutils.spawn.find_executable(str(cmd))
        if exe is None:
            msg = 'Executable {} does not exist'
            raise ValueError(msg.format(cmd))

        self._cmd = str(exe)
        self._options = options
        
    @property
    def command(self):
        return self._cmd

    def __str__(self):
        return 'Viewer({})'.format(self._cmd)

    def write_pdb(self, coords, filename):

        if coords.ndim == 2: coords = coords.reshape(1,-1,3)

        ensemble = structure.Ensemble()

        for i, xyz in enumerate(coords,1):

            chain  = make_chain(xyz)
            struct = structure.Structure('')
            struct.chains.append(chain)
            struct.model_id = i
            
            ensemble.models.append(struct)
        
        ensemble.to_pdb(filename)

    def __call__(self, coords, cleanup=True):
        """
        View 3d coordinates as a cloud of atoms.
        """
        tmpfile = tempfile.mktemp()

        self.write_pdb(coords, tmpfile)
        
        os.system('{0} {1}'.format(self._cmd, tmpfile))

        time.sleep(1.)

        if cleanup: os.unlink(tmpfile)

class ChainViewer(Viewer):
    """ChainViewer

    Specialized viewer for visualizing chain molecules. 
    """
    def __init__(self):

        super(ChainViewer, self).__init__('pymol')

        self.pymol_settings = ('set ribbon_trace_atoms=1',
                               'set ribbon_radius=0.75000',
                               'set cartoon_trace_atoms=1',
                               'set spec_reflect=0.00000',
                               'set opaque_background, off',
                               'bg_color white',
                               'as ribbon',
                               'util.chainbow()')

    def __call__(self, coords, cleanup=True):
        """
        View 3d coordinates as a ribbon
        """
        pdbfile = tempfile.mktemp() + '.pdb'
        pmlfile = pdbfile.replace('.pdb','.pml')
        
        self.write_pdb(coords, pdbfile)
        
        pmlscript = ('load {}'.format(pdbfile),
                     'hide') + \
                     self.pymol_settings

        with open(pmlfile, 'w') as f:
            f.write('\n'.join(pmlscript))
        
        os.system('{0} {1} &'.format(self._cmd, pmlfile))

        time.sleep(2.)

        if cleanup:
            os.unlink(pdbfile)
            os.unlink(pmlfile)
    

