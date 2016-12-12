"""
Testing neighbor list
"""
import isdhic
import numpy as np

from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform

from csb.bio.utils import distance_matrix

def kd_pairs(coords, cellsize):

    tree = cKDTree(coords)
    pairs = tree.query_pairs(cellsize)

    return pairs

def nblist_contacts(nblist, universe):

    nblist.update(universe)
    return nblist.ctype.contacts

def nblist_pairs(nblist, universe):

    contacts = nblist_contacts(nblist, universe)
    contacts = [contacts[i,:l] for i, l in enumerate(nblist.ctype.n_contacts.tolist())]

    return [(i,j) for i in range(len(contacts)) for j in contacts[i]]

def pairs_to_matrix(pairs, n_particles):

    c = np.zeros((n_particles,n_particles),'i')

    for i, j in pairs:
        c[i,j] = 1
        c[j,i] = 1

    return c

if __name__ == '__main__':

    n_particles = 1000
    box_length  = 10 / 2.1

    cellsize   = 2. / 10 * box_length
    n_cells    = 500
    n_per_cell = 500

    coords     = np.random.rand(n_particles,3) * box_length
    nblist     = isdhic.NBList(cellsize, n_cells, n_per_cell, n_particles)
    universe   = isdhic.Universe(n_particles)
    universe.coords[...] = coords

    pairs = nblist_pairs(nblist, universe)

    d = distance_matrix(universe.coords)
    A = pairs_to_matrix(pairs, n_particles)
    B = pairs_to_matrix(kd_pairs(coords, cellsize), n_particles)

    print 'Are contact matrices identical? ---',
    print np.all(squareform(A,checks=False) ==
                 squareform((d<nblist.cellsize).astype('i'),checks=False))

