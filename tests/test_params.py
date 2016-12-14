"""
Testing functionality of the Parameter class
"""
import utils
import isdhic
import numpy as np

from isdhic.core import take_time

def random_pairs(n_particles, n_pairs):
    """
    Create set of random distances
    """
    pairs = set()

    while len(pairs) < n_pairs:
        i = np.random.randint(n_particles)
        j = np.random.randint(n_particles)
        if i == j: continue
        i, j = min(i,j), max(i,j)
        pairs.add((i,j))

    return list(pairs)

def numpy_distances(distances, universe):

    return np.sum((universe.coords[distances.first_index] -
                   universe.coords[distances.second_index])**2,1)**0.5

if __name__ == '__main__':

    n_particles  = int(1e4)-1
    n_distances  = 10000
    diameter     = 4.
    universe     = utils.create_universe(n_particles, diameter)

    precision    = isdhic.Precision('lowerupper')
    coords       = isdhic.Coordinates(universe)
    pairs        = random_pairs(n_particles, n_distances)
    connectivity = zip(np.arange(1,n_particles),np.arange(n_particles-1))
    contacts     = isdhic.ModelDistances(coords, pairs, 'contacts')
    chain        = isdhic.ModelDistances(coords, connectivity, 'chain')
    distances    = isdhic.ModelDistances(coords, connectivity + pairs, 'all')

    params = isdhic.Parameters()

    for param in (precision, coords, distances):
        params.add(param)

    print 'Getting/setting of Cartesian coordinates works? -',
    print np.fabs(universe.coords.flatten()-coords.get()).max() < 1e-10
    print 

    print 'Comparing cython with numpy implementation\n'

    with take_time('\tevaluate "{}" using cython'.format(contacts.name)):
        contacts.update()

    with take_time('\tevaluate "{}" using numpy'.format(contacts.name)):
        d_numpy = numpy_distances(contacts, universe)
        
    print '\n\tmax discrepancy between distances: {0:.1e}\n'.format(
        np.fabs(contacts.get() - d_numpy).max())

    print 'Does it pay off to evaluate all distances at once?\n'

    with take_time('\tfirst "{0}", then "{1}"'.format(chain.name,contacts.name)):
        chain.update()
        contacts.update()

    with take_time('\t"{0}"'.format(distances.name)):
        distances.update()

    print '\n\tmax discrepancy between distances: {0:.1e}\n'.format(
        np.fabs(distances.get() - np.concatenate((chain.get(), contacts.get()),0)).max())

    print params

    if False:

        viewer = utils.ChainViewer()
        viewer(universe.coords)
