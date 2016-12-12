"""
Testing functionality of the Universe class
"""
import utils
import isdhic
import numpy as np

pymol = utils.ChainViewer()

n_particles = int(1e4)-1

beadsize = 4.
universe = isdhic.Universe(n_particles)

universe.coords[...] = utils.randomwalk(n_particles) * beadsize

for counter, particle in enumerate(universe):

    if counter % 100: continue

    print particle, np.round(universe.coords[particle.serial])

