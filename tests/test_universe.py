"""
Testing functionality of the Universe class
"""
import numpy as np

from isdhic import utils

pymol = utils.ChainViewer()

universe = utils.create_universe(n_particles=1e4-1, diameter=4.)

for counter, particle in enumerate(universe):

    if counter % 100: continue

    print particle, np.round(universe.coords[particle.serial])

