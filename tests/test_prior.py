"""
Testing functionality of the Parameter class
"""
import utils
import isdhic
import numpy as np

from isdhic.core import take_time

if __name__ == '__main__':

    n_particles  = int(1e4)-1
    diameter     = 4.
    universe     = isdhic.Universe(n_particles)
    coords       = isdhic.Coordinates(universe)
    forcefield   = isdhic.ForcefieldFactory.create_forcefield('rosetta',universe)
    prior        = isdhic.PriorCoordinates('boltzmann',forcefield)

    params = isdhic.Parameters()
    isdhic.Probability.set_params(params)
    
    for param in (coords,):
        params.add(param)

    coords.set(utils.randomwalk(n_particles) * diameter)

    print forcefield.energy(coords.get())
    print prior.log_prob()
