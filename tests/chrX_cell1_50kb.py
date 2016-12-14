import os
import sys
import isdhic
import numpy as np

from chromosome_model import ChromosomeSimulation

n_particles  = 3330
forcefield   = 'rosetta'
chrsize      = 166500 * 1e3
factor       = 1.5
diameter     = 4.
filename     = '../data/GSM1173493_cell-1.txt'
k_backbone   = 250.

parser       = isdhic.HiCParser(filename, 'X', 'X')
datasets     = parser.parse()
dataset      = datasets[('X','X')]

dataset.coarsen(n_particles, chrsize)
dataset.remove_self_contacts()

simulation   = ChromosomeSimulation(n_particles,
                                    forcefield = forcefield,
                                    k_backbone = k_backbone,
                                    diameter   = diameter,
                                    factor     = factor)

posterior  = simulation.create_chromosome(list(dataset))
universe   = simulation.universe
coords     = simulation.params['coordinates']
forces     = simulation.params['forces']

