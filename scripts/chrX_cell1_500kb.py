import os
import sys
import isdhic
import numpy as np

from isdhic.chromosome import ChromosomeSimulation

n_particles  = 333
forcefield   = 'rosetta'
chrsize      = 166500 * 1e3
factor       = 1.5
diameter     = 4.
filename     = '../data/GSM1173493_cell-1.txt'
k_backbone   = 250.
n_rog        = 10
mu_rog       = 3.7e3 / (2.58 * 107.5)
sigma_rog    = 0.3e3 / (2.58 * 107.5)
tau_rog      = n_rog / sigma_rog**2

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

model.data[0] = mu_rog
model.tau     = tau_rog

