"""
This script sets up posterior probability for the structure of
the X chromosome based on single-cell Hi-C contact data. 
"""
from isdhic import HiCParser
from isdhic.chromosome import ChromosomeSimulation

## Each bead represents 500 Kb of chromatin and has diameter of
## roughly 430 nm (assuming a chromatin density of 12 Mb / mu^3

resolution  =    500 * 1e3
chrsize     = 166500 * 1e3
n_particles = int(chrsize / resolution)

## Ramped Lennard-Jones potential preventing inter-bead clashes.
## The bead diameter is 4 and chosen for compatibility with
## parameters used in protein simulations

forcefield  = 'rosetta'
diameter    = 4.

## Single-cell Hi-C data from Nagano et al., the contact distance
## is factor x diameter which equals to 6. Data are from cell 1

filename    = '../data/GSM1173493_cell-1.txt'
factor      = 1.5

## Beads-on-string-model: consecutive beads are connected by a
## harmonic potential with a flat bottom. The harmonic force
## becomes active when the distance between beads i, i+1 exceeds
## the bead diameter. The force constant controls the strength of
## that force

k_backbone  = 250.

## Radius of gyration (rog) restraint derived from FISH data. The
## size of the X chromosome is roughly 2.58 x rog. Nagano et al.
## determined a size of 3.7 +/- 0.3 mu based on 10 FISH measurements

n_rog       = 10
mu_rog      = 3.7e3 / (2.58 * 430. / diameter)
sigma_rog   = 0.3e3 / (2.58 * 430. / diameter)
tau_rog     = n_rog / sigma_rog**2

## Read data and map chromosomal positions onto 500Kb beads

parser      = HiCParser(filename, 'X', 'X')
datasets    = parser.parse()
dataset     = datasets[('X','X')]

dataset.coarsen(n_particles, chrsize)
dataset.remove_self_contacts()

## Set up posterior probability using the above settings

simulation  = ChromosomeSimulation(n_particles,
                                    forcefield = forcefield,
                                    k_backbone = k_backbone,
                                    diameter   = diameter,
                                    factor     = factor)

posterior = simulation.create_chromosome(list(dataset))
universe  = simulation.universe
coords    = simulation.params['coordinates']
forces    = simulation.params['forces']

posterior['rog'].data[0] = mu_rog
posterior['rog'].tau     = tau_rog

