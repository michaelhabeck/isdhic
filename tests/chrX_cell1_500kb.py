import os
import sys
import isdhic
import numpy as np

from chromosome_model import ChromosomeSimulation

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

for model in posterior.likelihoods:
    print model

model.data[0] = mu_rog
model.tau     = tau_rog

if False:

    import compare_with_isd as compare
    from isdhic.core import take_time

    isd_posterior, _ = compare.create_isd_posterior()

    print '\n--- testing conditional posterior ---\n'

    with take_time('evaluating log probibility of {}'.format(posterior)):
        a = posterior.log_prob()

    with take_time('evaluating posterior with isd'):
        b = isd_posterior.torsion_posterior.energy(coords.get())

    compare.report_log_prob(a,-b)

    with take_time('\nevaluating forces of {}'.format(posterior)):
        forces.set(0.)
        posterior.update_forces()

    with take_time('evaluating forces with isd'):
        b = isd_posterior.torsion_posterior.gradient(coords.get())

    compare.report_gradient(forces.get(),-b)


        
