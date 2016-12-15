from isdhic.core import take_time
from isdhic.chromosome import ChromosomeSimulation

import compare_with_isd as compare

## generate posterior with isd

isd_posterior, contacts = compare.create_isd_posterior()
beadsize = isd_posterior.likelihoods['backbone'].data.values[0]

## create isdhic posterior

n_particles  = len(isd_posterior.universe.atoms)
simulation   = ChromosomeSimulation(n_particles, diameter=beadsize)
posterior    = simulation.create_chromosome(contacts)
universe     = simulation.universe
coords       = simulation.params['coordinates']
forces       = simulation.params['forces']

posterior['rog'].data[...] = isd_posterior.likelihoods['Rg'].data.values
posterior['rog'].tau = isd_posterior.likelihoods['Rg'].error_model.k

print posterior
for pdf in posterior:
    print pdf

print '\n--- testing conditional posterior ---\n'

with take_time('evaluating log probability of {}'.format(posterior)):
    a = posterior.log_prob()

with take_time('evaluating posterior with isd'):
    b = isd_posterior.torsion_posterior.energy(coords.get())

compare.report_log_prob(a,-b)

with take_time('\nevaluating forces of {}'.format(posterior)):
    forces.set(0.)
    posterior.update_forces()

with take_time('calculating forces with isd'):
    b = isd_posterior.torsion_posterior.gradient(coords.get())

compare.report_gradient(forces.get(),-b)



