from isdhic.core import take_time
from isdhic.chromosome import ChromosomeSimulation

import compare_with_isd as compare

## create posteriors with isd and isdhic

isd_posterior, _ = compare.create_isd_posterior()

filename   = '../scripts/chrX_cell1_500kb.py'
with open(filename) as script:
    exec script

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
