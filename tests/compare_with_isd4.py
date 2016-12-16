from isdhic.core import take_time
from isdhic.chromosome import ChromosomeSimulation

import compare_with_isd as compare

## create posteriors with isd and isdhic

with take_time('creating isdhic posterior'):

    filename   = '../scripts/chrX_cell1_50kb.py'
    with open(filename) as script:
        exec script

with take_time('creating isd posterior'):
    isd_posterior, _ = compare.create_isd_posterior(run='geo_rosetta_contacts_Rg_highres')

## set nuisance parameters
    
isd_posterior.likelihoods['contacts'].error_model.alpha = posterior['contacts'].alpha
isd_posterior.likelihoods['Rg'].error_model.k = posterior['rog'].tau

state = isd_posterior.as_state()
state.torsion_angles[...] = posterior.params['coordinates'].get()

isd_posterior.energy(state)

out = '{2}: {0:.5e} / {1:.5e}'

print '\n' + out.format(-posterior.log_prob(), state.E, 'posterior')

for p in posterior:

    E_isd = state[p.name].E_data if p.name in state.sub_states else None
    E_isd = state.E_phys if p.name == 'tsallis' else E_isd 
    E_isd = E_isd or state['Rg'].E_data + 0.5 * np.log(2*np.pi)

    print out.format(-p.log_prob(), E_isd, '  ' + p.name)

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
