"""
Comparison with old ISD version.
"""
import os
import sys
import isdhic
import numpy as np

from isdhic import utils
from isdhic.core import take_time

def report_log_prob(a, b):
    out = '  log_prob (isdhic / isd) = {0:0.5e} / {1:0.5e}\n'
    print out.format(a,b)
        
def report_gradient(a, b):
    print '  max discrepancy={0:.3e}, corr={1:.1f}'.format(
        np.fabs(a-b).max(), np.corrcoef(a,b)[0,1]*100)

def create_isd_posterior(run='geo_rosetta_contacts_Rg'):
    """
    Generate isd posterior
    """
    cwd    = os.getcwd()
    path   = os.path.expanduser('~/projects/hic2/py')
    script = os.path.join(path, '{}.py'.format(run))

    if not path in sys.path: sys.path.insert(0, path)

    with open(script) as f:
        exec f.read()

    os.chdir(cwd)

    return posterior, contacts

if __name__ == '__main__':

    posterior, contacts = create_isd_posterior()
    
    L_intra  = posterior.likelihoods['contacts']
    L_bbone  = posterior.likelihoods['backbone']
    L_rog    = posterior.likelihoods['Rg']
    prior    = posterior.conformational_priors['tsallis_prior']
    beadsize = L_bbone.data.values[0]
    
    ## make isdhic classes

    n_particles  = len(posterior.universe.atoms)

    params       = isdhic.Parameters()
    isdhic.Probability.set_params(params)

    universe     = utils.create_universe(n_particles, beadsize)
    coords       = isdhic.Coordinates(universe)
    forces       = isdhic.Forces(universe)

    forcefield   = isdhic.ForcefieldFactory.create_forcefield('rosetta',universe)
    forcefield.d = np.array([[beadsize]])
    forcefield.k = np.array([[0.0486]])

    contacts     = isdhic.ModelDistances(coords, contacts, 'contacts')
    logistic     = isdhic.Logistic(contacts.name, L_intra.data.values, contacts)

    connectivity = zip(range(n_particles-1),range(1,n_particles))
    backbone     = isdhic.ModelDistances(coords, connectivity, 'backbone')
    lowerupper   = isdhic.LowerUpper(backbone.name, L_bbone.data.values, backbone,
                                     L_bbone.error_model.lower_bounds,
                                     L_bbone.error_model.upper_bounds)

    radius       = isdhic.RadiusOfGyration(coords)
    normal       = isdhic.Normal(radius.name, L_rog.data.values, radius, L_rog.error_model.k)

    for param in (coords, forces): params.add(param)

    tsallis = isdhic.TsallisEnsemble('tsallis',forcefield)
    tsallis.E_min = prior.E_min
    tsallis.beta  = prior.beta

    tsallis.q = 1.06
    posterior.set_q(tsallis.q)

    logistic.alpha = L_intra.error_model.alpha
    lowerupper.tau = L_bbone.error_model.k

    ## Tsallis ensemble only

    print '\n--- testing Tsallis ensemble ---\n'

    L_intra.enabled, prior.enabled, L_bbone.enabled, L_rog.enabled = 0, 1, 0, 0

    with take_time('calculating energy with isdhic'):
        a = tsallis.log_prob()
    with take_time('calculating energy with isd'):
        b = posterior.torsion_posterior.energy(coords.get())

    report_log_prob(a,-b)

    with take_time('calculating forces with isdhic'):
        forces.set(0.)
        tsallis.update_forces()
    with take_time('calculating forces with isd'):
        b = posterior.torsion_posterior.gradient(coords.get())

    report_gradient(forces.get(),-b)

    ## logistic likelihood only

    print '\n--- testing Logistic likelihood ---\n'

    L_intra.enabled, prior.enabled, L_bbone.enabled, L_rog.enabled = 1, 0, 0, 0

    with take_time('evaluating logistic likelihood with isdhic'):
        contacts.update()
        a = logistic.log_prob()

    with take_time('evaluating logistic likelihood with isd'):
        L_intra.fill_mock_data()
        b = L_intra.error_model.energy(L_intra)

    report_log_prob(a,-b)

    with take_time('calculating forces with isdhic'):
        forces.set(0.)
        contacts.update()
        logistic.update_forces()

    with take_time('calculating forces with isd'):
        b = posterior.torsion_posterior.gradient(coords.get())

    report_gradient(forces.get(),-b)

    ## lowerupper likelihood only

    print '\n--- testing LowerUpper model ---\n'

    L_intra.enabled, prior.enabled, L_bbone.enabled, L_rog.enabled = 0, 0, 1, 0

    with take_time('evaluating lowerupper model with isdhic'):
        backbone.update()
        a = lowerupper.log_prob()

    with take_time('evaluating lowerupper model with isd'):
        L_bbone.fill_mock_data()
        b = L_bbone.error_model.energy(L_bbone)

    report_log_prob(a,-b)

    with take_time('calculating forces with isdhic'):
        forces.set(0.)
        backbone.update()
        lowerupper.update_forces()

    with take_time('calculating forces with isd'):
        b = posterior.torsion_posterior.gradient(coords.get())

    report_gradient(forces.get(),-b)

    ## radius of gyration only

    print '\n--- testing RadiusOfGyration model ---\n'

    L_intra.enabled, prior.enabled, L_bbone.enabled, L_rog.enabled = 0, 0, 0, 1

    with take_time('evaluating rog model with isdhic'):
        radius.update()
        a = normal.log_prob()

    with take_time('evaluating rog model with isd'):
        L_rog.fill_mock_data()
        b = L_rog.error_model.energy(L_rog)

    report_log_prob(a,-b)

    with take_time('calculating forces with isdhic'):
        forces.set(0.)
        radius.update()
        normal.update_forces()

    with take_time('calculating forces with isd'):
        b = posterior.torsion_posterior.gradient(coords.get())

    report_gradient(forces.get(),-b)

    ## full posterior

    print '\n--- testing full posterior ---\n'

    L_intra.enabled, prior.enabled, L_bbone.enabled, L_rog.enabled = 1, 1, 1, 1

    with take_time('evaluating posterior with isdhic'):
        for mock in (backbone, contacts, radius):
            mock.update()
        a = 0.
        for model in (tsallis, logistic, lowerupper, normal):
            a += model.log_prob()

    with take_time('evaluating posterior with isd'):
        b = posterior.torsion_posterior.energy(coords.get())

    report_log_prob(a,-b)

    with take_time('calculating forces with isdhic'):

        forces.set(0.)
        tsallis.update_forces()

        for model in (logistic, lowerupper, normal):
            model.mock.update()
            model.update_forces()

    with take_time('calculating forces with isd'):
        b = posterior.torsion_posterior.gradient(coords.get())

    report_gradient(forces.get(),-b)

    ## testing conditional posterior over conformational degrees of freedom

    print '\n--- testing conditional posterior ---\n'

    p_coords = isdhic.PosteriorCoordinates('Pr(x|D)', (lowerupper, logistic, normal), (tsallis,))

    with take_time('evaluating log probibility of {}'.format(p_coords)):
        lgp = p_coords.log_prob()
    print '  log_prob={0:.5e}'.format(lgp)

    a = forces.get().copy()

    with take_time('\nevaluating forces of {}'.format(p_coords)):
        forces.set(0.)
        p_coords.update_forces()

    report_gradient(forces.get(), a)
