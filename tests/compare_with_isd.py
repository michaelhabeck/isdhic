"""
Comparison with old ISD version.
"""
import os
import sys
import isdhic
import utils
import numpy as np

from isdhic.core import take_time

## generate posterior with isd

path   = os.path.expanduser('~/projects/hic2/py')
run    = 'geo_rosetta_contacts_Rg'
script = os.path.join(path, '{}.py'.format(run))

if not path in sys.path: sys.path.insert(0, path)

with open(script) as f:

    exec f.read()

    for name in posterior.likelihoods:
        posterior.likelihoods[name].enabled = 0
    L_bbone = posterior.likelihoods['backbone']
    
## make isdhic classes

n_particles  = len(posterior.universe.atoms)

params       = isdhic.Parameters()
isdhic.Probability.set_params(params)

universe     = utils.create_universe(n_particles, beadsize)
coords       = isdhic.Coordinates(universe)

forcefield   = isdhic.ForcefieldFactory.create_forcefield('rosetta',universe)
forcefield.d = np.array([[beadsize]])
forcefield.k = np.array([[0.0486]])

contacts     = isdhic.ModelDistances(coords, contacts, 'contacts')
logistic     = isdhic.Logistic(contacts.name, L.data.values, contacts)

connectivity = zip(range(n_particles-1),range(1,n_particles))
backbone     = isdhic.ModelDistances(coords, connectivity, 'backbone')
lowerupper   = isdhic.LowerUpper(backbone.name, L_bbone.data.values, backbone,
                                 L_bbone.error_model.lower_bounds,
                                 L_bbone.error_model.upper_bounds)
                                 
for param in (coords,): params.add(param)

tsallis = isdhic.TsallisEnsemble('tsallis',forcefield)
tsallis.E_min = prior.E_min
tsallis.beta  = prior.beta

tsallis.q = 1.06
posterior.set_q(tsallis.q)

logistic.alpha = L.error_model.alpha
lowerupper.tau = L_bbone.error_model.k

## Tsallis ensemble only

print '\n--- testing Tsallis ensemble ---\n'

with take_time('calculating energy with isdhic'):
    a = -tsallis.log_prob()
with take_time('calculating energy with isd'):
    b = posterior.torsion_posterior.energy(coords.get())
print '  results={0:0.5e}, {1:0.5e}\n'.format(a,b)

with take_time('calculating forces with isdhic'):
    a = -tsallis.gradient()
with take_time('calculating forces with isd'):
    b = posterior.torsion_posterior.gradient(coords.get())

print '  max discrepancy={0:.3e}, coor={1:.1f}'.format(
    np.fabs(a-b).max(), np.corrcoef(a,b)[0,1]*100)

## logistic likelihood only

print '\n--- testing Logistic likelihood ---\n'

L.enabled, prior.enabled, L_bbone.enabled = 1, 0, 0

with take_time('evaluating logistic likelihood with isdhic'):
    contacts.update()
    a = logistic.log_prob()
with take_time('evaluating logistic likelihood with isd'):
    L.fill_mock_data()
    b = -L.error_model.energy(L)
print '  results={0:0.5e}, {1:0.5e}\n'.format(a,b)

a = np.ascontiguousarray(universe.forces.reshape(-1,))
a[...] = 0.

with take_time('calculating forces with isdhic'):
    contacts.update()
    logistic.update_forces(a)
with take_time('calculating forces with isd'):
    b = posterior.torsion_posterior.gradient(coords.get())

b *= -1
print '  max discrepancy={0:.3e}, coor={1:.1f}'.format(
    np.fabs(a-b).max(), np.corrcoef(a,b)[0,1]*100)

## lowerupper likelihood only

print '\n--- testing LowerUpper model ---\n'

L.enabled, prior.enabled, L_bbone.enabled = 0, 0, 1

with take_time('evaluating lowerupper model with isdhic'):
    backbone.update()
    a = lowerupper.log_prob()
with take_time('evaluating lowerupper model with isd'):
    L_bbone.fill_mock_data()
    b = -L_bbone.error_model.energy(L_bbone)
print '  results={0:0.5e}, {1:0.5e}\n'.format(a,b)

a = np.ascontiguousarray(universe.forces.reshape(-1,))
a[...] = 0.

with take_time('calculating forces with isdhic'):
    backbone.update()
    lowerupper.update_forces(a)
with take_time('calculating forces with isd'):
    b = posterior.torsion_posterior.gradient(coords.get())

b *= -1
print '  max discrepancy={0:.3e}, coor={1:.1f}'.format(
    np.fabs(a-b).max(), np.corrcoef(a,b)[0,1]*100)

## full posterior

print '\n--- testing full posterior ---\n'

L.enabled, prior.enabled, L_bbone.enabled = 1, 1, 1

with take_time('evaluating posterior with isdhic'):
    for mock in (backbone, contacts):
        mock.update()
    a = 0.
    for model in (tsallis, logistic, lowerupper):
        a += model.log_prob()

with take_time('evaluating posterior with isd'):
    b = posterior.torsion_posterior.energy(coords.get())

b *= -1
print '  results={0:0.5e}, {1:0.5e}\n'.format(a,b)

with take_time('calculating forces with isdhic'):

    a = tsallis.gradient()

    for model in (logistic, lowerupper):
        model.mock.update()
        model.update_forces(a)
        
with take_time('calculating forces with isd'):
    b = posterior.torsion_posterior.gradient(coords.get())

b *= -1
print '  max discrepancy={0:.3e}, coor={1:.1f}'.format(
    np.fabs(a-b).max(), np.corrcoef(a,b)[0,1]*100)

