"""
Test pickling of c-wrappers
"""

from csb.io import load, dump
from isdhic import NBList

resolution = 500

with open('../scripts/chrX_cell1_{}kb.py'.format(resolution)) as script:
    exec script

forcefield = posterior['tsallis'].forcefield
nblist = forcefield.nblist

dump(nblist, '/tmp/nblist.pkl')
nblist2 = load('/tmp/nblist.pkl')

print nblist.ctype.update(posterior.params['coordinates'].get().reshape(-1,3),1)
print nblist2.ctype.update(posterior.params['coordinates'].get().reshape(-1,3),1)

dump(forcefield, '/tmp/forcefield.pkl')
forcefield2 = load('/tmp/forcefield.pkl')

print forcefield.energy(posterior.params['coordinates'].get().reshape(-1,3))
print forcefield2.energy(posterior.params['coordinates'].get().reshape(-1,3))

posterior['tsallis'].q = 1.03
posterior['rog'].beta = 0.2
posterior['contacts'].beta = 0.2

dump(posterior, '/tmp/posterior.pkl')
posterior2 = load('/tmp/posterior.pkl')

for p in posterior:
    print p.name, posterior[p.name].log_prob(), posterior2[p.name].log_prob()
