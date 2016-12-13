"""
Test the PROLSQ force field
"""
import isdhic
import numpy as np

from scipy import optimize
from isdhic.core import take_time

n_particles = 100
boxsize     = 10
coords      = np.ascontiguousarray(np.random.rand(3*n_particles) * boxsize)
universe    = isdhic.Universe(n_particles)
forcefield  = isdhic.ForcefieldFactory.create_forcefield('prolsq', universe)

print forcefield.energy(coords), 

E = forcefield.ctype.update_gradient(coords, universe.forces, forcefield.types, 1)
print E

a = universe.forces.flatten()

msg = 'eps={0:.0e}, norm={1:.2e}, corr={2:.1f}'

for eps in np.logspace(-3,-8,8):

    b = optimize.approx_fprime(coords, forcefield.energy, eps)
    print msg.format(eps, np.fabs((a-b)/(np.fabs(a)+1e-300)).max(), np.corrcoef(a,b)[0,1]*100)
