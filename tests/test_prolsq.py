"""
Test the PROLSQ force field
"""
import isdhic
import numpy as np

from scipy import optimize
from isdhic.core import take_time

n = 100#0
L = 10

coords   = np.random.rand(n,3) * L
nblist   = isdhic.NBList(5., 90, 500, n)
universe = isdhic.Universe(n)
universe.coords[...] = coords

nblist.update(universe)

forcefield = isdhic.PROLSQ()
forcefield.nblist = nblist
forcefield.n_types = 1
forcefield.k = np.array([[1.]])
forcefield.d = np.array([[1.]])

forcefield.link_parameters(universe)

def energy(x, forcefield=forcefield, universe=universe):
    universe.coords[...] = x.reshape(universe.coords.shape)
    return forcefield.energy(universe)

x = coords.flatten().copy()
print energy(x), 

E = forcefield.ctype.update_gradient(universe.ctype,forcefield.types,1)
print E

a = forcefield.update_gradient(universe).flatten().copy()

msg = 'eps={0:.0e}, norm={1:.2e}, corr={2:.1f}'

for eps in np.logspace(-3,-8,6):

    b = optimize.approx_fprime(x, energy, eps)
    print msg.format(eps, np.fabs((a-b)/(np.fabs(a)+1e-300)).max(), np.corrcoef(a,b)[0,1]*100)
