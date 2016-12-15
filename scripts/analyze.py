"""
Some classes and functions for analysing chromosome conformations generated with
Hamiltonian Monte Carlo.
"""
import numpy as np
import pylab as plt

from isdhic import utils
from csb.bio.utils import distance_matrix

from collections import OrderedDict

class Ensemble(object):

    def __init__(self, samples):

        self.samples = samples

    def average_distances(self, burnin=0, thining=1):
        
        d = 0.
        for x in self.samples[burnin::thining]:
            d += distance_matrix(x)
        d/= len(X)

        return d

    def calculate_energies(self, posterior, burnin=0, thining=1):

        energies = OrderedDict()
        for p in posterior: energies[p.name] = []
        
        coords = posterior.params['coordinates']
        
        for x in self.samples[burnin::thining]:

            coords.set(x)
            posterior.update()

            for p in posterior: energies[p.name].append(-p.log_prob())

        for name in energies: energies[name] = np.array(energies[name])

        return energies

if __name__ == '__main__':
    
    pymol = utils.ChainViewer()

    X = np.array([state.positions for state in hmc.samples]).reshape(len(hmc.samples),-1,3)
    V = np.array([state.potential_energy for state in hmc.samples])
    K = np.array([state.kinetic_energy for state in hmc.samples])

    ensemble = Ensemble(X)

    ## show distance matrix and superimpose contacts

    n_particles = X.shape[1]
    limits = (0,n_particles)
    
    fig, ax = plt.subplots(1,1,figsize=(10,10),subplot_kw=dict(xlim=limits,ylim=limits))
    ax.matshow(ensemble.average_distances(-500,10), origin='lower')
    ax.scatter(*np.transpose(list(posterior['contacts'].mock.pairs)), color='w', alpha=0.7, s=80)

    ## calculate energies and plot energy traces

    energies = ensemble.calculate_energies(posterior,burnin=100)

    fig, ax = plt.subplots(1,len(energies),figsize=(16,4), subplot_kw=dict(xlabel='HMC iteration'))

    for i, name in enumerate(energies.keys()):
        ax[i].plot(energies[name],color='k',lw=3,alpha=0.7)
        ax[i].set_ylabel('energy (-log probability)')
        ax[i].set_title(name)
        ax[i].xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax[i].yaxis.get_major_formatter().set_powerlimits((0, 1))
        
    fig.tight_layout()

    ## plot observed and backcalculated data

    models = [model for model in posterior.likelihoods if len(model.data) > 1]

    fig, ax = plt.subplots(1,len(models),figsize=(10,4), subplot_kw=dict(xlabel='data point'))

    for i, model in enumerate(models):
        ax[i].plot(model.mock.get(),color='k',lw=3,alpha=0.7,label='model')
        ax[i].plot(model.data,color='r',lw=2,ls='--',alpha=0.7,label='observed')
        ax[i].set_title(model.name)
        ax[i].legend(loc=3)
        
    fig.tight_layout()

    
