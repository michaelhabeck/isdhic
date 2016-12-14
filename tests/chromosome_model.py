import os
import sys
import utils
import isdhic
import numpy as np
import compare_with_isd as compare

from isdhic.core import take_time

class ChromosomeSimulation(object):

    def __init__(self, n_particles, forcefield='rosetta', diameter=4.0, **settings):

        self.n_particles  = int(n_particles)
        self.forcefield   = str(forcefield)
        self.diameter     = float(diameter)
        self.k_backbone   = settings.get('k_backbone', 250.)
        self.k_forcefield = settings.get('k_forcefield', 0.0486)
        self.beta         = settings.get('beta', 1.0)
        self.E_min        = settings.get('E_min', -100.)
        self.steepness    = settings.get('steepness', 100.)
        self.factor       = settings.get('factor', 1.5)
        
        self._universe = None
        self._params   = None
        
    @property
    def universe(self):
        
        if self._universe is None:
            msg = 'Universe has not been created'
            raise Exception(msg)
        
        return self._universe
    
    @property
    def params(self):
        
        if self._params is None:
            msg = 'Parameters have not been created'
            raise Exception(msg)
        
        return self._params
    
    def create_universe(self):

        self._universe = utils.create_universe(self.n_particles)

    def create_params(self):

        params = isdhic.Parameters()
        isdhic.Probability.set_params(params)

        coords = isdhic.Coordinates(self.universe)
        forces = isdhic.Forces(self.universe)

        for param in (coords, forces): params.add(param)

        self._params = params

    def create_prior(self):

        forcefield   = isdhic.ForcefieldFactory.create_forcefield(
            self.forcefield, self.universe)
        forcefield.d = np.array([[self.diameter]])
        forcefield.k = np.array([[self.k_forcefield]])

        prior = isdhic.TsallisEnsemble('tsallis', forcefield)
        prior.beta   = self.beta
        prior.E_min  = self.E_min

        return prior
    
    def create_chain(self):

        connectivity = zip(range(self.n_particles-1),range(1,self.n_particles))
        coords       = self.params['coordinates']
        backbone     = isdhic.ModelDistances(coords, connectivity, 'backbone')
        bonds        = np.ones(self.n_particles-1) * self.diameter
        lowerupper   = isdhic.LowerUpper(backbone.name, bonds, backbone, 0 * bonds, bonds, self.k_backbone)

        return lowerupper

    def create_contacts(self, pairs):

        threshold = np.ones(len(pairs)) * self.factor * self.diameter
        coords    = self.params['coordinates']
        contacts  = isdhic.ModelDistances(coords, pairs, 'contacts')
        logistic  = isdhic.Logistic(contacts.name, threshold, contacts, self.steepness)

        return logistic

    def create_radius_of_gyration(self, Rg=0.):

        coords = self.params['coordinates']
        radius = isdhic.RadiusOfGyration(coords)
        normal = isdhic.Normal(radius.name, np.array([Rg]), radius)

        return normal

    def create_chromosome(self, contacts):

        self.create_universe()
        self.create_params()

        priors = (self.create_prior(),)

        likelihoods = (self.create_chain(),
                       self.create_contacts(contacts),
                       self.create_radius_of_gyration())

        posterior = isdhic.PosteriorCoordinates(
            'posterior_xyz', likelihoods=likelihoods, priors=priors)

        return posterior

if __name__ == '__main__':

    ## generate posterior with isd

    posterior, contacts = compare.create_isd_posterior()
    beadsize = posterior.likelihoods['backbone'].data.values[0]
    
    n_particles  = len(posterior.universe.atoms)
    simulation   = ChromosomeSimulation(n_particles, diameter=beadsize)
    posterior_x  = simulation.create_chromosome(contacts)
    universe     = simulation.universe
    coords       = simulation.params['coordinates']
    forces       = simulation.params['forces']
    
    for model in posterior_x.likelihoods:
        print model

    print '\n--- testing conditional posterior ---\n'

    with take_time('evaluating log probability of {}'.format(posterior_x)):
        a = posterior_x.log_prob()

    with take_time('evaluating posterior with isd'):
        b = posterior.torsion_posterior.energy(coords.get())

    compare.report_log_prob(a,-b)

    with take_time('\nevaluating forces of {}'.format(posterior_x)):
        forces.set(0.)
        posterior_x.update_forces()

    with take_time('calculating forces with isd'):
        b = posterior.torsion_posterior.gradient(coords.get())

    compare.report_gradient(forces.get(),-b)


        
