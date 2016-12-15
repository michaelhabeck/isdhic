"""
Utility class for creating a beads-on-string model of a chromosome.
"""
import numpy as np

from .utils import create_universe
from .prior import TsallisEnsemble
from .model import Probability, Normal, LowerUpper, Logistic
from .params import Forces, Coordinates, Parameters, ModelDistances, RadiusOfGyration
from .posterior import PosteriorCoordinates
from .forcefield import ForcefieldFactory

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

        self._universe = create_universe(self.n_particles)

    def create_params(self):

        params = Parameters()
        Probability.set_params(params)

        coords = Coordinates(self.universe)
        forces = Forces(self.universe)

        for param in (coords, forces): params.add(param)

        self._params = params

    def create_prior(self):

        forcefield   = ForcefieldFactory.create_forcefield(
            self.forcefield, self.universe)
        forcefield.d = np.array([[self.diameter]])
        forcefield.k = np.array([[self.k_forcefield]])

        prior = TsallisEnsemble('tsallis', forcefield)
        prior.beta   = self.beta
        prior.E_min  = self.E_min

        return prior
    
    def create_chain(self):

        connectivity = zip(range(self.n_particles-1),range(1,self.n_particles))
        coords       = self.params['coordinates']
        backbone     = ModelDistances(coords, connectivity, 'backbone')
        bonds        = np.ones(self.n_particles-1) * self.diameter
        lowerupper   = LowerUpper(backbone.name, bonds, backbone, 0 * bonds, bonds, self.k_backbone)

        return lowerupper

    def create_contacts(self, pairs):

        threshold = np.ones(len(pairs)) * self.factor * self.diameter
        coords    = self.params['coordinates']
        contacts  = ModelDistances(coords, pairs, 'contacts')
        logistic  = Logistic(contacts.name, threshold, contacts, self.steepness)

        return logistic

    def create_radius_of_gyration(self, Rg=0.):

        coords = self.params['coordinates']
        radius = RadiusOfGyration(coords)
        normal = Normal(radius.name, np.array([Rg]), radius)

        return normal

    def create_chromosome(self, contacts):

        self.create_universe()
        self.create_params()

        priors = (self.create_prior(),)

        likelihoods = (self.create_chain(),
                       self.create_contacts(contacts),
                       self.create_radius_of_gyration())

        posterior = PosteriorCoordinates(
            'chromosome structure', likelihoods=likelihoods, priors=priors)

        return posterior

