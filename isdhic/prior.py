"""
Collection of priors.
"""
from .model import Probability

class PriorCoordinates(Probability):

    def __init__(self, name, forcefield):

        super(PriorCoordinates, self).__init__(name)

        self.forcefield = forcefield

    def log_prob(self):

        coords = self.params['coordinates'].get()

        return - self.forcefield.energy(coords)


