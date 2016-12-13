"""
Conditional posteriors
"""
import numpy as np

from .model import Probability

class ConditionalPosterior(Probability):

    def __init__(self, name, likelihoods=None, priors=None):

        super(ConditionalPosterior, self).__init__(name)

        self.likelihoods = likelihoods or ()
        self.priors = priors or ()

    def log_prob(self):

        self.update()

        log_p = 0.
        
        for p in self.priors + self.likelihoods:
            log_p += p.log_prob()

        return log_p

    def update(self):
        """
        Overwrite to update specific parameters.
        """
        pass

class PosteriorCoordinates(ConditionalPosterior):

    def update(self):

        for model in self.likelihoods:
            model.mock.update()

    def update_forces(self):

        for prior in self.priors:
            prior.update_forces()

        for model in self.likelihoods:
            model.mock.update()
            model.update_forces()


