import isdhic
import numpy as np

from isdhic import Probability

class Gaussian(Probability):

    @property
    def x(self):
        return self._location.get()

    @x.setter
    def x(self, value):
        self._location.set(float(value))

    @property
    def mu(self):
        return self._mean.get()

    @mu.setter
    def mu(self, value):
        self._mean.set(float(value))

    @property
    def tau(self):
        return self._precision.get()

    @tau.setter
    def tau(self, value):
        self._precision.set(value)

    @property
    def sigma(self):
        return 1 / self.tau**0.5

    def __init__(self, name='gaussian'):

        super(Gaussian, self).__init__(name)

        self._location = isdhic.Location(self.name + '.x')
        self._mean = isdhic.Location(self.name + '.mu')
        self._precision = isdhic.Precision(self.name + '.tau')
        
        self._params.add(self._location)
        self._params.add(self._mean)
        self._params.add(self._precision)

    def log_prob(self):

        log_norm = 0.5 * (np.log(2 * np.pi) - np.log(self.tau))

        return - 0.5 * self.tau * (self.x - self.mu) ** 2 - log_norm

if __name__ == '__main__':

    params = isdhic.Parameters()
    isdhic.Probability.set_params(params)

    model = Gaussian()

    print params

    statement = "model.tau = -1.0"
    try:
        exec statement
    except Exception, msg:
        print 'WARNING: "{0}" failed because {1}'.format(statement, msg)

    print '{0}.log_prob() = {1:.2f}'.format(model, model.log_prob())
