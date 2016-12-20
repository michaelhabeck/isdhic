import numpy as np

from .core import Nominable
from .params import Parameters, Scale, Precision

from csb.core import validatedproperty

class Probability(Nominable):
    """Probability

    Generic class that will be subclassed by all probabilistic models.
    """
    def log_prob(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def __init__(self, name, params=None):
        self.name   = name
        self.params = params or Parameters()
        
class Likelihood(Probability):

    @validatedproperty
    def data(values):
        """
        Observed data stored in a single vector.
        """
        return np.ascontiguousarray(values)

    @validatedproperty
    def grad(values):
        """
        Array for storing derivatives of likelihood with respect to mock data.
        """
        return np.ascontiguousarray(values)

    @property
    def beta(self):
        """
        Inverse temperature.
        """
        return self._beta.get()

    @beta.setter
    def beta(self, value):
        self._beta.set(value)
    
    def __init__(self, name, data, mock, beta=1.0, params=None):
        """Likelihood

        Initialize likelihood by providing a name, the raw data
        and a theory for calculating idealized obserations.

        Parameters
        ----------

        name : string
          name of the likelihood function

        data : iterable
          list of raw data points

        mock : instance of Parameters
          theory for calculating idealized data (needs to implement
          update_forces)

        beta : non-negative float
          inverse temperature used in tempering and annealing          
        """
        super(Likelihood, self).__init__(name, params)

        self.data = data
        self.mock = mock
        self.grad = np.zeros(data.shape)

        self._beta = Scale(self.name + '.beta')
        self.params.add(self._beta)
        self.beta  = beta

    def update(self):
        self.mock.update(self.params)
        
    def update_derivatives(self):
        """
        Calculate derivative of log likelihood with respect to mock
        data.
        """
        raise NotImplementedError
        
    def update_forces(self):
        """
        Update Cartesian forces by applying the chain rule.
        """
        self.update_derivatives()
        self.mock.update_forces(self.grad, self.params)

class Normal(Likelihood):
    """Normal

    Likelihood implementing a Normal distribution. It has a single
    nuisance parameter: the precision, i.e. inverse variance
    """
    @property
    def precision(self):
        """
        Inverse variance
        """
        return self._precision

    @property
    def tau(self):
        return self._precision.get()

    @tau.setter
    def tau(self, value):
        self._precision.set(value)

    @property
    def sigma(self):
        """
        Standard deviation
        """
        return 1 / self.tau**0.5

    def __init__(self, name, data, mock, precision=1.0, params=None):

        super(Normal, self).__init__(name, data, mock, params=params)

        self._precision = Precision(self.name + '.precision')
        self.tau = precision

    @property
    def logZ(self):
        """
        Normalization constant of the Normal distribution
        """
        return - 0.5 * len(self.data) * np.log(0.5 * self.tau / np.pi)
        
    def log_prob(self):

        diff = self.mock.get() - self.data

        log_prob = - 0.5 * self.tau * np.dot(diff,diff) - self.logZ

        return self.beta * log_prob

    def update_derivatives(self):

        self.grad[...] = self.beta * self.tau * (self.data - self.mock.get())

    def __str__(self):

        s = super(Normal, self).__str__()

        return s.replace(')', ', precision={0:0.3f})'.format(self.tau))

class LowerUpper(Normal):
    """LowerUpper

    Error model implementing a Normal distribution with a flat plateau. The
    start and end of the plateau are marked by lower bounds (stored in 'lower')
    and upper bounds (stored in 'upper')
    """
    @validatedproperty
    def lower(values):
        return np.ascontiguousarray(values)

    @validatedproperty
    def upper(values):
        return np.ascontiguousarray(values)

    @property
    def logZ(self):
        """
        Normalization constant
        """
        from .lowerupper import logZ

        return logZ(self.lower, self.upper, self.tau)

    def __init__(self, name, data, mock, lower, upper, precision=1.0, params=None):

        super(LowerUpper, self).__init__(name, data, mock, precision, params=params)

        self.lower = lower
        self.upper = upper

        self.validate()

    def log_prob(self):

        from .lowerupper import log_prob

        lgp = log_prob(self.data, self.mock.get(), self.lower, self.upper)

        return 0.5 * self.beta * self.tau * lgp - self.beta * self.logZ
    
    def update_derivatives(self):

        from .lowerupper import update_derivatives

        update_derivatives(self.mock.get(), self.grad, self.lower,
                           self.upper, self.beta * self.tau)

    def validate(self):
        if np.any(self.lower > self.upper):
            msg = 'Lower bounds must be smaller than upper bounds'
            raise ValueError(msg)
        
class Logistic(Likelihood):
    """Logistic

    Logistic likelihood for binary observations.
    """
    @property
    def steepness(self):
        """
        Steepness of logistic function.
        """
        return self._steepness
    
    @property
    def alpha(self):
        """
        Returns the current value of the steepness parameter.
        """
        return self._steepness.get()

    @alpha.setter
    def alpha(self, value):
        self._steepness.set(value)
    
    def __init__(self, name, data, mock, steepness=1.0, params=None):
        
        super(Logistic, self).__init__(name, data, mock, params=params)

        self._steepness = Scale(self.name + '.steepness')        
        self.alpha = steepness
        
    def log_prob(self):

        from .logistic import log_prob

        return self.beta * log_prob(self.data, self.mock.get(), self.alpha)

    def update_derivatives(self):

        from .logistic import update_derivatives

        update_derivatives(self.data, self.mock.get(), self.grad, self.alpha)

        self.grad *= self.beta

    def __str__(self):

        s = super(Logistic, self).__str__()
        s = s.replace(')', ', steepness={0:0.3f})'.format(self.alpha))
        
        return s

