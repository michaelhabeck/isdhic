from .core import Nominable
from .params import Parameters

class Probability(Nominable):
    """Probability

    Generic class that will be subclassed by all probabilistic
    models that are needed to describe projection data.
    """
    _params = None

    @classmethod
    def set_params(cls, params):
        
        if not isinstance(params, Parameters):
            msg = 'Argument must be an instance of the Parameters class'            
            raise TypeError(msg)
        
        cls._params = params
        
    @property
    def params(self):
        params = self.__class__._params
        if params is None:
            msg = 'Parameters have not been set'
            raise Exception(msg)
        return params

    def log_prob(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def __init__(self, name):
        self.name = name

