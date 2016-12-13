import numpy as np

from .core import Nominable
from csb.core import validatedproperty
from collections import OrderedDict

class Parameter(Nominable):
    """Parameter

    Base class for the parameters of a probabilistic model.
    """
    def __init__(self, name):
        """Parameter

        Base class for the parameters of a probabilistic model.

        Parameters
        ----------
        
        name : string
          parameter's name
        """
        self.name   = name
        self._value = None
        
        self.set_default()
        
    def set_default(self):
        pass

    def get(self):

        if self._value is None:
            msg = 'Parameter not set'
            raise ValueError(msg)

        return self._value

    def set(self, value):

        raise NotImplementedError

    def update(self):
        
        pass

    def __str__(self):
        s = super(Parameter, self).__str__()
        v = self._value
        if np.iterable(v):
            v = '{0:.2f}'.format(v[0]) if type(v[0]) == float else v[0]
            v = '[{0},...]'.format(v)
        return s.replace(')',', {})'.format(v))

class Location(Parameter):
    """Location

    Scalar location parameter
    """
    def set_default(self):
        self.set(0.)

    def set(self, value):
        self._value = float(value)

class Scale(Parameter):
    """Scale

    Non-negative scalar factor
    """
    def set_default(self):
        self.set(1.)

    def set(self, value):
        value = float(value)
        if value < 0.:
            msg = '{} must be non-negative'
            raise ValueError(msg.format(self.__class__.__name__))

        self._value = value
        
class Precision(Scale):
    """Precision

    Inverse variance
    """
    pass

class Coordinates(Parameter):
    """Coordinates

    Cartesian coordinates
    """
    def __init__(self, universe, name='coordinates'):
        
        super(Coordinates, self).__init__(name)
        self._value = np.ascontiguousarray(universe.coords.reshape(-1,))

    def set(self, coords):
        self._value[...] = coords.reshape(-1,)

class Forces(Parameter):
    """Forces

    Cartesian gradient
    """
    def __init__(self, universe, name='forces'):
        
        super(Forces, self).__init__(name)
        self._value = np.ascontiguousarray(universe.forces.reshape(-1,))

    def set(self, value):
        if np.iterable(value):
            self._value[...] = value.reshape(-1,)
        else:
            self._value[...] = value

class Distances(Parameter):
    """Distances

    Class for storing and evaluating inter-particle distances. In addition
    to the distances, this class also stores the indices of the particles
    between which the distances are defined
    """
    @validatedproperty
    def first_index(values):
        """
        First element of the index tuple, i.e. if distances are defined between
        i and j, this array stores all i
        """
        return np.ascontiguousarray(values)

    @validatedproperty
    def second_index(values):
        """
        Second element of the index tuple, i.e. if distances are defined between
        i and j, this array stores all j
        """
        return np.ascontiguousarray(values)

    @property
    def pairs(self):
        """
        Iterator over all index tuples
        """
        for i in xrange(len(self)):
            yield (self._first_index[i], self._second_index[i])

    def __init__(self, pairs, name='distances'):
        """Distances

        Pairwise distances between particles

        Parameters
        ----------

        pairs: iterable
          2-tuples specifying the particles whose pairwise distances will be
          computed
        """
        super(Distances, self).__init__(name)

        i, j = np.transpose(pairs).astype('i')

        self.first_index  = i
        self.second_index = j

        self._value = np.ascontiguousarray(np.zeros(len(i)))

    def set(self, distances):

        if np.any(distances < 0.):
            msg = 'Distances must be non-negative'
            raise ValueError(msg)

        self._value[...] = distances

class ModelDistances(Distances):
    """ModelDistances

    Class for storing and *evaluating* inter-particle distances. 
    """
    def __init__(self, coords, pairs, name='distances'):
        """Distances

        Pairwise distances between particles

        Parameters
        ----------

        coords : Coordinates instance
          coordinates stored in Coordinates indstance will be used to compute
          inter-particle distances

        pairs : iterable
          2-tuples specifying the particles whose pairwise distances will be
          computed
        """
        super(ModelDistances, self).__init__(pairs, name)

        self._coords = coords
        
    def update(self):

        from .distance import calc_data

        calc_data(self._coords.get(),
                  self.first_index,
                  self.second_index,
                  self._value)

    def update_forces(self, derivatives, forces):
        """
        Computes the Cartesian gradient assuming that an instance of
        'Distances' is passed as the dataset
        """
        from .distance import update_forces

        update_forces(self._coords.get(),
                      self.first_index,
                      self.second_index,
                      self._value,
                      derivatives,
                      forces)

class Parameters(object):
    """Parameters

    Class holding all model parameters, data and hyper parameters.
    This class is shared among all probabilities to make sure that
    the probabilities always use the same parameters.
    """
    def __init__(self):
        
        self._params = OrderedDict()

    def add(self, param):

        if param in self._params:
            msg = 'Parameter "{}" already added'.format(param)
            raise ValueError(msg)

        self._params[param.name] = param

    def __str__(self):
        s = ['Parameters:']
        for param in self:
            s.append('    {}'.format(param))
        return '\n'.join(s)
    
    def __iter__(self):

        return iter(self._params.values())

    def get(self):

        return [param.get() for param in self]

    def set(self, values):

        for param, value in zip(self, values):
            param.set(value)

    def __getitem__(self, name):

        return self._params[name]
