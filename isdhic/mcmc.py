"""
Markov chain Monte Carlo sampling
"""
import numpy as np

from copy import deepcopy

class History(object):
    """History

    Keeping track of accepted / rejected Monte Carlo trials.
    """
    def __init__(self):
        self.clear()

    def __len__(self):
        return len(self._history)

    def __getitem__(self, index):
        return self._history[index]

    def update(self, accept):
        self._history.append(int(accept))

    def clear(self):
        self._history = []

    def acceptance_rate(self, burnin=0):
        return np.mean(self._history[burnin:])

    def __str__(self):

        s = 'n_steps = {0}, acceptance rate = {1:.1f}%'.format(
            len(self), self.acceptance_rate() * 100)
        return s

class State(object):

    def __init__(self, value, log_prob):

        self.value = value if not np.iterable(value) else value.copy()
        self.log_prob = log_prob

class MetropolisHastings(object):

    def __init__(self, model, parameter):

        self.history   = History()
        self.model     = model
        self.parameter = parameter
        self.state     = self.create_state()

        self.history.clear()
        
    def create_state(self):
        """
        Create a state from current parameter settings.
        """
        return State(self.parameter.get(), self.model.log_prob())

    def propose(self, state):
        """
        Generates a new state from a given input state
        """
        raise NotImplementedError

    def next(self):
        """
        Proposes a new value for the parameter which is accepted or
        rejected according to the Metropolis criterion. 
        """
        current   = self.state
        candidate = self.propose(current)

        diff   = candidate.log_prob - current.log_prob
        accept = np.log(np.random.random()) < diff

        self.state = candidate if accept else current 

        self.history.update(accept)

        return self.state

    def __iter__(self):

        return self

class RandomWalk(MetropolisHastings):

    def __init__(self, model, parameter, stepsize=1e-1):

        super(RandomWalk, self).__init__(model, parameter)

        self.stepsize = float(stepsize)

    def propose(self, state):

        x = state.value
        y = x + self.stepsize * np.random.uniform(-1.,+1.,np.shape(x))

        self.parameter.set(y)

        return self.create_state()
        
class AdaptiveWalk(RandomWalk):

    def activate(self):
        self._active = True

    def deactivate(self):
        self._active = False

    @property
    def is_active(self):
        return self._active

    def __init__(self, model, parameter, stepsize=1e-1,
                 uprate=1.02, downrate=0.98, adapt_until=0):

        super(AdaptiveWalk, self).__init__(model, parameter, stepsize)

        self.uprate      = float(uprate)
        self.downrate    = float(downrate)
        self.adapt_until = int(adapt_until)

        self.deactivate()

    def next(self):

        state = super(AdaptiveWalk, self).next()

        if len(self.history) >= self.adapt_until: self.deactivate()

        if self.is_active:
            self.stepsize *= self.uprate if self.history[-1] else self.downrate

        return state

