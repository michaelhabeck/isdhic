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
        self.samples   = []
        
    def create_state(self):
        """
        Create a state from current parameter settings.
        """
        return State(self.parameter.get(), self.model.log_prob())

    def store_state(self, state=None):

        self.samples.append(state or self.create_state())

    def accept(self, candidate, current):
        diff = candidate.log_prob - current.log_prob
        return np.log(np.random.random()) < diff

    def propose(self, state):
        """
        Generates a new state from a given input state
        """
        raise NotImplementedError

    def sample(self, current=None):
        """
        Proposes a new value for the parameter which is accepted or
        rejected according to the Metropolis criterion. Returns the
        new state and a flag indicating if the proposed state was
        accepted or not.
        """
        current   = current or self.samples[-1]
        candidate = self.propose(current)
        
        if self.accept(candidate, current):
            return candidate, True
        else:
            return current, False

    def run(self, n_steps):
        """
        Generates a sequence of Monte Carlo samples
        """
        self.history.clear()

        self.samples = []
        self.store_state()
        
        for i in xrange(int(n_steps)-1):

            state, accept = self.sample()

            self.store_state(state)
            self.history.update(accept)

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

    def __init__(self, model, parameter, stepsize=1e-1, uprate=1.02, downrate=0.98, adapt_until=0):

        super(AdaptiveWalk, self).__init__(model, parameter, stepsize)

        self.uprate      = float(uprate)
        self.downrate    = float(downrate)
        self.adapt_until = int(adapt_until)

        self.deactivate()

    def accept(self, candidate, current):

        accept = super(AdaptiveWalk, self).accept(candidate, current)

        if len(self.samples) >= self.adapt_until: self.deactivate()

        if self.is_active:
            self.stepsize *= self.uprate if accept else self.downrate

        return accept

