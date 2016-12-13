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

    def __init__(self, value, log_prob=None):

        self.value = value if not np.iterable(value) else value.copy()
        self.log_prob = log_prob

class MetropolisHastings(object):

    def __init__(self, model, parameter):

        self.history   = History()
        self.model     = model
        self.parameter = parameter
        self.samples   = []
        
    def propose(self, state):
        """
        Generates a new state from a given input state
        """
        raise NotImplementedError

    def accept(self, proposed_state, current_state):
        dlgp = proposed_state.log_prob - current_state.log_prob
        return np.log(np.random.random()) < dlgp

    def sample(self):
        """
        Proposes a new value for the parameter which is accepted or
        rejected according to the Metropolis criterion. Returns a
        flag indicating whether the proposed state was accepted or not.
        """
        current  = self.samples[-1]
        proposal = self.propose(current)
        
        self.parameter.set(proposal.value)
        proposal.log_prob = self.model.log_prob()
        
        if self.accept(proposal, current):
            return proposal, True
        else:
            self.parameter.set(current.value)
            return current, False

    def store_state(self, state=None):

        if state is None:
            
            value   = self.parameter.get()
            logprob = self.model.log_prob()
            state   = State(value, logprob)
            
        self.samples.append(state)

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

        proposal = deepcopy(state)
        proposal.value += self.stepsize * np.random.uniform(-1.,+1.,np.shape(proposal.value))

        return proposal
        
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

    def accept(self, proposed_state, current_state):

        accept = super(AdaptiveWalk, self).accept(proposed_state, current_state)

        if len(self.samples) >= self.adapt_until: self.deactivate()

        if self.is_active:
            self.stepsize *= self.uprate if accept else self.downrate

        return accept

