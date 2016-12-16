"""
Replica exchange Monte Carlo
"""
import numpy as np

from .mcmc import MetropolisHastings, History

from collections import deque, defaultdict, OrderedDict

def generate_pairs(n_rex):
    """
    Generate list of pairs that will attempt to swap configurations
    during replica exchange Monte Carlo.
    """
    pairs = zip(range(n_rex),range(1,n_rex))
    return pairs[::2], pairs[1::2]

class Swaps(object):
    """Swaps

    Iterator over possible swaps between neighboring replicas.
    """
    def __init__(self, n_rex):

        self.n_states = int(n_rex)
        self.pairs = deque(generate_pairs(self.n_states))

    def next(self):

        pairs = self.pairs.popleft()
        self.pairs.append(pairs)

        return pairs

    def __iter__(self):

        return self

class ReplicaState(object):

    def __init__(self, states):
        self._states = list(states)

    def __iter__(self):
        return iter(self._states)

    def __getitem__(self, i):
        return self._states[i]

    def __setitem__(self, i, state):
        self._states[i] = state

    @property
    def log_prob(self):
        return sum(state.log_prob for state in self)

class ReplicaHistory(object):

    def __init__(self):
        self._swaps = defaultdict(History)

    def __str__(self):
        s = []
        for i, j in self.pairs:
            s.append('{0}<-->{1}: {2}'.format(
                str(i).rjust(3), str(j).ljust(3), self[(i,j)]))

        return '\n'.join(s)

    def __getitem__(self, pair):
        return self._swaps[pair]

    def clear(self):
        self._swaps.clear()

    def update(self, accepted):
        for (i,j), accept in accepted.items():
            self._swaps[(i,j)].update(accept)

    @property
    def pairs(self):
        return sorted(self._swaps.keys(), lambda a, b: cmp(a[0],b[0]))

class ReplicaExchange(MetropolisHastings):

    def __init__(self, samplers):

        self._samplers = samplers
        self._swaps    = Swaps(len(self))
        self.history   = ReplicaHistory()
        
    def __len__(self):
        return len(self._samplers)

    def __iter__(self):
        return iter(self._samplers)

    def __getitem__(self, i):
        return self._samplers[i]

    def create_state(self):
        return ReplicaState([sampler.create_state() for sampler in self])
    
    def store_state(self, state=None):
        self.samples.append(state or self.create_state())

    def propose_swap(self, current, i, j):

        self[i].parameter.set(current[j].value)
        self[j].parameter.set(current[i].value)

        return self[i].create_state(), self[j].create_state()

    def sample_swap(self, state, i, j):

        state_ij, state_ji = self.propose_swap(state, i, j)

        log_prob = (state_ji.log_prob + state_ij.log_prob) - \
                   (state[i].log_prob + state[j].log_prob)

        accept = np.log(np.random.random()) < log_prob

        if accept: state[i], state[j] = state_ij, state_ji
            
        return accept
    
    def propose(self, current):

        return ReplicaState([sampler.sample(state)[0] for sampler, state
                             in zip(self, current)])
    
    def sample(self, current=None):

        current   = current or self.samples[-1]
        candidate = self.propose(current)
        accept    = {}

        for i, j in self._swaps.next():
            accept[(i,j)] = self.sample_swap(candidate, i, j)

        return candidate, accept
            
