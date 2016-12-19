"""
Replica exchange Monte Carlo
"""
import numpy as np

from .mcmc import MetropolisHastings, History

from collections import deque, defaultdict, OrderedDict
from csb.numeric import log_sum_exp

def generate_pairs(n_rex):
    """
    Generate list of pairs that will attempt to swap configurations
    during replica exchange Monte Carlo.
    """
    pairs = zip(range(n_rex),range(1,n_rex))
    return pairs[::2], pairs[1::2]

def swap_rate(log_p, log_q, return_log=True):
    """
    Computes the (log) swap rate of a replica exchange simulation, i.e.

    rate(p<->q) = \int p(x) q(y) \min[1, p(y)q(x)/p(x)q(y)]
    """
    log_r = np.add.outer(log_p - log_sum_exp(log_p),
                         log_q - log_sum_exp(log_q))

    ## mask implementing the min operator

    mask = (log_r < log_r.T).astype('i')
    mask = (1 + mask - mask.T).flatten()

    log_r.shape = (-1,)

    rate = log_sum_exp(log_r[mask>0] + np.log(mask[mask>0]))

    return rate if return_log else np.exp(rate)

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
        self.n_swaps = 0
        
    def __str__(self):
        s = []
        for i, j in self.pairs:
            s.append('{0}<-->{1}: {2}'.format(
                str(i).rjust(3), str(j).ljust(3), self[(i,j)]))

        return '\n'.join(s)

    def __getitem__(self, pair):
        return self._swaps[pair]

    def __len__(self):
        return self.n_swaps

    def clear(self):
        self._swaps.clear()
        self.n_swaps = 0
        
    def update(self, accepted):
        self.n_swaps += 1
        for (i,j), accept in accepted.items():
            self._swaps[(i,j)].update(accept)

    @property
    def pairs(self):
        return sorted(self._swaps.keys(), lambda a, b: cmp(a[0],b[0]))

class ReplicaExchange(MetropolisHastings):

    @property
    def samplers(self):
        return self._samplers

    def __init__(self, samplers):

        self._samplers = samplers
        self._swaps    = Swaps(len(self))
        self.history   = ReplicaHistory()
        
        ## self.state = ReplicaState([sampler.state for sampler in self.samplers])
        
    @property
    def state(self):
        return ReplicaState([sampler.state for sampler in self.samplers])

    @state.setter
    def state(self, state):
        for i, sampler in enumerate(self._samplers):
            sampler.state = state[i]
    
    def __len__(self):
        return len(self._samplers)

    def __getitem__(self, i):
        return self._samplers[i]

    def propose_swap(self, state, i, j):

        self[i].parameter.set(state[j].value)
        state_ij = self[i].create_state()

        self[j].parameter.set(state[i].value)
        state_ji = self[j].create_state()

        return state_ij, state_ji

    def sample_swap(self, state, i, j):

        state_ij, state_ji = self.propose_swap(state, i, j)

        log_prob = (state_ji.log_prob + state_ij.log_prob) - \
                   (state[i].log_prob + state[j].log_prob)

        accept = np.log(np.random.random()) < log_prob

        if accept: state[i], state[j] = state_ij, state_ji
            
        return accept
    
    def next(self):

        state  = ReplicaState([sampler.next() for sampler in self._samplers])
        accept = {}

        for i, j in self._swaps.next():
            accept[(i,j)] = self.sample_swap(state, i, j)

        self.history.update(accept)
        self.state = state

        ## for i, sampler in enumerate(self._samplers):
        ##     sampler.state = state[i]

        return state
            
    def __iter__(self):

        return self
