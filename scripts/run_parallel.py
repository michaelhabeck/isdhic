"""
Run parallel replica simulation using multiprocessing
"""
import isdhic
import numpy as np
import multiprocessing as mp

from isdhic import utils
from isdhic.rex import ReplicaState

from run_rex import Replica

def create_posterior(resolution = 500):
    
    filename   = './chrX_cell1_{0}kb.py'.format(resolution)

    with open(filename) as script:
        exec script

    return posterior

def create_replica(q, beta, posterior=None):

    posterior = posterior or create_posterior()

    diameter = posterior['tsallis'].forcefield.d[0,0]
    coords = posterior.params['coordinates']
    n_particles = len(coords)/3
    extended = np.multiply.outer(np.arange(n_particles), np.eye(3)[0]) * diameter
    coords.set(extended)

    return Replica(posterior, n_steps=10, q=q, beta=beta)

def sample(initial, q, beta, stepsize):

    sampler = create_replica(q, beta)
    sampler.activate()
    
    sampler.state = initial
    sampler.stepsize = stepsize
    
    state = sampler.next()

    return state, sampler.stepsize, sampler.history

## class Bridge(object):
##     """
##     List of replicas
##     """
##     def __init__(self, samplers):
##         self._items = samplers

##     def __len__(self):
##         return len(self._items)

##     def __getitem__(self, i):
##         return self._items[i]

##     def __iter__(self):
##         return iter(self._items)

##     def __call__(self, i):
##         sampler = self[i]
##         return sampler.next()

##     def propose(self, workers):        
##         return ReplicaState(workers.map(self, range(len(self))))

class ParallelReplicaExchange(isdhic.ReplicaExchange):

    def __init__(self, samplers, n_cpus=1):

        super(ParallelReplicaExchange, self).__init__(samplers)

        self.workers = mp.Pool(processes=n_cpus)

    def move_parallel(self):

        tasks = []

        for state, replica in zip(self.state, self.samplers):
            tasks.append((state, replica.q, replica.beta, replica.stepsize))

        print 'Start parallel computation...'
        
        results = [self.workers.apply_async(sample, task) for task in tasks]

        states = []

        for i, result in enumerate(results):
            state, stepsize, history = result.get()
            map(self[i].history.update, history)
            self[i].stepsize = stepsize
            states.append(state)

        print 'Parallel computation finished.'

        return ReplicaState(states)

    def next(self):

        ## state  = self._samplers.propose(self.workers)

        state  = self.move_parallel()
        accept = {}

        for i, j in self._swaps.next():
            accept[(i,j)] = self.sample_swap(state, i, j)

        self.history.update(accept)
        self.state = state

        if not len(self.history) % 1:
            print '-' * 25, len(self.history), '-' * 25
            print self.history

        return state

def create_samples(n_samples, rex, samples):
            
    while len(samples) < n_samples:
        samples.append(rex.next())

if __name__ == '__main__':

    from csb.io import load, dump
    import os
    
    n_replicas = 10
    schedule   = np.transpose([
        np.logspace(0., np.log10(1.06), n_replicas),
        np.linspace(1., 0.1, n_replicas)])

    schedule = load(os.path.expanduser('~/tmp/geo_rosetta_contacts_Rg_6_schedule.pkl'))
    schedule[0,:] = 1.

    posterior = create_posterior()
    replicas = [create_replica(q,beta,posterior) for q, beta in schedule]

    rex = ParallelReplicaExchange(replicas, n_cpus=3)

    samples = []

if False:

    from isd.ro import threaded

    threaded(create_samples, 50, rex, samples)

if False:

    while len(samples) < 1000:
        samples.append(rex.next())

if False:

    E = np.array([[s.potential_energy for s in S] for S in samples])

    out = '{0:.3e}, {1:.3e}, {2:.3e} : {3}'

    for sampler in rex.samplers:

        print out.format(sampler.q, sampler.beta, sampler.stepsize, sampler.history)

if False:

    samples = load('/tmp/samples.pkl')

    E = np.array([[s.potential_energy for s in S] for S in samples])
    X = np.array([[s.positions for s in S] for S in samples])

    EE = []
    for x in X:
        e = []
        for k, y in enumerate(x):
            sampler = rex[k]
            sampler.parameter.set(y)
            sampler.set_replica_params()
            e.append(sampler.model.log_prob())
        EE.append(e)
    EE = np.array(EE)
