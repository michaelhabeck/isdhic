"""
Run parallel replica simulation using multiprocessing
"""
import isdhic
import numpy as np
import multiprocessing as mp

from copy import deepcopy

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

    return Replica(posterior, n_steps=10, n_leaps=250, q=q, beta=beta)

def sample(initial, q, beta, stepsize):

    sampler = create_replica(q, beta)
    sampler.activate()
    
    sampler.state = deepcopy(initial)
    sampler.stepsize = stepsize
    
    state = sampler.next()

    return deepcopy(state), sampler.stepsize, sampler.history

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

        self.state = ReplicaState(states)

        return self.state

    def next(self):

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
    
    n_replicas = 50
    schedule   = np.transpose([
        np.logspace(0., np.log10(1.06), n_replicas),
        np.linspace(1., 0.1, n_replicas)])

    try:
        schedule = load(os.path.expanduser('~/tmp/geo_rosetta_contacts_Rg_6_schedule.pkl'))
    except:
        schedule = load('/space/users/mhabeck/tmp/geo_rosetta_contacts_Rg_6_schedule.pkl')
        
    schedule[0,:] = 1.

    posterior = create_posterior()
    replicas = [create_replica(q,beta,posterior) for q, beta in schedule]

    rex = ParallelReplicaExchange(replicas, n_cpus=25)

    samples = []

if False:

    from isd.ro import threaded

    threaded(create_samples, 1e4, rex, samples)

if False:

    while len(samples) < 1000:
        samples.append(rex.next())

if False:

    E = np.array([[s.potential_energy for s in S] for S in samples])

    out = '{0:.3e}, {1:.3e}, {2:.3e} : {3}, -log_prob={4:.3e}'

    for sampler in rex.samplers:

        print out.format(sampler.q, sampler.beta, sampler.stepsize, sampler.history, 
                         sampler.state.potential_energy)

    rates = np.array([rex.history[pair].acceptance_rate() for pair in rex.history.pairs])
    
if False:

    from csb.bio.utils import rmsd
    from scipy.spatial.distance import squareform
    
    burnin = -400
    thining = 1#0

    mask = np.ones(333,'i')
    mask[:11] = 0
    mask[47:66] = 0

    x = X[burnin::thining].reshape(-1,333,3)
    x = np.compress(mask,x,1)
    d = [rmsd(xx,x[j]) for i, xx in enumerate(x) for j in range(i+1,len(x))]

    from sklearn.cluster import spectral_clustering

    K = 4
    membership = spectral_clustering(np.exp(-squareform(d)), n_clusters=K, eigen_solver='arpack')

    i = np.argsort(membership)

    matshow(squareform(d)[i][:,i],origin='lower')
