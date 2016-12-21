import numpy as np

from analyze import Ensemble
from csb.io import load, dump

if not 'rex' in globals():
    with open('./run_parallel.py') as script:
        exec script

def calc_energy(state, rex):

    E = []
    for x, s in zip(state,rex.samplers):
        s.set_replica_params()
        #s.model.params['coordinates'].set(x.positions)
        s.parameter.set(x.positions)
        E.append(-s.model.log_prob())

    return E

samples = load('/tmp/samples{}.pkl'.format(['',2][0]))

E = np.array([[s.potential_energy for s in S] for S in samples])
X = np.array([[s.positions for s in S] for S in samples])

EE = []
for x in X:
    e = []
    for k, y in enumerate(x):
        sampler = rex[k]
        sampler.parameter.set(y)
        sampler.set_replica_params()
        e.append(-sampler.model.log_prob())
    EE.append(e)
EE = np.array(EE)

posterior = create_posterior()

ens = Ensemble(X.reshape(-1,333,3))
energies = ens.calculate_energies(posterior)

E_vdw = energies['tsallis'].reshape(-1,len(schedule))
E_bb  = energies['backbone'].reshape(-1,len(schedule))
E_con = energies['contacts'].reshape(-1,len(schedule))
E_rog = energies['rog'].reshape(-1,len(schedule))

q, beta = schedule.copy().T
q[0] += 1e-10
E_min = -100

E_data = E_bb + (E_con + E_rog) * beta
E_prior = q/(q-1) * np.log(1 + (q-1) * (E_vdw - E_min)) + E_min
E_prior[:,0] = E_vdw[:,0]
E_post = E_data + E_prior

