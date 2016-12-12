import numpy as np

n_particles = 1000
dofs = np.ascontiguousarray(np.zeros(3*n_particles))
coords = dofs.reshape(-1,3)

coords = np.zeros((n_particles,3))
dofs = np.ascontiguousarray(coords.reshape(-1,))

i = 10
coords[i,...] = 1
print coords[i]
print dofs[3*i:3*(i+1)]

i = 20
dofs[3*i:3*(i+1)] = np.random.random(3)
print coords[i]
print dofs[3*i:3*(i+1)]
