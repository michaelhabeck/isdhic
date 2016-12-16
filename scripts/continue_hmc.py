"""
Continue HMC sampling at higher resolution.

This script assumes that 'run_hmc.py' has already been executed in
the *same* python session.
"""
from scipy import interpolate

def lift_coords(coarse_coords, n_fine):
    """
    Interpolate 3d fiber with a finer sampling.
    """
    tck, u = interpolate.splprep(np.reshape(coarse_coords,(-1,3)).T)    
    coords = interpolate.splev(np.linspace(0,1,n_fine), tck)

    return np.transpose(coords).flatten()

if __name__ == '__main__':
    
    resolution = 50
    filename   = './chrX_cell1_{0}kb.py'.format(resolution)

    with open(filename) as script:
        exec script

    coords.set(lift_coords(hmc.samples[-1].positions, n_particles))

    ## use Hamiltonian Monte Carlo to sample X chromosome structures from the
    ## posterior distribution

    n_steps  = 1e3                           ## number of HMC iterations
    n_leaps  = 1e1                           ## number of leapfrog integration steps
    stepsize = 1e-7                          ## initial integration stepsize
    
    hmc_fine = HamiltonianMonteCarlo(posterior,stepsize=stepsize)
    hmc_fine.leapfrog.n_steps = int(n_leaps)
    hmc_fine.adapt_until      = int(1e6) #0.5 * n_steps)
    hmc_fine.activate()

    with take_time('running HMC'):
        hmc_fine.run(n_steps)

