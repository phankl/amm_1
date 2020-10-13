import time
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI

from simulation import *

# setup MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
  start = time.time()

# simulation parameters
N = 100
L = 1.0
alpha = 0.5
dt = 0.1
gamma = 0.05
n = 1000000

# thermal expansion coefficient simulation parameters
average_start = int(0.5*n)
T_min = 0.005
T_max = 0.05
points = 10
repeats = 10

# expected standard deviation of random moves from Maxwell-Boltzmann distribution

Ts = np.linspace(T_min, T_max, points)

# run simulations over multiple cores
l_aves = np.zeros(points)
for i in range(rank, points, size):
  T = Ts[i]
  l_ave = 0.0
  MC_std = dt * np.sqrt(T)
  print('Temperature {}: {:.3f}'.format(i+1,T))
  for j in range(repeats):
    sim = Simulation()
    sim.init_MC(T, MC_std)
    sim.init_chain(N, L, alpha, 'gaussian', 0.1)
    sim.init_output(l=True)
    sim.run(n)

    # compute average length
    l = sim.l_hist[average_start:]
    l_ave += np.mean(l)

  l_aves[i] = l_ave / repeats

# reduce data to root process

l_aves_root = np.zeros(points)
comm.Allreduce([l_aves, MPI.DOUBLE], [l_aves_root, MPI.DOUBLE], op=MPI.SUM)

if rank == 0:
  # compute gradient

  # normalise with respect to chain length
  spacing_aves = l_aves_root / (L*(N-1))
  fit = np.polyfit(Ts, spacing_aves, 1)
  gradient = fit[0]
  offset = fit[1]

  # compute fit for plot
  fit_data = offset + gradient*Ts

  print('Alpha:', alpha)
  print('Thermal expansion coefficient:', gradient)

  end = time.time()
  print('Elapsed time:', end-start)

  # plot results
  plt.xlabel('Temperature')
  plt.ylabel('Average Spacings')
  plt.scatter(Ts, spacing_aves, label='Data')
  plt.plot(Ts, fit_data, label='Fit')
  plt.legend()
  plt.show()
