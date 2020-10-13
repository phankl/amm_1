# amm_1
Solution to the Atomistic Materials Modelling Practical 1 for the MPhil in Scientific Computing at the University of Cambridge

All internal workings, i.e. data structures, initialisation and MD and MC codes are contained in the Simulation class in simulation.py.
Each question has an individual solution script question_*.py invoking the framework in simulation.py.

Required packages:
numpy
matplotlib
mpi4py (optional)

Question 5 is optional and involves replacing the MD integrators by an MC trial step.
As questions 4 and 5 can take a long time to reach convergence, optional MPI support has been added in the files question_*_mpi.py.
With MPI (e.g. OpenMPI or MPICH) installed on the system, these files can be run with the following command:
mpirun -np #MPI_TASKS python question_*_mpi.py
#MPI_TASKS is the number of MPI tasks and roughly corresponds to the number of threads the simulation is run on.
