Spatial Neighbor-Based SIR (MPI)

This program is different from the well-mixed SIR model.
It uses a 2D grid of people and local neighbor interactions.

File:
sir_neighbor_mpi.py

Core idea:
- The global grid is split across MPI ranks with a 2D Cartesian decomposition.
- Each rank owns one local tile.
- At every step, each rank exchanges only boundary cells (halo) with N/S/E/W neighbor ranks.
- Infection is computed from local + halo data, so communication is local/sparse.

Why this matches "4 of 64 nodes are relevant":
- A rank only talks to its geometric neighbors, not all 64 ranks.
- If an infection front is local, only nearby ranks actively exchange impactful state.
- No global all-to-all infection dependency is used.

State model:
- 0 = Susceptible
- 1 = Infected
- 2 = Recovered

Update per step:
1) Halo exchange with rank neighbors.
2) For each local cell, count infected neighbors among N/S/E/W.
3) If susceptible with k infected neighbors:
   p_infect = 1 - (1 - beta)^k
4) If infected:
   recover with probability gamma.

Environment:
module load frameworks

Single-node quick run:
mpiexec -np 8 python sir_neighbor_mpi.py --rows 1024 --cols 1024 --steps 200 --out sir_neighbor

PBS-style mpiexec skeleton:
#!/bin/bash -x
#PBS -l select=4
#PBS -l walltime=01:00:00
#PBS -A <ProjectName>
#PBS -q prod

module use /soft/modulefiles
module load frameworks

NNODES=$(wc -l < "$PBS_NODEFILE")
RANKS_PER_NODE=8
NRANKS=$(( NNODES * RANKS_PER_NODE ))

mpiexec -np ${NRANKS} -ppn ${RANKS_PER_NODE} \
  python /lus/flare/projects/datascience/kaushik/ABM-GPU/sir/sir_neighbor_mpi.py \
  --rows 8192 --cols 8192 --steps 400 --beta 0.20 --gamma 0.04 --infected-frac0 0.0005 --out sir_neighbor

Outputs:
- Rank 0 writes `<out>.csv` with columns:
  step,susceptible,infected,recovered

Replot from CSV:
python plot_sir_csv.py --csv sir_neighbor.csv --out sir_neighbor.png --title "Neighbor SIR"
