Simple SIR model with PyTorch + matplotlib (mpiexec + torch.distributed)

What SIR means:
S = Susceptible: people who can still get infected
I = Infected: people currently infected
R = Recovered/Removed: people no longer spreading infection

Daily model update:
new_infections = beta * S * I / N
new_recoveries = gamma * I
S <- S - new_infections
I <- I + new_infections - new_recoveries
R <- R + new_recoveries

R0 quick guide:
For this basic model, a common approximation is R0 ~= beta/gamma.
- R0 > 1: outbreak tends to grow initially
- R0 < 1: outbreak tends to shrink

How to choose realistic values:
- If average infectious duration is D days, pick gamma ~= 1/D.
- If target R0 is known, pick beta ~= R0 * gamma.
- Example: D=5 -> gamma=0.2. For R0=2, beta=0.4.

Command-line options:
--population  Total population N (default: 10000)
--infected0   Initial infected people at day 0 (default: 10)
--beta        Infection rate; higher beta spreads faster (default: 0.30)
--gamma       Recovery rate; higher gamma recovers faster (default: 0.10)
--steps       Number of simulated days (default: 160)
--out         Output PNG file path (default: sir_curve.png)
--out-csv     Output CSV path; default is <out>.csv
--mpi         Enable MPI-launched torch.distributed mode (use with mpiexec)
--backend     torch.distributed backend: xccl|nccl|gloo (default: xccl)

Single-process run:
module load frameworks
python sir_gpu.py

MPI run with mpiexec (single node, xccl backend):
module load frameworks
mpiexec -np 8 python sir_gpu.py --mpi --backend xccl --population 10000000 --infected0 100 --steps 365 --out sir_curve.png

PBS-style skeleton (mpiexec):
#!/bin/bash -x
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -A <ProjectName>
#PBS -q prod
#PBS -l filesystems=flare:daos_user_fs

module use /soft/modulefiles
module load frameworks

NNODES=$(wc -l < "$PBS_NODEFILE")
RANKS_PER_NODE=8
NRANKS=$(( NNODES * RANKS_PER_NODE ))

echo "NNODES=${NNODES} NRANKS=${NRANKS} RANKS_PER_NODE=${RANKS_PER_NODE}"

mpiexec -np ${NRANKS} -ppn ${RANKS_PER_NODE} \
  python /lus/flare/projects/datascience/kaushik/ABM-GPU/sir/sir_gpu.py \
  --mpi --backend xccl --population 100000000 --infected0 1000 --beta 0.30 --gamma 0.10 --steps 365 --out sir_curve.png

CSV + plotting:
The run writes CSV in `step,susceptible,infected,recovered` format.
By default it uses `<out>.csv` (example: `sir_curve.csv`).

Replot from CSV:
python plot_sir_csv.py --csv sir_curve.csv --out sir_curve_replot.png --title "Well-mixed SIR"
