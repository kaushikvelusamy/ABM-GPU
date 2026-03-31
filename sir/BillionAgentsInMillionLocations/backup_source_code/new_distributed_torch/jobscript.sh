#!/usr/bin/env bash
#PBS -l select=2
#PBS -l walltime=00:59:00
#PBS -A datascience
#PBS -q debug-scaling
#PBS -k doe
#PBS -l filesystems=flare

module load frameworks
cd /lus/flare/projects/datascience/kaushik/ABM-GPU/sir/BillionAgentsInMillionLocations/backup_source_code/new_distributed_torch

NNODES=$(wc -l < "$PBS_NODEFILE")
RANKS_PER_NODE=12
NRANKS=$(( NNODES * RANKS_PER_NODE ))
CPU_BINDING1=list:4:9:14:19:20:25:56:61:66:71:74:79
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NRANKS}  RANKS_PER_NODE=${RANKS_PER_NODE}"

sh ./run_agent_location_sweep.sh


