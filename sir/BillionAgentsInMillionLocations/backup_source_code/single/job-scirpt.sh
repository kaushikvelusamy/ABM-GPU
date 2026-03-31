#!/usr/bin/env bash
#PBS -l select=11
#PBS -l walltime=00:59:00
#PBS -A datascience
#PBS -q debug-scaling
#PBS -k doe
#PBS -l filesystems=flare


set -euo pipefail

set +u
module load frameworks
set -u

THIS_DIR="/lus/flare/projects/datascience/kaushik/ABM-GPU/sir/BillionAgentsInMillionLocations/backup_source_code/single"
cd "${THIS_DIR}"

if [[ -z "${PBS_NODEFILE:-}" || ! -f "${PBS_NODEFILE}" ]]; then
  echo "PBS_NODEFILE is not set or does not exist."
  exit 1
fi

mapfile -t UNIQUE_NODES < <(awk '!seen[$0]++' "${PBS_NODEFILE}")
UNIQUE_NODE_COUNT="${#UNIQUE_NODES[@]}"
CPU_BINDING1="list:4:9:14:19:20:25:56:61:66:71:74:79"

COMMANDS=(
  "python3 sir_moving_agents.py --rows 1000 --cols 1000 --num-agents 10000000 --run-dir results/agent_location_sweep/run_20_agents_10000000_locations_1000000/ --plot-summary"
  "python3 sir_moving_agents.py --rows 100 --cols 100 --num-agents 100000000 --run-dir results/agent_location_sweep/run_21_agents_100000000_locations_10000/ --plot-summary"
  "python3 sir_moving_agents.py --rows 173 --cols 174 --num-agents 100000000 --run-dir results/agent_location_sweep/run_22_agents_100000000_locations_30102/ --plot-summary"
  "python3 sir_moving_agents.py --rows 316 --cols 317 --num-agents 100000000 --run-dir results/agent_location_sweep/run_23_agents_100000000_locations_100172/ --plot-summary"
  "python3 sir_moving_agents.py --rows 547 --cols 549 --num-agents 100000000 --run-dir results/agent_location_sweep/run_24_agents_100000000_locations_300303/ --plot-summary"
  "python3 sir_moving_agents.py --rows 1000 --cols 1000 --num-agents 100000000 --run-dir results/agent_location_sweep/run_25_agents_100000000_locations_1000000/ --plot-summary"
  "python3 sir_moving_agents.py --rows 100 --cols 100 --num-agents 1000000000 --run-dir results/agent_location_sweep/run_26_agents_1000000000_locations_10000/ --plot-summary"
  "python3 sir_moving_agents.py --rows 173 --cols 174 --num-agents 1000000000 --run-dir results/agent_location_sweep/run_27_agents_1000000000_locations_30102/ --plot-summary"
  "python3 sir_moving_agents.py --rows 316 --cols 317 --num-agents 1000000000 --run-dir results/agent_location_sweep/run_28_agents_1000000000_locations_100172/ --plot-summary"
  "python3 sir_moving_agents.py --rows 547 --cols 549 --num-agents 1000000000 --run-dir results/agent_location_sweep/run_29_agents_1000000000_locations_300303/ --plot-summary"
  "python3 sir_moving_agents.py --rows 1000 --cols 1000 --num-agents 1000000000 --run-dir results/agent_location_sweep/run_30_agents_1000000000_locations_1000000/ --plot-summary"
)

COMMAND_COUNT="${#COMMANDS[@]}"

echo "PBS nodefile: ${PBS_NODEFILE}"
echo "Unique nodes available: ${UNIQUE_NODE_COUNT}"
echo "Commands to launch: ${COMMAND_COUNT}"
echo "CPU binding: ${CPU_BINDING1}"
echo "Unique node list:"
for node in "${UNIQUE_NODES[@]}"; do
  echo "  ${node}"
done

if (( UNIQUE_NODE_COUNT < COMMAND_COUNT )); then
  echo "Need at least ${COMMAND_COUNT} unique nodes, but PBS provided ${UNIQUE_NODE_COUNT}."
  exit 1
fi

pids=()

for idx in "${!COMMANDS[@]}"; do
  node="${UNIQUE_NODES[idx]}"
  cmd="${COMMANDS[idx]}"
  run_number=$((idx + 20))

  echo
  echo "Launching run ${run_number} on node ${node}"
  echo "  ${cmd}"

  mpiexec -np 1 -hosts "${node}" -ppn 1 --cpu-bind "${CPU_BINDING1}" -genvall \
    bash -lc "cd '${THIS_DIR}' && ${cmd}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
