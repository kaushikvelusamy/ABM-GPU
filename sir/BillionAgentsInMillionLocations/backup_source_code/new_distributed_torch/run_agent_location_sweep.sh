#!/usr/bin/env bash
# Usage examples:
#   DRY_RUN=1 ./run_agent_location_sweep.sh
#   START_RUN=18 bash run_agent_location_sweep.sh

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${THIS_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MPIEXEC_BIN="${MPIEXEC_BIN:-mpiexec}"
RESULTS_BASE="${RESULTS_BASE:-results/agent_location_sweep}"
PLOT_SUMMARY="${PLOT_SUMMARY:-1}"
START_RUN="${START_RUN:-1}"
DRY_RUN="${DRY_RUN:-0}"

RANKS_PER_NODE="${RANKS_PER_NODE:-12}"
CPU_BINDING="${CPU_BINDING:-list:4:9:14:19:20:25:56:61:66:71:74:79}"

BETA="${BETA:-0.3}"
GAMMA="${GAMMA:-0.05}"
STEPS="${STEPS:-30}"
MOVE_PROB="${MOVE_PROB:-0.50}"
INFECTED0="${INFECTED0:-10}"
SEED="${SEED:-0}"
INFECTION_NEIGHBORHOOD="${INFECTION_NEIGHBORHOOD:-same_plus_news}"
MOVEMENT_NEIGHBORHOOD="${MOVEMENT_NEIGHBORHOOD:-stay_news}"

if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE}" ]]; then
  NNODES="${NNODES:-$(wc -l < "${PBS_NODEFILE}")}"
else
  NNODES="${NNODES:-1}"
fi
NRANKS="${NRANKS:-$(( NNODES * RANKS_PER_NODE ))}"

AGENT_COUNTS=(
  10000
  100000
  1000000
  10000000
  100000000
  1000000000
)

LOCATION_COUNTS=(
  10000
  30000
  100000
  300000
  1000000
)

mkdir -p "${RESULTS_BASE}"
MANIFEST_CSV="${RESULTS_BASE}/sweep_manifest.csv"

if (( START_RUN <= 1 )) || [[ ! -f "${MANIFEST_CSV}" ]]; then
  cat > "${MANIFEST_CSV}" <<'EOF'
run_index,run_name,num_agents,target_locations,rows,cols,actual_locations,nnodes,ranks_per_node,nranks,run_dir
EOF
fi

echo "Running distributed moving-agent SIR sweep"
echo "Results base: ${RESULTS_BASE}"
echo "Nodes: ${NNODES}"
echo "Ranks per node: ${RANKS_PER_NODE}"
echo "Total MPI ranks: ${NRANKS}"
echo "CPU binding: ${CPU_BINDING}"
echo "Total runs: $((${#AGENT_COUNTS[@]} * ${#LOCATION_COUNTS[@]}))"
echo "Starting from run: ${START_RUN}"
echo "Dry run: ${DRY_RUN}"
echo

run_index=0
total_runs=$((${#AGENT_COUNTS[@]} * ${#LOCATION_COUNTS[@]}))

for num_agents in "${AGENT_COUNTS[@]}"; do
  for target_locations in "${LOCATION_COUNTS[@]}"; do
    run_index=$((run_index + 1))

    if (( run_index < START_RUN )); then
      continue
    fi

    grid_values="$("${PYTHON_BIN}" - "${target_locations}" <<'PY'
import math
import sys

target = int(sys.argv[1])
rows = int(math.sqrt(target))
if rows <= 0:
    rows = 1
while (rows + 1) * (rows + 1) <= target:
    rows += 1
while rows * rows > target:
    rows -= 1
cols = (target + rows - 1) // rows
actual = rows * cols
print(rows, cols, actual)
PY
)"

    read -r rows cols actual_locations <<< "${grid_values}"

    run_name="run_${run_index}_agents_${num_agents}_locations_${actual_locations}"
    run_dir="${RESULTS_BASE}/${run_name}"
    out_log="${run_dir}/out.txt"

    mkdir -p "${run_dir}"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${run_index}" \
      "${run_name}" \
      "${num_agents}" \
      "${target_locations}" \
      "${rows}" \
      "${cols}" \
      "${actual_locations}" \
      "${NNODES}" \
      "${RANKS_PER_NODE}" \
      "${NRANKS}" \
      "${run_dir}" >> "${MANIFEST_CSV}"

    cmd=(
      "${MPIEXEC_BIN}"
      "-np" "${NRANKS}"
      "-ppn" "${RANKS_PER_NODE}"
      "--cpu-bind" "${CPU_BINDING}"
      "-genvall"
      "${PYTHON_BIN}"
      "sir_moving_agents_torch_distributed.py"
      "--rows" "${rows}"
      "--cols" "${cols}"
      "--num-agents" "${num_agents}"
      "--infected0" "${INFECTED0}"
      "--beta" "${BETA}"
      "--gamma" "${GAMMA}"
      "--steps" "${STEPS}"
      "--move-prob" "${MOVE_PROB}"
      "--infection-neighborhood" "${INFECTION_NEIGHBORHOOD}"
      "--movement-neighborhood" "${MOVEMENT_NEIGHBORHOOD}"
      "--seed" "${SEED}"
      "--run-dir" "${run_dir}/"
    )

    if [[ "${PLOT_SUMMARY}" == "1" ]]; then
      cmd+=("--plot-summary")
    fi

    echo "Run ${run_index}/${total_runs}"
    echo "  agents: ${num_agents}"
    echo "  target locations: ${target_locations}"
    echo "  grid: ${rows} x ${cols} = ${actual_locations}"
    echo "  nodes: ${NNODES}"
    echo "  ranks per node: ${RANKS_PER_NODE}"
    echo "  total ranks: ${NRANKS}"
    echo "  run dir: ${run_dir}"
    echo "  log: ${out_log}"

    {
      echo "Command:"
      printf '  %q' "${cmd[@]}"
      echo
      echo
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "Dry run enabled: command not executed."
      else
        "${cmd[@]}"
      fi
    } | tee "${out_log}"

    echo
  done
done

echo "Sweep complete"
echo "Manifest: ${MANIFEST_CSV}"
