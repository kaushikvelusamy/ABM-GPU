# Single Sweep Summary

Primary source: [results/agent_location_sweep](/lus/flare/projects/datascience/kaushik/ABM-GPU/sir/BillionAgentsInMillionLocations/backup_source_code/single/results/agent_location_sweep)
Historical terminal log: [out_30_runs.txt](/lus/flare/projects/datascience/kaushik/ABM-GPU/sir/BillionAgentsInMillionLocations/backup_source_code/single/out_30_runs.txt)

## Status

- Runs `1-20` have completed results on disk with populated `run_config.json` metrics.
- Runs `21-30` exist as run directories, but they are not completed runs.
- For runs `21-30`, the `out.txt` files show `DRY_RUN=1`, and their `run_config.json` values such as `runtime_seconds`, `peak_memory_mb`, and `total_output_size_bytes` are still `null`.

## Default Configuration

- Execution: CPU only (`single process`, non-distributed)
- `infected0 = 10`
- `beta = 0.3`
- `gamma = 0.05`
- `steps = 30`
- `seed = 0`
- `infection_neighborhood = same_plus_news`
- `movement_neighborhood = stay_news`
- `move_prob = 0.5`
- `plot_summary = true`

## Completed Runs

| Run | Agents | Actual Locations | Density | Final S | Final I | Final R | Runtime (s) | Peak Mem (MB) | Output Size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10000 | 10000 | 1.0000 | 5168 | 2899 | 1933 | 1.607 | 46.777 | 15.17 MB |
| 2 | 10000 | 30102 | 0.3322 | 9648 | 192 | 160 | 2.231 | 49.777 | 17.27 MB |
| 3 | 10000 | 100172 | 0.0998 | 9966 | 8 | 26 | 1.838 | 55.105 | 18.49 MB |
| 4 | 10000 | 300303 | 0.0333 | 9984 | 4 | 12 | 2.033 | 73.918 | 19.01 MB |
| 5 | 10000 | 1000000 | 0.0100 | 9987 | 2 | 11 | 2.589 | 139.262 | 19.29 MB |
| 6 | 100000 | 10000 | 10.0000 | 86 | 44939 | 54975 | 12.626 | 57.895 | 129.42 MB |
| 7 | 100000 | 30102 | 3.3220 | 44772 | 33173 | 22055 | 13.850 | 57.438 | 146.28 MB |
| 8 | 100000 | 100172 | 0.9983 | 93649 | 4027 | 2324 | 15.794 | 63.301 | 170.09 MB |
| 9 | 100000 | 300303 | 0.3330 | 99685 | 170 | 145 | 17.438 | 80.824 | 185.51 MB |
| 10 | 100000 | 1000000 | 0.1000 | 99977 | 1 | 22 | 18.736 | 142.277 | 193.39 MB |
| 11 | 1000000 | 10000 | 100.0000 | 59663 | 390961 | 549376 | 122.205 | 131.875 | 1.26 GB |
| 12 | 1000000 | 30102 | 33.2204 | 228599 | 390554 | 380847 | 125.005 | 135.793 | 1.34 GB |
| 13 | 1000000 | 100172 | 9.9828 | 582192 | 265792 | 152016 | 133.939 | 141.699 | 1.41 GB |
| 14 | 1000000 | 300303 | 3.3300 | 909127 | 59510 | 31363 | 143.295 | 156.301 | 1.53 GB |
| 15 | 1000000 | 1000000 | 1.0000 | 992994 | 4744 | 2262 | 166.514 | 192.531 | 1.74 GB |
| 16 | 10000000 | 10000 | 1000.0000 | 0 | 3828035 | 6171965 | 1219.924 | 845.816 | 12.98 GB |
| 17 | 10000000 | 30102 | 332.2038 | 1473997 | 4378925 | 4147078 | 1236.721 | 866.199 | 13.55 GB |
| 18 | 10000000 | 100172 | 99.8283 | 4810437 | 3187564 | 2001999 | 1263.070 | 831.578 | 13.92 GB |
| 19 | 10000000 | 300303 | 33.2997 | 7964954 | 1304686 | 730360 | 1295.405 | 847.957 | 14.21 GB |
| 20 | 10000000 | 1000000 | 10.0000 | 9438733 | 370511 | 190756 | 1338.461 | 914.777 | 14.72 GB |

## Not Completed

| Run | Agents | Actual Locations | Status |
| --- | ---: | ---: | --- |
| 21 | 100000000 | 10000 | dry-run only |
| 22 | 100000000 | 30102 | dry-run only |
| 23 | 100000000 | 100172 | dry-run only |
| 24 | 100000000 | 300303 | dry-run only |
| 25 | 100000000 | 1000000 | dry-run only |
| 26 | 1000000000 | 10000 | dry-run only |
| 27 | 1000000000 | 30102 | dry-run only |
| 28 | 1000000000 | 100172 | dry-run only |
| 29 | 1000000000 | 300303 | dry-run only |
| 30 | 1000000000 | 1000000 | dry-run only |

## Why 60 Minutes Was Still Not Enough For Some Runs

The important clarification is that `60` minutes was not the limiting factor for every individual completed `10,000,000`-agent run.

The completed `10,000,000`-agent runs on disk took:

- run `16`: `1219.924 s` = about `20.3` minutes
- run `17`: `1236.721 s` = about `20.6` minutes
- run `18`: `1263.070 s` = about `21.1` minutes
- run `19`: `1295.405 s` = about `21.6` minutes
- run `20`: `1338.461 s` = about `22.3` minutes

So for these completed `10,000,000`-agent single-process runs, one run by itself fits within a `60`-minute walltime.

What caused walltime trouble earlier was the execution pattern:

- multiple runs were being executed sequentially inside one PBS allocation
- even though each large run was around `20-22` minutes, several such runs back-to-back exceed a `60`-minute job limit
- that is why the old terminal log showed PBS walltime kills during the sequential sweep

There is also a second issue for the next scale jump:

- the `100,000,000` and `1,000,000,000` agent cases were never actually executed in the current results set
- those directories are dry-run placeholders only
- based on how runtime and output size already grow from `1,000,000` to `10,000,000` agents, the `100,000,000` and especially `1,000,000,000` single-process CPU cases are very unlikely to fit comfortably inside `60` minutes

Another reason larger runs become harder is output cost:

- by run `20`, the total output size is already about `14.72 GB`
- the run is writing large `agent_history.csv` and `location_history.csv` files
- that logging overhead adds both time and storage pressure on top of the actual simulation work

So the practical answer is:

- `60` minutes is enough for one `10,000,000`-agent run in the current single-process CPU code
- `60` minutes is not enough for a batch of several of those runs executed sequentially in one PBS job
- `60` minutes is also unlikely to be enough for the still-unrun `100,000,000` and `1,000,000,000` single-process CPU cases

## Takeaways

- Runtime and output size rise sharply with agent count.
- Peak memory rises substantially at `10,000,000` agents, reaching about `915 MB` by run `20`.
- Higher density still tends to produce larger outbreaks by the final step.
- For the current single-process CPU path, batching several large runs into one `60`-minute PBS job is the main walltime risk.
- The `100,000,000` and `1,000,000,000` cases have not been executed yet, so they should be treated as pending rather than failed.
