# Resource Sweep Status

Updated: 2026-04-02 01:38 Europe/Madrid

## Current state

- Completed phases:
  - `plan`
  - `burst-feasibility`
  - `spark-feasibility`
  - `config-sweep` for `burst`
  - `config-sweep` for `spark`
  - `size-sweep` for `burst`
  - `size-sweep` for `spark`
- In progress:
  - none

## Spark size-sweep final status

- Completed points: `64/64`
- Remaining points: `0/64`
- Overall status: `64 passed`, `0 failed`
- Completed by algorithm:
  - `bfs`: `16/16`
  - `sssp`: `16/16`
  - `labelpropagation`: `16/16`
  - `wcc`: `16/16`

## Last completed point

- Algorithm: `labelpropagation`
- Nodes: `2000000`
- Partitions: `32`
- Executors: `16`
- Executor memory: `3g`
- Status: `passed`
- Primary metric: `408632.0 ms`
- Validation: `passed`

## Outputs

- Spark size sweep:
  - `/home/sergio/src/labelpropagation/resource_sweep/size_sweep/spark_size_sweep.json`
- Burst size sweep:
  - `/home/sergio/src/labelpropagation/resource_sweep/size_sweep/burst_size_sweep.json`
- Plan:
  - `/home/sergio/src/labelpropagation/resource_sweep/plans/resource_sweep_plan.json`
- Generated report index:
  - `/home/sergio/src/labelpropagation/resource_sweep/reports/index.json`
- Generated report summary:
  - `/home/sergio/src/labelpropagation/resource_sweep/reports/docs/summary.md`

## Notes

- The Spark size sweep was resumed from the existing JSON and the previous `labelpropagation` timeouts were recovered with a higher command timeout.
- The final sweep JSON is no longer partial: `metadata.partial = false`.
- Spark containers are still up after the sweep; host memory is currently healthy (`57 GiB` available).
- Resource-sweep figures and tables have been regenerated from the completed JSON artifacts.

## Remaining work

- Update `doc-tfm` if the completed Spark sweep should be reflected in the document.
