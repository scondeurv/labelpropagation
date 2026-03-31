# Runtime Evaluation Figure Status

This note tracks the status of the figures we wanted to mirror methodologically from the reference article.

## Already generated

- `startup_latency.svg`
- `load_latency.svg`
- `startup_timeline.svg`
- `collectives_latency.svg`
- `broadcast_latency.svg`
- `all_to_all_latency.svg`
- `ptp_throughput.svg`
- `ptp_throughput_vs_burst_size.svg`
- `labelpropagation_phase_breakdown.svg`
- `sssp_phase_breakdown.svg`
- `labelpropagation_load_vs_granularity.svg`
- `sssp_load_vs_granularity.svg`
- `bfs_end_to_end_comparison.svg`
- `sssp_end_to_end_comparison.svg`
- `wcc_end_to_end_comparison.svg`
- `tables/logical_traffic.json`

## Partially covered

- collectives
  Covered both as a combined view (`collectives_latency.svg`) and as split views (`broadcast_latency.svg`, `all_to_all_latency.svg`).

- loading characterization
  Covered by the dedicated `load_latency.svg` probe over S3 partitions.

- application decomposition
  Covered on a homogeneous scale for `labelpropagation` and `sssp` at `500k` nodes, `4` partitions and granularities `1/2/4`.

- application comparison
  Covered for `bfs`, `sssp` and `wcc` using the saved `standalone/Burst/Spark` campaign outputs.
