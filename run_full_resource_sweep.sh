#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/sergio/src/labelpropagation"
PYTHON_BIN="python3"
DRIVER="$ROOT/run_resource_sweep_campaign.py"

COMMON_ARGS=(
  --app-partitions 4,8,12
  --feasibility-nodes 500000
  --config-nodes 500000
  --size-nodes 100000,500000,1000000,2000000
  --algorithms bfs,sssp,labelpropagation,wcc
  --ow-host localhost
  --ow-port 31001
  --worker-s3-endpoint http://minio-service.default.svc.cluster.local:9000
  --host-s3-endpoint http://localhost:9000
  --bucket test-bucket
  --backend redis-list
  --chunk-size-kb 1024
  --sssp-max-iterations 100
  --lp-iterations 20
  --command-timeout-sec 300
)

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

run_phase() {
  log "phase: $*"
  "$PYTHON_BIN" "$DRIVER" "$@"
}

log "saving plan"
run_phase --phase plan "${COMMON_ARGS[@]}"

log "running Burst sweep"
run_phase --phase burst-feasibility --prepare-burst-cluster "${COMMON_ARGS[@]}"
run_phase --phase config-sweep --framework-scope burst --prepare-burst-cluster "${COMMON_ARGS[@]}"
run_phase --phase size-sweep --framework-scope burst --prepare-burst-cluster "${COMMON_ARGS[@]}"

log "stopping Minikube to free host resources before Spark"
minikube stop

log "running Spark sweep"
run_phase --phase spark-feasibility --prepare-spark-cluster --stop-spark-cluster "${COMMON_ARGS[@]}"
run_phase --phase config-sweep --framework-scope spark --prepare-spark-cluster --stop-spark-cluster "${COMMON_ARGS[@]}"
run_phase --phase size-sweep --framework-scope spark --prepare-spark-cluster --stop-spark-cluster "${COMMON_ARGS[@]}"

log "full resource sweep finished"
