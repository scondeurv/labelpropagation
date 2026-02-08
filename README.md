# Label Propagation - Distributed Implementation

Implementation of the Label Propagation algorithm using OpenWhisk Burst for distributed execution.

## Structure

- `lpst/`: Standalone Rust implementation (local, single-node)
- `ow-lp/`: Distributed implementation using Burst middleware
- `generate_payload.py`: Script to generate input payloads
- `labelpropagation_utils.py`: Utility functions for payload generation
- `labelpropagation.py`: Main execution script for OpenWhisk

## Local Implementation (lpst/)

See `lpst/README.md` for details on the local single-node implementation.

## Distributed Implementation (ow-lp/)

### Building

```bash
cd ow-lp
cargo build --release
```

### Input Format

The distributed version expects graph data in S3 with the following format:

```
source_node dest_node [initial_label]
```

Each line represents either:
- An edge: `node1 node2`
- An edge with initial label: `node1 node2 label`

Example:
```
0 1 0
1 2
2 3
3 4 1
```

### Generating Payload

```bash
python generate_payload.py \
    --partitions 4 \
    --num_nodes 1000 \
    --convergence_threshold 0.01 \
    --s3_bucket my-bucket \
    --s3_key graphs/my-graph \
    --s3_endpoint http://localhost:9000 \
    --s3_region us-east-1 \
    --aws_access_key_id minioadmin \
    --aws_secret_access_key minioadmin \
    --output labelpropagation_payload.json
```

### Running Distributed Label Propagation

```bash
python labelpropagation.py \
    --ow-host localhost \
    --ow-port 31001 \
    --lp-endpoint http://localhost:9000 \
    --partitions 4 \
    --num-nodes 1000 \
    --bucket my-bucket \
    --key graphs/my-graph \
    --granularity 4 \
    --backend redis \
    --chunk-size 1048576
```

## Algorithm Overview

The distributed implementation follows these steps:

1. **Partitioning**: Nodes are partitioned across workers using modulo distribution
2. **Initialization**: Each worker loads its partition's edges and initial labels from S3
3. **Iteration**:
   - Each worker broadcasts its nodes' labels to neighbors on other workers
   - Workers gather label information from all other workers
   - Labels are updated synchronously with clamping (initial labels don't change)
   - Convergence is checked globally across all workers
4. **Termination**: Algorithm stops when change ratio falls below threshold or max iterations reached

### Communication Pattern

- **Broadcast**: Share label information with specific workers
- **Gather**: Collect label updates from all workers
- **Convergence Check**: Root worker aggregates change ratios and broadcasts decision

## Performance Considerations

- Node distribution via modulo ensures balanced workload
- Message chunking enabled for large graphs
- Synchronous updates guarantee correctness
- Deterministic tie-breaking ensures reproducibility

## Comparison with PageRank

Like PageRank, this implementation:
- Uses Burst middleware for distributed communication
- Partitions data across workers
- Performs iterative synchronous updates
- Supports convergence-based termination
- Reads from S3 and tracks timestamps

Key differences:
- Propagates discrete labels instead of continuous scores
- Uses majority voting instead of weighted sums
- Implements clamping for semi-supervised learning
- Requires less floating-point computation
