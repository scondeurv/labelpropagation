use ahash::AHashMap as HashMap;
use std::{
    cmp::Ordering,
    time::{SystemTime, UNIX_EPOCH},
};

use aws_config::Region;
use aws_credential_types::Credentials;
use aws_sdk_s3::Client as S3Client;
use burst_communication_middleware::{Middleware, MiddlewareActorHandle};
use bytes::Bytes;
use futures::future::join_all;
use serde_derive::{Deserialize, Serialize};
use serde_json::{Error, Value};

const ROOT_WORKER: u32 = 0;
const MAX_ITER: u32 = 50;
const UNKNOWN: u32 = u32::MAX;

/// Input parameters for the Label Propagation action
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Input {
    /// S3 connection and path details
    input_data: S3InputParams,
    /// Total number of nodes in the graph
    num_nodes: u32,
    /// Maximum number of LP iterations (defaults to MAX_ITER)
    max_iterations: Option<u32>,
    /// Convergence threshold: stop if fewer than this many labels change
    convergence_threshold: Option<u32>,
    /// Total number of partitions the graph is divided into
    partitions: u32,
    /// Number of partitions handled by this specific worker
    granularity: u32,
    #[serde(default)]
    group_id: Option<u32>,
    /// Timeout for collective operations in seconds (defaults to auto-calculated)
    timeout_seconds: Option<u64>,
}

/// S3 configuration for fetching graph data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct S3InputParams {
    bucket: String,
    key: String,
    region: String,
    endpoint: Option<String>,
    aws_access_key_id: String,
    aws_secret_access_key: String,
    aws_session_token: Option<String>,
}

/// Resulting output of the action execution
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Output {
    bucket: String,
    key: String,
    /// Performance tracing timestamps
    timestamps: Vec<Timestamp>,
    /// Optional labels (usually skipped in final output for performance)
    #[serde(skip_serializing_if = "Option::is_none")]
    labels: Option<Vec<u32>>,
    /// Results report (only for root worker)
    #[serde(skip_serializing_if = "Option::is_none")]
    results: Option<String>,
}

/// A simple key-value pair for logging execution events with timestamps
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Timestamp {
    key: String,
    value: String,
}

/// Helper function to create a timestamp entry for the current time
fn timestamp(key: &str) -> Timestamp {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    Timestamp {
        key: key.to_string(),
        value: now.to_string(),
    }
}

/// Full label vector message (one `u32` per node)
/// This is the primary data structure exchanged between workers via the middleware.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelsMessage(pub Vec<u32>);

impl From<Bytes> for LabelsMessage {
    /// Deserializes a byte buffer into a vector of u32 labels (Little Endian).
    /// On LE platforms, uses a single memcpy into a properly aligned u32 buffer.
    fn from(bytes: Bytes) -> Self {
        let count = bytes.len() / 4;
        #[cfg(target_endian = "little")]
        {
            let mut result = Vec::<u32>::with_capacity(count);
            // SAFETY: Vec<u32> is properly aligned for u32. On LE platforms, u32 in-memory
            // representation matches the LE byte format. We copy exactly count*4 bytes
            // into a buffer with capacity for `count` u32 values.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    result.as_mut_ptr() as *mut u8,
                    count * 4,
                );
                result.set_len(count);
            }
            LabelsMessage(result)
        }
        #[cfg(not(target_endian = "little"))]
        {
            let vecu32 = bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let arr: [u8; 4] = chunk.try_into().unwrap();
                    u32::from_le_bytes(arr)
                })
                .collect();
            LabelsMessage(vecu32)
        }
    }
}

impl From<LabelsMessage> for Bytes {
    /// Serializes a vector of u32 labels into a byte buffer (Little Endian).
    /// On LE platforms, reinterprets the Vec<u32> memory directly as bytes (zero-copy).
    fn from(val: LabelsMessage) -> Self {
        #[cfg(target_endian = "little")]
        {
            let mut vec = std::mem::ManuallyDrop::new(val.0);
            let byte_len = vec.len() * 4;
            let byte_cap = vec.capacity() * 4;
            let ptr = vec.as_mut_ptr() as *mut u8;
            // SAFETY: On LE, Vec<u32> in-memory layout is identical to LE byte representation.
            // Vec<u32> allocations are at least 4-byte aligned, satisfying u8's requirement.
            // ManuallyDrop prevents double-free: memory ownership transfers to byte_vec.
            let byte_vec = unsafe { Vec::from_raw_parts(ptr, byte_len, byte_cap) };
            Bytes::from(byte_vec)
        }
        #[cfg(not(target_endian = "little"))]
        {
            let mut bytes = Vec::with_capacity(val.0.len() * 4);
            for num in val.0 {
                bytes.extend_from_slice(&num.to_le_bytes());
            }
            Bytes::from(bytes)
        }
    }
}

fn encode_seed_pairs(seed_pairs: &[(u32, u32)]) -> LabelsMessage {
    let mut payload = Vec::with_capacity(seed_pairs.len() * 2);
    for &(node, label) in seed_pairs {
        payload.push(node);
        payload.push(label);
    }
    LabelsMessage(payload)
}

fn apply_seed_pairs(labels: &mut [u32], encoded_pairs: &[u32]) {
    assert!(
        encoded_pairs.len() % 2 == 0,
        "seed payload must contain node/label pairs"
    );
    for pair in encoded_pairs.chunks_exact(2) {
        let node = pair[0] as usize;
        let label = pair[1];
        if node < labels.len() {
            labels[node] = label;
        }
    }
}

/// Count message for convergence (single u64)
#[derive(Debug, Clone)]
struct CountMessage(pub u64);

/// Efficient Graph representation using Compressed Sparse Row (CSR) format
struct CSRGraph {
    owned_nodes: Vec<u32>,
    offsets: Vec<u32>,
    flat_edges: Vec<u32>,
}

impl From<Bytes> for CountMessage {
    fn from(bytes: Bytes) -> Self {
        let mut arr = [0u8; 8];
        if bytes.len() >= 8 {
            arr.copy_from_slice(&bytes[..8]);
        }
        CountMessage(u64::from_le_bytes(arr))
    }
}

impl From<CountMessage> for Bytes {
    fn from(val: CountMessage) -> Self {
        Bytes::from(val.0.to_le_bytes().to_vec())
    }
}

fn assigned_partition_range(partitions: u32, worker_id: u32) -> std::ops::Range<u32> {
    let start = worker_id;
    let end = start + 1;
    assert!(
        end <= partitions,
        "worker {worker_id} assigned partitions [{start}, {end}) outside total partitions {partitions}"
    );
    start..end
}

fn parse_partition_body(
    worker_id: u32,
    part_key: &str,
    body: &str,
    num_nodes: u32,
    edges: &mut Vec<(u32, u32)>,
    initial_labels: &mut Vec<(u32, u32)>,
) -> Result<(), String> {
    if body.trim().is_empty() {
        return Err(format!(
            "[Worker {worker_id}] partition {part_key} is empty"
        ));
    }

    let initial_edge_count = edges.len();
    for line in body.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let mut it = line.split('\t');
        let src = it.next().and_then(|s| s.parse::<u32>().ok());
        let dst = it.next().and_then(|s| s.parse::<u32>().ok());
        let label = it.next().and_then(|s| s.parse::<i64>().ok());
        if let (Some(s), Some(d)) = (src, dst) {
            if s >= num_nodes || d >= num_nodes {
                eprintln!(
                    "[Worker {}] Invalid edge in {}: {} -> {} (max={})",
                    worker_id,
                    part_key,
                    s,
                    d,
                    num_nodes - 1
                );
                continue;
            }
            edges.push((s, d));
            if let Some(l) = label {
                if l >= 0 {
                    initial_labels.push((s, l as u32));
                }
            }
        }
    }

    if edges.len() == initial_edge_count {
        return Err(format!(
            "[Worker {worker_id}] partition {part_key} contains no valid edges"
        ));
    }

    Ok(())
}

fn build_csr_graph(num_nodes: u32, mut edges: Vec<(u32, u32)>) -> CSRGraph {
    edges.sort_unstable_by_key(|e| e.0);
    let mut owned_nodes = Vec::new();
    let mut offsets = vec![0u32; (num_nodes + 1) as usize];
    let mut flat_edges = Vec::with_capacity(edges.len());
    let mut current_offset = 0u32;
    let mut edge_idx = 0;
    for n in 0..num_nodes {
        offsets[n as usize] = current_offset;
        let mut found = false;
        while edge_idx < edges.len() && edges[edge_idx].0 == n {
            flat_edges.push(edges[edge_idx].1);
            edge_idx += 1;
            current_offset += 1;
            found = true;
        }
        if found {
            owned_nodes.push(n);
        }
    }
    offsets[num_nodes as usize] = current_offset;
    CSRGraph {
        owned_nodes,
        offsets,
        flat_edges,
    }
}

fn build_results_report(final_labels: &[u32], num_nodes: u32, completed_iterations: u32) -> String {
    let mut label_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for &label in final_labels {
        *label_counts.entry(label).or_insert(0) += 1;
    }

    let mut report = String::new();
    report.push_str("\n=== Label Propagation Results ===\n");
    report.push_str(&format!("Total nodes: {}\n", num_nodes));
    report.push_str(&format!("Total iterations: {}\n", completed_iterations));
    report.push_str("\nLabel Distribution:\n");
    let mut sorted_labels: Vec<_> = label_counts.iter().collect();
    sorted_labels.sort_by_key(|&(label, _)| label);
    for (label, count) in sorted_labels.iter().take(20) {
        if **label == UNKNOWN {
            report.push_str(&format!("  UNKNOWN: {} nodes\n", count));
        } else {
            report.push_str(&format!("  Label {}: {} nodes\n", label, count));
        }
    }

    report.push_str("\nSample nodes (first 20):\n");
    for i in 0..20.min(num_nodes as usize) {
        let label = final_labels[i];
        let label_str = if label == UNKNOWN {
            "UNKNOWN".to_string()
        } else {
            label.to_string()
        };
        report.push_str(&format!("  Node {}: Label {}\n", i, label_str));
    }
    report.push_str("=================================\n");
    report
}

async fn load_partition_flat(
    params: &Input,
    s3_client: &S3Client,
    worker_id: u32,
) -> Result<(CSRGraph, Vec<(u32, u32)>), String> {
    let part_range = assigned_partition_range(params.partitions, worker_id);
    let mut fetch_futures = Vec::new();
    for p in part_range.clone() {
        let part_key = format!("{}/part-{:05}", params.input_data.key, p);
        fetch_futures.push(async move {
            let output = s3_client
                .get_object()
                .bucket(&params.input_data.bucket)
                .key(&part_key)
                .send()
                .await
                .map_err(|err| format!("[Worker {worker_id}] failed to fetch {part_key}: {err}"))?;
            let data =
                output.body.collect().await.map_err(|err| {
                    format!("[Worker {worker_id}] failed to read {part_key}: {err}")
                })?;
            let body = std::str::from_utf8(&data.to_vec())
                .map_err(|err| format!("[Worker {worker_id}] invalid UTF-8 in {part_key}: {err}"))?
                .to_owned();
            Ok::<(String, String), String>((part_key, body))
        });
    }

    let results = join_all(fetch_futures).await;
    let mut edges = Vec::with_capacity(100_000 * params.granularity as usize);
    let mut initial_labels = Vec::new();
    for result in results {
        let (part_key, body) = result?;
        parse_partition_body(
            worker_id,
            &part_key,
            &body,
            params.num_nodes,
            &mut edges,
            &mut initial_labels,
        )?;
    }

    let graph = build_csr_graph(params.num_nodes, edges);
    println!(
        "[Worker {}] Loaded partitions {:?}: {} owned nodes, {} edges",
        worker_id,
        part_range,
        graph.owned_nodes.len(),
        graph.flat_edges.len()
    );
    Ok((graph, initial_labels))
}

fn should_continue(iter: u32, max_iter: Option<u32>, changed: u32, threshold: u32) -> bool {
    let under_threshold = changed <= threshold;
    let under_iter = match max_iter {
        Some(m) => iter < m,
        None => iter < MAX_ITER,
    };
    under_iter && !under_threshold
}

fn majority_label(counts: &mut HashMap<u32, usize>, current: u32) -> u32 {
    if counts.is_empty() {
        return current;
    }
    let mut best = current;
    let mut best_count = 0usize;
    for (label, count) in counts.iter() {
        if *label == UNKNOWN {
            continue;
        }
        match count.cmp(&best_count) {
            Ordering::Greater => {
                best = *label;
                best_count = *count;
            }
            Ordering::Equal => {
                if *label < best {
                    best = *label;
                }
            }
            Ordering::Less => {}
        }
    }
    counts.clear();
    best
}

fn run_label_propagation_core(
    params: &Input,
    middleware: &MiddlewareActorHandle<LabelsMessage>,
    graph: CSRGraph,
    initial_labels_vec: Vec<(u32, u32)>,
    mut timestamps: Vec<Timestamp>,
) -> (Vec<Timestamp>, Vec<u32>, u32) {
    let worker = middleware.info.worker_id;
    let burst_size = middleware.info.burst_size;
    println!(
        "[Worker {worker}] starting label propagation (burst_size={burst_size}, num_nodes={})",
        params.num_nodes
    );

    // Share only the seeded nodes so the initial collective stays small.
    let initial_msg = encode_seed_pairs(&initial_labels_vec);
    let combined = middleware
        .reduce(initial_msg, |mut left, right| {
            left.0.extend(right.0);
            left
        })
        .unwrap();

    let global_seed_pairs = if let Some(msg) = combined {
        middleware.broadcast(Some(msg), ROOT_WORKER).unwrap()
    } else {
        middleware.broadcast(None, ROOT_WORKER).unwrap()
    };

    let mut global_labels = LabelsMessage(vec![UNKNOWN; params.num_nodes as usize]);
    apply_seed_pairs(&mut global_labels.0, &global_seed_pairs.0);
    let global_has_seeds = !global_seed_pairs.0.is_empty();
    let is_seed: Vec<bool> = if global_has_seeds {
        global_labels.0.iter().map(|&label| label != UNKNOWN).collect()
    } else {
        vec![false; params.num_nodes as usize]
    };

    // Without seeds, start in unsupervised mode using each node ID as its label.
    if !global_has_seeds {
        println!(
            "[Worker {}] No initial labels found globally, using unsupervised mode",
            worker
        );
        for (idx, label) in global_labels.0.iter_mut().enumerate() {
            *label = idx as u32;
        }
    }

    let max_iter = params.max_iterations.unwrap_or(MAX_ITER);
    let threshold = params.convergence_threshold.unwrap_or(0);
    let mut iter = 0;
    let mut completed_iterations = 0;
    let unsupervised_mode = !global_has_seeds;

    let avg_degree = if graph.owned_nodes.is_empty() {
        10
    } else {
        (graph.flat_edges.len() / graph.owned_nodes.len()).max(1)
    };
    let mut counts_map = HashMap::with_capacity(avg_degree.min(100));

    let mut labels_a = global_labels.0;
    let mut labels_b = vec![UNKNOWN; params.num_nodes as usize];
    let mut use_a_as_read = true;

    let extended_size = (params.num_nodes as usize) + 1;
    let mut local_updates = vec![UNKNOWN; extended_size];

    // Run the propagation loop until the labels converge or we hit the iteration limit.
    while iter < max_iter {
        timestamps.push(timestamp(&format!("iter_{}_start", iter)));
        let read_buf = if use_a_as_read { &labels_a } else { &labels_b };
        local_updates.fill(UNKNOWN);
        let mut local_changed: u32 = 0;

        // Recompute the labels for the nodes owned by this worker.
        for &node in &graph.owned_nodes {
            let idx = node as usize;
            let current_label = read_buf[idx];

            if !unsupervised_mode && is_seed[idx] {
                local_updates[idx] = current_label;
                continue;
            }

            let start = graph.offsets[idx];
            let end = graph.offsets[idx + 1];

            for i in start..end {
                let neighbor = graph.flat_edges[i as usize] as usize;
                let label = read_buf[neighbor];
                if label != UNKNOWN {
                    *counts_map.entry(label).or_insert(0) += 1;
                }
            }

            let new_label = majority_label(&mut counts_map, current_label);
            local_updates[idx] = new_label;
            if new_label != current_label {
                local_changed += 1;
            }
        }
        local_updates[params.num_nodes as usize] = local_changed;
        timestamps.push(timestamp(&format!("iter_{}_compute", iter)));

        // Merge local updates into one global label vector and sum the change counts.
        let reduced = middleware
            .reduce(
                LabelsMessage(std::mem::replace(&mut local_updates, Vec::new())),
                |mut left, right| {
                    let n = left.0.len() - 1;
                    for i in 0..n {
                        if left.0[i] == UNKNOWN {
                            left.0[i] = right.0[i];
                        }
                    }
                    left.0[n] = left.0[n].saturating_add(right.0[n]);
                    left
                },
            )
            .unwrap();
        timestamps.push(timestamp(&format!("iter_{}_reduce", iter)));

        // Broadcast the merged labels so the next round reads a synchronized state.
        let global = if let Some(msg) = reduced {
            middleware.broadcast(Some(msg), ROOT_WORKER).unwrap()
        } else {
            middleware.broadcast(None, ROOT_WORKER).unwrap()
        };
        timestamps.push(timestamp(&format!("iter_{}_broadcast", iter)));

        let total_changed = global.0[params.num_nodes as usize];
        if use_a_as_read {
            labels_b.copy_from_slice(&global.0[..params.num_nodes as usize]);
        } else {
            labels_a.copy_from_slice(&global.0[..params.num_nodes as usize]);
        }

        local_updates = global.0;
        local_updates.resize(extended_size, UNKNOWN);

        use_a_as_read = !use_a_as_read;
        completed_iterations += 1;

        if worker == ROOT_WORKER {
            println!("[Worker {worker}] iter {iter}: changed={total_changed}");
        }
        if !should_continue(iter, params.max_iterations, total_changed, threshold) {
            break;
        }
        timestamps.push(timestamp(&format!("iter_{}_end", iter)));
        iter += 1;
    }

    timestamps.push(timestamp("worker_end"));
    let final_labels = if use_a_as_read { labels_a } else { labels_b };
    (timestamps, final_labels, completed_iterations)
}

fn label_propagation(params: Input, middleware: &MiddlewareActorHandle<LabelsMessage>) -> Output {
    // Set up the worker runtime and the clients needed for this invocation.
    let mut timestamps = vec![timestamp("worker_start")];

    let worker = middleware.info.worker_id;

    let credentials_provider = Credentials::from_keys(
        params.input_data.aws_access_key_id.clone(),
        params.input_data.aws_secret_access_key.clone(),
        params.input_data.aws_session_token.clone(),
    );

    let config = match params.input_data.endpoint.clone() {
        Some(endpoint) => aws_sdk_s3::config::Builder::new()
            .endpoint_url(endpoint)
            .credentials_provider(credentials_provider)
            .region(Region::new(params.input_data.region.clone()))
            .force_path_style(true)
            .build(),
        None => aws_sdk_s3::config::Builder::new()
            .credentials_provider(credentials_provider)
            .region(Region::new(params.input_data.region.clone()))
            .build(),
    };
    let s3_client = S3Client::from_conf(config);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // Load the partitions assigned to this worker and build the local CSR graph.
    timestamps.push(timestamp("get_input"));
    let (graph, initial_labels_vec) = rt
        .block_on(load_partition_flat(&params, &s3_client, worker))
        .unwrap_or_else(|err| panic!("{err}"));
    timestamps.push(timestamp("get_input_end"));

    let (mut timestamps, final_labels, completed_iterations) =
        run_label_propagation_core(&params, middleware, graph, initial_labels_vec, timestamps);

    // Worker 0 writes the final labels and emits the summary used by validation and benchmarks.
    let results_report = if worker == ROOT_WORKER {
        timestamps.push(timestamp("write_labels_start"));

        let labels_map: std::collections::HashMap<String, u32> = (0..params.num_nodes)
            .map(|i| (i.to_string(), final_labels[i as usize]))
            .collect();

        let report = build_results_report(&final_labels, params.num_nodes, completed_iterations);
        println!("{}", report);

        let output_key = format!("{}/output/labels_final.json", params.input_data.key);

        if labels_map.len() < 10_000_000 {
            let labels_json = serde_json::json!({ "labels": labels_map });
            let labels_str = serde_json::to_string(&labels_json).unwrap();
            let write_result = rt.block_on(async {
                s3_client
                    .put_object()
                    .bucket(&params.input_data.bucket)
                    .key(&output_key)
                    .body(labels_str.into_bytes().into())
                    .send()
                    .await
            });
            match write_result {
                Ok(_) => println!(
                    "[Worker {}] ✓ Wrote final labels to s3://{}/{}",
                    worker, params.input_data.bucket, output_key
                ),
                Err(e) => eprintln!("[Worker {}] ✗ Failed to write labels: {:?}", worker, e),
            }
        } else {
            println!(
                "[Worker {}] ! Skipping large JSON serialization for S3 ({} nodes)",
                worker,
                labels_map.len()
            );
        }

        timestamps.push(timestamp("write_labels_end"));
        Some(report)
    } else {
        None
    };

    Output {
        bucket: params.input_data.bucket.clone(),
        key: format!("worker-{}", worker),
        timestamps,
        labels: None,
        results: results_report,
    }
}

pub fn main(args: Value, burst_middleware: Middleware<LabelsMessage>) -> Result<Value, Error> {
    let input: Input = serde_json::from_value(args)?;

    if input.partitions % input.granularity != 0 {
        panic!(
            "ERROR: partitions ({}) must be divisible by granularity ({}) for balanced distribution",
            input.partitions, input.granularity
        );
    }

    let handle = burst_middleware.get_actor_handle();
    let result = label_propagation(input, &handle);
    serde_json::to_value(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use burst_communication_middleware::{
        BurstMiddleware, BurstOptions, Middleware, RemoteBroadcastProxy, RemoteMessage,
        RemoteSendReceiveFactory, RemoteSendReceiveProxy, TokioChannelImpl, TokioChannelOptions,
    };
    use std::{
        collections::{HashMap as StdHashMap, HashSet},
        sync::Arc,
        thread,
    };

    struct DummyRemoteProxy;

    #[async_trait]
    impl burst_communication_middleware::RemoteSendProxy for DummyRemoteProxy {
        async fn remote_send(
            &self,
            _dest: u32,
            _msg: RemoteMessage,
        ) -> burst_communication_middleware::Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl burst_communication_middleware::RemoteReceiveProxy for DummyRemoteProxy {
        async fn remote_recv(
            &self,
            _source: u32,
        ) -> burst_communication_middleware::Result<RemoteMessage> {
            panic!("remote recv should not be used in the local distributed LP test");
        }
    }

    impl RemoteSendReceiveProxy for DummyRemoteProxy {}

    #[async_trait]
    impl burst_communication_middleware::RemoteBroadcastSendProxy for DummyRemoteProxy {
        async fn remote_broadcast_send(
            &self,
            _msg: RemoteMessage,
        ) -> burst_communication_middleware::Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl burst_communication_middleware::RemoteBroadcastReceiveProxy for DummyRemoteProxy {
        async fn remote_broadcast_recv(
            &self,
        ) -> burst_communication_middleware::Result<RemoteMessage> {
            panic!("remote broadcast recv should not be used in the local distributed LP test");
        }
    }

    impl RemoteBroadcastProxy for DummyRemoteProxy {}

    struct DummyRemoteFactory;

    #[async_trait]
    impl RemoteSendReceiveFactory<()> for DummyRemoteFactory {
        async fn create_remote_proxies(
            burst_options: Arc<BurstOptions>,
            _options: (),
        ) -> burst_communication_middleware::Result<
            StdHashMap<
                u32,
                (
                    Box<dyn RemoteSendReceiveProxy>,
                    Box<dyn RemoteBroadcastProxy>,
                ),
            >,
        > {
            let current_group = burst_options
                .group_ranges
                .get(&burst_options.group_id)
                .unwrap();
            Ok(current_group
                .iter()
                .map(|worker_id| {
                    (
                        *worker_id,
                        (
                            Box::new(DummyRemoteProxy) as Box<dyn RemoteSendReceiveProxy>,
                            Box::new(DummyRemoteProxy) as Box<dyn RemoteBroadcastProxy>,
                        ),
                    )
                })
                .collect())
        }
    }

    fn run_lp_local(
        edges: Vec<(u32, u32, Option<u32>)>,
        num_nodes: u32,
        max_iter: u32,
    ) -> Vec<u32> {
        let mut owned_nodes = Vec::new();
        let mut offsets = vec![0u32; (num_nodes + 1) as usize];
        let mut flat_edges = Vec::new();

        let mut adj: HashMap<u32, Vec<u32>> = HashMap::default();
        let mut seeds: HashMap<u32, u32> = HashMap::default();

        for (src, dst, label) in edges {
            adj.entry(src).or_insert_with(Vec::new).push(dst);
            if let Some(l) = label {
                seeds.insert(src, l);
            }
        }
        for node in 0..num_nodes {
            if let Some(neighbors) = adj.get(&node) {
                owned_nodes.push(node);
                offsets[node as usize] = flat_edges.len() as u32;
                flat_edges.extend_from_slice(neighbors);
            }
        }
        offsets[num_nodes as usize] = flat_edges.len() as u32;
        let unsupervised = seeds.is_empty();
        let mut labels = vec![UNKNOWN; num_nodes as usize];

        if unsupervised {
            for i in 0..num_nodes {
                labels[i as usize] = i;
            }
        } else {
            for (&node, &label) in &seeds {
                labels[node as usize] = label;
            }
        }
        for _ in 0..max_iter {
            let prev_labels = labels.clone();
            let mut changed = 0;

            for &node in &owned_nodes {
                if !unsupervised && seeds.contains_key(&node) {
                    continue;
                }

                let start = offsets[node as usize] as usize;
                let end = offsets[(node + 1) as usize] as usize;
                let neighbors = &flat_edges[start..end];

                let mut counts: HashMap<u32, usize> = HashMap::default();
                for &neighbor in neighbors {
                    let l = prev_labels[neighbor as usize];
                    if l != UNKNOWN {
                        *counts.entry(l).or_insert(0) += 1;
                    }
                }
                if counts.is_empty() {
                    continue;
                }

                let mut best = prev_labels[node as usize];
                let mut best_count = 0usize;

                for (&label, &count) in &counts {
                    if label == UNKNOWN {
                        continue;
                    }
                    match count.cmp(&best_count) {
                        Ordering::Greater => {
                            best = label;
                            best_count = count;
                        }
                        Ordering::Equal => {
                            if label < best {
                                best = label;
                            }
                        }
                        Ordering::Less => {}
                    }
                }

                if best != prev_labels[node as usize] {
                    labels[node as usize] = best;
                    changed += 1;
                }
            }

            if changed == 0 {
                break;
            }
        }

        labels
    }

    #[test]
    fn partition_range_matches_worker_id() {
        assert_eq!(assigned_partition_range(8, 0), 0..1);
        assert_eq!(assigned_partition_range(8, 1), 1..2);
        assert_eq!(assigned_partition_range(8, 6), 6..7);
        assert_eq!(assigned_partition_range(8, 7), 7..8);
    }

    #[test]
    fn grouped_workers_keep_distinct_partitions() {
        let group0 = (0..4)
            .map(|worker_id| assigned_partition_range(8, worker_id))
            .collect::<Vec<_>>();
        let group1 = (4..8)
            .map(|worker_id| assigned_partition_range(8, worker_id))
            .collect::<Vec<_>>();
        assert_eq!(group0, vec![0..1, 1..2, 2..3, 3..4]);
        assert_eq!(group1, vec![4..5, 5..6, 6..7, 7..8]);
    }

    #[test]
    fn empty_partition_body_is_rejected() {
        let mut edges = Vec::new();
        let mut labels = Vec::new();
        let err = parse_partition_body(1, "graphs/part-00002", "", 10, &mut edges, &mut labels)
            .unwrap_err();
        assert!(err.contains("partition graphs/part-00002 is empty"));
    }

    #[test]
    fn results_report_uses_completed_iteration_count() {
        let report = build_results_report(&[0, 1, 1, 2], 4, 1);
        assert!(report.contains("Total iterations: 1"));
    }

    #[test]
    fn distributed_lp_converges_to_expected_labels() {
        let params = Input {
            input_data: S3InputParams {
                bucket: "unused".to_string(),
                key: "unused".to_string(),
                region: "us-east-1".to_string(),
                endpoint: None,
                aws_access_key_id: "unused".to_string(),
                aws_secret_access_key: "unused".to_string(),
                aws_session_token: None,
            },
            num_nodes: 3,
            max_iterations: Some(10),
            convergence_threshold: Some(0),
            partitions: 2,
            granularity: 1,
            group_id: None,
            timeout_seconds: None,
        };

        let group_ranges = vec![(0.to_string(), vec![0, 1].into_iter().collect())]
            .into_iter()
            .collect::<StdHashMap<String, HashSet<u32>>>();

        let tokio_runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        let proxies = tokio_runtime
            .block_on(BurstMiddleware::create_proxies::<
                TokioChannelImpl,
                DummyRemoteFactory,
                _,
                _,
            >(
                BurstOptions::new(2, group_ranges, 0.to_string())
                    .burst_id("lp-local-test".to_string())
                    .enable_message_chunking(false)
                    .build(),
                TokioChannelOptions::new()
                    .broadcast_channel_size(32)
                    .build(),
                (),
            ))
            .unwrap();

        let mut actors = proxies
            .into_iter()
            .map(|(worker_id, middleware)| {
                (
                    worker_id,
                    Middleware::new(middleware, tokio_runtime.handle().clone()),
                )
            })
            .collect::<StdHashMap<u32, Middleware<LabelsMessage>>>();

        let worker0_graph = build_csr_graph(3, vec![(0, 1), (0, 2), (2, 0), (2, 1)]);
        let worker1_graph = build_csr_graph(3, vec![(1, 0), (1, 2)]);
        let worker0_labels = vec![(0, 100)];
        let worker1_labels = Vec::new();

        let handle0 = actors.remove(&0).unwrap().get_actor_handle();
        let handle1 = actors.remove(&1).unwrap().get_actor_handle();

        let params0 = params.clone();
        let params1 = params.clone();
        let thread0 = thread::spawn(move || {
            run_label_propagation_core(
                &params0,
                &handle0,
                worker0_graph,
                worker0_labels,
                vec![timestamp("worker_start")],
            )
        });
        let thread1 = thread::spawn(move || {
            run_label_propagation_core(
                &params1,
                &handle1,
                worker1_graph,
                worker1_labels,
                vec![timestamp("worker_start")],
            )
        });

        let (_timestamps0, labels0, iterations0) = thread0.join().unwrap();
        let (_timestamps1, labels1, iterations1) = thread1.join().unwrap();

        assert_eq!(iterations0, 2);
        assert_eq!(iterations1, 2);
        assert_eq!(labels0, vec![100, 100, 100]);
        assert_eq!(labels1, vec![100, 100, 100]);
    }

    #[test]
    fn test_triangle_graph() {
        let edges = vec![
            (0, 1, Some(100)),
            (0, 2, Some(100)),
            (1, 0, None),
            (1, 2, None),
            (2, 0, None),
            (2, 1, None),
        ];

        let result = run_lp_local(edges, 3, 10);

        assert_eq!(result[0], 100);
        assert_eq!(result[1], 100);
        assert_eq!(result[2], 100);
    }

    #[test]
    fn test_star_graph() {
        let edges = vec![
            (0, 1, Some(42)),
            (0, 2, Some(42)),
            (0, 3, Some(42)),
            (0, 4, Some(42)),
            (1, 0, None),
            (2, 0, None),
            (3, 0, None),
            (4, 0, None),
        ];

        let result = run_lp_local(edges, 5, 10);

        assert_eq!(result[0], 42);
        assert_eq!(result[1], 42);
        assert_eq!(result[2], 42);
        assert_eq!(result[3], 42);
        assert_eq!(result[4], 42);
    }

    #[test]
    fn test_unsupervised_triangle() {
        let edges = vec![
            (0, 1, None),
            (0, 2, None),
            (1, 0, None),
            (1, 2, None),
            (2, 0, None),
            (2, 1, None),
        ];

        let result = run_lp_local(edges, 3, 10);

        // Todos deben converger a 0 (la etiqueta más pequeña)
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 0);
        assert_eq!(result[2], 0);
    }

    #[test]
    fn test_deterministic() {
        let edges = vec![
            (0, 1, Some(7)),
            (0, 2, None),
            (1, 0, None),
            (1, 2, None),
            (2, 0, None),
            (2, 1, None),
        ];

        let result1 = run_lp_local(edges.clone(), 3, 10);
        let result2 = run_lp_local(edges, 3, 10);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_tie_breaking() {
        let edges = vec![
            (0, 2, Some(50)),
            (1, 2, Some(30)),
            (2, 0, None),
            (2, 1, None),
        ];

        let result = run_lp_local(edges, 3, 5);

        assert_eq!(result[0], 50);
        assert_eq!(result[1], 30);
        // Nodo 2 tiene empate, debe elegir la más pequeña
        assert_eq!(result[2], 30);
    }
}
