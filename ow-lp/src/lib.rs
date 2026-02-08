use std::{
    cmp::Ordering,
    time::{SystemTime, UNIX_EPOCH},
};
use ahash::AHashMap as HashMap;

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
    /// Deserializes a byte buffer into a vector of u32 labels (Little Endian)
    fn from(bytes: Bytes) -> Self {
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

impl From<LabelsMessage> for Bytes {
    /// Serializes a vector of u32 labels into a byte buffer (Little Endian)
    fn from(val: LabelsMessage) -> Self {
        let mut bytes = Vec::with_capacity(val.0.len() * 4);
        for num in val.0 {
            bytes.extend_from_slice(&num.to_le_bytes());
        }
        Bytes::from(bytes)
    }
}

/// Count message for convergence (single u64)
#[derive(Debug, Clone)]
struct CountMessage(pub u64);

/// Efficient Graph representation using Compressed Sparse Row (CSR) format
struct CSRGraph {
    /// List of node IDs owned by this worker
    owned_nodes: Vec<u32>,
    /// Offsets into the `flat_edges` vector for each node
    offsets: Vec<u32>,
    /// Adjacency list stored in a single flat vector
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

/// Load adjacency for this worker partition from S3.
/// 
/// It tries to fetch pre-partitioned files (e.g., `part-0`, `part-1`) based on 
/// the worker's ID and granularity. If that fails, it falls back to reading 
/// the full graph and filtering locally.
async fn load_partition_flat(
    params: &Input,
    s3_client: &S3Client,
    worker_id: u32,
) -> (CSRGraph, Vec<(u32, u32)>) {
    let start_part = worker_id * params.granularity;
    let end_part = (worker_id + 1) * params.granularity;

    // Fetch Multiple partitions in parallel
    let mut fetch_futures = Vec::new();
    for p in start_part..end_part {
        let part_key = format!("{}/part-{:05}", params.input_data.key, p);
        fetch_futures.push(async move {
            s3_client
                .get_object()
                .bucket(&params.input_data.bucket)
                .key(&part_key)
                .send()
                .await
        });
    }

    let results = join_all(fetch_futures).await;
    let mut edges = Vec::with_capacity(100_000 * params.granularity as usize);
    let mut initial_labels = Vec::new();

    let mut all_found = true;
    for result in results {
        match result {
            Ok(output) => {
                if let Ok(data) = output.body.collect().await {
                    let bytes = data.to_vec();
                    if let Ok(body_str) = std::str::from_utf8(&bytes) {
                        // Graph lines format: src_node \t dst_node [\t initial_label]
                        for line in body_str.lines() {
                            if line.trim().is_empty() { continue; }
                            let mut it = line.split('\t');
                            let src = it.next().and_then(|s| s.parse::<u32>().ok());
                            let dst = it.next().and_then(|s| s.parse::<u32>().ok());
                            let label = it.next().and_then(|s| s.parse::<i64>().ok());
                            if let (Some(s), Some(d)) = (src, dst) {
                                // Validate node IDs are within bounds
                                if s >= params.num_nodes || d >= params.num_nodes {
                                    eprintln!("[Worker {}] Invalid edge: {} -> {} (max={})", worker_id, s, d, params.num_nodes - 1);
                                    continue;
                                }
                                edges.push((s, d));
                                // Only process valid positive labels
                                if let Some(l) = label { if l >= 0 { initial_labels.push((s, l as u32)); } }
                            }
                        }
                    }
                } else { all_found = false; break; }
            }
            Err(_) => { all_found = false; break; }
        }
    }

    // Fallback: Read full graph file if partitions aren't available
    if !all_found || edges.is_empty() {
        println!("[Worker {}] Falling back to full graph: {}", worker_id, params.input_data.key);
        if let Ok(output) = s3_client.get_object().bucket(&params.input_data.bucket).key(&params.input_data.key).send().await {
            if let Ok(data) = output.body.collect().await {
                let bytes = data.to_vec();
                if let Ok(body_str) = std::str::from_utf8(&bytes) {
                    for line in body_str.lines() {
                        if line.trim().is_empty() { continue; }
                        let mut it = line.split('\t');
                        let src = it.next().and_then(|s| s.parse::<u32>().ok());
                        let dst = it.next().and_then(|s| s.parse::<u32>().ok());
                        let label = it.next().and_then(|s| s.parse::<i64>().ok());
                        if let (Some(s), Some(d)) = (src, dst) {
                            // Validate node IDs are within bounds
                            if s >= params.num_nodes || d >= params.num_nodes {
                                eprintln!("[Worker {}] Invalid edge in fallback: {} -> {} (max={})", worker_id, s, d, params.num_nodes - 1);
                                continue;
                            }
                            // Only keep edges where 'src' belongs to this worker
                            let target_worker = (s % params.partitions) / params.granularity;
                            if target_worker == worker_id {
                                edges.push((s, d));
                                if let Some(l) = label { if l >= 0 { initial_labels.push((s, l as u32)); } }
                            }
                        }
                    }
                }
            }
        }
    }

    // Convert Edgelist to CSR format for efficient neighbor lookup
    edges.sort_unstable_by_key(|e| e.0);
    let mut owned_nodes = Vec::new();
    let mut offsets = vec![0u32; (params.num_nodes + 1) as usize];
    let mut flat_edges = Vec::with_capacity(edges.len());
    let mut current_offset = 0u32;
    let mut edge_idx = 0;
    for n in 0..params.num_nodes {
        offsets[n as usize] = current_offset;
        let mut found = false;
        while edge_idx < edges.len() && edges[edge_idx].0 == n {
            flat_edges.push(edges[edge_idx].1);
            edge_idx += 1;
            current_offset += 1;
            found = true;
        }
        if found { owned_nodes.push(n); }
    }
    offsets[params.num_nodes as usize] = current_offset;

    println!("[Worker {}] Final graph size: {} owned nodes, {} edges", worker_id, owned_nodes.len(), flat_edges.len());
    if flat_edges.is_empty() {
        panic!("[Worker {}] CRITICAL: No edges loaded! Check S3 connectivity and bucket: {}", worker_id, params.input_data.key);
    }
    (CSRGraph { owned_nodes, offsets, flat_edges }, initial_labels)
}

/// Estimate appropriate timeout based on problem size and cluster configuration
fn estimate_timeout(num_nodes: u32, burst_size: u32, max_iter: u32) -> u64 {
    // Base timeout + scaling factor based on nodes, iterations, and inverse of workers
    let base_time = 60u64; // 1 minute base
    let node_factor = (num_nodes / 10_000).max(1) as u64;
    let worker_factor = 10u64 / burst_size.max(1) as u64;
    let iter_factor = max_iter as u64;
    
    base_time + (node_factor * iter_factor * worker_factor)
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

/// Main distributed Label Propagation logic
fn label_propagation(
    params: Input,
    middleware: &MiddlewareActorHandle<LabelsMessage>,
) -> Output {
    let mut timestamps = vec![timestamp("worker_start")];

    let worker = middleware.info.worker_id;
    let burst_size = middleware.info.burst_size;
    println!(
        "[Worker {worker}] starting label propagation (burst_size={burst_size}, num_nodes={})",
        params.num_nodes
    );

    // Initialize the S3 client using provided credentials
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

    // Load graph partition for this worker from S3
    timestamps.push(timestamp("get_input"));
    let (graph, initial_labels_vec) = rt.block_on(load_partition_flat(&params, &s3_client, worker));
    timestamps.push(timestamp("get_input_end"));

    // Initialize local portion of the labels vector
    let mut labels = vec![UNKNOWN; params.num_nodes as usize];
    let mut initial_labels = HashMap::default();
    for (node, label) in initial_labels_vec {
        if (node as usize) < labels.len() {
            labels[node as usize] = label;
            initial_labels.insert(node, label);
        }
    }

    // Phase 1: Distributed Coordination to check if any worker has seed labels
    // This determines if we run in supervised mode (seed-based) or unsupervised mode (ID-based)
    let local_has_seeds_val = if !initial_labels.is_empty() { 1 } else { 0 };
    let reduced_seeds_msg = middleware
        .reduce(LabelsMessage(vec![local_has_seeds_val]), |mut a, b| {
            if a.0[0] == 1 || b.0[0] == 1 { a.0[0] = 1; }
            a
        })
        .unwrap();

    let global_seeds_msg = if let Some(msg) = reduced_seeds_msg {
        middleware.broadcast(Some(msg), ROOT_WORKER).unwrap()
    } else {
        middleware.broadcast(None, ROOT_WORKER).unwrap()
    };
    let global_has_seeds = global_seeds_msg.0[0] == 1;
    
    // In unsupervised mode, each node starts with its own ID as its label
    if !global_has_seeds && initial_labels.is_empty() {
        println!("[Worker {}] No initial labels found globally, using unsupervised mode", worker);
        for &idx in &graph.owned_nodes {
            labels[idx as usize] = idx;
        }
    }

    // Phase 2: Synchronize initial labels across all workers
    // This ensures every worker has a complete view of all initial/seed labels
    let initial_msg = LabelsMessage(labels);
    let combined = middleware
        .reduce(initial_msg, |mut left, right| {
            for (a, b) in left.0.iter_mut().zip(right.0.iter()) {
                if *a == UNKNOWN { *a = *b; }
            }
            left
        })
        .unwrap();

    let global_labels = if let Some(msg) = combined {
        middleware.broadcast(Some(msg), ROOT_WORKER).unwrap()
    } else {
        middleware.broadcast(None, ROOT_WORKER).unwrap()
    };

    let max_iter = params.max_iterations.unwrap_or(MAX_ITER);
    let threshold = params.convergence_threshold.unwrap_or(0);
    let mut iter = 0;
    let unsupervised_mode = !global_has_seeds;
    
    // Pre-allocate counts_map with estimated capacity to reduce rehashing
    let avg_degree = if graph.owned_nodes.is_empty() {
        10
    } else {
        (graph.flat_edges.len() / graph.owned_nodes.len()).max(1)
    };
    let mut counts_map = HashMap::with_capacity(avg_degree.min(100));
    
    // Double-buffering: separate read and write buffers to prevent race conditions
    let mut labels_a = global_labels.0;
    let mut labels_b = vec![UNKNOWN; params.num_nodes as usize];
    let mut use_a_as_read = true;

    // Main Iterative Loop
    while iter < max_iter {
        timestamps.push(timestamp(&format!("iter_{}_start", iter)));

        // Select read and write buffers
        let (read_buf, write_buf) = if use_a_as_read {
            (&labels_a, &mut labels_b)
        } else {
            (&labels_b, &mut labels_a)
        };

        // 1. Compute local updates: each worker updates labels for its owned nodes
        let mut local_updates = vec![UNKNOWN; params.num_nodes as usize];
        let mut local_changed: u32 = 0;

        for &node in &graph.owned_nodes {
            let idx = node as usize;
            let current_label = read_buf[idx];

            // Primary constraint: don't update seed labels in supervised mode
            if !unsupervised_mode && initial_labels.contains_key(&node) {
                local_updates[idx] = current_label;
                continue;
            }

            // Majority Label Rule: choose the most common label among neighbors
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
        timestamps.push(timestamp(&format!("iter_{}_compute", iter)));

        // 2. Reduce labels from all workers into a single global vector
        // Only one worker per node owns its label update
        let reduced_labels = middleware
            .reduce(LabelsMessage(local_updates), |mut left, right| {
                for (a, b) in left.0.iter_mut().zip(right.0.iter()) {
                    if *a == UNKNOWN { *a = *b; }
                }
                left
            })
            .unwrap();
        timestamps.push(timestamp(&format!("iter_{}_reduce_labels", iter)));

        // 3. Broadcast the merged global labels back to all workers for the next iteration
        let global_labels = if let Some(msg) = reduced_labels {
            middleware.broadcast(Some(msg), ROOT_WORKER).unwrap()
        } else {
            middleware.broadcast(None, ROOT_WORKER).unwrap()
        };
        
        // Update the write buffer with the broadcasted labels
        write_buf.copy_from_slice(&global_labels.0);
        
        // Swap buffers for next iteration (no allocation, just pointer swap)
        use_a_as_read = !use_a_as_read;
        
        timestamps.push(timestamp(&format!("iter_{}_broadcast_labels", iter)));

        // 4. Check for convergence (global label change count)
        let reduced_changed = middleware
            .reduce(LabelsMessage(vec![local_changed]), |mut a, b| {
                a.0[0] = a.0[0].saturating_add(b.0[0]);
                a
            })
            .unwrap();

        // Root worker makes the stop/continue decision
        let should_stop = if worker == ROOT_WORKER {
            if let Some(count_msg) = reduced_changed {
                let total_changed = count_msg.0[0];
                println!("[Worker {worker}] iter {iter}: changed={total_changed}");
                !should_continue(iter, params.max_iterations, total_changed, threshold)
            } else { false }
        } else { false };

        // Synchronize the stop signal across the cluster
        let stop_signal = middleware.broadcast(
            if worker == ROOT_WORKER { Some(LabelsMessage(vec![if should_stop { 1 } else { 0 }])) } else { None },
            ROOT_WORKER
        ).unwrap();

        if stop_signal.0[0] == 1 { break; }
        iter += 1;
        timestamps.push(timestamp(&format!("iter_{}_end", iter)));
    }

    timestamps.push(timestamp("worker_end"));

    // Determine which buffer contains the final labels
    let final_labels = if use_a_as_read {
        &labels_a
    } else {
        &labels_b
    };

    // Worker 0 writes final labels to S3 for validation and generates results report
    let results_report = if worker == ROOT_WORKER {
        timestamps.push(timestamp("write_labels_start"));
        
        let labels_map: std::collections::HashMap<String, u32> = (0..params.num_nodes)
            .map(|i| (i.to_string(), final_labels[i as usize]))
            .collect();
        
        // Generate label distribution
        let mut label_counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for &label in final_labels.iter() {
            *label_counts.entry(label).or_insert(0) += 1;
        }
        
        let mut report = String::new();
        report.push_str("\n=== Label Propagation Results ===\n");
        report.push_str(&format!("Total nodes: {}\n", params.num_nodes));
        report.push_str(&format!("Total iterations: {}\n", iter));
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
        
        // Add sample of nodes
        report.push_str("\nSample nodes (first 20):\n");
        for i in 0..20.min(params.num_nodes as usize) {
            let label = final_labels[i];
            let label_str = if label == UNKNOWN { "UNKNOWN".to_string() } else { label.to_string() };
            report.push_str(&format!("  Node {}: Label {}\n", i, label_str));
        }
        report.push_str("=================================\n");
        
        println!("{}", report);
        
        let output_key = format!("{}/output/labels_final.json", params.input_data.key);

        if labels_map.len() < 10_000_000 {
            let labels_json = serde_json::json!({ "labels": labels_map });
            let labels_str = serde_json::to_string(&labels_json).unwrap();
            let write_result = rt.block_on(async {
                s3_client.put_object()
                    .bucket(&params.input_data.bucket)
                    .key(&output_key)
                    .body(labels_str.into_bytes().into())
                    .send()
                    .await
            });
            match write_result {
                Ok(_) => println!("[Worker {}] ✓ Wrote final labels to s3://{}/{}", worker, params.input_data.bucket, output_key),
                Err(e) => eprintln!("[Worker {}] ✗ Failed to write labels: {:?}", worker, e),
            }
        } else {
            println!("[Worker {}] ! Skipping large JSON serialization for S3 ({} nodes)", worker, labels_map.len());
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

/// OpenWhisk entrypoint
pub fn main(args: Value, burst_middleware: Middleware<LabelsMessage>) -> Result<Value, Error> {
    let input: Input = serde_json::from_value(args)?;
    
    // Validate partitions are evenly divisible by granularity
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

    // Helper para crear un grafo simple y ejecutar LP localmente (sin S3, sin middleware)
    fn run_lp_local(
        edges: Vec<(u32, u32, Option<u32>)>, // (src, dst, optional_label)
        num_nodes: u32,
        max_iter: u32,
    ) -> Vec<u32> {
        // Construir grafo CSR
        let mut owned_nodes = Vec::new();
        let mut offsets = vec![0u32; (num_nodes + 1) as usize];
        let mut flat_edges = Vec::new();
        
        // Agrupar edges por nodo origen
        let mut adj: HashMap<u32, Vec<u32>> = HashMap::default();
        let mut seeds: HashMap<u32, u32> = HashMap::default();
        
        for (src, dst, label) in edges {
            adj.entry(src).or_insert_with(Vec::new).push(dst);
            if let Some(l) = label {
                seeds.insert(src, l);
            }
        }
        
        // Convertir a CSR
        for node in 0..num_nodes {
            if let Some(neighbors) = adj.get(&node) {
                owned_nodes.push(node);
                offsets[node as usize] = flat_edges.len() as u32;
                flat_edges.extend_from_slice(neighbors);
            }
        }
        offsets[num_nodes as usize] = flat_edges.len() as u32;
        
        // Inicializar labels
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
        
        // Ejecutar iteraciones de LP
        for _ in 0..max_iter {
            let prev_labels = labels.clone();
            let mut changed = 0;
            
            for &node in &owned_nodes {
                if !unsupervised && seeds.contains_key(&node) {
                    continue; // Clamping
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
                
                // Majority vote con tie-breaking
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
