//! # Label Propagation Demo
//!
//! This is a simple demonstration of the label propagation algorithm.
//! It creates a small example graph and runs the algorithm to show basic functionality.

use label_propagation::run_lp;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <graph_file> <num_nodes> <max_iter>", args[0]);
        std::process::exit(1);
    }

    let graph_file = &args[1];
    let num_nodes: u32 = args[2].parse().expect("Invalid num_nodes");
    let max_iter: u32 = args[3].parse().expect("Invalid max_iter");

    let start_load = Instant::now();
    let (adj, initial_labels) = load_graph(graph_file);
    let load_duration = start_load.elapsed();

    let start_lp = Instant::now();
    let labels = run_lp(&adj, &initial_labels, num_nodes, max_iter);
    let lp_duration = start_lp.elapsed();

    // Output results in JSON format for the benchmark script
    let result = serde_json::json!({
        "load_time_ms": load_duration.as_millis(),
        "execution_time_ms": lp_duration.as_millis(),
        "total_time_ms": (load_duration + lp_duration).as_millis(),
        "labels": labels
    });

    println!("{}", result.to_string());
}

fn load_graph(path: &str) -> (HashMap<u32, Vec<u32>>, HashMap<u32, u32>) {
    let file = File::open(path).expect("Could not open graph file");
    let reader = BufReader::new(file);

    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut initial_labels: HashMap<u32, u32> = HashMap::new();

    for line in reader.lines() {
        let line = line.expect("Could not read line");
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 2 {
            continue;
        }

        let src: u32 = parts[0].parse().expect("Invalid src node");
        let dst: u32 = parts[1].parse().expect("Invalid dst node");

        adj.entry(src).or_default().push(dst);
        
        if parts.len() >= 3 {
            let label: u32 = parts[2].parse().expect("Invalid label");
            initial_labels.insert(src, label);
        }
    }

    (adj, initial_labels)
}
