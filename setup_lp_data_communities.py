#!/usr/bin/env python3
"""
Generate a graph with community structure for Label Propagation testing.
Each community has dense internal connections and sparse connections to other communities.
"""
import argparse
import io
import boto3
from botocore.client import Config
import random

def generate_community_graph(num_nodes, num_communities, num_partitions, intra_prob=0.3, inter_prob=0.01, seed_ratio=0.05):
    """
    Generate a graph with community structure.
    
    Args:
        num_nodes: Total number of nodes
        num_communities: Number of communities (also used as number of labels)
        num_partitions: Number of partitions for distributed processing
        intra_prob: Probability of edge within community (high)
        inter_prob: Probability of edge between communities (low)
        seed_ratio: Fraction of nodes to label initially
    
    Returns:
        Dict mapping partition_id -> list of edge lines
    """
    nodes_per_community = num_nodes // num_communities
    partitions = {i: [] for i in range(num_partitions)}
    
    print(f"Creating {num_communities} communities with ~{nodes_per_community} nodes each")
    
    # Assign each node to a community
    node_to_community = {}
    for node in range(num_nodes):
        community = node // nodes_per_community
        if community >= num_communities:
            community = num_communities - 1
        node_to_community[node] = community
    
    # Generate edges
    edge_count = 0
    seed_nodes = set()
    
    for src in range(num_nodes):
        src_community = node_to_community[src]
        part_idx = src % num_partitions
        
        # Decide if this node should be a seed (initially labeled)
        if random.random() < seed_ratio:
            seed_nodes.add(src)
        
        # Connect to other nodes
        for dst in range(src + 1, num_nodes):
            dst_community = node_to_community[dst]
            
            # Higher probability for intra-community edges
            if src_community == dst_community:
                prob = intra_prob
            else:
                prob = inter_prob
            
            if random.random() < prob:
                # Create undirected edge (add both directions)
                line_src = f"{src}\t{dst}"
                if src in seed_nodes:
                    line_src += f"\t{src_community * 100}"  # Use community as label
                partitions[part_idx].append(line_src)
                
                line_dst = f"{dst}\t{src}"
                if dst in seed_nodes:
                    line_dst += f"\t{dst_community * 100}"
                dst_part_idx = dst % num_partitions
                partitions[dst_part_idx].append(line_dst)
                
                edge_count += 1
    
    print(f"Generated {edge_count} edges ({edge_count * 2} directed edges)")
    print(f"Labeled {len(seed_nodes)} seed nodes ({len(seed_nodes)/num_nodes*100:.1f}%)")
    
    return partitions, seed_nodes, node_to_community

def upload_to_s3(s3, bucket, key_prefix, partitions):
    """Upload graph partitions to S3"""
    for part_idx, edges in partitions.items():
        if not edges:
            continue
            
        data = '\n'.join(edges) + '\n'
        object_name = f"{key_prefix}/part-{str(part_idx).zfill(5)}"
        
        s3.put_object(
            Bucket=bucket,
            Key=object_name,
            Body=data.encode('utf-8'),
            ContentType="text/plain"
        )
        print(f"✅ Uploaded partition {part_idx} to {bucket}/{object_name} ({len(data)} bytes, {len(edges)} edges)")

def main():
    parser = argparse.ArgumentParser(description="Setup Label Propagation data with community structure")
    parser.add_argument("--endpoint", default="minio-service.default:9000", help="S3 endpoint")
    parser.add_argument("--access-key", default="minioadmin")
    parser.add_argument("--secret-key", default="minioadmin")
    parser.add_argument("--bucket", default="test-bucket")
    parser.add_argument("--prefix", default="graphs/cluster-test")
    parser.add_argument("--nodes", type=int, default=10000, help="Total number of nodes")
    parser.add_argument("--communities", type=int, default=4, help="Number of communities")
    parser.add_argument("--partitions", type=int, default=8, help="Number of partitions for workers")
    parser.add_argument("--intra-prob", type=float, default=0.3, help="Probability of intra-community edge")
    parser.add_argument("--inter-prob", type=float, default=0.01, help="Probability of inter-community edge")
    parser.add_argument("--seed-ratio", type=float, default=0.05, help="Fraction of nodes to label initially")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Setup S3 client
    endpoint_url = f"http://{args.endpoint}" if not args.endpoint.startswith("http") else args.endpoint
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_key,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    
    # Create bucket if needed
    try:
        s3.create_bucket(Bucket=args.bucket)
        print(f"✅ Created bucket: {args.bucket}")
    except Exception as e:
        print(f"ℹ️  Bucket {args.bucket} already exists")
    
    # Generate graph
    print("\nGenerating community graph...")
    partitions, seed_nodes, node_to_community = generate_community_graph(
        args.nodes, 
        args.communities, 
        args.partitions,
        args.intra_prob,
        args.inter_prob,
        args.seed_ratio
    )
    
    # Upload to S3
    print("\nUploading to S3...")
    upload_to_s3(s3, args.bucket, args.prefix, partitions)
    
    # Print summary
    print("\n" + "="*50)
    print("Graph Summary:")
    print(f"  Total nodes: {args.nodes}")
    print(f"  Communities: {args.communities}")
    print(f"  Seed nodes: {len(seed_nodes)} ({len(seed_nodes)/args.nodes*100:.1f}%)")
    print(f"  Partitions: {args.partitions}")
    print(f"  S3 location: s3://{args.bucket}/{args.prefix}/")
    print("="*50)
    
    # Show community distribution
    print("\nCommunity distribution (first 20 nodes):")
    for node in range(min(20, args.nodes)):
        community = node_to_community[node]
        label = f"Label {community * 100}" if node in seed_nodes else "Unlabeled"
        print(f"  Node {node}: Community {community}, {label}")

if __name__ == "__main__":
    main()
