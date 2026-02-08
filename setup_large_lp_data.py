#!/usr/bin/env python3
"""
Generate large graph datasets for Label Propagation benchmarking.
Creates both local .txt file for standalone version and S3 partitions for burst version.
"""
import argparse
import io
import boto3
from botocore.client import Config

def generate_large_graph(num_nodes, num_partitions, output_local, bucket=None, s3_prefix=None, 
                        endpoint="localhost:9000", access_key="minioadmin", secret_key="minioadmin",
                        density=10):
    """
    Generate a ring graph with seeds for Label Propagation.
    Each node connects to multiple neighbors (controlled by density).
    10% of nodes are seeded with labels based on their position.
    
    Args:
        density: Number of neighbors each node connects to (higher = denser graph)
    """
    print(f"Generating graph: {num_nodes} nodes, density={density}...")
    
    # Generate edges - each node connects to 'density' neighbors
    edges = []
    for i in range(num_nodes):
        src = i
        for offset in range(1, density + 1):
            dst = (i + offset) % num_nodes
            
            # Add label for 10% of nodes (deterministic seeds)
            if i % 10 == 0 and offset == 1:
                label = (i // (num_nodes // 4)) * 100  # 4 label groups
                edges.append(f"{src}\t{dst}\t{label}")
            else:
                edges.append(f"{src}\t{dst}")
    
    # Write local file for standalone version
    print(f"Writing local file: {output_local}")
    with open(output_local, 'w') as f:
        f.write('\n'.join(edges))
    
    # Upload to S3 if requested
    if bucket and s3_prefix:
        print(f"Uploading to S3: {bucket}/{s3_prefix}/ ({num_partitions} partitions)")
        s3 = boto3.client(
            's3',
            endpoint_url=f"http://{endpoint}" if not endpoint.startswith("http") else endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        
        # Ensure bucket exists
        try:
            s3.create_bucket(Bucket=bucket)
        except:
            pass
        
        # Partition edges by source node modulo
        partitions = [[] for _ in range(num_partitions)]
        for edge in edges:
            src_node = int(edge.split('\t')[0])
            part_idx = src_node % num_partitions
            partitions[part_idx].append(edge)
        
        # Upload each partition
        for i, part_edges in enumerate(partitions):
            if part_edges:
                data = '\n'.join(part_edges).encode('utf-8')
                object_name = f"{s3_prefix}/part-{str(i).zfill(5)}"
                s3.put_object(
                    Bucket=bucket,
                    Key=object_name,
                    Body=data,
                    ContentType="text/plain"
                )
                print(f"  ✅ Partition {i}: {len(part_edges)} edges")
    
    print(f"✅ Graph generation complete: {num_nodes} nodes, {len(edges)} edges")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate large LP graph datasets")
    parser.add_argument("--nodes", type=int, required=True, help="Number of nodes")
    parser.add_argument("--partitions", type=int, default=8, help="Number of S3 partitions")
    parser.add_argument("--output", type=str, default=None, help="Local output file")
    parser.add_argument("--bucket", type=str, default="test-bucket", help="S3 bucket")
    parser.add_argument("--prefix", type=str, default=None, help="S3 key prefix")
    parser.add_argument("--endpoint", type=str, default="localhost:9000", help="S3 endpoint")
    parser.add_argument("--no-s3", action="store_true", help="Skip S3 upload")
    parser.add_argument("--density", type=int, default=10, help="Number of neighbors per node (graph density)")
    
    args = parser.parse_args()
    
    # Default file naming
    if args.output is None:
        args.output = f"large_{args.nodes}.txt"
    if args.prefix is None:
        args.prefix = f"graphs/large-{args.nodes}"
    
    generate_large_graph(
        num_nodes=args.nodes,
        num_partitions=args.partitions,
        output_local=args.output,
        bucket=None if args.no_s3 else args.bucket,
        s3_prefix=None if args.no_s3 else args.prefix,
        endpoint=args.endpoint,
        density=args.density
    )
