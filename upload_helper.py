import sys
import boto3
from botocore.client import Config

def upload_partition(nodes, partitions, endpoint="localhost:33609"):
    bucket = "test-bucket"
    s3_prefix = f"graphs/large-{nodes}"
    local_file = f"large_{nodes}.txt"
    
    s3 = boto3.client(
        's3',
        endpoint_url=f"http://{endpoint}",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    
    print(f"Reading {local_file} for {nodes} nodes...")
    parts_data = [[] for _ in range(partitions)]
    
    # Ensure bucket exists
    try:
        s3.create_bucket(Bucket=bucket)
    except Exception:
        pass

    with open(local_file, 'r') as f:
        for line in f:
            if not line.strip(): continue
            src_node = int(line.split('\t')[0])
            part_idx = src_node % partitions
            parts_data[part_idx].append(line.strip())
            
    for i in range(partitions):
        if parts_data[i]:
            data = '\n'.join(parts_data[i]).encode('utf-8')
            key = f"{s3_prefix}/part-{str(i).zfill(5)}"
            print(f"Uploading {key} ({len(parts_data[i])} edges)...")
            s3.put_object(Bucket=bucket, Key=key, Body=data)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python upload_helper.py <nodes> <partitions> [endpoint]")
        sys.exit(1)
    
    nodes = int(sys.argv[1])
    parts = int(sys.argv[2])
    endpoint = sys.argv[3] if len(sys.argv) > 3 else "localhost:41511"
    
    upload_partition(nodes, parts, endpoint)
