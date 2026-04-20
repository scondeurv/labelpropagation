#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import boto3

ROOT = Path("/home/sergio/src")
sys.path.insert(0, str(ROOT))

from baselines.run_spark_graph_benchmarks import validate_lp_spark_output


def s3_client(endpoint: str, access_key: str, secret_key: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def upload_input(args: argparse.Namespace) -> None:
    client = s3_client(args.endpoint, args.access_key, args.secret_key)
    client.upload_file(str(args.input_file), args.bucket, args.key)
    print(
        json.dumps(
            {
                "uploaded": True,
                "bucket": args.bucket,
                "key": args.key,
                "input_file": str(args.input_file),
            }
        )
    )


def delete_prefix(args: argparse.Namespace) -> None:
    client = s3_client(args.endpoint, args.access_key, args.secret_key)
    paginator = client.get_paginator("list_objects_v2")
    deleted = 0
    for page in paginator.paginate(Bucket=args.bucket, Prefix=args.prefix):
        contents = page.get("Contents", [])
        if not contents:
            continue
        for item in contents:
            client.delete_object(Bucket=args.bucket, Key=item["Key"])
            deleted += 1
    print(json.dumps({"deleted": deleted, "bucket": args.bucket, "prefix": args.prefix}))


def download_prefix(client, bucket: str, prefix: str, output_dir: Path) -> int:
    paginator = client.get_paginator("list_objects_v2")
    downloaded = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        for item in contents:
            key = item["Key"]
            relative = key[len(prefix):].lstrip("/")
            if not relative:
                continue
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            downloaded += 1
    return downloaded


def validate_output(args: argparse.Namespace) -> None:
    client = s3_client(args.endpoint, args.access_key, args.secret_key)
    temp_dir = Path(tempfile.mkdtemp(prefix="cloudlab-lp-spark-"))
    try:
        downloaded = download_prefix(client, args.bucket, args.prefix, temp_dir)
        if downloaded == 0:
            raise RuntimeError(
                f"no Spark output objects found under s3://{args.bucket}/{args.prefix}"
            )
        summary = validate_lp_spark_output(
            graph_file=args.graph_file,
            num_nodes=args.num_nodes,
            max_iter=args.max_iter,
            spark_output_dir=temp_dir,
        )
        summary["downloaded_objects"] = downloaded
        summary["bucket"] = args.bucket
        summary["prefix"] = args.prefix
        print(json.dumps(summary))
        if not summary.get("passed", False):
            raise SystemExit(1)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def validate_local_output(args: argparse.Namespace) -> None:
    summary = validate_lp_spark_output(
        graph_file=args.graph_file,
        num_nodes=args.num_nodes,
        max_iter=args.max_iter,
        spark_output_dir=args.local_output_dir,
    )
    print(json.dumps(summary))
    if not summary.get("passed", False):
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CloudLab LP Spark smoke test helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--endpoint", required=True)
    common.add_argument("--access-key", required=True)
    common.add_argument("--secret-key", required=True)
    common.add_argument("--bucket", required=True)

    upload = subparsers.add_parser("upload-input", parents=[common])
    upload.add_argument("--input-file", type=Path, required=True)
    upload.add_argument("--key", required=True)
    upload.set_defaults(func=upload_input)

    delete = subparsers.add_parser("delete-prefix", parents=[common])
    delete.add_argument("--prefix", required=True)
    delete.set_defaults(func=delete_prefix)

    validate = subparsers.add_parser("validate-output", parents=[common])
    validate.add_argument("--prefix", required=True)
    validate.add_argument("--graph-file", type=Path, required=True)
    validate.add_argument("--num-nodes", type=int, required=True)
    validate.add_argument("--max-iter", type=int, required=True)
    validate.set_defaults(func=validate_output)

    validate_local = subparsers.add_parser("validate-local-output")
    validate_local.add_argument("--local-output-dir", type=Path, required=True)
    validate_local.add_argument("--graph-file", type=Path, required=True)
    validate_local.add_argument("--num-nodes", type=int, required=True)
    validate_local.add_argument("--max-iter", type=int, required=True)
    validate_local.set_defaults(func=validate_local_output)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
