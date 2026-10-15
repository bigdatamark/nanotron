# [MODULE DOC]
# Purpose: minimal S3/local file helpers (adapted from datatrove) for streaming files and listing S3 folders.
# Main contents: _get_s3_path_components, _get_s3_object, _stream_file, _get_s3_file_list; BOTO3_AVAILABLE flag.
# Fits in nanotron: used by the tokenized bytes datasets to read shards from S3 or disk.
# Import-time side effects: imports datasets' xPath; tries to import boto3 and records the result in BOTO3_AVAILABLE.
import os
import re
from collections import deque
from re import Pattern
from typing import Union

from datasets.download.streaming_download_manager import xPath

try:
    import boto3

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


# mostly borrowed from datatrove: https://github.com/huggingface/datatrove/blob/main/src/datatrove/io/cloud/s3.py


def _get_s3_path_components(s3_path: Union[str, xPath]):
    # [FUNCTION DOC]
    # HELPER: splits "s3://bucket/key" into (bucket, key) after collapsing double slashes.
    # Inputs: s3_path (string or xPath). Outputs: tuple of two strings.
    # Exceptions: None explicitly raised. Side effects: None.
    # KNOWN ISSUE: partitions on the OS path separator rather than "/", which is wrong on Windows.
    s3_path = str(s3_path)
    bucket_name, _, prefix = s3_path[len("s3://") :].replace("//", "/").partition(os.sep)
    return bucket_name, prefix


def _get_s3_object(s3_path: Union[str, xPath]):
    # [FUNCTION DOC]
    # HELPER: returns a boto3 S3 Object resource for the path.
    # Inputs: s3_path. Outputs: boto3 Object.
    # Exceptions: NameError if boto3 is not installed. Side effects: creates a boto3 resource (may read credentials).
    s3_path = str(s3_path)
    bucket_name, prefix = _get_s3_path_components(s3_path)
    s3_resource = boto3.resource("s3")
    s3_object = s3_resource.Object(bucket_name=bucket_name, key=prefix)
    return s3_object


def _stream_file(file_path: Union[str, xPath], chunk_size, offset):
    # [FUNCTION DOC]
    # HELPER: generator yielding byte chunks of chunk_size from a file, starting at byte chunk_size times offset.
    # Behavior: S3 paths use a ranged GET and iterate the body in chunks; local paths seek and read until EOF.
    # Inputs: file_path (string or xPath), chunk_size (int bytes), offset (int, in chunks).
    # Outputs: yields bytes objects.
    # Exceptions: OSError / boto errors propagate.
    # Side effects: network or file I/O.
    file_path = str(file_path)
    if file_path.startswith("s3://"):
        s3_object = _get_s3_object(file_path)
        yield from s3_object.get(Range=f"bytes={chunk_size * offset}-")["Body"].iter_chunks(chunk_size)
    else:
        with open(file_path, "rb") as f:
            f.seek(chunk_size * offset)
            while True:
                chunk = f.read(chunk_size)  # Read a chunk of the specified size
                if not chunk:
                    break  # If the chunk is empty, end of file is reached
                yield chunk


def _get_s3_file_list(
    s3_path: Union[str, xPath], pattern: Union[str, Pattern] = None, recursive: bool = True, max_recursion: int = -1
):
    # [FUNCTION DOC]
    # HELPER: lists files under an S3 prefix, optionally filtered by a regex and recursing breadth-first.
    # Behavior: paginate list_objects_v2 with delimiter "/" per prefix; when recursive and below max_recursion, queue
    # common sub-prefixes; skip the prefix object itself; if a pattern is given keep only keys whose path relative to the
    # main prefix fully matches it; collect full "s3://bucket/key" URIs.
    # Inputs: s3_path, pattern (string or compiled regex, optional), recursive (default True), max_recursion (default -1,
    # unlimited).
    # Outputs: sorted list of S3 URIs.
    # Exceptions: boto errors propagate; NameError without boto3.
    # Side effects: network calls to S3.
    """Get list of relative paths to files in a cloud folder with a given (optional) pattern

    Args:
        s3_path: path to the cloud folder (e.g. s3://bucket/prefix)
        pattern: optional pattern to filter files (str or re.Pattern)
        recursive: whether to recursively search for files (default: True)
        max_recursion: how many levels to recursively search for files (-1 means no limit)
    """
    s3_path = str(s3_path)

    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    s3_client = boto3.client("s3")
    bucket, main_prefix = _get_s3_path_components(s3_path)

    paginator = s3_client.get_paginator("list_objects_v2")
    objects = []
    prefixes = deque()

    prefixes.append((0, main_prefix))
    while prefixes:
        level, prefix = prefixes.popleft()
        for resp in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            if recursive and (max_recursion == -1 or level < max_recursion):
                prefixes.extend([(level + 1, next_prefix["Prefix"]) for next_prefix in resp.get("CommonPrefixes", [])])
            filtered_objects = [x for x in resp.get("Contents", []) if x["Key"] != prefix]
            if pattern is not None:
                filtered_objects = [
                    x for x in filtered_objects if pattern.fullmatch(os.path.relpath(x["Key"], main_prefix))
                ]
            objects.extend([f"s3://{bucket}/{x['Key']}" for x in filtered_objects])
    return sorted(objects)
