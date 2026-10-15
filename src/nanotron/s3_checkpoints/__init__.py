# [PACKAGE DOC]
# Purpose: Package for moving checkpoints between local disk and remote storage (S3 or any fsspec filesystem).
# Main contents / exports: S3Mover (background s5cmd upload/download/removal), fs_open, fs_copy,
# check_path_is_local (fsspec helpers).
# Fits in nanotron: the trainer uses S3Mover for checkpoint upload when an s3_upload config is set;
# serialization code uses the fsspec helpers to read/write checkpoints on arbitrary filesystems.
# Import-time side effects: imports submodules only.
from .fsspec import check_path_is_local, fs_copy, fs_open
from .s3_mover import S3Mover

__all__ = ["S3Mover", "fs_open", "fs_copy", "check_path_is_local"]
