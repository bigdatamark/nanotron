# [MODULE DOC]
# Purpose: One-off maintenance script that renames checkpoint safetensors files whose names had the type/suffix
# part duplicated (for example model_model_weight.safetensors_pp-rank...safetensors becomes
# model_weight_pp-rank...safetensors).
# Main contents: update_checkpoint (walk and rename), main (CLI). No import-time side effects.
"""Fixes the problem where '{type.value}_{suffix_name}.safetensors' was duplicated in checkpoint files

For example this script will change the following:
```
checkpoints/10/model/model/decoder/0/pp_block/attn/o_proj/model_model_weight.safetensors_pp-rank-0-of-1_tp-rank-0-of-2.safetensors
to
checkpoints/10/model/model/decoder/0/pp_block/attn/o_proj/model_weight_pp-rank-0-of-1_tp-rank-0-of-2.safetensors
```

Example Usage:

python scripts/fix_checkpoint_bad_naming.py /fsx/nouamane/projects/nanotron/checkpoints/10
"""

import argparse
import os
import re
from pathlib import Path


def update_checkpoint(checkpoint_dir: str):
    # [FUNCTION DOC]
    # Behavior: recursively walks checkpoint_dir; for each .safetensors file containing "_model" directly preceded by
    # "model", removes those occurrences, asserts the result contains the safetensors extension exactly twice,
    # removes the first occurrence, and renames the file in place.
    # Inputs: checkpoint_dir (str or Path, checkpoint root).
    # Outputs: None.
    # Exceptions: AssertionError when a matching file does not contain exactly two safetensors extensions; OSError on
    # rename failure.
    # Side effects: renames files on disk; prints progress.
    # KNOWN ISSUE: the extension pattern uses an unescaped dot, so it matches any character before "safetensors".
    print(f"Updating checkpoint in {checkpoint_dir}")
    for root, _, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith(".safetensors"):
                # r'(?<=model)_(model)' means match the string '_model' that is preceded by 'model'
                if len(re.findall(r"(?<=model)_(model)", file)) == 0:
                    continue
                # we remove second _model
                new_file = re.sub(r"(?<=model)_(model)", "", file)
                # we would have "model_weight.safetensors_pp-rank-0-of-1_tp-rank-0-of-2.safetensors"

                # let's assert we have two matches of ".safetensors"
                assert len(re.findall(r".safetensors", new_file)) == 2
                # then we remove first match
                new_file = re.sub(r".safetensors", "", new_file, count=1)
                # so that we get "model_weight_pp-rank-0-of-1_tp-rank-0-of-2.safetensors"

                print(f"Renaming {file} to {new_file}")
                os.rename(os.path.join(root, file), os.path.join(root, new_file))


def main():
    # [FUNCTION DOC]
    # Behavior: parses a positional checkpoint_dir path and calls update_checkpoint.
    # Inputs: None (sys.argv). Outputs: None. Exceptions: SystemExit on bad args. Side effects: file renames.
    # KNOWN ISSUE: the parser description says "Update checkpoint from 1.3 to 1.4", which does not describe the fix.
    parser = argparse.ArgumentParser(description="Update checkpoint from 1.3 to 1.4")
    parser.add_argument("checkpoint_dir", type=Path, help="Path to the checkpoint directory")
    args = parser.parse_args()
    update_checkpoint(args.checkpoint_dir)


if __name__ == "__main__":
    main()
