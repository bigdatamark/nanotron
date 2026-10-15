# [MODULE DOC]
# Purpose: command-line tool that tokenizes a text dataset (HuggingFace dataset or jsonl files) into nanotron-compatible tokenized files using datatrove.
# Main contents: get_args (CLI parsing) and main (builds and runs a datatrove LocalPipelineExecutor). Usage examples are in the module docstring below.
# Fits in nanotron: produces pre-tokenized data consumed by nanotron's datatrove/nanoset data loaders.
# Import-time side effects: None; runs main only when executed as a script.
"""
To process HuggingFace Datasets:
    python3 tools/preprocess_data.py --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B --output-folder datasets/emotion --n-tasks 16 hf --dataset dair-ai/emotion
To process Jsonl files:
    python3 tools/preprocess_data.py --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B --output-folder datasets/c4-es --n-tasks 16 jsonl --dataset raw_datasets/c4-es-json-files
"""

import argparse

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.tokens import DocumentTokenizer


def get_args():
    # [FUNCTION DOC]
    # Summary: Builds and parses the CLI. Common options: --tokenizer-name-or-path (required), --eos-token (default None), --output-folder (required), --logging-dir (default None), --n-tasks (default 8). A required subcommand selects the reader: "hf" with --dataset (required), --column (default text), --split (default train); or "jsonl" with --dataset (required), --column (default text), --glob-pattern (default None). The chosen subcommand is stored in the "readers" attribute.
    # Inputs: None (reads sys.argv).
    # Outputs: argparse Namespace.
    # Exceptions: SystemExit on invalid arguments or help.
    # Side effects: May print usage/help to stdout/stderr.
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="Tokenizer")
    group.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )
    group.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="EOS token to add after each document. Default: None",
    )

    group = parser.add_argument_group(title="Output data")
    group.add_argument(
        "--output-folder", type=str, required=True, help="Path to the output folder to store the tokenized documents"
    )
    group = parser.add_argument_group(title="Miscellaneous configs")
    group.add_argument(
        "--logging-dir",
        type=str,
        default=None,
        help="Path to a folder for storing the logs of the preprocessing step. Default: None",
    )
    group.add_argument(
        "--n-tasks", type=int, default=8, help="Total number of tasks to run the preprocessing step. Default: 8"
    )
    # Subparsers for processing either Hugging Face datasets or jsonl files
    sp = parser.add_subparsers(
        dest="readers",
        required=True,
        description="Type of dataset to process. It can be either a Hugging Face Dataset loaded with datasets.load_data ('hf') or a .jsonl dataset ('jsonl')",
    )

    p1 = sp.add_parser(name="hf")
    p1.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to local stored dataset or repository on the Hugging Face hub that can be loaded with datasets.load_dataset",
    )
    p1.add_argument("--column", type=str, default="text", help="Column to preprocess from the Dataset. Default: text")
    p1.add_argument("--split", type=str, default="train", help="Which split of the data to process. Default: train")

    p2 = sp.add_parser(name="jsonl")
    p2.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a .jsonl file or a folder containing multiple .jsonl files",
    )
    p2.add_argument("--column", type=str, default="text", help="Column to preprocess from the Dataset. Default: text")
    p2.add_argument(
        "--glob-pattern", type=str, default=None, help="A glob pattern to filter files to read. Default: None"
    )

    args = parser.parse_args()

    return args


def main(args):
    # [FUNCTION DOC]
    # Summary: Creates a HuggingFaceDatasetReader (with the split as dataset option) when readers is "hf", otherwise a JsonlReader with the glob pattern; then runs a LocalPipelineExecutor with that reader followed by a DocumentTokenizer (output folder, tokenizer, eos token, no shuffling, max 1e9 tokens per file) over n_tasks tasks with the given logging dir.
    # Inputs: args - Namespace from get_args.
    # Outputs: None.
    # Exceptions: None explicitly raised; datatrove/tokenizer errors propagate.
    # Side effects: Reads the dataset (possibly downloading), writes tokenized files and logs to disk, may spawn worker processes.
    # Build datatrove reader
    if args.readers == "hf":
        datatrove_reader = HuggingFaceDatasetReader(
            dataset=args.dataset,
            text_key=args.column,
            dataset_options={"split": args.split},
        )
    else:
        datatrove_reader = JsonlReader(data_folder=args.dataset, text_key=args.column, glob_pattern=args.glob_pattern)

    preprocess_executor = LocalPipelineExecutor(
        pipeline=[
            datatrove_reader,
            DocumentTokenizer(
                output_folder=args.output_folder,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                eos_token=args.eos_token,
                shuffle=False,
                max_tokens_per_file=1e9,
            ),
        ],
        tasks=args.n_tasks,
        logging_dir=args.logging_dir,
    )
    preprocess_executor.run()


if __name__ == "__main__":
    _args = get_args()
    main(_args)
