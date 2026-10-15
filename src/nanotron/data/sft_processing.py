# [MODULE DOC]
# Purpose: supervised fine-tuning (SFT) preprocessing: turns prompt/completion pairs into padded token sequences with
# position ids and a loss mask restricted to completion tokens.
# Main contents: process_sft, prepare_sft_dataset, pack_sft_sequences (placeholder).
# Fits in nanotron: exported by nanotron.data for SFT training runs.
# Import-time side effects: None beyond importing torch.
import torch


def process_sft(examples, tokenizer, trainer_sequence_length):
    # [FUNCTION DOC]
    # Summary: batched map function building SFT model inputs.
    # Behavior: concatenate each prompt, completion and the tokenizer EOS token; tokenise with padding to the longest
    # and truncation to trainer_sequence_length + 1 tokens as torch tensors. Tokenise the prompts alone to get their
    # lengths. For each row, position_ids are 0 to the number of attended tokens minus one, -1 on padding; label_mask is
    # True from the prompt length up to the valid length (False everywhere if the prompt fills the sequence). label_ids
    # is a copy of input_ids; attention_mask is converted to bool. Prints a warning counting rows with no loss tokens.
    # Inputs: examples (dictionary with "prompt" and "completion" lists of strings), tokenizer (Hugging Face tokenizer with
    # an EOS token and a pad token), trainer_sequence_length (int).
    # Outputs: the tokenizer BatchEncoding with input_ids, attention_mask, label_ids, position_ids, label_mask tensors of
    # shape batch by padded length.
    # Exceptions: None explicitly raised.
    # Side effects: prints to stdout.
    # KNOWN ISSUE: prompts and completions are tokenised twice at the start and the results discarded (wasted work).
    # KNOWN ISSUE: prompt length assumes left-aligned tokens and that the prompt tokenises identically alone and in
    # context; with left padding or added special tokens the label mask is misaligned.
    """
    Process examples for supervised fine-tuning by:
    1. Tokenizing prompts and completions separately
    2. Combining them into full examples with EOS token
    3. Creating position_ids for each token in the sequence
    4. Creating label_mask that only enables loss on completion tokens

    Args:
        examples: Dictionary with 'prompt' and 'completion' fields
        tokenizer: HuggingFace tokenizer
        trainer_sequence_length: Maximum sequence length for training

    Returns:
        Dictionary with processed tokens including:
        - input_ids: Combined tokenized sequence
        - position_ids: Sequential position IDs for each token
        - label_mask: Boolean mask with True only for completion tokens
        - attention_mask: Attention mask for padding
        - label_ids: Same as input_ids, used for loss calculation
    """
    # First tokenize prompts and completions separately to get lengths
    tokenizer(examples["prompt"], padding=False, truncation=False, return_tensors=None)

    tokenizer(examples["completion"], padding=False, truncation=False, return_tensors=None)

    # Combine prompt and completion with EOS token
    texts = [
        f"{prompt}{completion}{tokenizer.eos_token}"
        for prompt, completion in zip(examples["prompt"], examples["completion"])
    ]

    # Use trainer_sequence_length + 1 to match collator's expectation
    max_length = trainer_sequence_length + 1

    # Tokenize combined text
    tokenized = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    # Get filtered prompt tokens for length calculation
    filtered_prompt_tokens = tokenizer(examples["prompt"], padding=False, truncation=False, return_tensors=None)

    # Get attention mask and convert to bool
    attention_mask = tokenized["attention_mask"].bool()
    batch_size, seq_length = attention_mask.shape

    # Create sequential position_ids initialized to -1 (for padding)
    position_ids = torch.full((batch_size, seq_length), fill_value=-1, dtype=torch.long)

    # Create label_mask (initialize to False)
    label_mask = torch.zeros((batch_size, seq_length), dtype=torch.bool)

    # For each sequence in the batch
    for i in range(batch_size):
        # Get the actual prompt length, but ensure we don't exceed sequence length
        prompt_length = min(len(filtered_prompt_tokens["input_ids"][i]), seq_length)

        # Set position ids for all tokens (prompt and completion) to sequential values
        # But only where attention_mask is True (non-padding tokens)
        valid_length = attention_mask[i].sum().item()
        position_ids[i, :valid_length] = torch.arange(valid_length)

        # Set label_mask to True only for completion tokens
        # If prompt consumes the entire sequence, no tokens are used for loss
        if prompt_length < seq_length:
            # Set completion tokens label mask to True (rest remains False)
            label_mask[i, prompt_length:valid_length] = True

    # Create label_ids (same as input_ids)
    tokenized["label_ids"] = tokenized["input_ids"].clone()

    # Add the created tensors
    tokenized["position_ids"] = position_ids
    tokenized["label_mask"] = label_mask

    # Keep attention_mask for model's use
    tokenized["attention_mask"] = attention_mask

    # Log examples where prompt consumes all tokens
    too_long_prompts = sum(1 for i in range(batch_size) if not label_mask[i].any())
    if too_long_prompts > 0:
        print(
            f"Warning: {too_long_prompts}/{batch_size} examples have prompts that are too long, no completion tokens"
        )

    return tokenized


def prepare_sft_dataset(raw_dataset, tokenizer, trainer_sequence_length, debug_max_samples=None, num_proc=1):
    # [FUNCTION DOC]
    # Summary: maps process_sft over a dataset and filters unusable rows.
    # Behavior: optionally truncate to debug_max_samples rows (printing a notice); batched map with process_fn, removing
    # the original columns, using num_proc processes; filter rows lacking any non-negative position id or any True label
    # mask; print how many rows were removed.
    # Inputs: raw_dataset (datasets.Dataset with prompt/completion), tokenizer, trainer_sequence_length, debug_max_samples
    # (optional int), num_proc (default 1).
    # Outputs: processed datasets.Dataset.
    # Exceptions: None explicitly raised.
    # Side effects: prints; datasets cache I/O; spawns processes.
    """
    Prepare a dataset for supervised fine-tuning by processing the examples
    and filtering invalid samples.

    Args:
        raw_dataset: Dataset containing 'prompt' and 'completion' fields
        tokenizer: HuggingFace tokenizer
        trainer_sequence_length: Maximum sequence length for training
        debug_max_samples: If set, limit the dataset to this many samples
        num_proc: Number of processes for parallelization

    Returns:
        Processed dataset ready for training
    """
    # If in debug mode, limit the dataset size before processing
    if debug_max_samples is not None:
        print(f"DEBUG MODE: Limiting dataset to {debug_max_samples} samples")
        raw_dataset = raw_dataset.select(range(min(debug_max_samples, len(raw_dataset))))

    # Create a wrapper function that handles empty examples correctly
    def process_fn(examples):
        # [FUNCTION DOC]
        # HELPER: map wrapper returning empty columns for an empty batch, else process_sft output.
        # Inputs: examples (batch dictionary). Outputs: dictionary of columns.
        # Exceptions: None explicitly raised. Side effects: as process_sft.
        # Check if there are any examples to process
        if len(examples["prompt"]) == 0:
            return {k: [] for k in ["input_ids", "position_ids", "label_ids", "label_mask", "attention_mask"]}

        # Process the examples
        result = process_sft(examples, tokenizer, trainer_sequence_length)
        return result

    # Apply the map function to process the dataset
    train_dataset = raw_dataset.map(
        process_fn, batched=True, remove_columns=raw_dataset.column_names, num_proc=num_proc
    )

    # Filter out examples where:
    # 1. All position_ids are -1 (completely padding)
    # 2. All label_mask values are False (no tokens contribute to loss)
    def is_valid_sample(example):
        # [FUNCTION DOC]
        # HELPER: filter predicate; True when the row has at least one non-negative position id and at least one True label
        # mask value.
        # Inputs: example (row dictionary). Outputs: bool. Exceptions: None explicitly raised. Side effects: None.
        # Check if there's at least one non-padding token
        has_content = any(pos_id >= 0 for pos_id in example["position_ids"])

        # Check if there's at least one token that contributes to loss
        has_label = any(mask for mask in example["label_mask"])

        # Sample is valid if it has content AND at least one token for loss
        return has_content and has_label

    # Apply the filter
    original_size = len(train_dataset)
    train_dataset = train_dataset.filter(is_valid_sample)
    filtered_size = len(train_dataset)

    # Log how many samples were filtered out
    if original_size > filtered_size:
        print(
            f"Filtered out {original_size - filtered_size} samples ({(original_size - filtered_size) / original_size:.2%}) with no valid tokens for loss calculation"
        )

    return train_dataset


# Optional: Additional functions for sequence packing can be added here
def pack_sft_sequences(tokenized_examples, max_length, tokenizer):
    # [FUNCTION DOC]
    # Summary: placeholder for packing several SFT examples into one sequence; not implemented.
    # Inputs: tokenized_examples, max_length, tokenizer (unused).
    # Outputs: never returns.
    # Exceptions: always raises NotImplementedError.
    # Side effects: None.
    """
    Pack multiple SFT examples into a single sequence to improve training efficiency.
    This is a placeholder for future implementation.

    Args:
        tokenized_examples: List of tokenized examples
        max_length: Maximum sequence length
        tokenizer: HuggingFace tokenizer

    Returns:
        Packed sequences with appropriate position_ids and label_masks
    """
    # TODO: Implement sequence packing for SFT
    # This function would concatenate multiple examples into a single sequence
    # while maintaining proper position_ids and label_masks
    raise NotImplementedError("Sequence packing for SFT is not yet implemented")
