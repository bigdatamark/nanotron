# [MODULE DOC]
# Purpose: Example script producing a YAML config for a tiny Llama model (hidden 16, 2 layers, 4 heads, vocab 256) used for quick smoke tests of nanotron training.
# Behavior at import time: builds LlamaConfig, computes and prints an approximate parameter count, builds LR scheduler (cosine, 2 warmup steps), AdamW optimizer (ZeRO 0), parallelism (dp=2, pp=2, tp=2, 1f1b, REDUCE_SCATTER), tokens (seq 256, 15 steps, mbs 2), two data stages on stas/openwebtext-10k (stable from step 1, annealing from step 10) and a Config with the dummy word-level tokenizer.
# Main entry (run as script): saves config_tiny_llama.yaml next to this file.
# Exports: model_config, num_params, seed, learning_rate, optimizer, parallelism, tokens, data_stages, checkpoints_path, config.
# Side effects at import: prints; creates ./checkpoints in the current working directory.
# KNOWN ISSUE: inline comment claims 1.62M parameters, which does not match this tiny config (about 14K by the formula used).
""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information."""
import os

from nanotron.config import (
    AdamWOptimizerArgs,
    CheckpointsArgs,
    Config,
    DataArgs,
    DatasetStageArgs,
    GeneralArgs,
    LlamaConfig,
    LoggingArgs,
    LRSchedulerArgs,
    ModelArgs,
    OptimizerArgs,
    ParallelismArgs,
    PretrainDatasetsArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.logging import human_format

model_config = LlamaConfig(
    # Config for a tiny model model with 1.62M parameters
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    hidden_size=16,
    initializer_range=0.02,
    intermediate_size=64,
    max_position_embeddings=256,
    num_attention_heads=4,
    num_hidden_layers=2,
    num_key_value_heads=4,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling=None,
    tie_word_embeddings=True,
    use_cache=True,
    vocab_size=256,
)

num_params = human_format(
    model_config.vocab_size * model_config.hidden_size * 2
    + model_config.num_hidden_layers
    * (
        3 * model_config.hidden_size * model_config.intermediate_size
        + 4 * model_config.hidden_size * model_config.hidden_size
    )
).replace(".", "p")

print(f"Model has {num_params} parameters")

seed = 42

learning_rate = LRSchedulerArgs(
    learning_rate=3e-4, lr_warmup_steps=2, lr_warmup_style="linear", lr_decay_style="cosine", min_decay_lr=1e-5
)

optimizer = OptimizerArgs(
    zero_stage=0,
    weight_decay=0.01,
    clip_grad=1.0,
    accumulate_grad_in_fp32=True,
    learning_rate_scheduler=learning_rate,
    optimizer_factory=AdamWOptimizerArgs(
        adam_eps=1e-08,
        adam_beta1=0.9,
        adam_beta2=0.95,
        torch_adam_is_fused=True,
    ),
)

parallelism = ParallelismArgs(
    dp=2,
    pp=2,
    tp=2,
    pp_engine="1f1b",
    tp_mode="REDUCE_SCATTER",
    tp_linear_async_communication=True,
)

tokens = TokensArgs(sequence_length=256, train_steps=15, micro_batch_size=2, batch_accumulation_per_replica=1)

data_stages = [
    DatasetStageArgs(
        name="Stable Training Stage",
        start_training_step=1,
        data=DataArgs(
            dataset=PretrainDatasetsArgs(hf_dataset_or_datasets="stas/openwebtext-10k", text_column_name="text"),
            seed=seed,
        ),
    ),
    DatasetStageArgs(
        name="Annealing Phase",
        start_training_step=10,
        data=DataArgs(
            dataset=PretrainDatasetsArgs(hf_dataset_or_datasets="stas/openwebtext-10k", text_column_name="text"),
            seed=seed,
        ),
    ),
]

checkpoints_path = "./checkpoints"
os.makedirs(checkpoints_path, exist_ok=True)

config = Config(
    general=GeneralArgs(project="debug", run="tiny_llama_%date_%jobid", seed=seed),
    checkpoints=CheckpointsArgs(checkpoints_path=checkpoints_path, checkpoint_interval=10),
    parallelism=parallelism,
    model=ModelArgs(init_method=RandomInit(std=0.025), model_config=model_config),
    tokenizer=TokenizerArgs("robot-test/dummy-tokenizer-wordlevel"),
    optimizer=optimizer,
    logging=LoggingArgs(),
    tokens=tokens,
    data_stages=data_stages,
    profiler=None,
)

if __name__ == "__main__":
    dir = os.path.dirname(__file__)

    # Save config as YAML file
    config.save_as_yaml(f"{dir}/config_tiny_llama.yaml")

    # You can now train a model with this config using `/run_train.py`
