# [MODULE DOC]
# Purpose: manual script (not a pytest test) comparing the convergence of torch LayerNorm and nanotron TritonLayerNorm over 10000 Adam steps on a fixed CUDA input, printing and logging both losses via LoggerWriter.
# Main contents: get_time_name and a main block containing the training loop and a local loss_function.
# Import-time side effects: None beyond imports; the training runs only as a script.
import torch
from nanotron.logging import LoggerWriter
from nanotron.nn.layer_norm import TritonLayerNorm
from torch.nn import LayerNorm


def get_time_name():
    # [FUNCTION DOC]
    # HELPER: timestamp string.
    # Behavior: returns the current local time formatted day/month/year_hour:minute:second.
    # Inputs: none.
    # Outputs: string.
    # Exceptions: None explicitly raised.
    # Side effects: None.
    # KNOWN ISSUE: defined but never used in this script.
    import datetime

    today = datetime.datetime.now()
    return today.strftime("%d/%m/%Y_%H:%M:%S")


if __name__ == "__main__":
    BATCH_SIZE = 1
    SEQ_LEN = 2
    DEVICE, DTYPE = torch.device("cuda:0"), torch.float32
    HIDDEN_SIZE = 1024
    NUM_STEPS = 10_000

    inputs = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)

    layer_norm = LayerNorm(normalized_shape=inputs.size(-1), device=DEVICE, dtype=DTYPE)
    fused_layer_norm = TritonLayerNorm(
        normalized_shape=inputs.size(-1),
        device=DEVICE,
        dtype=DTYPE,
    )
    ref_optim = torch.optim.Adam(layer_norm.parameters(), lr=0.1)
    optim = torch.optim.Adam(fused_layer_norm.parameters(), lr=0.1)
    logger = LoggerWriter()

    def loss_function(x):
        # [FUNCTION DOC]
        # HELPER: loss for the convergence loop.
        # Behavior: sums all elements of x.
        # Inputs: x, tensor.
        # Outputs: scalar tensor.
        # Exceptions: None explicitly raised.
        # Side effects: None.
        return x.sum()

    for step in range(NUM_STEPS):
        # NOTE: just make the output fluctuate a bit
        random = torch.randn(1, device=DEVICE) * 0.01
        ref_outputs = layer_norm(inputs) * random
        outputs = fused_layer_norm(inputs) * random

        loss = loss_function(outputs)
        ref_loss = loss_function(ref_outputs)

        ref_optim.zero_grad()
        ref_loss.backward()
        ref_optim.step()

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"Step: {step}, outputs: {outputs.sum()}, ref_loss: {ref_outputs.sum()}")
        print(f"Step: {step}, loss: {loss}, ref_loss: {ref_loss}")

        # wandb.log({"loss": loss.item(), "ref_loss": ref_loss.item(), "step": step})
        logger.add_scalar("loss", loss.item(), step)
        logger.add_scalar("ref_loss", ref_loss.item(), step)
