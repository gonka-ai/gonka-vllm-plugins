import torch

from gonka_poc.poc.decode_random import (
    random_pick_indices_decode,
    random_pick_indices_decode_steps,
)


def test_batched_steps_match_per_step_calls():
    """Batched index emission (encode_sph_slices) must equal per-step reference
    calls bitwise, order included — otherwise artifacts are incompatible."""
    cpu = torch.device("cpu")
    bh, pk, nonce, dim, k = "0xdeadbeef", "pk1", 4242, 256, 16
    steps = list(range(1 + 256))  # prefill + decode steps, as in prod
    batched = random_pick_indices_decode_steps(bh, pk, nonce, dim, k, cpu, steps)
    assert batched.shape == (len(steps), k)
    for s in steps:
        single = random_pick_indices_decode(bh, pk, [nonce], dim, k, cpu, step=s)[0]
        assert torch.equal(batched[s], single), f"step {s}: index set/order differs"
