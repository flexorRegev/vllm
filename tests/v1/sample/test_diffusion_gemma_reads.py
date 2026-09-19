# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request DiffusionGemma state behind structured reads: seed canvases,
read-only slots, the per-slot step cap and the per-step read trajectory."""

import math

import numpy as np
import pytest
import torch

from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaRequestStates,
    DiffusionSampler,
    _compiled_sample_step,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the sampler state lives on the GPU"
)

CL = 8
VOCAB = 64
MAX_REQS = 4
MAX_STEPS = 48


def _states() -> DiffusionGemmaRequestStates:
    return DiffusionGemmaRequestStates(
        max_num_reqs=MAX_REQS,
        canvas_length=CL,
        vocab_size=VOCAB,
        max_denoising_steps=MAX_STEPS,
        device=torch.device("cuda"),
        hidden_size=4,
        stability_threshold=2,
    )


def _slots(*idx: int) -> tuple[np.ndarray, torch.Tensor]:
    slots = np.array(idx, dtype=np.int64)
    return slots, torch.tensor(slots, device="cuda")


def test_seed_canvas_replaces_only_seeded_slots():
    states = _states()
    for slot in range(3):
        states.add_request(slot)
    seed = list(range(CL))
    states.set_seed_canvas(1, seed)
    slots, slots_gpu = _slots(0, 1, 2)
    states.init_canvas(slots_gpu)
    before = states.canvas[slots_gpu].clone()

    states.apply_seed_canvases(slots, slots_gpu)

    after = states.canvas[slots_gpu]
    assert after[1].tolist() == seed
    assert torch.equal(after[0], before[0])
    assert torch.equal(after[2], before[2])


def test_apply_seed_canvases_leaves_unseeded_batches_alone():
    states = _states()
    states.add_request(0)
    slots, slots_gpu = _slots(0)
    states.init_canvas(slots_gpu)
    before = states.canvas[0].clone()

    states.apply_seed_canvases(slots, slots_gpu)

    assert torch.equal(states.canvas[0], before)


def test_add_request_clears_seed_and_read_only():
    states = _states()
    states.add_request(0)
    states.set_seed_canvas(0, [1] * CL)
    states.set_read_only(0)
    assert states.seeded_slots == {0}
    assert states.read_only_slots == {0}

    states.add_request(0)

    assert not states.seeded_slots
    assert not states.read_only_slots
    assert not bool(states.has_seed[0])
    assert not bool(states.read_only[0])


def test_canvas_width_resets_with_the_slot():
    states = _states()
    states.add_request(0)
    states.canvas_width_np[0] = 4
    states.set_seed_canvas(0, [7, 7, 7, 7])
    assert states.seed_canvas[0, :4].tolist() == [7, 7, 7, 7]

    states.add_request(0)

    assert states.canvas_width_np[0] == CL


def test_remove_request_forgets_the_slot():
    states = _states()
    states.add_request(0)
    states.set_seed_canvas(0, [1] * CL)
    states.set_read_only(0)

    states.remove_request(0)

    assert not states.seeded_slots
    assert not states.read_only_slots


def _denoise_once(
    states: DiffusionGemmaRequestStates,
    slots: list[int],
    compute_sc: bool = True,
    width: int = CL,
    embed_weight: torch.Tensor | None = None,
    logits: torch.Tensor | None = None,
    normalizer: torch.Tensor | None = None,
    eager: bool = False,
) -> None:
    """One compiled denoise step over ``slots`` with flat logits, so nothing
    converges by stability or confidence and only the step cap can end it.
    ``width`` below CL runs the step on [:, :width] views, as the sampler
    does for a narrow tile."""
    n = len(slots)
    device = states.device
    decode_slots = torch.tensor(slots, dtype=torch.int64, device=device)
    decode_idx = torch.arange(n, dtype=torch.int64, device=device)
    # Dynamo falls back to eager once a call site passes its recompile limit,
    # so the un-compiled body has to be correct on its own.
    step = (
        _compiled_sample_step._torchdynamo_orig_callable
        if eager
        else _compiled_sample_step
    )
    step(
        torch.zeros(n * width, VOCAB, device=device) if logits is None else logits,
        decode_slots,
        decode_idx,
        decode_slots,
        torch.full((n,), width, dtype=torch.int64, device=device),
        states.canvas[:, :width],
        states.argmax_canvas[:, :width],
        states.step,
        states.is_encoder_phase,
        states.confident,
        states.self_conditioning_embeds[:, :width],
        torch.zeros(VOCAB, 4, device=device) if embed_weight is None else embed_weight,
        torch.tensor(1.0, device=device) if normalizer is None else normalizer,
        states.accepted_canvas_history[:, :, :width],
        states.accepted_canvas_history_len,
        states.max_steps,
        states.seed_canvas[:, :width],
        states.free_mask[:, :width],
        states.has_seed,
        states.fixed_steps,
        states.never_accept,
        torch.zeros(n, CL, dtype=torch.int32, device=device)[:, :width],
        torch.zeros(n, dtype=torch.int32, device=device),
        torch.zeros(MAX_REQS, CL, dtype=torch.int64, device=device),
        max_denoising_steps=float(MAX_STEPS),
        t_min=0.5,
        t_max=1.0,
        confidence_threshold=0.1,
        vocab_size=VOCAB,
        CL=width,
        ST=states.stability_threshold,
        entropy_bound=0.1,
        sc_vocab_start=0,
        sc_vocab_end=VOCAB,
        tp_size=1,
        tp_group_name="",
        compute_sc=compute_sc,
    )


def test_single_step_tile_skips_self_conditioning():
    states = _states()
    states.add_request(0)
    states.is_encoder_phase[0] = False
    states.self_conditioning_embeds[0] = 1.0

    _denoise_once(states, [0], compute_sc=False)

    assert not states.self_conditioning_embeds[0].any()


def test_narrow_tile_leaves_the_rest_of_the_canvas_alone():
    states = _states()
    states.add_request(0)
    states.is_encoder_phase[0] = False
    states.canvas[0] = 5
    states.argmax_canvas[0] = 5

    _denoise_once(states, [0], width=4)

    # Columns past the width are untouched; the step counted.
    assert states.canvas[0, 4:].tolist() == [5] * (CL - 4)
    assert states.argmax_canvas[0, 4:].tolist() == [5] * (CL - 4)
    assert states.step[0] == 1


def test_slot_positions_are_the_only_free_canvas_columns():
    states = _states()
    states.add_request(0)
    states.set_slot_positions(0, [2, 5])

    assert states.free_mask[0].tolist() == [i in (2, 5) for i in range(CL)]

    # The slot forgets its mask when it is reused.
    states.add_request(0)
    assert states.free_mask[0].all()


def test_clamped_positions_hold_the_seed_across_steps():
    states = _states()
    states.add_request(0)
    states.is_encoder_phase[0] = False
    seed = [9] * CL
    states.set_seed_canvas(0, seed)
    states.set_slot_positions(0, [3])
    states.canvas[0] = 9

    for _ in range(4):
        _denoise_once(states, [0])
        canvas = states.canvas[0].tolist()
        argmax = states.argmax_canvas[0].tolist()
        fixed = [i for i in range(CL) if i != 3]
        assert [canvas[i] for i in fixed] == [9] * len(fixed)
        assert [argmax[i] for i in fixed] == [9] * len(fixed)

    # Flat logits argmax to token 0, so the free slot is not the seed token.
    assert states.argmax_canvas[0, 3] == 0


def test_an_unmasked_seeded_slot_denoises_the_whole_canvas():
    states = _states()
    states.add_request(0)
    states.is_encoder_phase[0] = False
    states.set_seed_canvas(0, [9] * CL)
    states.canvas[0] = 9

    # Same tile, clamping on: with no slot positions the mask is all-free, so
    # the step re-noises as it always did.
    _denoise_once(states, [0])

    assert states.argmax_canvas[0].tolist() == [0] * CL
    assert (states.canvas[0] != 9).any()


def test_self_conditioning_is_one_hot_at_clamped_positions():
    states = _states()
    states.add_request(0)
    states.is_encoder_phase[0] = False
    states.set_seed_canvas(0, [4] * CL)
    states.set_slot_positions(0, [1])
    # One distinguishable embedding row per token id.
    embed_weight = torch.arange(VOCAB, dtype=torch.float32, device=states.device)
    embed_weight = embed_weight[:, None].repeat(1, 4)

    _denoise_once(states, [0], embed_weight=embed_weight)

    sc = states.self_conditioning_embeds[0]
    fixed = [i for i in range(CL) if i != 1]
    # Held positions carry the seed token's embedding row exactly; the free
    # one carries the model's own (uniform) mixture.
    assert sc[fixed].eq(4.0).all()
    assert not sc[1].eq(4.0).any()


def test_step_cap_is_per_slot():
    states = _states()
    for slot in (0, 1):
        states.add_request(slot)
        states.is_encoder_phase[slot] = False
    states.max_steps[0] = 1

    _denoise_once(states, [0, 1])

    # Slot 0 hit its cap and moves to commit. Slot 1 keeps denoising.
    assert states.is_encoder_phase[:2].tolist() == [True, False]
    assert states.step[:2].tolist() == [1, 1]


def _trajectory_sampler(states: DiffusionGemmaRequestStates) -> DiffusionSampler:
    """Just enough sampler to drive the trajectory stash: recording reads the
    diffusion states and writes the stash, nothing else."""
    sampler = DiffusionSampler.__new__(DiffusionSampler)
    sampler.diffusion_states = states
    sampler._trajectory = {}
    return sampler


def _peaked_logits(states: DiffusionGemmaRequestStates) -> torch.Tensor:
    """One canvas of flat logits, with position 1 peaked on token 6."""
    logits = torch.zeros(CL, VOCAB, device=states.device)
    logits[1, 6] = 10.0
    return logits


def test_trajectory_records_one_row_per_step():
    states = _states()
    states.add_request(0)
    states.set_trajectory(0, [1, 3], [5, 6, 7])
    sampler = _trajectory_sampler(states)
    logits = _peaked_logits(states)

    for _ in range(3):
        sampler._record_trajectory(logits, [0], CL)
    out = sampler._take_trajectory(0)

    assert out["positions"] == [1, 3]
    assert out["label_token_ids"] == [5, 6, 7]
    assert [step["step"] for step in out["steps"]] == [1, 2, 3]
    for step in out["steps"]:
        assert step["argmax_id"] == [6, 0]
        assert [len(row) for row in step["label_logprobs"]] == [3, 3]
        # label_logprobs follow the request's id order: position 1 puts its
        # mass on 6, the second of the three labels.
        assert step["label_logprobs"][0][1] == pytest.approx(
            step["argmax_logprob"][0], abs=1e-5
        )
        # Position 3 saw flat logits, so it carries the full-vocab entropy.
        assert step["entropy"][1] == pytest.approx(math.log(VOCAB), abs=1e-4)
        assert step["entropy"][0] < step["entropy"][1]
    # The read has emitted; the slot keeps nothing.
    assert not sampler._trajectory


def test_trajectory_records_only_the_slots_that_asked():
    states = _states()
    for slot in (0, 1):
        states.add_request(slot)
    states.set_trajectory(1, [0], [])
    sampler = _trajectory_sampler(states)

    sampler._record_trajectory(
        torch.zeros(2 * CL, VOCAB, device=states.device), [0, 1], CL
    )

    assert set(sampler._trajectory) == {1}
    steps = sampler._take_trajectory(1)["steps"]
    assert len(steps) == 1
    assert steps[0]["label_logprobs"] == [[]]


def test_trajectory_resets_with_the_slot():
    states = _states()
    states.add_request(0)
    states.set_trajectory(0, [0, 1], [3])
    assert states.trajectory_slots == {0}

    states.add_request(0)

    assert not states.trajectory_slots
    assert not states.trajectory_meta


def _slot_logits(states: DiffusionGemmaRequestStates, peak: float, pos: int = 3):
    """A canvas whose only uncertainty is at ``pos``: ``peak`` spread over two
    tokens there, every other position certain on token 0."""
    logits = torch.full((CL, VOCAB), -50.0, device=states.device)
    logits[:, 0] = 50.0
    logits[pos] = -50.0
    logits[pos, 1] = peak
    logits[pos, 2] = peak
    return logits


def test_confidence_is_read_over_the_free_slots_only():
    states = _states()
    states.add_request(0)
    states.is_encoder_phase[0] = False
    states.set_seed_canvas(0, [9] * CL)
    states.set_slot_positions(0, [3])
    # The free slot is split between two tokens: entropy log(2) = 0.69, well
    # over the 0.1 threshold. Averaged over all eight positions it would be
    # 0.087 and the read would call itself confident.
    logits = _slot_logits(states, peak=0.0)

    _denoise_once(states, [0], logits=logits)
    assert not bool(states.confident[0])

    # The same canvas without a mask: the held positions dilute the slot's
    # entropy and the old criterion passes.
    plain = _states()
    plain.add_request(0)
    plain.is_encoder_phase[0] = False
    _denoise_once(plain, [0], logits=_slot_logits(plain, peak=0.0))
    assert bool(plain.confident[0])


def test_fixed_steps_runs_the_cap_out():
    states = _states()
    for slot in (0, 1):
        states.add_request(slot)
        states.is_encoder_phase[slot] = False
        states.max_steps[slot] = 4
    states.set_fixed_steps(1)
    # Certain and stable everywhere: slot 0 converges as soon as the history
    # is long enough, slot 1 keeps denoising to its cap.
    logits = torch.full((2 * CL, VOCAB), -50.0, device=states.device)
    logits[:, 0] = 50.0

    for _ in range(2):
        _denoise_once(states, [0, 1], logits=logits)
    assert states.is_encoder_phase[:2].tolist() == [True, False]

    # Slot 0 has left the denoise phase; run slot 1 out on its own.
    for _ in range(2):
        _denoise_once(states, [1], logits=logits[:CL])
    assert states.step[1] == 4
    assert bool(states.is_encoder_phase[1])


def test_free_slots_are_never_accepted_when_the_request_says_so():
    def run(never_accept: bool) -> list[int]:
        states = _states()
        states.add_request(0)
        states.is_encoder_phase[0] = False
        states.set_seed_canvas(0, [9] * CL)
        states.set_slot_positions(0, [3])
        if never_accept:
            states.set_slots_never_accept(0)
        # The slot is certain on token 1, so the entropy bound accepts it.
        logits = torch.full((CL, VOCAB), -50.0, device=states.device)
        logits[:, 0] = 50.0
        logits[3] = -50.0
        logits[3, 1] = 50.0
        drawn = []
        for _ in range(5):
            # These logits converge the read, and the step after a converged
            # one is the commit that re-inits the canvas. Hold the slot in the
            # denoise phase: acceptance is what this test is about.
            states.is_encoder_phase[0] = False
            _denoise_once(states, [0], logits=logits)
            drawn.append(int(states.canvas[0, 3]))
            # The template is held either way.
            assert states.canvas[0, 0] == 9
            # And the model's own answer is still what the read reports.
            assert states.argmax_canvas[0, 3] == 1
        return drawn

    assert run(never_accept=False) == [1] * 5
    # Re-noised every step: five uniform draws over the vocabulary.
    assert set(run(never_accept=True)) != {1}


def test_trajectory_gathers_its_slots_before_the_fp32_cast():
    """The tile's logits are [tile * W, vocab]; casting that to fp32 for a
    handful of slot rows is gigabytes at a wide canvas."""

    class _RecordCasts(torch.overrides.TorchFunctionMode):
        def __init__(self):
            self.shapes: list[tuple[int, ...]] = []

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.Tensor.float and args:
                self.shapes.append(tuple(args[0].shape))
            return func(*args, **kwargs)

    states = _states()
    for slot in (0, 1):
        states.add_request(slot)
    states.set_trajectory(1, [1, 3], [5, 6])
    sampler = _trajectory_sampler(states)
    tile = torch.zeros(2 * CL, VOCAB, device=states.device)

    with _RecordCasts() as casts:
        sampler._record_trajectory(tile, [0, 1], CL)

    assert casts.shapes, "the slot rows are cast to fp32"
    # Nothing wider than the slots the request asked for is ever upcast.
    assert all(shape[0] <= 2 for shape in casts.shapes)
    assert tuple(tile.shape) not in casts.shapes
    rows = sampler._trajectory[1]
    assert len(rows) == 1
    assert tuple(rows[0].shape) == (2, 5)
    assert rows[0].dtype is torch.float32


def test_fixed_steps_and_never_accept_reset_with_the_slot():
    states = _states()
    states.add_request(0)
    states.set_fixed_steps(0)
    states.set_slots_never_accept(0)

    states.add_request(0)

    assert not bool(states.fixed_steps[0])
    assert not bool(states.never_accept[0])


def test_three_seeded_multi_step_rows_share_one_tile():
    """Concurrent structured reads land in one tile: every row is seeded,
    clamped to its own free position and fixed to the step cap. Each row must
    keep its own seed and its own step count."""
    states = _states()
    seeds = {0: [11] * CL, 1: [22] * CL, 2: [33] * CL}
    free = {0: 1, 1: 3, 2: 5}
    for slot, seed in seeds.items():
        states.add_request(slot)
        states.is_encoder_phase[slot] = False
        states.set_seed_canvas(slot, seed)
        states.set_slot_positions(slot, [free[slot]])
        states.set_fixed_steps(slot)
        states.max_steps[slot] = 3
        states.canvas[slot] = seed[0]

    for step in range(1, 4):
        _denoise_once(states, [0, 1, 2])
        for slot, seed in seeds.items():
            canvas = states.canvas[slot].tolist()
            held = [i for i in range(CL) if i != free[slot]]
            assert [canvas[i] for i in held] == [seed[0]] * len(held)
            assert states.step[slot] == step
        # The cap is what ends a fixed-steps read, so no row commits early.
        assert states.is_encoder_phase[:3].tolist() == [step == 3] * 3


def test_self_conditioning_stores_into_the_fp32_buffer_from_bf16_embeddings():
    """The soft embed follows the embedding dtype; the buffer is fp32. An index
    put does not cast, so the eager body has to. Only the compiled graph hid
    this, and dynamo runs the body eagerly once a call site recompiles enough."""
    states = _states()
    states.add_request(0)
    states.is_encoder_phase[0] = False

    _denoise_once(
        states,
        [0],
        embed_weight=torch.ones(VOCAB, 4, device=states.device, dtype=torch.bfloat16),
        normalizer=torch.tensor(1.0, device=states.device, dtype=torch.bfloat16),
        eager=True,
    )

    assert states.self_conditioning_embeds.dtype == torch.float32
    assert states.self_conditioning_embeds[0].ne(0).any()
