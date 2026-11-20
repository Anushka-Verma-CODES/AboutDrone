"""Frame-block splitting for VisioDECT.

VisioDECT images are consecutive video frames numbered 1..1200 per drone
model and weather scenario, so neighbouring frames are near-duplicates.
Each sequence is cut into blocks of BLOCK frames; whole blocks go to train,
val or test (8/2/2), and the first and last EDGE frames of every block are
dropped so that no test frame has a near-identical neighbour in training.

Used by both the detector dataset (src.detection.block_split) and the
classifier evaluation, with the same seed, so both use the same test frames.
"""
from collections import defaultdict

import numpy as np

from src.config import SEED

BLOCK, EDGE = 100, 5


def block_of(num):
    i = num - 1
    return i // BLOCK, i % BLOCK


def near_edge(num):
    off = block_of(num)[1]
    return off < EDGE or off >= BLOCK - EDGE


def assign_blocks(keys_nums, seed=SEED):
    """keys_nums: iterable of (sequence_key, frame_number).
    Returns {(*sequence_key, block): split}."""
    rng = np.random.default_rng(seed)
    blocks = defaultdict(set)
    for key, num in keys_nums:
        blocks[key].add(block_of(num)[0])
    out = {}
    for key, bs in sorted(blocks.items()):
        bs = list(rng.permutation(sorted(bs)))
        n_test = n_val = max(1, round(len(bs) / 6))
        for j, b in enumerate(bs):
            out[key + (b,)] = "test" if j < n_test else "val" if j < n_test + n_val else "train"
    return out
