from __future__ import annotations

from types import SimpleNamespace

from vqa_project.data import VQAWithVisualGenomeSampler


def test_sampler_uses_all_vqa_plus_quarter_vg_and_shards_evenly() -> None:
    dataset = SimpleNamespace(vqa_example_count=8, visual_genome_example_count=8)
    rank_zero = VQAWithVisualGenomeSampler(dataset, num_replicas=2, rank=0, seed=42)
    rank_one = VQAWithVisualGenomeSampler(dataset, num_replicas=2, rank=1, seed=42)

    zero_indices = list(rank_zero)
    one_indices = list(rank_one)
    selected = zero_indices + one_indices

    assert rank_zero.visual_genome_sample_count == 2
    assert len(zero_indices) == len(one_indices) == 5
    assert set(zero_indices).isdisjoint(one_indices)
    assert set(range(8)).issubset(selected)
    assert len({index for index in selected if index >= 8}) == 2


def test_sampler_repeats_with_same_epoch_and_resamples_for_next_epoch() -> None:
    dataset = SimpleNamespace(vqa_example_count=8, visual_genome_example_count=8)
    sampler = VQAWithVisualGenomeSampler(dataset, num_replicas=1, rank=0, seed=42)

    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    repeated = list(sampler)
    sampler.set_epoch(4)
    next_epoch = list(sampler)

    assert first == repeated
    assert first != next_epoch
