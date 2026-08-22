"""Regression tests for fixed-global vs configured-local jet type labels."""

import torch

from util.data.jet_attributes import (
    generate_jets,
    global_jet_type_indices,
    local_jet_type_indices,
)


class _RecordingAttributeModel:
    def __init__(self):
        self.context = None

    def sample(self, num_jets, context):
        self.context = context.clone()
        # eta, pT, mass, multiplicity; only shape matters for this sampler test.
        return torch.zeros(num_jets, 4, device=context.device), torch.zeros(num_jets)


def test_single_q_generation_uses_global_q_one_hot_not_gluon():
    model = _RecordingAttributeModel()
    generated, _ = generate_jets(model, "cpu", jet_types=["q"], num_jets=8)

    expected = torch.tensor([0., 1., 0., 0., 0.]).repeat(8, 1)
    torch.testing.assert_close(model.context, expected)
    torch.testing.assert_close(generated[:, :5], expected)


def test_single_t_generation_uses_global_t_one_hot_not_gluon():
    model = _RecordingAttributeModel()
    generated, _ = generate_jets(model, "cpu", jet_types=["t"], num_jets=8)

    expected = torch.tensor([0., 0., 1., 0., 0.]).repeat(8, 1)
    torch.testing.assert_close(model.context, expected)
    torch.testing.assert_close(generated[:, :5], expected)


def test_global_and_local_class_spaces_are_distinct_for_specialists():
    assert global_jet_type_indices(["q"]) == [1]
    assert global_jet_type_indices(["t"]) == [2]
    assert global_jet_type_indices(["g", "q", "t"]) == [0, 1, 2]

    torch.testing.assert_close(
        local_jet_type_indices(torch.tensor([1, 1, 1]), ["q"]),
        torch.zeros(3, dtype=torch.long),
    )
    torch.testing.assert_close(
        local_jet_type_indices(torch.tensor([2, 0, 1]), ["t", "g", "q"]),
        torch.tensor([0, 1, 2]),
    )
