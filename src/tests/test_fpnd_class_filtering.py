import torch

from util.data.fpnd_input import iter_fpnd_class_subsets


def test_fpnd_filters_mixed_generated_batch_by_class():
    jets = torch.arange(6 * 3 * 3, dtype=torch.float64).reshape(6, 3, 3)
    class_labels = torch.tensor([0, 2, 1, 0, 2, 1])
    jet_types = ["g", "q", "t"]

    subsets = list(iter_fpnd_class_subsets(jets, class_labels, jet_types))

    assert [jet_type for jet_type, _ in subsets] == jet_types
    for class_index, (_, class_jets) in enumerate(subsets):
        torch.testing.assert_close(
            class_jets,
            jets[class_labels == class_index],
        )


def test_fpnd_skips_unrepresented_class():
    subsets = list(
        iter_fpnd_class_subsets(
            torch.zeros(2, 3, 3),
            torch.tensor([0, 0]),
            ["g", "q"],
        )
    )

    assert len(subsets) == 1
    assert subsets[0][0] == "g"
