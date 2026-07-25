import json

import numpy as np
import pytest
import torch

pytest.importorskip("energyflow")

from experiments.analyze_filtered_mass_shell_batch import json_ready


def test_json_ready_converts_metric_arrays_recursively():
    value = {
        "metric": (np.array([1.0, 2.0]), np.float64(0.2)),
        "tensor": torch.tensor([3.0]),
    }
    converted = json_ready(value)
    assert converted == {"metric": [[1.0, 2.0], 0.2], "tensor": [3.0]}
    json.dumps(converted)
