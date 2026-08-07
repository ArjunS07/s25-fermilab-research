"""EMA update math (experiment plan 2.1)."""
import torch
import torch.nn as nn

from util.infra.ema import ModelEMA


def _tiny_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))


def test_ema_tracks_toward_current_weights():
    model = _tiny_model(0)
    ema = ModelEMA(model, decay=0.9)

    # Snapshot initial shadow == initial weights.
    w0 = model[0].weight.detach().clone()
    assert torch.allclose(ema.shadow["0.weight"], w0)

    # Change the model weights, then one EMA update.
    with torch.no_grad():
        model[0].weight.add_(1.0)
    w1 = model[0].weight.detach().clone()
    ema.update(model)

    expected = 0.9 * w0 + 0.1 * w1
    assert torch.allclose(ema.shadow["0.weight"], expected, atol=1e-6)


def test_ema_converges_to_constant_weights():
    model = _tiny_model(1)
    ema = ModelEMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(2.0)
    for _ in range(50):
        ema.update(model)
    for v in ema.shadow.values():
        if torch.is_floating_point(v):
            assert torch.allclose(v, torch.full_like(v, 2.0), atol=1e-6)


def test_copy_to_restores_shadow():
    model = _tiny_model(2)
    ema = ModelEMA(model, decay=0.99)
    shadow_before = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(5.0)
    ema.copy_to(model)  # model should now hold the (unchanged) shadow
    for k, v in model.state_dict().items():
        assert torch.allclose(v, shadow_before[k], atol=1e-6)
