# LorentzNet normalized-log-map factorial

All four arms use mass-shell RFM, invariant reference-contraction scalar initialization,
online geodesic ICP, a masked lognormal axis-aligned prior, normalized log-map particle
directions, and final particle-field normalization by `sqrt(degree)`.

| Arm | Reference readout | Layer geometry |
|---|---|---|
| G | raw ambient references | evolving auxiliary `y` |
| H | normalized tangent references | evolving auxiliary `y` |
| I | raw ambient references | fixed physical `x`, geodesic edges |
| J | normalized tangent references | fixed physical `x`, geodesic edges |

This is a complete 2x2 factorial. G is the new baseline; H and I measure the two main
effects, and J measures their combination/interaction.
