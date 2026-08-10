# LorentzNet flow-geometry ablation

The four configs in this directory hold the LorentzNet architecture and training
recipe fixed while varying only flow geometry and the optional plain reference
readout.

| arm | geometry | references | parameters |
|---|---|---|---:|
| A | Euclidean FM | none | 441,621 |
| B | mass-shell RFM | none | 441,621 |
| C | Euclidean FM | plain readout | 451,319 |
| D | mass-shell RFM | plain readout | 451,319 |

All use six LGEB blocks at width 96, one million gluon-30 training examples per
epoch stream, a 200,000 optimizer-step cap, effective batch 250 on one GPU, the
same five RNG streams, the equal-share axis-aligned prior, no ICP/coupling, EMA, and
64 Euler sampling steps for 50,000 evaluation jets.

The Euclidean objective is ordinary masked component-wise MSE. It is frame
dependent by design even though the LorentzNet field is equivariant. Euclidean
sampling publishes the raw ambient endpoint: it does not project to a mass shell,
repair energy, or clamp components. The RFM configs reuse the existing shell
lifting, geodesic interpolation/target, induced tangent loss, and exp-map Euler
integration.

The backbone coordinate update uses `y_i - y_j`, matching the released LorentzNet
implementation. It has no emergency component clamp; its final coordinate
coefficient layer is initialized at standard deviation 1e-3 and the coordinate
residual has a fixed 1e-2 scale. The learned field head is shared across geometries
and explicitly includes self vectors. The optional reference head directly emits
the ordered `(e_t, jet_p4)` coefficients, without role embeddings or reference
tokens.

The `axis_aligned_equal` prior uses the conditioned jet axis and equal pT shares
over real particles only. One per-jet scalar correction makes the real-particle
transverse vector sum exactly match conditioned jet pT. It contains no random
fragmentation logits, target matching, or batch-level normalization.
