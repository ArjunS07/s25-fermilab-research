# LorentzNet reference-scalar + ICP follow-up

This RFM-only follow-up adds two invariant particle inputs before the first LGEB:

```text
h_i^0 = MLP(signed_log(<y_i,e_t>), signed_log(<y_i,p_J>))
        - time_embedding + condition_embedding
```

The ordered references transform jointly with the event, so both contractions are
Lorentz invariant. Both arms use the equal-share axis-aligned prior and fresh-noise
online geodesic ICP coupling.

- **E** (`g30-lorentznet-e-rfm-refscalar-icp-none.yaml`): contractions only; no
  reference vectors in the physical vector-field readout. **441,717 parameters.**
- **F** (`g30-lorentznet-f-rfm-refscalar-icp-refs.yaml`): contractions plus the
  existing plain `(e_t, p_J)` vector-field readout. **451,415 parameters.**

Launch decision after D completes:

1. If D restores genuinely good corrected FPND, launch neither arm.
2. If D materially improves over B, launch F only.
3. If B and D are comparably poor, launch both E and F to retain the readout ablation.

Every launched arm must first pass its 20-step smoke with zero invalid samples and
zero unstable stability-probe trajectories.
