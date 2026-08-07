# FPND ptrel Normalization Bug

*This description uses ASD-STE100 Simplified Technical English.*

## 1. Summary

The FPND metric gave a wrong, high value of about 24. This is not a fault in the
model. It is a mismatch in how we make the input for FPND. To understand it, you must
first understand how JetNet stores a jet.

## 2. How JetNet stores a jet (the important part)

A real jet has many particles. A gluon jet often has more than 30.

- JetNet keeps only the **30 hardest** particles. It drops the rest. The dropped
  particles are the soft (low-pT) ones.
- JetNet stores each kept particle with a relative pT:

  ```
  ptrel_i = pt_i / jet_pt
  ```

- Here `jet_pt` is the **true pT of the full jet**, before the drop. JetNet keeps this
  number separately, at the jet level. You cannot get it back from the 30 particles.
- The 30 dropped-down particles carry about 9% of the jet pT. So the 30 kept `ptrel`
  values do **not** sum to 1. They sum to about **0.91**.

The key idea: **the true jet pT and the sum of the 30 kept particles are two different
numbers.** The 30-particle limit makes them different on purpose. For a jet with a true
pT of 1000 GeV, the 30 kept particles carry about 910 GeV; the other ~90 GeV is in the
dropped soft particles, but the denominator is still 1000.

## 3. What FPND expects

FPND uses a fixed ParticleNet model. The JetNet authors trained this model on the
stored JetNet arrays. So ParticleNet expects `ptrel` in the JetNet convention:

- the 30 values sum to about **0.91**, and
- each value stays below the ParticleNet input limit (`ptrel` maximum = 0.8935).

## 4. What our code did wrong

We made the FPND input with jetnet's `EtaPhiPtE_to_relEtaPhiPt`.

- That function only gets the 30 particles. It does not get the true jet pT.
- So it rebuilds the denominator from the 30 particles that it has (their vector-sum
  pT, about 910 GeV in the example).
- This denominator is about 9% too small, because it does not include the dropped
  soft pT.
- As a result, every `ptrel` is about 10% too large. The 30 values sum to about 1.0,
  not 0.91.
- These large values go above the ParticleNet input limit. The jets become out of the
  training distribution. FPND becomes very large — about 20, even for **real** jets.

## 5. The fix — use the true jet pT

Divide by the true jet pT, not by the sum of the 30 particles. We already have the
true jet pT: it is `gen_pt_cond`, the jet pT that Stage 1 makes and that Stage 2 uses
as a condition. This is our version of JetNet's stored `jet_pt`.

This is not a trick:

- It is the **same** convention that JetNet uses for real jets.
- It does not hide a bad model. If the model makes constituents with the wrong pT
  fraction, the sum of `ptrel` moves away from 0.91, and ParticleNet sees this.

To divide by the sum of the 30 particles is the wrong choice. It forces the sum to
1.0 for every jet and removes the soft-radiation information.

## 6. Why only FPND changes

- W1M, W1P, W1EFP, FPD, and cov_mmd compare the generated jets against the test jets.
  Both sides use `EtaPhiPtE_to_relEtaPhiPt`, so both sides have the same 10% offset.
  The offset cancels.
- FPND is the only metric that compares against an **external** fixed reference (the
  cached ParticleNet). There is no cancellation. So FPND is the only metric that sees
  the mismatch.

## 7. The code (short)

Old — the FPND input is `gen_polar_rel`, made by `EtaPhiPtE_to_relEtaPhiPt` (the
vector-sum convention):

```python
jets = gen_polar_rel.float()
```

New — a separate FPND input, made from the true (conditioning) jet pT and eta. The
one line that matters:

```python
pt_rel = pt / jet_pt          # jet_pt = gen_pt_cond, the true clustered jet pT
```

The helper `build_fpnd_input(gen_polar_abs, gen_jet_eta, gen_pt_cond, gen_mask)` makes
the full `(eta_rel, phi_rel, pt_rel)` this way. `metrics.py` gives its output to FPND:

```python
jets = gen_fpnd_input.float()
```

The W1/FPD/cov paths do not change. They keep `gen_polar_rel`.

## 8. Result

- The fix changes only the FPND input. The other metrics do not change.
- FPND for the models falls from about 24 to a range of 0.55 to 0.93.
- FPD stays the same to 7 significant figures. This shows that the fix changes only
  the FPND path.
