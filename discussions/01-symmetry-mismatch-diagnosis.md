# Diagnosis: the symmetry group of LEFT-JeN is strictly larger than the symmetry of the data

*Written 2026-07-03. Status: confirmed empirically — generated 3-momentum directions are ~uniform on the sphere.*

## TL;DR

LEFT-JeN is **exactly equivariant under spatial rotations** (a subgroup of the Lorentz
group), and every prior we have used (`isotropic_com`, Gaussian) has an **exactly
rotation-invariant** 3-momentum distribution. Equivariant sampler + invariant prior ⇒
the generated distribution is **provably isotropic, for any weights, after any amount
of training**. But jet data in relative coordinates is collimated around a fixed axis.
The model literally cannot fit the orientation-dependent structure of the data. This is
an architectural theorem, not a tuning problem.

## The argument

Two ingredients:

**(a) The sampler commutes with rotations.**
Every learned coefficient in LEFT-JeN is a function of Minkowski invariants
(ψ(‖x_i − x_j‖²_M), ψ(⟨x_i, x_j⟩_M)), time, and global conditioning — all unchanged by
a spatial rotation R of the input cloud. Outputs are these coefficients times
differences of input 4-vectors, and `velocity = x_L − x_0` preserves this. So the
velocity field satisfies v(Rx) = R·v(x), and Euler integration composes:
the full ODE map satisfies Φ(Rx₀) = R·Φ(x₀).
(The component-wise clamps at ±1e8 formally break this but never activate in practice;
the ψ-clamps act on scalars and are harmless.)

**(b) The prior is rotation-invariant.**
`isotropic_com` samples directions uniformly on the sphere; an iid Gaussian on
(px, py, pz) is also exactly isotropic. Masked/padded zeros are rotation-invariant too.

Therefore, for any rotation R, the generated sample X = Φ(x₀) has the same law as
Φ(Rx₀) = R·X: **the generated distribution is invariant under all spatial rotations.**
The data is not — relative-coordinate jets are collimated around the jet axis. QED.

This covers the earlier Gaussian-prior runs as well: any prior with an isotropic
3-momentum distribution lands in the same trap. The only exits are (i) a non-isotropic
axis-aligned prior, or (ii) symmetry-breaking inputs (reference vectors).

## Why results were "crappy but not insane"

Spatial rotations fix the energy component and all rotation-invariant observables.
So the model **can** learn:

- the E spectrum and |p⃗| spectrum,
- pairwise opening angles, invariant masses,
- multiplicity structure.

During training it converges to the **rotationally-averaged velocity field** — the
least-squares projection of the true (non-equivariant) marginal field onto the space
of equivariant fields is the group average. Anything defined relative to the fixed
frame — η_rel, φ_rel, particle pT — is scrambled by the random orientation of each
generated cloud.

This also explains why the training loss plateaus at a nonzero floor regardless of
training length: the residual is the variance of the true field around its group
average, which no equivariant network can reduce.

Note: ICP permutation alignment of the prior does not escape this. It changes the
*conditional* coupling during training, but the *marginal* field the network can
express is still equivariant, and at inference the prior is isotropic.

## The fix: symmetry breaking via reference vectors (not a retreat from equivariance)

Add reference 4-vectors — the time axis e_t = (1,0,0,0) and the jet axis / jet
4-momentum — as extra unmasked "virtual particles" that participate in message passing
(their displacement outputs discarded), and feed their invariants with each real
particle as per-node scalar features. Note ⟨x_i, e_t⟩ = E_i and ⟨x_i, x_jet⟩ measures
axis alignment: the model gains access to energy positivity and collimation.

### Why this does not break equivariance

The scalar features ⟨x_i, e_t⟩ *do* change if you transform the particles alone — that
is the intended symmetry breaking. The resolution is *whose* transformation we mean.
The network is an equivariant function of **all** its 4-vector inputs, references
included:

    f(Λx₁, …, Λx_N, Λe_t, Λx_jet) = Λ · f(x₁, …, x_N, e_t, x_jet)   for all Λ.

Invariants are invariant under this *joint* transformation: ⟨Λx_i, Λe_t⟩ = ⟨x_i, e_t⟩.
The architecture theorem (invariant coefficients × input vectors) still applies, so the
function is exactly Lorentz-equivariant in this sense.

We then *evaluate* at fixed references. As a function of the particles alone, the
network is equivariant only under the **stabilizer subgroup** of the references:
fixing e_t leaves SO(3); fixing the jet axis too leaves SO(2) about the axis — which is
the true (approximate) symmetry of relative-coordinate jet data. The symmetry is broken
*exactly* down to the data's symmetry, no further, and through inputs rather than
architecture.

Analogy: a CNN is translation-equivariant; feeding it absolute-position channels breaks
that in a controlled way while the conv weights still share across space. Here we keep
everything equivariance buys — weight sharing across the group orbit, geometric
inductive bias, the guarantee that the *only* frame information available is what we
explicitly injected — while gaining the ability to represent frame-dependent structure.
Bonus: to generate lab-frame jets at arbitrary orientation, pass the actual jet
4-momentum as the reference; the same weights generalize across orientations for free.

## Second, independent expressiveness gap: no per-particle hidden state

LorentzNet carries per-node scalar features h_i updated every layer; LEFT-JeN dropped
them. Messages m_ij are functions of only two pairwise scalars plus global/time
embeddings — the network can barely distinguish particles and never sees E_i, m_i², or
axis alignment as scalars.

This does **not** require per-particle dataset attributes (JetNet has none beyond
momenta, unlike JetClass). h_i is a hidden state; seed it from quantities derived from
the momenta themselves: ψ(m_i²) = ψ(⟨x_i, x_i⟩), and — once reference vectors exist —
ψ(E_i), ψ(⟨x_i, x_jet⟩). Even a learned constant seed works; the value is in having a
per-particle channel that accumulates information across layers at all.

## On the backbone choice: LorentzNet-style is sufficient (with conditions)

Villar et al. 2021 ("Scalars are universal"): any Lorentz-equivariant vector-valued
function of 4-vectors can be written as a sum of the input vectors weighted by
functions of pairwise invariants. That is exactly the LorentzNet/LEFT-JeN
parametrization. With per-node scalars, pairwise invariants, and reference vectors,
**this family can represent anything L-GATr can on point-cloud inputs.** L-GATr's
multivector channels matter for higher-grade inputs/outputs; for generating 4-momenta
they are plausibly dead weight.

Honest caveats:
1. L-GATr's real advantage is optimization/scaling, not expressiveness; the gap grows
   with data/params. At JetNet scale we should be fine; JetClass scale is riskier.
2. Steal attention: replace the sigmoid-gated message sum with softmax attention whose
   logits are functions of the pairwise invariants and node scalars (invariant logits ⇒
   equivariant output). Plus FiLM/adaLN conditioning of t and g per node, instead of
   concatenating three 128-dim embeddings into every pairwise message (the current
   memory hog forcing physical batch 16).
3. To claim superseding L-GATr, we need head-to-head numbers on their generative
   benchmark (public code exists).

## Riemannian flow matching on mass shells

The set of valid 4-momenta of a particle with mass m — the upper mass shell
{p : p² = m², E > 0} — is the hyperboloid model of hyperbolic 3-space H³ with curvature
−1/m², with metric induced from Minkowski space. The restricted Lorentz group **is**
its isometry group. This is the physically canonical hyperbolic space; the
Poincaré-ball-on-raw-coordinates branch is a chart of the same manifold applied to the
wrong object (arbitrary scaled coordinates rather than on-shell momenta), and it breaks
Lorentz structure.

A jet is a point on the product manifold (H³)^N. RFM (Chen & Lipman 2024) on H³ is
exact and closed-form:

- Distance:       d(p, q) = m · arccosh(⟨p, q⟩ / m²)
- Tangent space:  T_p = {v : ⟨v, p⟩ = 0}; loss norm ‖v‖²_g = −⟨v, v⟩ (positive-definite there)
- Log map:        log_p(q) = d(p, q) · w/‖w‖,  w = q − (⟨p, q⟩/m²)·p
- Exp map:        exp_p(v) = cosh(‖v‖/m)·p + m·sinh(‖v‖/m)·v/‖v‖
- Interpolant:    y_t = exp_{y₀}(t · log_{y₀}(y₁));  target u_t = log_{y_t}(y₁)/(1−t)

Why it is powerful:

1. **On-shell by construction.** Every point on every trajectory has E > 0 and exact
   mass m — negative energies and spacelike outputs are structurally impossible, at
   every intermediate ODE step. (The "polar coordinates have convex support" instinct,
   done coordinate-free and Lorentz-compatibly: geodesic convexity of H³.)
2. **Equivariance from the geometry.** Interpolant, target field, exp/log, and sampler
   all commute with Lorentz transformations before the network enters.
3. **Dimension 4 → 3 per particle** — only physical degrees of freedom flow.
4. **Invariant OT/ICP.** Pairing with geodesic distance makes the optimal coupling
   Lorentz-invariant (the current Euclidean ICP cost is frame-dependent).

Massless particles: JetNet particles are effectively massless (E = pT·cosh η), so lift
with a fixed regulator mass: E = √(m² + |p⃗|²). Lossless in practice (JetNet discards
particle masses). m in scaled units is an ablation knob (~0.1–1); small m → high
curvature → numerical stiffness (use stable arccosh near 1, log-domain formulas);
large m → distorted kinematics.

Recipe: lift data and prior to the shell; model eats on-shell y_t (ideal input for the
invariant backbone), outputs a 4-vector, **project to the tangent space**
v → v − (⟨v, y_t⟩/m²)·y_t (itself equivariant), masked Riemannian MSE against u_t.
Sampling: geodesic Euler steps y ← exp_y(v·dt). CFG unchanged (tangent vectors at the
same point form a vector space). Padding: park masked particles at (m, 0, 0, 0).
The existing `hyperbolic_loss`/`pushforward` scaffolding maps onto this nearly
one-to-one — swap Poincaré-ball-on-coordinates maps for hyperboloid-on-momenta maps.

What geometry does *not* fix: the symmetry mismatch above. The data is still not
invariant, so reference vectors are still required. The three pieces are complementary:

> **(i)** minimal universal LorentzNet-style backbone with invariant-logit attention,
> **(ii)** flow matching on the physically canonical manifold — the mass shell — giving
> on-shell, positive-energy samples and equivariance by construction,
> **(iii)** exact symmetry breaking via reference 4-vectors matching the true SO(2)
> symmetry of relative-coordinate jets.
> Simpler than L-GATr, more physical, and (to be shown) better.

## FLOPs / data are not the constraints

- Data: ~850k JetNet jets supports models 10–30× the current ~1M params; equivariance
  removes the need for augmentation.
- Compute: current bottleneck is *memory* (B, N, N, ~386) pairwise-concat tensors, not
  FLOPs (~1 GFLOP/jet forward; a full run ≈ a GPU-day). Fixing conditioning unlocks
  physical batch 128–256 and ~10× model scale within budget.
- Closed-form exp/log on the shell adds negligible overhead.
