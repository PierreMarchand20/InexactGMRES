# Inexact GMRES with relaxed H-matrix products: ideas and results so far

*Status on 2026-11-17. Package `InexactGMRES`, HMatrices fork `PierreMarchand20/HMatrices.jl`, branch `inexact-hmatrix`.*

## 1. Setting

We solve boundary integral equations with GMRES, using an H-matrix whose
matrix-vector product can be truncated at a tolerance η (`HMatrices.ITerm`).

- **η.** In each low-rank block, the product keeps the ACA terms up to the
  last one with `rel_er > η`, plus one. η is a per-block relative tolerance.
    - Dense (near-field) blocks are always applied exactly.
    - η = 0 is the full H-matrix product.
    - η = 1 is the coarsest product allowed: about one term per low-rank block.
- **Test problems.** Dirichlet Helmholtz scattering, CFIE `I/2 + D − ikS`, or
  its direct (transposed) form.
    - Disk: no trapping.
    - Elliptic cavity, and a bigger, more closed cavity: strong trapping, with
    quasimodes of the inner ellipse.
    - Unit sphere (3D): no trapping.
    - 3D elliptic cavities (prolate spheroidal shells with an opening):
    trapping in 3D.
- **Measures.** Unless stated otherwise, tol = 1e-6 and npplo = 10.
    - **Work:** all products of a solve, each weighted by the share of H-matrix
    entries it uses, divided by the work of exact GMRES. It is independent of
    the implementation and does not count orthogonalization or other overhead.
    - **Speed-up:** exact time divided by relaxed time.

## 2. Relaxation strategy

Simoncini–Szyld (2003): if ‖E_k v_k‖ ≤ ℓ · tol / res_{k−1} with
ℓ = σ_m(H_m)/m, the gap between the true and the computed residual stays
below tol. Implemented as `η_k = min(scale · ℓ · tol / res_{k−1}, 1)`.

- **σ/m:** ℓ = σ_m/m, the safe value. It needs σ_m from an exact solve, so it
  is not practical.
- **ℓ = 1:** practical, but not covered by the theory.

### Residual gap bounds (diagnostics)

- **Triangle-inequality bound (4.4).** 20–100× pessimistic:
    - 5–9× (about √m) comes from the triangle inequality itself;
    - 3–6× comes from using η in place of ‖E v‖.
- **Root-sum-of-squares.** Not a bound: the errors are correlated, since all
  products truncate the same H-matrix.
- **Decade ladder.** An Abel-summation bound on a decade ladder of η,
  ‖gap‖ ≤ Σ_ℓ ‖Δ_ℓ‖ ‖y_tail‖, about 3× tighter than (4.4).

## 3. Making relaxation safe: true-residual check and restarts

When the projected residual reaches tol, `igmres` computes the true residual
with one accurate product.

- If the true residual is above tol, the solve restarts from the current
  iterate with that exact residual.
- All later η are then multiplied by `0.5 · tol‖b‖/‖r‖`.

This is a form of iterative refinement. It makes ℓ = 1 safe in practice; the
price is one or more restarts.

## 4. Recycling (GCRO-DR)

At each restart, the 10 harmonic Ritz vectors U with the smallest harmonic
Ritz values are kept, and C = A·U is recomputed (`recycle=10`).

- It makes restarted GMRES(50) converge on the bigger cavity.
- With accurate products, one restart costs 11 accurate products: the check
  plus 10 for recycling.

**Relaxed recycling** (option `relaxed_recycling=true`)

- The 10 recycled products use
  `η_U = scale · |θ_min|/‖A‖ · strategy(‖r0‖/‖b‖, tol)`.
- The factor |θ_min|/‖A‖ comes from the normalized U having norm about
  1/|θ_min|, which amplifies the product error in the residual gap.
- Near a resonance η_U falls back to almost accurate products by itself
  (5e-8 on the cavity).
- On the sphere at k = 15, where the check failed only because of a bug since
  fixed (section 13), the work fell from 1.43 to 0.86. On the cavities the gain
  is 0–4%.

## 5. Corrected operator

`Inti.adaptive_correction` is applied to the S and D/D′ kernels.

- Without it, the Nyström matrix has an O(1) error at high k, from the −ikS
  term. The uncorrected results were meaningless and have been dropped.
- With it, iterations drop by 12–15% and the cavity resonances become visible
  (σ_m = 4.1e-7 at k = 53.935).

## 6. Deflation with quasimodes (cavities)

The deflation space is made of normal traces of Mathieu modes of the inner
ellipse, extended by zero. They are approximate null vectors of the direct
CFIE.

- **Mode selection.** Keep modes with |k − k_q| < 1 and a share of |Θ|² on the
  mouth arc below 1e-3.
    - Bouncing-ball modes are the most trapped (leakage about 4e-7).
    - High-order modes leak (0.16–0.30) and should not be deflated: deflating
    too many far modes is worse than none.
- **Speed.** Selection is now about 10× faster (a closure fix and a coarser
  frequency scan): about 20 s at k = 300.

**Deflated frequency sweeps** (direct operator, 66 values of k in [50, 300],
averages, relaxed recycling; "estimated" is the strategy of section 15)

| case | p | exact its | ℓ=1 its | work σ/m · ℓ=1 · estimated | speed-up σ/m · ℓ=1 · estimated |
|---|---|---|---|---|---|
| cavity | 0 | 95 | 112 (+18%) | 0.93 · 0.99 · 0.94 | 1.04 · 0.82 · 1.01 |
| cavity, deflated | 15 | 68 | 72 (+7%) | 0.94 · 0.97 · 0.95 | 1.01 · 0.88 · 1.00 |
| bigger cavity | 0 | 211 | 266 (+26%) | 0.96 · 0.98 · – | 0.98 · 0.77 · – |
| bigger cavity, deflated | 51 | 84 | 91 (+9%) | 0.94 · 0.98 · – | 0.96 · 0.84 · – |

The two bigger-cavity rows predate the estimated strategy; those 2D reruns
were stopped in favour of the 3D cavities (section 11).

- **Deflation itself is a large gain:** on the bigger cavity, the exact work
  drops from 211 to 84 + 51 = 135 products.
- **Relaxation slows convergence less once trapped modes are deflated:**
  +9% iterations instead of +26%.
- **The work ratio still stays at about 1:**
    - the p accurate deflation products can't be relaxed;
    - the restart (11 products) weighs more on a shorter solve.

## 7. Frequency sweeps without deflation (corrected operator)

Averages over k in [50, 300] (26 values for the disk, 66 for the cavities),
relaxed recycling.

| case | exact its | ℓ=1 its | restarts ℓ=1 · estimated | work σ/m · ℓ=1 · estimated | speed-up σ/m · ℓ=1 · estimated |
|---|---|---|---|---|---|
| disk | 27 | 27 | 0 · 0.08 | 0.90 · 0.83 · 0.89 | 1.04 · 1.09 · 0.91 |
| cavity | 94 | 110 | 1.0 · 0.14 | 0.93 · 0.97 · 0.92 | 1.03 · 0.82 · 1.01 |
| bigger cavity | 211 | 270 | 1.2 · 0.11 | 0.95 · 0.98 · 0.94 | 1.00 · 0.77 · 0.99 |

- **Estimated strategy.** It keeps the exact iteration count (within 0.6%)
  and restarts at only 8–14% of the frequencies. Its work matches the σ/m
  oracle and beats ℓ = 1 on the cavities. On the disk its error estimates
  (about 9% of a product in 2D) eat the gain in time.

- **Disk.** ℓ = 1 saves 13–20% of the work, with no extra iterations and no
  restarts. It improves slowly with k: 0.87 at k = 50, 0.81 at k = 300.
- **Cavities.** Relaxation delays convergence along the trapped modes. On the
  bigger cavity at k = 300, ℓ = 1 needs 607 iterations against 373 for exact,
  a work ratio of 1.25.
    - The extra iterations cost more than the restarts.
    - Overall, relaxation does not pay off.
- **Per-product margin in 2D.**
    - Dense blocks hold 30–50% of the entries, so a product at η = 1 still
    costs 0.35–0.46 of a full one.
    - Compression grows with k (bigger cavity: 4.5 to 16 at η = 0, 8 to 48 at
    η = 1).
- **Fair timing.** Exact and relaxed GMRES now use the same serial product
  (`accurate_mul`). The default threaded `mul!` was 7–14% slower and inflated
  the earlier speed-ups.

## 8. Laplace (bigger cavity, n up to 77k)

- The dense share stays at about a third, so a product at η = 1 costs about
  0.42.
- **Single layer:** work ≈ 0.87–0.92.
- **Second-kind operator −½I + D:** 21 iterations whatever n, work ≈ 0.81–0.85.
- No more margin than Helmholtz in 2D, so a Laplace sweep is not worth
  adding.

## 9. Inexact iterative refinement (negative result)

Outer loop: x += d, where A d ≈ r is solved to η_in with coarse products, and
r = b − A x is computed with one accurate product.

- **Variants tested:** η_in ∈ {1e-1, 1e-2, 1e-3}, with a relaxed or a fixed
  inner product, with and without deflation.
- **Result:** worse than the current scheme in every case.

| case | current ℓ=1 work | best iterative refinement work |
|---|---|---|
| disk k=300 | 0.81 | 0.86 |
| cavity k=100 | 0.94 | 1.12 |
| bigger cavity k=100 | 1.01 | 2.08 |

- **Why it loses:**
    - each outer step rebuilds the Krylov space and has to rediscover the slow
    modes;
    - each outer step reduces the residual much less than η_in.
- **η_in = 1e-1** fails to converge on the bigger cavity.

## 10. 3D: the unit sphere (no trapping)

Same CFIE as the disk, on the unit sphere with second-order triangles
(`sphere_problem`, `src/helmholtz_dirichlet_cfie_sphere_frequency_sweep.jl`).
The scattered field matches the Mie series to 6e-5 at k = 4 and 8e-6 at
k = 18.

**Frequency sweep** (relaxed recycling; "estimated" is the practical safe
strategy of section 15)

| k | n | exact its | work σ/m · ℓ=1 · estimated | speed-up σ/m · ℓ=1 · estimated | compression (η = 0 · η = 1) |
|---|---|---|---|---|---|
| 4 | 3.7k | 9 | 0.82 · 0.72 · 0.76 | 1.09 · 1.20 · 1.05 | 1.8 · 4.7 |
| 8 | 15k | 12 | 0.71 · 0.59 · 0.66 | 1.22 · 1.37 · 1.12 | 4.3 · 18 |
| 12 | 33k | 13 | 0.69 · 0.57 · 0.65 | 1.23 · 1.37 · 1.18 | 7.5 · 36 |
| 16 | 59k | 14 | 0.67 · 0.54 · 0.63 | 1.42 · 1.61 · 1.30 | 11 · 68 |
| 18 | 74k | 14 | 0.68 · 0.55 · 0.65 | 1.25 · 1.38 · 0.99 | 13 · 78 |

- **Why 3D does better:** the dense share falls with k (28% at k = 5, 14%
  at k = 15), since 3D ranks grow faster than in 2D. A product at η = 1 costs
  only 0.17 of a full one at k = 15.
- **No restarts and no extra iterations** for any strategy, at any k.
- **Work:**
    - ℓ = 1 saves about 45% at large k;
    - the safe estimated strategy saves about 35%.
- **Time:** 1.2–1.45× faster for ℓ = 1, 1.05–1.3× for the estimated
  strategy.
    - Truncated products save less in time than in entries, because the dense
      blocks and the per-block overhead weigh more.
    - Each point is a single timing, so the values are noisy (k = 14 is an
      outlier at 1.9–2.4).
- **This is the most promising case so far.**

**Timing pitfalls on large problems** (now fixed in `frequency_sweep`)

- **First pass after assembly.** It is 3× slower (0.62 s against 0.20 s per
  product at k = 18), and was charged to the exact solve, which runs first.
  The sweep now does one accurate product and a garbage collection before
  timing.
- **Compilation.** A warm-up with different closures left compilation in the
  first point. The first k is now run twice instead.

## 11. 3D: the elliptic cavities (trapping)

Prolate spheroidal shells (semi-axes 1, 0.5, 0.5 and 1.3, 0.6, 0.6) cut at
x = cos(7π/10) and x = cos(9π/10), from `data/elliptic_cavity{,_bigger}_3D.geo`
(`cavity_problem` with `geo="elliptic_cavity_3D"` or
`"elliptic_cavity_bigger_3D"`). Same settings as the sphere: 10 elements per
wavelength, second-order triangles, 3 quadrature points per triangle.
OpenCASCADE leaves the inner surface and the rim with inverted normals;
`cavity_problem` reorients them.

**First look at four frequencies** (relaxed recycling, restarts in brackets)

| geometry | k | n | compression (η = 0 · η = 1) | σ_m | exact its | its ℓ=1 · estimated | work σ/m · ℓ=1 · estimated | speed-up σ/m · ℓ=1 · estimated |
|---|---|---|---|---|---|---|---|---|
| cavity | 4 | 3.7k | 1.3 · 3.5 | 0.38 | 23 | 23 · 23 | 0.75 · 0.61 · 0.70 | 1.26 · 1.18 · 1.19 |
| cavity | 8 | 13k | 2.9 · 11 | 0.19 | 38 | 40 (1) · 38 | 0.73 · 0.64 · 0.69 | 1.22 · 1.09 · 1.30 |
| cavity | 12 | 30k | 5.0 · 24 | 0.12 | 46 | 52 (1) · 46 | 0.78 · 0.64 · 0.75 | 1.11 · 1.10 · 1.17 |
| cavity | 16 | 53k | 7.4 · 39 | 0.26 | 51 | 57 (1) · 51 | 0.71 · 0.59 · 0.70 | 1.21 · 1.22 · 1.22 |
| bigger | 4 | 4.4k | 1.4 · 4.8 | 0.24 | 26 | 27 (1) · 26 | 0.72 · 0.72 · 0.65 | 1.18 · 1.05 · 1.22 |
| bigger | 8 | 16k | 3.2 · 15 | 0.22 | 41 | 48 (1) · 41 | 0.72 · 0.64 · 0.68 | 1.18 · 1.07 · 1.19 |
| bigger | 12 | 38k | 5.6 · 26 | 0.098 | 59 | 66 (1) · 59 | 0.74 · 0.58 · 0.69 | 1.17 · 1.15 · 1.19 |
| bigger | 16 | 63k | 8.2 · 45 | 0.20 | 72 | 80 (1) · 72 | 0.71 · 0.54 · 0.71 | 1.25 · 1.31 · 1.17 |

σ/m never restarts and keeps the exact iteration counts.

- **Trapping is visible:** 46–72 iterations at k = 12–16, against 13–14 on
  the sphere, growing with k.
- **Relaxation pays off despite it,** unlike in 2D: the per-product margin
  is large (at k = 16 a product at η = 1 uses 5–6× fewer entries than an
  accurate one).
    - The estimated strategy saves 25–35% of the work, with the exact
    iteration counts and no restarts: speed-up 1.17–1.30.
    - ℓ = 1 saves 36–46% of the work but adds 10–15% iterations and one
    restart. Relaxed recycling makes the restart cheap, but in time it is no
    better than the estimated strategy.
**Fine frequency sweep** (bigger cavity, k = 4–12 in steps of 0.25, 33
values, `src/helmholtz_dirichlet_cfie_cavity_3D_frequency_sweep.jl`)

| | σ/m | ℓ = 1 | estimated |
|---|---|---|---|
| iterations (exact: 43 on average, 26–70) | +0% | +10% | +0.1% |
| restarts (average) | 0 | 1.03 | 0.06 (2 of 33 k) |
| work, average (range) | 0.77 (0.69–0.93) | 0.72 (0.52–1.20) | 0.73 (0.65–0.92) |
| speed-up, average | 1.16 | 1.10 | 1.15 |

- **No strong 3D resonance in this range.** σ_m dips at k ≈ 5.5, 6.5, 7.75,
  8.25, 9, 10.5, 10.75, 11.75. The deepest is 0.0098 at k = 10.5, against
  about 0.25 elsewhere and 1e-7 at a 2D resonance. The cavity leaks too much
  at these frequencies.
- **The gain holds at the dips.** At k = 10.5 exact GMRES needs 65
  iterations: the estimated strategy needs 66 (1 restart, work 0.84), and ℓ = 1
  needs 82 (+26%) but still has work 0.66, thanks to relaxed recycling.
- **The worst points are not dips:** they come from the singular correction
  (k = 4.25, 5.75, 8.5, see below). Without these three points the averages
  are 0.75 · 0.68 · 0.71 for the work and 1.17 · 1.12 · 1.17 for the speed-up.

**Singular correction spilling into low-rank blocks.** At k = 4.25, 5.75 and
8.5 the H-matrix stores more entries than a dense matrix (compression 0.58,
0.66 and 0.93, against 1.4–3.3 around them).

- `Inti.adaptive_correction` picks its correction distance from the largest
  element, with a heuristic that jumps by steps: about 2 element sizes for the
  double layer at k = 4, but 2.9–3.7 at these three k.
- The correction then has 2.3–3.2× more nonzeros per row (420–570 against
  180), and many fall in admissible blocks. `axpy!` adds each of those as a
  rank-one term, which is never truncated.
- It hurts the exact and relaxed solves alike, but it lowers the relaxation
  gain (work about 0.92 at those k).
- **Possible fixes:**
    - a fixed correction distance proportional to the mesh size;
    - adding the correction to the H-matrix as a compressed low-rank update
    instead of rank-one terms;
    - applying it as a separate sparse product.

## 12. Where the time goes (bigger 3D cavity, k = 12)

The work counts H-matrix entries; time is what matters. With n = 37,728, an
accurate product takes 0.14 s and exact GMRES spends 98% of its time in
products, so the whole gap comes from the products themselves and from
igmres's extra machinery.

**A coarse product saves less time than entries**

| η | entries (vs accurate) | time (vs accurate) | dense blocks | low-rank blocks |
|---|---|---|---|---|
| 0 | 1 | 1 (0.141 s) | 0.039 s | 0.106 s |
| 1e-2 | 0.35 | 0.52 | 0.041 s | 0.039 s |
| 1 | 0.21 | 0.36 | 0.039 s | 0.017 s |

- **The dense near field is the floor.** Its 27k small blocks (at most
  50 × 50) hold 16% of the entries but take 28% of the accurate time, and
  relaxation never touches them. At η = 1 they are 75% of the product time.
- Roughly, time ≈ 0.2 + 0.8 × entries.
- **Allocations do not matter.** An allocation-free low-rank product (a shared
  buffer instead of `adjoint(B) * x`) cuts allocations from 4–10 MB to
  1.9 MB per product, but saves only 1–5% of the time. Not worth changing the
  fork.
- **igmres overhead.** Error estimates (13% of a product each in 3D), the σ
  estimate and recycling make up the rest: for the estimated strategy,
  products are about 0.78 of exact GMRES's time against 0.90 in total.

**Leaf size** (`GeometricSplitter(nmax)`, Inti's default 50; same mesh,
admissibility η = 3, ACA tolerance 1e-8)

| nmax | dense share of entries | memory | accurate product | η = 1 product | exact solve | estimated solve |
|---|---|---|---|---|---|---|
| 25 | 7% | 4.28 GiB | 0.242 s | 0.070 s | 10.8 s | 9.09 s |
| **50** | 16% | 3.81 GiB | 0.144 s | 0.054 s | 8.77 s | **7.29 s** |
| 100 | 28% | 3.77 GiB | 0.137 s | 0.066 s | 8.54 s | 7.32 s |
| 200 | 49% | 4.30 GiB | 0.164 s | 0.110 s | 9.93 s | 9.15 s |

- Smaller leaves make coarse products relatively cheaper, but every product
  slower: twice as many blocks.
- Bigger leaves grow the near field that relaxation cannot touch.
- **The default 50 is about optimal,** and relaxation still beats the best
  exact solve (nmax = 100) by 8.54 / 7.29 ≈ 1.17.

## 13. Bugs found in the HMatrices fork (both fixed)

1. **`rel_er` is not monotone.** The truncation now keeps every term up to
   the last one above η, plus one (commit ed25a9f).
2. **Sparse corrections appended to low-rank blocks were dropped.**
   - `axpy!` adds the singular correction to low-rank blocks as rank-one terms,
     without extending `rel_er`, and the truncated product dropped them even at
     η = 0.
   - The error floor was about 2e-6 on the sphere at k = 15, which made the
     true-residual check fail even for σ/m.
   - The fork commit f1053a2 now always applies those terms.
   - It does not affect any 2D case (no such terms there).

## 14. Product error: measured vs η

- **The real error is much smaller than η:** c = ‖E v‖/(η‖v‖) is about
    - 0.02–0.14 on the disk;
    - 0.13–0.25 on the bigger cavity;
    - 0.13–0.18 on the sphere.
- **c is stable:** it mostly depends on η, not on v, and rises toward 0.3–0.8
  only at η close to 1.
- **Cheap a posteriori estimate.** Applying the first dropped ACA term of each
  low-rank block gives ‖E v‖ within 0.8–1.05 of the true value, for about 3%
  (sphere) to 9% (2D) of a full product.

**Calibration** (option `calibrate=true`)

The requested tolerance becomes a target for ‖E v‖ itself. c is measured once
per decade of η (7–9 estimates per solve), and the product uses
η = target / c.

| case | σ/m | σ/m calibrated | ℓ = 1 | ℓ = 1 calibrated |
|---|---|---|---|---|
| sphere k=5 | 0.77 | **0.66** | 0.66 | 1.07 (1 restart) |
| sphere k=10 | 0.71 | **0.62** | 0.59 | 0.87 (1 restart) |
| sphere k=15 | 0.69 | **0.60** | 0.56 | 0.79 (1 restart) |
| disk k=300 | 0.89 | **0.83** | 0.81 | 1.06 (1 restart) |
| bigger cavity k=100 | 0.98 | 0.98 | 1.00 | 1.14 (1 restart) |

- **σ/m calibrated:** 6–11% less work than σ/m, with no extra iterations and
  the Simoncini–Szyld guarantee.
- **ℓ = 1 calibrated:** worse everywhere. ℓ = 1 only works because of the
  hidden margin c ≈ 0.15, which calibration removes.
- **What calibration does and doesn't do:**
    - It maps the allowed error to η accurately.
    - It does not choose the allowed error: that is still ℓ, which theory
    requires to be σ_m/m.

## 15. Estimating σ_m on the fly (option `estimate_sigma=true`)

ℓ_k = σ_min(H̄_{k−1}) / m̂, where m̂ = k plus the iterations still needed at
the convergence rate of the last 10 iterations. The first product of each
cycle is accurate, since there is no estimate yet.

Combined with calibration and a safety factor of 0.25 (passed as ℓ), it
covers three error sources:
- σ_min(H̄_k) overestimates σ_m (interlacing);
- the error estimate is about 0.8 of the true error;
- the bound only guarantees a true residual ≤ 2·tol.

Work relative to exact GMRES, restarts in parentheses:

| case | ℓ = 1 | σ/m known, calibrated | estimated | estimated + calibrated, safety 0.25 |
|---|---|---|---|---|
| sphere k=15 | **0.56** | 0.60 | 0.68 | 0.65 |
| disk k=300 | **0.81** | 0.83 | 0.89 | 0.86 |
| cavity k=53.935 (resonance) | 1.17 (1) | 0.98 | 1.07 (1) | 1.07 (1) |
| cavity k=100 | 0.90 (1) | 0.88 | 0.88 | **0.88** |
| bigger cavity k=100 | 1.00 (1) | 0.98 | 0.97 | **0.97** |
| bigger cavity k=100, deflated | 1.03 (1) | 0.96 | 0.96 | **0.96** |

- **No restarts except at the resonance**, where the near-null mode only
  appears late in H̄_k. Even there it needs 69 iterations against 86 for
  ℓ = 1.
- **It beats ℓ = 1 wherever trapping slows convergence**, and costs about 10%
  more work where nothing is trapped.
- See `notes/relaxation_theory.md`, section 9.

## 16. Open questions and next steps

1. **Track the gap instead of bounding it.** Use the calibrated estimates in
   Σ‖E_j v_j‖|y_j| and tighten η only when needed. It could close the 10%
   between the safe strategy and ℓ = 1.
2. **3D resonances.** The fine sweep found only mild dips (σ_m ≥ 0.01) for
   k ≤ 12. Stronger trapping needs higher k or a narrower mouth; deflation
   with spheroidal quasimodes only if relaxation then breaks down.
3. **Singular correction in 3D** (section 11): keep it out of the low-rank
   blocks.
4. **A deflation-aware σ estimate**, and cheaper restarts after deflation
   (skip recycling when deflation already handles the trapped modes).
5. **Work in time, not entries.** Weight dense entries by their measured cost
   (about 1.8× a low-rank entry) so the work curves match the speed-ups.
6. **More robust timings** (best of several runs) for the paper.
7. **GCR variant with residual replacement** (van der Vorst–Ye style). An
   open idea, not tried.

## Main message so far

- Relaxing the H-matrix product pays off when the per-product margin is
  large, which is the case in 3D:
    - about 20% of the work on the disk;
    - about 45% on the sphere at large k, with a 1.3–1.4× speed-up, and
      increasing with k;
    - about 25–30% on the 3D cavity despite trapping, with a 1.15× speed-up on
      average over k = 4–12, including the mild resonances found there.
- In 2D with trapping, the iterations relaxation adds along the slow modes
  cancel the savings. Deflation removes most of that slowdown, but the
  accurate deflation products then keep the work ratio at about 1.
- **Time gains are smaller than work gains:** the dense near field and the
  per-block overhead keep a coarse product at about 0.3–0.4 of an accurate
  one in 3D. Neither allocations nor the leaf size can change that.
- **Calibration** makes the requested error the actual error.
- **The on-the-fly σ estimate** gives a practical strategy that is almost
  always safe (no restarts), at about 10% more work than ℓ = 1 where nothing
  is trapped, and less where trapping matters.
