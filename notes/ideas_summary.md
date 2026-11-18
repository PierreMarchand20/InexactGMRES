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
    - **Speed-up:** exact GMRES's solve time divided by the relaxed solve's,
    on the same assembled H-matrix with the same serial product, for one
    right-hand side. It leaves out meshing and assembly, which dominate a
    single solve: on the bigger 3D cavity at k = 12, assembly takes about
    60 s and exact GMRES about 8.5 s, so a 1.2× speed-up saves about 1.4 s,
    about 2% of the total. The gain matters when the assembled matrix is
    reused: with 100 right-hand sides (incidence directions) the solves take
    about 850 s, and 1.2× saves about 18% of the total.

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

**3D** (scratch study `laplace3d.jl`): exterior Dirichlet, u = D[σ] + S[σ],
operator ½I + D + S, data G(x − x0) with x0 inside, so the exact solution is
known (field error 1.8e-5 on the sphere, 6.8e-4 on the cavity). Same meshes
as the Helmholtz runs at k = 4, 8, 12, 16 (n = 4k to 63k).

| geometry | exact its | compression η = 0 · η = 1 (finest) | work σ/m · ℓ=1 · estimated | speed-up σ/m · ℓ=1 · estimated |
|---|---|---|---|---|
| sphere | 4 | 17 · 68 | 0.97 → 0.85 · 0.90 → 0.76 · 0.95 → 0.83 | 0.91–1.01 · 0.96–1.11 · 0.72–0.78 |
| bigger cavity | 12 | 9.0 · 46 | 0.73 → 0.67 · 0.57 → 0.50 · 0.62 → 0.56 | 1.12–1.28 · 1.24–1.58 · 1.08–1.28 |

(arrows: coarsest to finest mesh; no restarts anywhere)

- **The per-product margin is as large as for Helmholtz** (compression at
  η = 1 about 46–68), contrary to what the small Laplace ranks suggested.
- **What limits the gain is the iteration count.** On the sphere GMRES
  converges in 4 iterations: the first product (accurate for the estimated
  strategy) and the final true-residual check weigh a lot, and the residual
  falls so fast that few products can be coarse. The estimated strategy is
  even slower than exact GMRES there, its error estimates being pure
  overhead.
- On the cavity (12 iterations) the gain is like Helmholtz on convex bodies:
  44–50% of the work with ℓ = 1, 1.5× faster at the finer meshes.
- **Rule of thumb:** relaxation needs both a large per-product margin and
  enough iterations to spread the budget over.

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
The scattered field matches the Mie series to 1.4e-4 at k = 4 and 1.0e-5 at
k = 18 (with the fixed correction distance of section 11; 6e-5 and 8e-6 with
Inti's default).

**Frequency sweep** (relaxed recycling; "estimated" is the practical safe
strategy of section 15)

| k | n | exact its | work σ/m · ℓ=1 · estimated | speed-up σ/m · ℓ=1 · estimated | compression (η = 0 · η = 1) |
|---|---|---|---|---|---|
| 4 | 3.7k | 9 | 0.82 · 0.72 · 0.76 | 1.09 · 1.20 · 1.05 | 1.8 · 4.7 |
| 6 | 8.4k | 11 | 0.74 · 0.63 · 0.69 | 1.32 · 1.46 · 1.19 | 3.0 · 10 |
| 8 | 15k | 12 | 0.70 · 0.58 · 0.65 | 1.21 · 1.40 · 1.15 | 4.3 · 18 |
| 10 | 23k | 12 | 0.71 · 0.59 · 0.66 | 1.15 · 1.36 · 1.18 | 5.9 · 27 |
| 12 | 33k | 13 | 0.69 · 0.57 · 0.65 | 1.12 · 1.60 · 1.58 | 7.5 · 36 |
| 14 | 45k | 13 | 0.69 · 0.56 · 0.65 | 1.36 · 1.85 · 1.46 | 9.4 · 51 |
| 16 | 59k | 14 | 0.67 · 0.54 · 0.63 | 1.36 · 1.55 · 1.33 | 11 · 68 |
| 18 | 74k | 14 | 0.68 · 0.55 · 0.65 | 1.20 · 1.38 · 1.18 | 13 · 78 |
| average | | | 0.71 · 0.59 · 0.67 | 1.23 · 1.48 · 1.26 | |

- **Why 3D does better:** the dense share falls with k (28% at k = 5, 14%
  at k = 15), since 3D ranks grow faster than in 2D. A product at η = 1 costs
  only 0.17 of a full one at k = 15.
- **No restarts and no extra iterations** for any strategy, at any k.
- **Work:**
    - ℓ = 1 saves about 45% at large k;
    - the safe estimated strategy saves about 35%.
- **Time:** 1.48× faster on average for ℓ = 1, 1.26× for the estimated
  strategy. About 85% of the saved work becomes saved time (speed-up × work
  ≈ 0.84–0.87): truncated products save less in time than in entries,
  because the dense blocks and the per-block overhead weigh more (section 12).
    - The largest problems (k = 12–14) remain the noisiest points.
- **This is the most promising case so far.**

**Timing pitfalls** (now fixed in `frequency_sweep`)

- **First pass after assembly.** It is 3× slower (0.62 s against 0.20 s per
  product at k = 18), and was charged to the exact solve, which runs first.
  The sweep now does one accurate product and a garbage collection before
  timing.
- **Compilation.** A warm-up with different closures left compilation in the
  first point. The first k is now run twice instead.
- **Single timings are not enough.** Other jobs on the machine (a CI run),
  or a slow spell such as a change of core on this M1 Pro, gave speed-ups
  from 0.14 to 115. Back-to-back repeats of one solver did not help: they
  share the slow spell, and the estimated strategy, always timed right after
  the heavy untimed runs, came out at 0.6 where a direct test gives 1.2.
  Each solve's time is now its best over 5 rounds in which the four solvers
  are interleaved, with a GC before each solve. Speed-up × work then varies
  by only ±0.03–0.04 across k. All 3D sweeps were rerun this way; the 2D
  sweeps (sections 6–7) still use single timings.

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

σ/m never restarts and keeps the exact iteration counts. These were single
timings, before the correction fix and the interleaved timing; the sweep below
supersedes them.

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
| iterations (exact: 43 on average, 26–70) | +0% | +11% | +0.1% |
| restarts (average) | 0 | 1.03 | 0.03 (only k = 10.5) |
| work, average (range) | 0.73 (0.68–0.85) | 0.64 (0.56–0.73) | 0.68 (0.64–0.84) |
| speed-up, average (range) | 1.20 (1.07–1.37) | 1.16 (0.99–1.27) | 1.22 (1.02–1.36) |

All with the singular correction fix below, and interleaved timings (section
10). Before the fix, k = 4.25, 5.75 and 8.5 had work about 0.92 for σ/m and
the estimated strategy, and up to 1.20 for ℓ = 1.

- **No strong 3D resonance in this range.** σ_m dips below 0.1 at k ≈ 5.5,
  6.5, 7.75, 8.25, 9–9.25, 10.5, 10.75, 11.5–12. The deepest is 0.0098 at k = 10.5, against
  about 0.25 elsewhere and 1e-7 at a 2D resonance. The cavity leaks too much
  at these frequencies.
- **The gain holds at the dips.** At k = 10.5 exact GMRES needs 65
  iterations: the estimated strategy needs 66 (1 restart, work 0.84), and ℓ = 1
  needs 82 (+26%) but still has work 0.66, thanks to relaxed recycling.
- **The estimated strategy** restarts only at k = 10.5, saves about 30% of
  the work, and is the fastest on average: ℓ = 1 loses its larger work
  savings to the restart it makes at every k.

**Control: the closed spheroid** (`data/spheroid_3D.geo`, the outer shape
(1.3, 0.6, 0.6) of the bigger cavity without its interior; convex, no
trapping; k = 4–12 in steps of 0.5,
`src/helmholtz_dirichlet_cfie_spheroid_frequency_sweep.jl`). Averages over the
same 17 k:

| | spheroid | bigger cavity |
|---|---|---|
| exact GMRES iterations | 15 (13–16) | 43 (26–65) |
| ℓ = 1 extra iterations | +0.4% | +12% |
| restarts, ℓ = 1 · estimated | 0.06 · 0 | 1.06 · 0.06 |
| σ_m | 0.49–0.64 | 0.01–0.28 |
| compression at η = 1 | 13.4 | 15.3 |
| work σ/m · ℓ=1 · estimated | 0.70 · 0.59 · 0.64 | 0.73 · 0.64 · 0.68 |
| speed-up σ/m · ℓ=1 · estimated | 1.27 · 1.42 · 1.27 | 1.21 · 1.16 · 1.22 |

- **The per-product margin is the same** (compression at η = 1 about 14 on
  both): the relaxed product is just as cheap on the cavity.
- **Trapping costs only 3–5 points of work savings,** and 0.05–0.25 of
  speed-up. σ_m is 3–50× smaller
  on the cavity, so the safe strategies use tighter products, and ℓ = 1 adds
  iterations and a restart per solve. Unlike in 2D, relaxation still pays.
- **The savings grow with k on the spheroid,** as on the sphere: estimated
  work 0.74 at k = 4, 0.59 at k = 12; speed-up 1.12 to 1.41.
- **ℓ = 1 is not safe even on a convex body:** one restart at k = 10 (work
  0.84, speed-up 0.91). The estimated strategy never restarts there.

**Sphere cluster** (`data/sphere_cluster_3D.geo`: 3 × 3 × 2 spheres of radius
0.25 with gaps of 0.1, CFIE; k = 4–14, n = 4k to 55k,
`src/helmholtz_dirichlet_cfie_sphere_cluster_frequency_sweep.jl`)

| | σ/m | ℓ = 1 | estimated |
|---|---|---|---|
| iterations (exact: 22–41) | +0% | +8% | +0.3% |
| restarts | 0 | 1 at every k (2 at k = 14) | 1 (k = 8) |
| work, average | 0.71 | 0.67 | 0.69 |
| speed-up, average | 1.22 | 1.12 | 1.20 |

- **Multiple scattering acts like mild trapping:** iterations grow from 22 to
  41 with k, and σ_m dips to 0.034 at k = 9. Results are close to the
  cavity's.
- **ℓ = 1 restarts every time** and ends up the slowest.

**Disk screen** (`screen_problem`: the unit disk in z = 0, an open surface, so
a first-kind single-layer equation S σ = −uᵢ, oblique incidence; k = 4–32,
n = 942 to 57k, `src/helmholtz_dirichlet_sl_screen_frequency_sweep.jl`)

| | σ/m | ℓ = 1 | estimated |
|---|---|---|---|
| iterations (exact: 27 → 76, growing with n) | +0% | +10% | +0% |
| restarts | 0 | 1 at every k | 1 (k = 16) |
| work, average (k ≥ 20) | 0.89 (0.88) | 0.67 (0.55) | 0.73 (0.69) |
| speed-up, average (k ≥ 20) | 1.07 (1.07) | 1.21 (1.33) | 1.19 (1.22) |

- **σ_m is small and shrinks with n** (0.0076 to 0.001), so the σ/m oracle
  asks for nearly accurate products and hardly gains.
- **The estimated strategy does far better than the oracle** (work 0.69
  against 0.88 at large k), restarts once, and its true residual stays at or
  below tol. The Simoncini–Szyld bound is very pessimistic for this
  first-kind operator, and the estimated σ_min(H̄_k), which overestimates
  σ_m, uses part of that slack safely.
- **ℓ = 1** saves the most work but restarts every time: 1.3–1.44× at the
  largest meshes.
- No reference solution yet (no simple closed form for the disk screen).

**Field cuts** (`src/helmholtz_dirichlet_cfie_3D_field_cuts.jl`, total field
in the plane z = 0, which contains the incident direction and the cavity's
axis; `field_cut_<geometry>_k<k>.png`)

- Outside, the sphere, the spheroid and the cavity look alike: standing waves
  on the lit side, a bright band along the lower flank, a shadow.
- The cavity differs near and inside the mouth. At k = 8 the interior field
  is weak and disordered (|u| ≈ 0.2–0.4, 41 iterations). At k = 10.5, its
  deepest σ_m dip, an axial mode with 5 lobes appears (|u| up to 0.8, 65
  iterations).
- Even there the interior field stays below the incident amplitude: k = 10.5
  is near a cavity resonance, not on it. With a mouth this small the
  resonances must be very narrow, so the 0.25 grid steps over them.

**Singular correction spilling into low-rank blocks.** At k = 4.25, 5.75 and
8.5 the H-matrix stored more entries than a dense matrix (compression 0.58,
0.66 and 0.93, against 1.4–3.3 around them). It is an assembly artefact that
hurts the exact and relaxed solves alike.

- `Inti.adaptive_correction` corrects each target against the elements within
  `maxdist`. By default `maxdist` is counted in radii of the largest element,
  an outlier whose size varies from mesh to mesh: 1.95 mesh sizes at k = 4,
  but 2.9–3.7 at these three k.
- The correction then has 2–3× more nonzeros per row, many in admissible
  blocks, where `axpy!` adds each one as a rank-one term that is never
  truncated.
- **Fix:** the 3D problems pass `maxdist = 2·meshsize`. Compression at the
  three k becomes 1.52, 2.11, 3.45. The scattered field does not change (7e-4
  against a mesh twice as fine, on the cavity, for any `maxdist` from 1.5 to
  4 mesh sizes).
- The sphere sweep (section 10) was rerun with the fix too; Inti's default
  picked 1.65–1.84 mesh sizes there, without spill-over, so only its Mie
  error changed slightly.
- **Side notes on Inti**, to report upstream at some point: its distance scan
  tests only the +x, +y, +z directions and keeps the last one; and passing
  `rtol` without `atol` sets `atol = 0`, which makes the adaptive quadrature
  stall on vanishing double-layer integrals.

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
3. **Report the Inti issues upstream** (section 11), outside the thesis
   work.
4. **A deflation-aware σ estimate**, and cheaper restarts after deflation
   (skip recycling when deflation already handles the trapped modes).
5. **Work in time, not entries.** Weight dense entries by their measured cost
   (about 1.8× a low-rank entry) so the work curves match the speed-ups.
6. **Retime the 2D sweeps** with the interleaved best-of-5 timing (done for
   all 3D sweeps, section 10).
7. **GCR variant with residual replacement** (van der Vorst–Ye style). An
   open idea, not tried.

## Main message so far

- Relaxing the H-matrix product pays off when the per-product margin is
  large, which is the case in 3D:
    - about 20% of the work on the disk;
    - about 45% on the sphere at large k with ℓ = 1 (1.48× speed-up on
      average), 33% with the estimated strategy (1.26×);
    - about 36% on the closed spheroid (k = 4–12), increasing with k, with a
      1.27× speed-up for the estimated strategy;
    - about 32% on the 3D cavity despite trapping, with a 1.22× speed-up on
      average over k = 4–12, including the mild resonances found there:
      trapping costs only 3–5 points against the spheroid.
- In 2D with trapping, the iterations relaxation adds along the slow modes
  cancel the savings. Deflation removes most of that slowdown, but the
  accurate deflation products then keep the work ratio at about 1.
- **More geometries in 3D give the same picture:** a sphere cluster
  (multiple scattering) saves about 30% of the work, 1.2× faster; a disk
  screen (first-kind operator, up to 76 iterations) about 30% with the
  estimated strategy, which beats the σ/m oracle there.
- **What the speed-up is a speed-up of:** the iterative solve only, for one
  right-hand side. With assembly included, a 1.2× solve speed-up is about 2%
  of a single solve's total time; it pays when the matrix is reused for many
  right-hand sides (about 18% of the total with 100).
- **Time gains are smaller than work gains:** the dense near field and the
  per-block overhead keep a coarse product at about 0.3–0.4 of an accurate
  one in 3D. Neither allocations nor the leaf size can change that.
- **Calibration** makes the requested error the actual error.
- **The on-the-fly σ estimate** gives a practical strategy that is almost
  always safe (no restarts), at about 10% more work than ℓ = 1 where nothing
  is trapped, and less where trapping matters.
