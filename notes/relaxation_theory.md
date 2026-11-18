# Relaxed GMRES with an inexact H-matrix product: derivation and calibration

This note derives the Simoncini–Szyld relaxation criterion, describes the
inexact H-matrix product in those terms, and explains what calibration
changes and what it does not. Notation follows `src/InexactGMRES.jl`.

## 1. GMRES with inexact products

We solve $Ax = b$ starting from $x_0$, with $r_0 = b - Ax_0$ and
$\beta = \|r_0\|$.

At iteration $k$, Arnoldi applies a perturbed operator $A + E_k$ to the unit
vector $v_k$. Everything else (orthogonalization, least squares) is exact, so
the Arnoldi relation becomes

$$
(A + E_k)\, v_k = V_{k+1} h_k ,
\qquad\text{i.e.}\qquad
A V_m = V_{m+1}\bar H_m - F_m ,
\qquad F_m = [E_1 v_1, \dots, E_m v_m].
$$

GMRES computes

$$
y_m = \arg\min_y \|\beta e_1 - \bar H_m y\| ,
\qquad
x_m = x_0 + V_m y_m .
$$

**Computed residual.** The algorithm only sees

$$
\tilde r_m = r_0 - V_{m+1}\bar H_m y_m ,
\qquad
\|\tilde r_m\| = \|\beta e_1 - \bar H_m y_m\| .
$$

**True residual.** Using the Arnoldi relation,

$$
r_m = b - A x_m = r_0 - A V_m y_m = \tilde r_m + F_m y_m .
$$

So the **residual gap** is exactly

$$
r_m - \tilde r_m = \sum_{k=1}^{m} [y_m]_k\, E_k v_k .
$$

The size of each product error $E_k v_k$ is weighted by the coefficient
$[y_m]_k$ of the corresponding Krylov vector in the solution.

## 2. The coefficients decay like the residual

**Lemma (Simoncini–Szyld 2003).** For $1 \le k \le m$,

$$
|[y_m]_k| \;\le\; \frac{\|\tilde r_{k-1}\|}{\sigma_m(\bar H_m)} .
$$

**Proof.** Let $\hat y_{k-1}$ be the solution at step $k-1$, padded with zeros
to length $m$. Its components $k, \dots, m$ are zero. Its residual in
coefficient space is $\beta e_1 - \bar H_m \hat y_{k-1}$, whose norm is
$\|\tilde r_{k-1}\|$, because the leading block of $\bar H_m$ is
$\bar H_{k-1}$.

Since $y_m$ minimizes over all $y$,

$$
y_m - \hat y_{k-1} = \bar H_m^{+}\,(\beta e_1 - \bar H_m \hat y_{k-1}) ,
$$

and therefore

$$
|[y_m]_k|
= |[y_m - \hat y_{k-1}]_k|
\le \|\bar H_m^{+}\|\,\|\tilde r_{k-1}\|
= \frac{\|\tilde r_{k-1}\|}{\sigma_m(\bar H_m)} . \qquad\square
$$

Late Krylov vectors enter the solution with small coefficients, so their
products can be less accurate. This is the whole idea of relaxation.

## 3. The relaxation criterion

**Bounding the gap.** Combining the gap formula with the lemma, through the
triangle inequality,

$$
\|r_m - \tilde r_m\|
\;\le\; \sum_{k=1}^{m} |[y_m]_k|\,\|E_k v_k\|
\;\le\; \frac{1}{\sigma_m(\bar H_m)} \sum_{k=1}^{m} \|\tilde r_{k-1}\|\,\|E_k v_k\| .
$$

**The criterion.** To keep the gap below $\varepsilon$, it suffices to
require at every iteration

$$
\boxed{\;
\|E_k v_k\| \;\le\; \frac{\sigma_m(\bar H_m)}{m}\,\frac{\varepsilon}{\|\tilde r_{k-1}\|}
\;}
$$

**In the code's units.** With $\varepsilon = \mathrm{tol}\,\|b\|$ and the
relative residual $\mathrm{res}_{k-1} = \|\tilde r_{k-1}\|/\|b\|$:

$$
\|E_k v_k\| \;\le\; \ell \,\frac{\mathrm{tol}}{\mathrm{res}_{k-1}},
\qquad \ell = \frac{\sigma_m(\bar H_m)}{m} .
$$

This is `rel_to_eps(ℓ, res, tol)`, capped at 1.

**Remarks.**

1. **The guarantee is $\|r_m\| \le 2\,\mathrm{tol}\,\|b\|$, not tol.** It
   comes from $\|r_m\| \le \|\tilde r_m\| + \|r_m - \tilde r_m\|$. GMRES stops
   as soon as $\|\tilde r_m\| < \mathrm{tol}\,\|b\|$, so a gap close to its
   bound makes the true-residual check fail even when the criterion holds.
   This is why the restart multiplies all later tolerances by
   $0.5\,\mathrm{tol}\,\|b\|/\|r\|$.
2. **$\ell = \sigma_m/m$ is not available during the solve.** It involves the
   final $m$ and $\bar H_m$.
   - By interlacing, $\sigma_{\min}(\bar H_k)$ decreases with $k$, so the
     current $\sigma_{\min}(\bar H_k)$ *overestimates* $\sigma_m$. Plugging it
     in is not safe in general.
   - $\ell = 1$ is practical but not covered by the theory.
3. **Where the pessimism comes from.** The bound loses accuracy at two steps:
   - the triangle inequality, which ignores cancellations between the terms
     $[y_m]_k E_k v_k$ (measured 5–9×, about $\sqrt m$);
   - the lemma, which is sharp only when the coefficients are as large as
     allowed.

   The estimate actually tracked in the code, $\sum_k \eta_k |[y_m]_k|$, adds
   a third source: using $\eta_k$ in place of $\|E_k v_k\|$ (section 5).

## 4. The inexact H-matrix product

**Low-rank blocks.** Each one, $A_b \approx \sum_{j=1}^{r_b} u_j w_j^{H}$, is
stored by ACA together with the increments

$$
\mathrm{rel\_er}_j \approx \frac{\|u_j\|\,\|w_j\|}{\|A_b^{(j)}\|_F},
$$

where $A_b^{(j)}$ is the partial approximation after $j$ terms.

**Truncation at $\eta$** (`HMatrices.ITerm`, with the fork's fixes):

- keep the terms up to the last one with $\mathrm{rel\_er}_j > \eta$, plus
  one;
- always keep terms beyond `rel_er`, such as sparse corrections that
  `axpy!` appended as rank-one terms;
- apply dense blocks exactly.

**The product error** is the dropped tail of each low-rank block:

$$
E v \;=\; \sum_{b\ \text{low-rank}} \Big(\sum_{j > k_b(\eta)} u_j w_j^{H}\Big)\, v_{J_b}
\quad (\text{placed in the rows } I_b),
$$

where $k_b(\eta)$ is the number of terms kept in block $b$.

**A priori bound.** Each dropped tail satisfies roughly
$\|E_b\| \lesssim \eta\,\|A_b\|$. Summing over the at most $C_{sp}$ blocks of
each block row and block column (the sparsity constant of the block tree):

$$
\|E v\| \;\le\; C_{sp}\,\max_b \|E_b\|\,\|v\|
\;\lesssim\; C_{sp}\,\eta\,\|A\|\,\|v\| .
$$

This is an upper bound, and a loose one. $C_{sp}$ is a few tens, and grows
from 2D to 3D.

## 5. What the error really is

**Measured ratio** along actual solves:

$$
c(\eta, v) \;=\; \frac{\|E(\eta)\,v\|}{\eta\,\|v\|}
\;\approx\;
\begin{cases}
0.02\text{–}0.14 & \text{disk},\\
0.13\text{–}0.25 & \text{bigger cavity},\\
0.13\text{–}0.18 & \text{sphere},
\end{cases}
$$

rising to 0.3–0.8 only as $\eta \to 1$.

- **c is far below the a priori $C_{sp}\|A\|$.** The increments
  $\mathrm{rel\_er}$ are relative to Frobenius norms of whole blocks, while a
  given $v$ does not align with the dropped terms.
- **c depends mostly on $\eta$, not on $v$.**

**Why ℓ = 1 works without calibration.** Requesting $\eta_k = \mathrm{tol}/\mathrm{res}_{k-1}$
actually produces

$$
\|E_k v_k\| \approx c\, \frac{\mathrm{tol}}{\mathrm{res}_{k-1}},
$$

so the effective factor is $\ell_{\text{eff}} \approx c \approx 0.15$, not 1.
On the sphere at $k = 15$, $\sigma_m/m = 0.75/13 \approx 0.058$:
$\ell_{\text{eff}}$ is only about 2.5× above the safe value. The pessimism of
the triangle inequality absorbs that factor, so the check passes.

## 6. A posteriori estimate: the first dropped term

ACA terms decay roughly geometrically, $\|u_{j+1}\|\|w_{j+1}\| \approx \rho\, \|u_j\|\|w_j\|$.
So the first dropped term dominates the tail:

$$
E_b v_{J_b} \;\approx\; u_{k_b+1}\,(w_{k_b+1}^{H} v_{J_b}),
\qquad
\Big\|\sum_b \cdots\Big\| \approx (1-\rho)\,\|E v\| .
$$

This is `dropped_term_estimate`.

- **Accuracy.** It gives 0.80–1.05 of the true $\|E v\|$ (once 1.56 on the
  bigger cavity), consistent with $\rho \approx 0.2$.
- **Cost.** One rank-one term per low-rank block: about 3% of a full product
  on the sphere at $k = 15$, about 9% in 2D.

## 7. Calibration

The criterion of section 3 constrains $\|E_k v_k\|$, not $\eta$. Calibration
chooses $\eta$ so that the *actual* error meets the target.

1. **Target.** $\varepsilon_k = \mathrm{scale}\cdot \ell\cdot \mathrm{tol}/\mathrm{res}_{k-1}$,
   from the precision strategy.
2. **Ratio.** If the decade of $\eta$ is new, measure
   $\hat c = \mathrm{estimate}(\eta, v_k)/(\eta\|v_k\|)$ and store it for that
   decade. This happens 7–9 times per solve.
3. **Product.** Apply it at $\eta_k = \min(\varepsilon_k/\hat c,\, 1)$.

**Guarantee.** If $\hat c$ is within a factor $\gamma$ of the true ratio, then
$\|E_k v_k\| \le \gamma\,\varepsilon_k$ and the gap is at most
$\gamma\,\mathrm{tol}\,\|b\|$. The estimate tends to underestimate
($\gamma \approx 1/0.8$), so a safety factor of about 0.8 on $\hat c$ would
restore the bound.

**What calibration does not do.** It maps the allowed error to $\eta$
accurately, but the allowed error itself is still set by $\ell$, which the
theory requires to be $\sigma_m/m$. The measurements confirm this (work
relative to exact GMRES):

| strategy | actual $\|E_k v_k\|$ | sphere $k=15$ | disk $k=300$ |
|---|---|---|---|
| $\ell = \sigma_m/m$ | $\approx c\,\sigma_m/m \cdot \mathrm{tol}/\mathrm{res}$ | 0.69 | 0.89 |
| $\ell = \sigma_m/m$, calibrated | $\sigma_m/m \cdot \mathrm{tol}/\mathrm{res}$ | **0.60** | **0.83** |
| $\ell = 1$ | $\approx c \cdot \mathrm{tol}/\mathrm{res}$ | 0.56 | 0.81 |
| $\ell = 1$, calibrated | $\mathrm{tol}/\mathrm{res}$ | 0.79, 1 restart | 1.06, 1 restart |

Calibrated $\sigma_m/m$ is safe and almost as cheap as uncalibrated $\ell=1$.
Calibrated $\ell = 1$ is about $m/\sigma_m \approx 17\times$ too aggressive on
the sphere, so the check fails.

## 8. Recycled and deflated directions

With a recycled or deflation space $U$ and $C = AU$ orthonormal (GCRO), the
iterate is $x = x_0 + U z + V_m y$, with $z = C^{H} r_0 - B y$ and
$B = C^{H} A V_m$.

**The extra gap term.** If $C$ is computed with a perturbed product
$(A + E_U)\,U$, the gap gains a term

$$
E_U\, U z ,\qquad \|E_U U z\| \le \max_{u}\frac{\|E_U u\|}{\|u\|}\;\|U\|\,\|z\| .
$$

**Why it is dangerous.** $U$ is normalized so that $\|Au\| = 1$, and its
columns approximate the smallest harmonic Ritz values $\theta$. So
$\|U\| \approx 1/|\theta_{\min}|$, while $\|z\| \lesssim \|r_0\|$, the
residual at the restart.

**The recycling criterion.** Keeping this term within the same budget gives

$$
\frac{\|E_U u\|}{\|u\|} \;\le\; |\theta_{\min}|\;\ell\,\frac{\mathrm{tol}}{\|r_0\|/\|b\|}.
$$

- **Without calibration:** $\eta_U = \text{target}/\|A\|$, using the a priori
  $\|E u\| \lesssim \eta\|A\|\|u\|$.
- **With calibration:** $\eta_U = \text{target}/\hat c$.
- **Near a resonance:** $|\theta_{\min}|$ is tiny, so $\eta_U \to 0$ and the
  rule falls back to accurate products. The trapped directions need them.

The deflation space is treated the same way, with accurate products.

## 9. Estimating $\ell$ on the fly

A practical *and* nearly safe strategy needs $\sigma_m(\bar H_m)/m$ during the
solve. Option `estimate_sigma=true` (`sigma_over_m` in `src/utils.jl`) uses,
at iteration $k$ of a cycle,

$$
\hat\ell_k = \frac{\sigma_{\min}(\bar H_{k-1})}{\hat m},
\qquad
\hat m = k + \frac{\log(\mathrm{tol}/\mathrm{res}_{k-1})}{\log \rho},
$$

where $\rho$ is the mean convergence rate over the last 10 iterations.

**The first product of a cycle is accurate.** No $\bar H$ is available yet,
and $[y_m]_1$ is typically the largest coefficient. With $\ell = 1$, this
single term alone can exceed the whole budget: about $\mathrm{tol}/\sigma_m$,
with $\sigma_m = 0.75$ on the sphere.

**Three sources of error remain:**

1. **$\sigma_{\min}(\bar H_k)$ overestimates $\sigma_m$.** By interlacing it
   decreases with $k$, so near a resonance the near-null direction only
   appears late.
2. **$\hat m$ is a prediction**, based on the recent convergence rate.
3. **The error estimate is about 0.8 of the true error** (section 6), and the
   guarantee is only $\|r_m\| \le 2\,\mathrm{tol}\,\|b\|$ (section 3).

Hence a **safety factor**: `precision_strategy = rel_to_eps(0.25, ·, ·)`
together with `estimate_sigma=true, calibrate=true`.

**Measured** (tol = 1e-6, relaxed recycling, work relative to exact GMRES,
restarts in parentheses):

| case | $\sigma_m/m$ | $\ell=1$ | $\sigma_m/m$ known, calibrated | estimated | estimated + calibrated, safety 0.25 |
|---|---|---|---|---|---|
| sphere $k=5$ | 0.055 | 0.66 | 0.66 | 0.76 | 0.71 |
| sphere $k=15$ | 0.058 | 0.56 | 0.60 | 0.68 | 0.65 |
| disk $k=300$ | 0.029 | 0.81 | 0.83 | 0.89 | 0.86 |
| cavity $k=53.935$ (resonance) | 6e-6 | 1.17 (1) | 0.98 | 1.07 (1) | 1.07 (1) |
| cavity $k=100$ | 0.0013 | 0.90 (1) | 0.88 | 0.88 | 0.88 |
| bigger cavity $k=100$ | 8e-5 | 1.00 (1) | 0.98 | 0.97 | 0.97 |
| bigger cavity $k=100$, deflated | 1.4e-4 | 1.03 (1) | 0.96 | 0.96 | 0.96 |

**Takeaways.**

- **Safety 0.25 avoids restarts in all cases but the resonance**, where the
  near-null mode is found late. Even there it needs 69 iterations against 86
  for $\ell = 1$.
- **It beats $\ell = 1$ wherever trapping slows convergence.**
- **It costs about 10% more work than $\ell = 1$ on the sphere and the disk.**
  There, $\ell = 1$ benefits from the hidden margin $c$ without paying for
  it.

**Other directions.**

- **A deflation-aware estimate of $\sigma_m$.** With quasimodes deflated, the
  remaining spectrum is better separated, which should make
  $\sigma_{\min}(\bar H_k)$ settle earlier.
- **Track the gap instead of bounding it.** Use the calibrated estimates
  $\hat c\,\eta_j$ for $\|E_j v_j\|$ in $\sum_j \|E_j v_j\|\,|[y_k]_j|$ at
  every step, and tighten the next $\eta$ when the running total approaches
  the budget. This is not an a priori guarantee either: the coefficients
  $[y_k]_j$ still change in later iterations.
