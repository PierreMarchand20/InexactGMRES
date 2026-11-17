using Inti
using Gmsh
using HMatrices
using LinearAlgebra
using SparseArrays
using StaticArrays
using BenchmarkTools
using DataFrames
using CSV
using Plots
using InexactGMRES

## Problem builders: Helmholtz Dirichlet CFIE (D - ik*S), disk and cavity,
## both parametrized by wavenumber k alone (meshsize = 2π/(npplo*k) keeps
## points-per-wavelength fixed as k varies, matching data/elliptic_cavity_2D.geo's
## own npplo/k DefineConstant convention).

"""
    assemble_cfie(Q, k; ε=1e-8, θ=π/4)

Assembles the Dirichlet CFIE (D - ik*S) H-matrix and RHS on quadrature `Q`
(2D or 3D), for a plane wave incident at angle `θ` (in the xy-plane). Returns a NamedTuple `(; L, g, pde,
uᵢ)` -- `pde` and `uᵢ` (the incident wave) are included since callers doing
field reconstruction (not just solving) need them too.
"""
function assemble_cfie(Q, k; ε=1e-8, θ=π / 4)
    dim = length(first(Q).coords)
    pde = Inti.Helmholtz(; k, dim)
    SL = Inti.SingleLayerKernel(pde)
    DL = Inti.DoubleLayerKernel(pde)
    K = let SL = SL, DL = DL, k = k
        (t, q) -> DL(t, q) - im * k * SL(t, q)
    end
    d = dim == 2 ? SVector(cos(θ), sin(θ)) : SVector(cos(θ), sin(θ), 0.0)
    uᵢ = (x) -> exp(im * k * dot(x, d))
    g = map(q -> -uᵢ(q.coords), Q)
    Lop = Inti.IntegralOperator(K, Q, Q)
    L = Inti.assemble_hmatrix(Lop; rtol=ε)
    Id = sparse((0.5 + 0im) * I, size(L))
    axpy!(1.0, Id, L)
    axpy!(1.0, singular_correction(DL, SL, k, Q), L)
    return (; L, g, pde, uᵢ)
end

"""
    assemble_cfie_direct(Q, k; ε=1e-8, θ=π/4)

Direct Dirichlet CFIE `(I/2 + D' - ik*S) ∂ₙu = ∂ₙuᵢ - ik*uᵢ`, the transpose of
[`assemble_cfie`](@ref)'s operator, whose unknown is the normal trace `∂ₙu` of
the total field. Same return values.
"""
function assemble_cfie_direct(Q, k; ε=1e-8, θ=π / 4)
    dim = length(first(Q).coords)
    pde = Inti.Helmholtz(; k, dim)
    SL = Inti.SingleLayerKernel(pde)
    ADL = Inti.AdjointDoubleLayerKernel(pde)
    K = let SL = SL, ADL = ADL, k = k
        (t, q) -> ADL(t, q) - im * k * SL(t, q)
    end
    d = dim == 2 ? SVector(cos(θ), sin(θ)) : SVector(cos(θ), sin(θ), 0.0)
    uᵢ = (x) -> exp(im * k * dot(x, d))
    g = map(q -> (im * k * dot(d, q.normal) - im * k) * uᵢ(q.coords), Q)
    L = Inti.assemble_hmatrix(Inti.IntegralOperator(K, Q, Q); rtol=ε)
    axpy!(1.0, sparse((0.5 + 0im) * I, size(L)), L)
    axpy!(1.0, singular_correction(ADL, SL, k, Q), L)
    return (; L, g, pde, uᵢ)
end

"""
    singular_correction(DL, SL, k, Q)

Sparse correction of the singular and nearly singular entries of `DL - ik*SL`
on `Q` (`Inti.adaptive_correction`). Without it, the Nyström matrix just
drops the diagonal, an O(1) error at high k because of the `-ik*SL` term.
"""
function singular_correction(DL, SL, k, Q)
    return Inti.adaptive_correction(Inti.IntegralOperator(DL, Q, Q)) -
           im * k * Inti.adaptive_correction(Inti.IntegralOperator(SL, Q, Q))
end

"""
    disk_problem(k; npplo=10, qorder=4, ε=1e-8, θ=π/4)

Dirichlet CFIE problem on the unit circle, meshed at `meshsize =
2π/(npplo*k)` points-per-wavelength. Returns `(; L, g, pde, uᵢ, Q,
meshsize)`.
"""
function disk_problem(k; npplo=10, qorder=4, ε=1e-8, θ=π / 4)
    meshsize = 2π / (npplo * k)
    circle = Inti.parametric_curve(0.0, 1.0; labels=["circle"]) do s
        return SVector(cos(2π * s[1]), sin(2π * s[1]))
    end
    msh = Inti.meshgen(circle; meshsize)
    Q = Inti.Quadrature(msh; qorder)
    return (; assemble_cfie(Q, k; ε, θ)..., Q, meshsize)
end

"""
    sphere_problem(k; npplo=10, qorder=2, gorder=2, ε=1e-8, θ=π/4)

Dirichlet CFIE problem on the unit sphere, meshed with second-order
triangles of size `meshsize = 2π/(npplo*k)`. Returns `(; L, g, pde, uᵢ, Q,
meshsize)`.
"""
function sphere_problem(k; npplo=10, qorder=2, gorder=2, ε=1e-8, θ=π / 4)
    meshsize = 2π / (npplo * k)
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
    gmsh.model.occ.addSphere(0, 0, 0, 1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(gorder)
    msh = Inti.import_mesh(; dim=3)
    gmsh.finalize()
    Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 3, Inti.entities(msh))
    Q = Inti.Quadrature(view(msh, Inti.boundary(Ω)); qorder)
    return (; assemble_cfie(Q, k; ε, θ)..., Q, meshsize)
end

"""
    cavity_problem(k; geo="elliptic_cavity_2D", npplo=10, qorder=4, gorder=2, ε=1e-8, θ=π/4)

Loads `data/\$geo.geo` (a half-elliptical-shell cavity, parametrized by the
Gmsh DefineConstant's `npplo`/`k`) at the given wavenumber `k`, assembles the
Dirichlet CFIE H-matrix and RHS. `geo` can be swapped for
`"elliptic_cavity_bigger_2D"` to use the more closed variant (shell cut at
x = cos(9π/10) instead of cos(7π/10), so a narrower mouth), and for the 3D
versions `"elliptic_cavity_3D"` and `"elliptic_cavity_bigger_3D"` (prolate
spheroidal shells, meshed with triangles; `qorder` then defaults to 2). `direct=true`
assembles [`assemble_cfie_direct`](@ref) instead. Returns `(; L, g, pde, uᵢ,
Q, meshsize)`.
"""
function cavity_problem(k; geo="elliptic_cavity_2D", npplo=10, qorder=nothing, gorder=2, ε=1e-8, θ=π / 4, direct=false)
    meshsize = 2π / (npplo * k)
    dim = endswith(geo, "3D") ? 3 : 2
    qorder = something(qorder, dim == 3 ? 2 : 4) # 3D: same as sphere_problem
    filename = joinpath(pkgdir(InexactGMRES), "data", "$geo.geo")
    gmsh.initialize()
    gmsh.onelab.set("""[{"type":"number","name":"npplo","values":[$npplo]},{"type":"number","name":"k","values":[$k]}]""")
    gmsh.open(filename)
    gmsh.model.mesh.generate(dim - 1)
    # 3D: after OCC booleans some surfaces are stored with the opposite orientation
    # to the volume's boundary (negative tags), which Inti would keep: reverse them
    if dim == 3
        for (_, vol) in gmsh.model.getEntities(3), (_, s) in gmsh.model.getBoundary([(3, vol)], false, true)
            s < 0 && gmsh.model.mesh.reverse([(2, -s)])
        end
    end
    gmsh.model.mesh.setOrder(gorder)
    msh = Inti.import_mesh(; dim)
    gmsh.finalize()

    ents = Inti.entities(msh)
    Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == dim, ents)
    Γ = Inti.boundary(Ω)
    Γ_msh = view(msh, Γ)
    Q = Inti.Quadrature(Γ_msh; qorder)
    return (; (direct ? assemble_cfie_direct : assemble_cfie)(Q, k; ε, θ)..., Q, meshsize)
end

## Quasimodes of the elliptic cavity: Dirichlet eigenmodes of the inner ellipse
## x²/a² + y²/b² < 1 (a=1, b=0.5 in data/elliptic_cavity*_2D.geo), in elliptic
## coordinates x = c cosh ξ cos θ, y = c sinh ξ sin θ, c = √(a²-b²):
## u = R(ξ)Θ(θ), Θ'' + (α - 2q cos 2θ)Θ = 0, R'' - (α - 2q cosh 2ξ)R = 0, q = (kc/2)².

"""
    mathieu_angular(m, q, kind)

Characteristic value `α` and angular function `Θ` of `ce_m` (`kind=:even`) or
`se_m` (`kind=:odd`), from the symmetric tridiagonal eigenproblem on their
Fourier coefficients.
"""
function mathieu_angular(m, q, kind; vectors=true)
    N = ceil(Int, 2sqrt(q)) + 40
    if kind == :even && iseven(m)
        n = 2 .* (0:N-1); trig = cos; j = m ÷ 2
    elseif kind == :even
        n = 2 .* (0:N-1) .+ 1; trig = cos; j = (m - 1) ÷ 2
    elseif isodd(m)
        n = 2 .* (0:N-1) .+ 1; trig = sin; j = (m - 1) ÷ 2
    else
        n = 2 .* (0:N-1) .+ 2; trig = sin; j = m ÷ 2 - 1
    end
    M = SymTridiagonal(Float64.(n .^ 2), fill(Float64(q), N - 1))
    kind == :even && iseven(m) && (M.ev[1] *= sqrt(2))
    n[1] == 1 && (M.dv[1] += kind == :even ? q : -q)
    vectors || return eigvals(M, j+1:j+1)[1], nothing # α only, O(N)
    F = eigen(M)
    coef = F.vectors[:, j+1]
    kind == :even && iseven(m) && (coef[1] /= sqrt(2))
    # `let`: `n` and `trig` are assigned in branches, so capturing them directly boxes them (~15× slower)
    return F.values[j+1], let n = n, trig = trig; θ -> sum(coef[i] * trig(n[i] * θ) for i in 1:N); end
end

"""
    mathieu_radial(α, q, ξ0, kind; steps=20_000)

`(R(ξ0), R'(ξ0))` for the radial Mathieu function regular on the focal
segment (`R'(0) = 0` for `ce`, `R(0) = 0` for `se`), integrated with RK4.
"""
function mathieu_radial(α, q, ξ0, kind; steps=20_000)
    f(ξ, u) = SVector(u[2], (α - 2q * cosh(2ξ)) * u[1])
    u = kind == :even ? SVector(1.0, 0.0) : SVector(0.0, 1.0)
    h = ξ0 / steps
    for i in 0:steps-1
        ξ = i * h
        k1 = f(ξ, u); k2 = f(ξ + h / 2, u + h / 2 * k1)
        k3 = f(ξ + h / 2, u + h / 2 * k2); k4 = f(ξ + h, u + h * k3)
        u += h / 6 * (k1 + 2k2 + 2k3 + k4)
    end
    return u[1], u[2]
end

"""
    ellipse_dirichlet_frequencies(m, kind, kmin, kmax; a=1.0, b=0.5, dk=0.05)

Wavenumbers in `[kmin, kmax]` of the Dirichlet eigenmodes `R(ξ)ce_m(θ)`
(`kind=:even`) or `R(ξ)se_m(θ)` (`kind=:odd`) of the ellipse.
"""
function ellipse_dirichlet_frequencies(m, kind, kmin, kmax; a=1.0, b=0.5, dk=0.05)
    c = sqrt(a^2 - b^2); ξ0 = acosh(a / c)
    R(k) = (q = (k * c / 2)^2; mathieu_radial(first(mathieu_angular(m, q, kind; vectors=false)), q, ξ0, kind)[1])
    ks = Float64[]
    k0, R0 = kmin, R(kmin)
    for k1 in kmin+dk:dk:kmax
        R1 = R(k1)
        if sign(R1) != sign(R0)
            lo, hi, Rlo = k0, k1, R0
            for _ in 1:50
                mid = (lo + hi) / 2; Rm = R(mid)
                sign(Rm) == sign(Rlo) ? (lo = mid; Rlo = Rm) : (hi = mid)
            end
            push!(ks, (lo + hi) / 2)
        end
        k0, R0 = k1, R1
    end
    return ks
end

"""
    quasimode_density(Q, kq, m, kind; a=1.0, b=0.5, tol=1e-2)

Normal trace `∂ₙu` (along `Q`'s normals) of the eigenmode at wavenumber `kq`,
on the quadrature nodes of the inner wall, extended by zero elsewhere: an
approximate null vector of the direct CFIE near `kq`.
"""
function quasimode_density(Q, kq, m, kind; a=1.0, b=0.5, tol=1e-2)
    c = sqrt(a^2 - b^2); ξ0 = acosh(a / c); q = (kq * c / 2)^2
    α, Θ = mathieu_angular(m, q, kind)
    _, dR = mathieu_radial(α, q, ξ0, kind)
    return map(Q) do node
        x, y = node.coords
        abs(x^2 / a^2 + y^2 / b^2 - 1) > tol && return zero(ComplexF64)
        θ = atan(y / (c * sinh(ξ0)), x / (c * cosh(ξ0)))
        ∂ξX = SVector(c * sinh(ξ0) * cos(θ), c * cosh(ξ0) * sin(θ))
        return complex(dR * Θ(θ) * dot(node.normal, ∂ξX) / dot(∂ξX, ∂ξX))
    end
end

"""
    mouth_share(m, kind, kq; θ_mouth, a=1.0, b=0.5)

Share of `|Θ|²` of the angular Mathieu function (`ce_m` or `se_m` at the
wavenumber `kq`) on the mouth arc `|θ| > θ_mouth` of the inner ellipse: a
free indicator of how much the quasimode leaks through the opening.
"""
function mouth_share(m, kind, kq; θ_mouth, a=1.0, b=0.5)
    c = sqrt(a^2 - b^2)
    _, Θ = mathieu_angular(m, (kq * c / 2)^2, kind)
    θs = range(-π, π; length=4001)[1:end-1]
    w = abs2.(Θ.(θs))
    return sum(w[abs.(θs) .> θ_mouth]) / sum(w)
end

"""
    select_quasimodes(Q, k; θ_mouth, δ=1.0, ε=1e-3, a=1.0, b=0.5)

Deflation space for the direct CFIE at `k`: the normal traces of the Mathieu
quasimodes with `|k - k_q| < δ` and a share below `ε` of `|Θ|²` on the mouth
arc `|θ| > θ_mouth` (`7π/10` for the cavity, `9π/10` for the bigger one).
Returns a matrix, or `nothing` when no mode qualifies.
"""
function select_quasimodes(Q, k; θ_mouth, δ=1.0, ε=1e-3, a=1.0, b=0.5)
    mmax = ceil(Int, k * sqrt(a^2 - b^2)) # angular orders beyond k·c are evanescent near the minor axis
    cols = Vector{ComplexF64}[]
    for kind in (:even, :odd), m in (kind == :even ? (0:mmax) : (1:mmax))
        mouth_share(m, kind, k; θ_mouth, a, b) > 10ε && continue # leaky at k already: skip the scan
        for kq in ellipse_dirichlet_frequencies(m, kind, k - δ, k + δ; a, b)
            mouth_share(m, kind, kq; θ_mouth, a, b) < ε && push!(cols, quasimode_density(Q, kq, m, kind; a, b))
        end
    end
    return isempty(cols) ? nothing : reduce(hcat, cols)
end

# (I - CCᴴ)A, the operator of a Krylov space deflated by C = A*U (orthonormal)
struct Projected{T,M} <: AbstractMatrix{T}
    A::M
    C::Matrix{T}
end
Base.size(P::Projected) = size(P.A)
function LinearAlgebra.mul!(y::AbstractVector, P::Projected, x::AbstractVector, α::Number, β::Number)
    w = InexactGMRES.accurate_mul(P.A, x)
    w .-= P.C * (P.C' * w)
    return iszero(β) ? (y .= α .* w) : (y .= α .* w .+ β .* y)
end

"""
    deflated_exact_gmres(L, g, U; tol)

Exact GMRES deflated by `span(U)`: `C = L*U` (orthonormalized), then GMRES on
`(I - CCᴴ)L` with right-hand side `(I - CCᴴ)g`, which has the same residuals
as `igmres(...; deflation=U)` with exact products. Returns
`(iterations, H_m)`; `U === nothing` falls back to `exact_gmres`.
"""
function deflated_exact_gmres(L, g, U; tol)
    U === nothing && return InexactGMRES.exact_gmres(L, g; tol, return_H=true)[3:4]
    C = Matrix(qr(reduce(hcat, [InexactGMRES.accurate_mul(L, u) for u in eachcol(U)])).Q)
    Pg = g - C * (C' * g)
    return InexactGMRES.exact_gmres(Projected(L, C), Pg; tol=tol * norm(g) / norm(Pg), return_H=true)[3:4]
end

"""
    tol_sweep(builder, k, tols; deflation=nothing)

Build one problem (fixed geometry/frequency, via `builder(k)`) and benchmark
`igmres` vs `exact_gmres` at each tolerance in `tols`. Returns a DataFrame
with columns `tol`, `speedup`, `rel_error`, `igmres_it`, `exact_it`,
`igmres_restarts` (restarts after a failed true-residual check).

`deflation(prob, k)`, when given, returns a deflation space (or `nothing`).
Both solvers are then deflated by it, and their timings include building
`C = L*U`.
"""
function tol_sweep(builder, k, tols; deflation=nothing)
    deflation === nothing || return deflated_tol_sweep(builder, k, tols, deflation)
    (; L, g) = builder(k)
    rows = map(tols) do tol
        bench_igmres = @benchmark igmres($L, $g; tol=$tol)
        bench_exact = @benchmark InexactGMRES.exact_gmres($L, $g; tol=$tol)
        restarts = Int[]
        y, _, it = igmres(L, g; tol, correction_log=restarts)
        _, _, exact_it = InexactGMRES.exact_gmres(L, g; tol)
        rel_error = norm(L * y - g) / norm(g)
        speedup = minimum(bench_exact).time / minimum(bench_igmres).time
        (; tol, speedup, rel_error, igmres_it=it, exact_it, igmres_restarts=length(restarts))
    end
    return DataFrame(rows)
end

function deflated_tol_sweep(builder, k, tols, deflation)
    prob = builder(k)
    (; L, g) = prob
    U = deflation(prob, k)
    rows = map(tols) do tol
        bench_igmres = @benchmark igmres($L, $g; tol=$tol, deflation=$U)
        bench_exact = @benchmark deflated_exact_gmres($L, $g, $U; tol=$tol)
        restarts = Int[]
        y, _, it = igmres(L, g; tol, deflation=U, correction_log=restarts)
        exact_it, _ = deflated_exact_gmres(L, g, U; tol)
        rel_error = norm(L * y - g) / norm(g)
        speedup = minimum(bench_exact).time / minimum(bench_igmres).time
        (; tol, speedup, rel_error, igmres_it=it, exact_it, igmres_restarts=length(restarts), p=U === nothing ? 0 : size(U, 2))
    end
    return DataFrame(rows)
end

"""
    frequency_sweep(builder, ks; tol, bound_factor=1.0, deflation=nothing)

For each `k` in `ks`, build a fresh problem via `builder(k)` and run
`exact_gmres` once plus `igmres` twice -- same two heuristics as
`igmres_precision_study`: the σ(H_m) heuristic (`rel_to_eps(sigma_m/m, res,
t)`) and the constant bound factor heuristic (`rel_to_eps(bound_factor,
res, t)`). `sigma_m`, the smallest singular value of the final Hessenberg
matrix, and the iteration counts for all three solves are candidate
signatures of a nearby scattering resonance.

Timing uses a single `@elapsed` call per solve rather than
`BenchmarkTools` (a full `@benchmark` per point, times 3 solves times
dozens of k's, would add many minutes of pure benchmarking overhead); the
first `k` is run twice so the first sweep point isn't contaminated by JIT
compilation. This makes individual speedup values noisier than
`tol_sweep`'s, but the trend across the whole sweep should still be
visible.

To check igmres actually solves the right problem, each heuristic is also
run once more (untimed) with `track_true_residual=true`, recording the final
true relative residual `‖b - A x‖/‖b‖` and the final gap `‖r - r̃‖/‖b‖`
between the true and the projection residual.

Returns a DataFrame with columns `k`, `exact_it`, `igmres_it_sigma`,
`igmres_it_constant_factor`, `sigma_m`, `speedup_sigma`,
`speedup_constant_factor`, `true_res_sigma`, `true_res_constant_factor`,
`gap_sigma`, `gap_constant_factor`, `restarts_sigma`,
`restarts_constant_factor` (restarts after a failed true-residual check), `size`, `p`,
the solve times `exact_time`, `time_sigma`, `time_constant_factor`, the
H-matrix compression ratio of an accurate product (`compression`) and of
products at `η = 1e-2` and `η = 1` (`compression_1e2`, `compression_1`),
and the work of each relaxed solve relative to the exact one
(`work_ratio_sigma`, `work_ratio_constant_factor`, see [`relaxed_work`](@ref)).

`deflation(prob, k)`, when given, returns a deflation space at each `k` (or
`nothing`; `p` is its dimension). All solves are then deflated by it, the
exact one through [`deflated_exact_gmres`](@ref), whose `H_m` gives
`sigma_m`; the timings include building `C = L*U`.

`igmres_kwargs` are passed to every `igmres` call (e.g.
`(; relaxed_recycling=true)`). With `estimated=true`, a third relaxed solve
uses the practical safe strategy: `σ_m/m` estimated on the fly and
calibrated products (`estimate_sigma=true, calibrate=true`), with a safety
factor `safety` (default 0.25). It adds the columns `igmres_it_estimated`,
`restarts_estimated`, `time_estimated`, `speedup_estimated`,
`true_res_estimated` and `work_ratio_estimated`.
"""
function frequency_sweep(builder, ks; tol, bound_factor=1.0, deflation=nothing, igmres_kwargs=(;), estimated=false, safety=0.25)
    # the first k is run twice and its first row dropped: it compiles the
    # closures of the loop body, so that no timing includes compilation
    rows = map(vcat(first(ks), ks)) do k
        prob = builder(k)
        (; L, g) = prob
        U = deflation === nothing ? nothing : deflation(prob, k)
        # collect the assembly's garbage and touch the whole matrix once: otherwise
        # the first timed solve pays for both (3x slower first pass on the sphere at k = 18)
        InexactGMRES.accurate_mul(L, g)
        GC.gc()
        exact_time = @elapsed ((exact_it, Hmat) = deflated_exact_gmres(L, g, U; tol))
        sigma_m = svd(Hmat).S[end]

        sigma_strategy = (res, t) -> InexactGMRES.rel_to_eps(sigma_m / exact_it, res, t)
        constant_strategy = (res, t) -> InexactGMRES.rel_to_eps(bound_factor, res, t)

        restarts_sigma, restarts_constant = Int[], Int[] # iterations of the restarts after a failed true-residual check
        eta_sigma, eta_constant = Float64[], Float64[]   # matvec tolerances actually used
        GC.gc()
        sigma_time = @elapsed ((_, _, igmres_it_sigma) = igmres(L, g; tol, precision_strategy=sigma_strategy, deflation=U,
            correction_log=restarts_sigma, eta_log=eta_sigma, igmres_kwargs...))
        GC.gc()
        constant_time = @elapsed ((_, _, igmres_it_constant_factor) = igmres(L, g; tol, precision_strategy=constant_strategy, deflation=U,
            correction_log=restarts_constant, eta_log=eta_constant, igmres_kwargs...))
        p = U === nothing ? 0 : size(U, 2)
        compression = InexactGMRES.effective_compression_ratio(L, 0.0)

        # separate untimed runs: track_true_residual adds an exact matvec per
        # iteration, which would distort the speedups above
        (_, _, _, true_res_sigma, gap_sigma) = igmres(L, g; tol, precision_strategy=sigma_strategy, track_true_residual=true, deflation=U, igmres_kwargs...)
        (_, _, _, true_res_constant_factor, gap_constant_factor) = igmres(L, g; tol, precision_strategy=constant_strategy, track_true_residual=true, deflation=U, igmres_kwargs...)

        extra = (;)
        if estimated
            restarts_est, eta_est = Int[], Float64[]
            GC.gc()
            time_est = @elapsed ((x_est, _, it_est) = igmres(L, g; tol, deflation=U, estimate_sigma=true, calibrate=true,
                precision_strategy=(res, t) -> InexactGMRES.rel_to_eps(safety, res, t),
                correction_log=restarts_est, eta_log=eta_est, igmres_kwargs...))
            extra = (; igmres_it_estimated=it_est, restarts_estimated=length(restarts_est), time_estimated=time_est,
                speedup_estimated=exact_time / time_est,
                true_res_estimated=norm(g - InexactGMRES.accurate_mul(L, x_est)) / norm(g),
                work_ratio_estimated=relaxed_work(L, eta_est) / (exact_it + p))
        end

        gnorm = norm(g)
        (; k, exact_it, igmres_it_sigma, igmres_it_constant_factor, sigma_m,
            speedup_sigma=exact_time / sigma_time,
            speedup_constant_factor=exact_time / constant_time,
            true_res_sigma=true_res_sigma[end],
            true_res_constant_factor=true_res_constant_factor[end],
            gap_sigma=gap_sigma[end] / gnorm,
            gap_constant_factor=gap_constant_factor[end] / gnorm,
            restarts_sigma=length(restarts_sigma), restarts_constant_factor=length(restarts_constant),
            size=length(g), p, exact_time, time_sigma=sigma_time, time_constant_factor=constant_time,
            compression, compression_1e2=InexactGMRES.effective_compression_ratio(L, 1e-2),
            compression_1=InexactGMRES.effective_compression_ratio(L, 1.0),
            work_ratio_sigma=relaxed_work(L, eta_sigma) / (exact_it + p),
            work_ratio_constant_factor=relaxed_work(L, eta_constant) / (exact_it + p), extra...)
    end
    return DataFrame(rows[2:end])
end

"""
    relaxed_work(L, etas)

Work of a relaxed `igmres` solve in units of one accurate product, from its
`eta_log` (every product of the solve): each product at tolerance `η` counts
for the share of the H-matrix entries it uses
([`effective_compression_ratio`](@ref)), so accurate products (`η = 0`:
true-residual checks, recycling, deflation space) count for 1, and error
estimates (`NaN`, see [`dropped_term_estimate`](@ref)) for one term per
low-rank block.
"""
function relaxed_work(L, etas)
    full = InexactGMRES.effective_compression_ratio(L, 0.0)
    # error estimate (NaN): one term per low-rank block, relative to the used(0) = length(L)/full entries
    estimate = full / length(L) * sum(sum(size(l)) for l in HMatrices.leaves(L) if HMatrices.isadmissible(l); init=0)
    return sum(η -> isnan(η) ? estimate : full / InexactGMRES.effective_compression_ratio(L, η), etas; init=0.0)
end

"""
    plot_frequency_extras(df, stem, name, suffix, op_label, resonances; tol)

Figures added to every frequency sweep: H-matrix compression ratios, work of
the relaxed solves relative to the exact one, and restarts, against `k`.
Saved as `frequency_sweep_\$(stem)_{compression,work,restarts}\$suffix.png`.
"""
function plot_frequency_extras(df, stem, name, suffix, op_label, resonances; tol)
    vl(p) = vline!(p, resonances; label="bouncing-ball modes", linestyle=:dot, color=:gray, alpha=0.4)
    p5 = plot(df.k, df.compression; label="accurate product", marker=:circle, yaxis=:log)
    plot!(p5, df.k, df.compression_1e2; label="product at η = 1e-2", marker=:circle, linestyle=:dash)
    plot!(p5, df.k, df.compression_1; label="product at η = 1", marker=:circle, linestyle=:dot)
    vl(p5); xlabel!(p5, "k"); ylabel!(p5, "Compression ratio (dense / H-matrix entries used)")
    title!(p5, "H-matrix compression vs frequency, $name (tol=$tol$op_label)")
    savefig(p5, "frequency_sweep_$(stem)_compression$suffix.png")

    p6 = plot(df.k, df.work_ratio_sigma; label="σ(H_m) heuristic", marker=:circle)
    plot!(p6, df.k, df.work_ratio_constant_factor; label="constant bound factor", marker=:circle, linestyle=:dash)
    hasproperty(df, :work_ratio_estimated) && plot!(p6, df.k, df.work_ratio_estimated; label="estimated σ_m/m, calibrated", marker=:circle, linestyle=:dot)
    hline!(p6, [1.0]; label="exact GMRES", color=:black)
    vl(p6); xlabel!(p6, "k"); ylabel!(p6, "Work / work of exact GMRES")
    title!(p6, "Work of the relaxed solves vs frequency, $name (tol=$tol$op_label)")
    savefig(p6, "frequency_sweep_$(stem)_work$suffix.png")

    p7 = plot(df.k, df.restarts_sigma; label="σ(H_m) heuristic", marker=:circle)
    plot!(p7, df.k, df.restarts_constant_factor; label="constant bound factor", marker=:circle, linestyle=:dash)
    hasproperty(df, :restarts_estimated) && plot!(p7, df.k, df.restarts_estimated; label="estimated σ_m/m, calibrated", marker=:circle, linestyle=:dot)
    vl(p7); xlabel!(p7, "k"); ylabel!(p7, "Restarts (failed true-residual checks)")
    title!(p7, "Restarts vs frequency, $name (tol=$tol$op_label)")
    savefig(p7, "frequency_sweep_$(stem)_restarts$suffix.png")
end

"""
    read_resonances(path; kmin=-Inf, kmax=Inf)

Read a single-column, headerless CSV of known resonance/mode k-values (e.g.
the `data/bad_frequencies_*.csv` files), filtered to `[kmin, kmax]`.
"""
function read_resonances(path; kmin=-Inf, kmax=Inf)
    ks = CSV.read(path, DataFrame; header=false)[:, 1]
    return filter(k -> kmin <= k <= kmax, ks)
end

"""
    precision_strategy_comparison(prob, k, tol; name, label, bound_factor=1.0)

Run `igmres_precision_study` on a problem built by `disk_problem`/
`cavity_problem` (`prob`, at wavenumber `k`) and save four plots, prefixed
by `name` and titled with `label`: per-iteration residuals/heuristic/gap/
bound for the σ(H_m) and the constant bound factor heuristics, the
reconstructed total field, and the effective H-matrix compression used by
each heuristic's matvec. Returns the study.
"""
function precision_strategy_comparison(prob, k, tol; name, label, bound_factor=1.0)
    (; L, g, pde, uᵢ, Q, meshsize) = prob
    println(L) # includes number of leaves, rank range, and compression ratio

    study = InexactGMRES.igmres_precision_study(L, g, tol; bound_factor)
    (; y_exact, residuals_sigma, true_residuals_sigma, residual_gap_sigma, bound_right4_sigma, sigma_heuristic, it_sigma,
        residuals_constant_factor, true_residuals_constant_factor, residual_gap_constant_factor,
        bound_right4_constant_factor, constant_factor_heuristic, it_constant_factor) = study

    # residual_gap_* and bound_right4_* are in absolute units (matching the
    # paper); normalize by ||g|| so they sit on the same relative scale as
    # the other (already-relative) curves
    gnorm = norm(g)
    for (strategy, title, residuals, true_residuals, gap, bound, heuristic, it) in (
        ("sigma", "σ(H_m) heuristic", residuals_sigma, true_residuals_sigma,
            residual_gap_sigma, bound_right4_sigma, sigma_heuristic, it_sigma),
        ("constant_factor", "constant bound factor ($bound_factor) heuristic", residuals_constant_factor,
            true_residuals_constant_factor, residual_gap_constant_factor, bound_right4_constant_factor,
            constant_factor_heuristic, it_constant_factor),
    )
        p = Plots.plot(1:it, residuals; label="igmres residual (internal)", yaxis=:log, marker=:diamond)
        Plots.plot!(p, 1:it, true_residuals; label="true residual", marker=:circle)
        Plots.plot!(p, 1:it, heuristic; label="heuristic value (matvec rtol)", marker=:utriangle, linestyle=:dash)
        Plots.plot!(p, 1:it, gap ./ gnorm; label="||true - internal||", marker=:star5, linestyle=:dot)
        Plots.plot!(p, 1:it, bound ./ gnorm; label="Simoncini-Szyld bound (4.4)", marker=:rect, linestyle=:dashdot)
        Plots.hline!(p, [tol]; label="H-matrix assembly rtol", color=:black)
        Plots.xlabel!(p, "Iteration")
        Plots.ylabel!(p, "Relative residual / matvec rtol")
        Plots.title!(p, "igmres with $title ($label)")
        Plots.savefig(p, "$(name)_$(strategy)_heuristic.png")
    end

    # Reconstructed total field u = uᵢ + D[y] - ik*S[y] (from the exact
    # solution), evaluated off-surface with Inti's own single_double_layer:
    # H-matrix compressed and corrected for near-singular interactions
    # between the grid and the nearby boundary
    # ~6 points per wavelength, capped so the number of targets stays sane
    # (at very high k, e.g. the disk at k≈1257, the plot will still alias)
    npts = clamp(ceil(Int, 6 * 4 * k / 2π), 200, 800)
    xx = yy = range(-2, 2; length=npts)
    grid = [SVector(x1, x2) for x1 in xx, x2 in yy]
    outside = findall(x -> !Inti.isinside(x, Q), grid)
    S_viz, D_viz = Inti.single_double_layer(;
        op=pde, target=grid[outside], source=Q,
        compression=(method=:hmatrix, tol),
        correction=(method=:dim, maxdist=5 * meshsize, target_location=:outside),
    )
    scattered = D_viz * y_exact - im * k * (S_viz * y_exact)
    field = fill(NaN, size(grid))
    field[outside] = real.(uᵢ.(grid[outside]) .+ scattered)
    p = Plots.heatmap(xx, yy, field'; c=:RdBu, aspect_ratio=:equal, clims=(-2, 2))
    Plots.xlabel!(p, "x")
    Plots.ylabel!(p, "y")
    Plots.title!(p, "Total field Re(u) ($label)")
    Plots.savefig(p, "$(name)_solution.png")

    # Effective compression ratio used by each heuristic's matvec, per
    # iteration, vs. the compression already baked into L's own assembly
    p = Plots.plot(1:it_sigma, InexactGMRES.effective_compression_ratio(L, sigma_heuristic);
        label="σ(H_m) heuristic", marker=:utriangle)
    Plots.plot!(p, 1:it_constant_factor, InexactGMRES.effective_compression_ratio(L, constant_factor_heuristic);
        label="constant bound factor", marker=:rect)
    Plots.hline!(p, [HMatrices.compression_ratio(L)]; label="static compression_ratio(L)", linestyle=:dash)
    Plots.xlabel!(p, "Iteration")
    Plots.ylabel!(p, "Effective compression ratio")
    Plots.title!(p, "Effective H-matrix compression during matvec ($label)")
    Plots.savefig(p, "$(name)_compression.png")

    return study
end
