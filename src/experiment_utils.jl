using Inti
using Gmsh
using HMatrices
using LinearAlgebra
using SparseArrays
using StaticArrays
using BenchmarkTools
using DataFrames
using CSV
using InexactGMRES

## Problem builders: Helmholtz Dirichlet CFIE (D - ik*S), disk and cavity,
## both parametrized by wavenumber k alone (meshsize = 2π/(npplo*k) keeps
## points-per-wavelength fixed as k varies, matching data/elliptic_cavity_2D.geo's
## own npplo/k DefineConstant convention).

"""
    assemble_cfie(Q, k; ε=1e-8, θ=π/4)

Assembles the Dirichlet CFIE (D - ik*S) H-matrix and RHS on quadrature `Q`,
for a plane wave incident at angle `θ`. Returns a NamedTuple `(; L, g, pde,
uᵢ)` -- `pde` and `uᵢ` (the incident wave) are included since callers doing
field reconstruction (not just solving) need them too.
"""
function assemble_cfie(Q, k; ε=1e-8, θ=π / 4)
    pde = Inti.Helmholtz(; k, dim=2)
    SL = Inti.SingleLayerKernel(pde)
    DL = Inti.DoubleLayerKernel(pde)
    K = let SL = SL, DL = DL, k = k
        (t, q) -> DL(t, q) - im * k * SL(t, q)
    end
    d = SVector(cos(θ), sin(θ))
    uᵢ = (x) -> exp(im * k * dot(x, d))
    g = map(q -> -uᵢ(q.coords), Q)
    Lop = Inti.IntegralOperator(K, Q, Q)
    L = Inti.assemble_hmatrix(Lop; rtol=ε)
    Id = sparse((0.5 + 0im) * I, size(L))
    axpy!(1.0, Id, L)
    return (; L, g, pde, uᵢ)
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
    cavity_problem(k; geo="elliptic_cavity_2D", npplo=10, qorder=4, gorder=2, ε=1e-8, θ=π/4)

Loads `data/\$geo.geo` (a half-elliptical-shell cavity, parametrized by the
Gmsh DefineConstant's `npplo`/`k`) at the given wavenumber `k`, assembles the
Dirichlet CFIE H-matrix and RHS. `geo` can be swapped for
`"elliptic_cavity_bigger_2D"` to use the more closed variant (shell cut at
x = cos(9π/10) instead of cos(7π/10), so a narrower mouth). Returns `(; L,
g, pde, uᵢ, Q, meshsize)`.
"""
function cavity_problem(k; geo="elliptic_cavity_2D", npplo=10, qorder=4, gorder=2, ε=1e-8, θ=π / 4)
    meshsize = 2π / (npplo * k)
    filename = joinpath(pkgdir(InexactGMRES), "data", "$geo.geo")
    gmsh.initialize()
    gmsh.onelab.set("""[{"type":"number","name":"npplo","values":[$npplo]},{"type":"number","name":"k","values":[$k]}]""")
    gmsh.open(filename)
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.setOrder(gorder)
    msh = Inti.import_mesh(; dim=2)
    gmsh.finalize()

    ents = Inti.entities(msh)
    Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, ents)
    Γ = Inti.boundary(Ω)
    Γ_msh = view(msh, Γ)
    Q = Inti.Quadrature(Γ_msh; qorder)
    return (; assemble_cfie(Q, k; ε, θ)..., Q, meshsize)
end

"""
    tol_sweep(builder, k, tols)

Build one problem (fixed geometry/frequency, via `builder(k)`) and benchmark
`igmres` vs `exact_gmres` at each tolerance in `tols`. Returns a DataFrame
with columns `tol`, `speedup`, `rel_error`, `igmres_it`, `exact_it`.
"""
function tol_sweep(builder, k, tols)
    (; L, g) = builder(k)
    rows = map(tols) do tol
        bench_igmres = @benchmark igmres($L, $g; tol=$tol)
        bench_exact = @benchmark InexactGMRES.exact_gmres($L, $g; tol=$tol)
        y, _, it = igmres(L, g; tol)
        _, _, exact_it = InexactGMRES.exact_gmres(L, g; tol)
        rel_error = norm(L * y - g) / norm(g)
        speedup = minimum(bench_exact).time / minimum(bench_igmres).time
        (; tol, speedup, rel_error, igmres_it=it, exact_it)
    end
    return DataFrame(rows)
end

"""
    frequency_sweep(builder, ks; tol, bound_factor=1.0)

For each `k` in `ks`, build a fresh problem via `builder(k)` and run
`exact_gmres` once plus `igmres` twice -- same two heuristics as
`igmres_precision_study`: the σ(H_m) heuristic (`rel_to_eps(sigma_m/m, res,
t)`) and the constant bound factor heuristic (`rel_to_eps(bound_factor,
res, t)`). `sigma_m`, the smallest singular value of the final Hessenberg
matrix, and the iteration counts for all three solves are candidate
signatures of a nearby scattering resonance.

Timing uses a single `@elapsed` call per solve rather than
`BenchmarkTools` (a full `@benchmark` per point, times 3 solves times
dozens of k's, would add many minutes of pure benchmarking overhead); one
warm-up solve is run first so the first sweep point isn't contaminated by
JIT compilation. This makes individual speedup values noisier than
`tol_sweep`'s, but the trend across the whole sweep should still be
visible.

To check igmres actually solves the right problem, each heuristic is also
run once more (untimed) with `track_true_residual=true`, recording the final
true relative residual `‖b - A x‖/‖b‖` and the final gap `‖r - r̃‖/‖b‖`
between the true and the projection residual.

Returns a DataFrame with columns `k`, `exact_it`, `igmres_it_sigma`,
`igmres_it_constant_factor`, `sigma_m`, `speedup_sigma`,
`speedup_constant_factor`, `true_res_sigma`, `true_res_constant_factor`,
`gap_sigma`, `gap_constant_factor`, `size`.
"""
function frequency_sweep(builder, ks; tol, bound_factor=1.0)
    (; L, g) = builder(first(ks))
    InexactGMRES.exact_gmres(L, g; tol)
    # warm up both heuristics' distinct closure types, not just the default
    # precision_strategy, so the first sweep point isn't contaminated
    dummy_sigma_m, dummy_it = 1.0, 1
    igmres(L, g; tol, precision_strategy=(res, t) -> InexactGMRES.rel_to_eps(dummy_sigma_m / dummy_it, res, t))
    igmres(L, g; tol, precision_strategy=(res, t) -> InexactGMRES.rel_to_eps(bound_factor, res, t))

    rows = map(ks) do k
        (; L, g) = builder(k)
        exact_time = @elapsed ((_, _, exact_it, Hmat) = InexactGMRES.exact_gmres(L, g; tol, return_H=true))
        sigma_m = svd(Hmat).S[end]

        sigma_strategy = (res, t) -> InexactGMRES.rel_to_eps(sigma_m / exact_it, res, t)
        constant_strategy = (res, t) -> InexactGMRES.rel_to_eps(bound_factor, res, t)

        sigma_time = @elapsed ((_, _, igmres_it_sigma) = igmres(L, g; tol, precision_strategy=sigma_strategy))
        constant_time = @elapsed ((_, _, igmres_it_constant_factor) = igmres(L, g; tol, precision_strategy=constant_strategy))

        # separate untimed runs: track_true_residual adds an exact matvec per
        # iteration, which would distort the speedups above
        (_, _, _, true_res_sigma, gap_sigma) = igmres(L, g; tol, precision_strategy=sigma_strategy, track_true_residual=true)
        (_, _, _, true_res_constant_factor, gap_constant_factor) = igmres(L, g; tol, precision_strategy=constant_strategy, track_true_residual=true)

        gnorm = norm(g)
        (; k, exact_it, igmres_it_sigma, igmres_it_constant_factor, sigma_m,
            speedup_sigma=exact_time / sigma_time,
            speedup_constant_factor=exact_time / constant_time,
            true_res_sigma=true_res_sigma[end],
            true_res_constant_factor=true_res_constant_factor[end],
            gap_sigma=gap_sigma[end] / gnorm,
            gap_constant_factor=gap_constant_factor[end] / gnorm,
            size=length(g))
    end
    return DataFrame(rows)
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
