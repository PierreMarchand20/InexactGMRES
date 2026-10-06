using Inti
using StaticArrays
using LinearAlgebra
using Gmsh
using SparseArrays
using HMatrices
using Plots
using InexactGMRES

include(joinpath(pkgdir(InexactGMRES), "src", "experiment_utils.jl"))

## Problem setup: Helmholtz scattering by a 2D cavity, Dirichlet CFIE
## (combined field integral equation, D - ik*S) on data/elliptic_cavity_2D.geo
## (a coarser mesh than the k=50-300 frequency sweep, so this runs quickly)
λ = 0.01
k = 2π / λ
tol = sqrt(eps())

(; L, g, pde, uᵢ, Q, meshsize) = cavity_problem(k; ε=tol)
println("Number of quadrature points: ", length(Q))

println(L) # includes number of leaves, rank range, and compression ratio

## Run exact_gmres (for sigma_m) and both igmres precision strategies
bound_factor = 1.
study = InexactGMRES.igmres_precision_study(L, g, tol; bound_factor)
(; sigma_m, y_exact, residuals_sigma, true_residuals_sigma, residual_gap_sigma, bound_right4_sigma, sigma_heuristic, it_sigma,
    residuals_constant_factor, true_residuals_constant_factor, residual_gap_constant_factor,
    bound_right4_constant_factor, constant_factor_heuristic, it_constant_factor) = study

# residual_gap_* and bound_right4_* are in absolute units (matching the
# paper); normalize by ||g|| so they sit on the same relative scale as the
# other (already-relative) curves on these plots
gnorm = norm(g)
gap_sigma = residual_gap_sigma ./ gnorm
gap_constant_factor = residual_gap_constant_factor ./ gnorm
bound_sigma = bound_right4_sigma ./ gnorm
bound_constant_factor = bound_right4_constant_factor ./ gnorm

## Plot 1: sigma(H_m) heuristic, residual decrease and heuristic value
p1 = Plots.plot(1:it_sigma, residuals_sigma; label="igmres residual (internal)", yaxis=:log, marker=:diamond)
Plots.plot!(p1, 1:it_sigma, true_residuals_sigma; label="true residual", marker=:circle)
Plots.plot!(p1, 1:it_sigma, sigma_heuristic; label="heuristic value (matvec rtol)", marker=:utriangle, linestyle=:dash)
Plots.plot!(p1, 1:it_sigma, gap_sigma; label="||true - internal||", marker=:star5, linestyle=:dot)
Plots.plot!(p1, 1:it_sigma, bound_sigma; label="Simoncini-Szyld bound (4.4)", marker=:rect, linestyle=:dashdot)
Plots.hline!(p1, [tol]; label="H-matrix assembly rtol", color=:black)
Plots.xlabel!(p1, "Iteration")
Plots.ylabel!(p1, "Relative residual / matvec rtol")
Plots.title!(p1, "igmres with σ(H_m) heuristic (2D cavity)")
Plots.savefig(p1, "helmholtz_cavity_sigma_heuristic.png")

## Plot 2: constant bound factor heuristic, residual decrease and heuristic value
p2 = Plots.plot(1:it_constant_factor, residuals_constant_factor; label="igmres residual (internal)", yaxis=:log, marker=:diamond)
Plots.plot!(p2, 1:it_constant_factor, true_residuals_constant_factor; label="true residual", marker=:circle)
Plots.plot!(p2, 1:it_constant_factor, constant_factor_heuristic; label="heuristic value (matvec rtol)", marker=:utriangle, linestyle=:dash)
Plots.plot!(p2, 1:it_constant_factor, gap_constant_factor; label="||true - internal||", marker=:star5, linestyle=:dot)
Plots.plot!(p2, 1:it_constant_factor, bound_constant_factor; label="Simoncini-Szyld bound (4.4)", marker=:rect, linestyle=:dashdot)
Plots.hline!(p2, [tol]; label="H-matrix assembly rtol", color=:black)
Plots.xlabel!(p2, "Iteration")
Plots.ylabel!(p2, "Relative residual / matvec rtol")
Plots.title!(p2, "igmres with constant bound factor ($bound_factor) heuristic (2D cavity)")
Plots.savefig(p2, "helmholtz_cavity_constant_factor_heuristic.png")

## Plot 3: reconstructed total field u = uᵢ + D[y] - ik*S[y] around the
## cavity (using the exact solution; the two heuristic solutions agree
## closely with it, per the residual/gap plots above, so a single
## reconstruction suffices). Evaluate off-surface via Inti's own
## single_double_layer, same as the documented workflow
## (https://integralequations.github.io/Inti.jl/stable/pluto-examples/
## helmholtz_scattering/): compressed with an H-matrix and, unlike a plain
## assembled kernel, corrected for the near-singular interactions between
## the grid and the boundary close to it.
xx = yy = range(-2, 2; length=200)
grid = [SVector(x1, x2) for x1 in xx, x2 in yy]
outside = findall(x -> !Inti.isinside(x, Q), grid)

S_viz, D_viz = Inti.single_double_layer(;
    op=pde, target=grid[outside], source=Q,
    compression=(method=:hmatrix, tol),
    correction=(method=:dim, maxdist=5*meshsize, target_location=:outside),
)
scattered = D_viz*y_exact - im*k*(S_viz*y_exact)

field = fill(NaN, size(grid))
field[outside] = real.(uᵢ.(grid[outside]) .+ scattered)
p4 = Plots.heatmap(xx, yy, field'; c=:RdBu, aspect_ratio=:equal, clims=(-2, 2))
Plots.xlabel!(p4, "x")
Plots.ylabel!(p4, "y")
Plots.title!(p4, "Total field Re(u) (2D cavity)")
Plots.savefig(p4, "helmholtz_cavity_solution.png")

## Plot 5: effective compression ratio used by each heuristic's matvec, per
## iteration -- how much of L is actually touched at the matvec rtol each
## strategy requests, vs. the compression already baked into L's own
## (static) assembly
eff_comp_sigma = InexactGMRES.effective_compression_ratio(L, sigma_heuristic)
eff_comp_constant_factor = InexactGMRES.effective_compression_ratio(L, constant_factor_heuristic)
static_ratio = HMatrices.compression_ratio(L)

p5 = Plots.plot(1:it_sigma, eff_comp_sigma; label="σ(H_m) heuristic", marker=:utriangle)
Plots.plot!(p5, 1:it_constant_factor, eff_comp_constant_factor; label="constant bound factor", marker=:rect)
Plots.hline!(p5, [static_ratio]; label="static compression_ratio(L)", linestyle=:dash)
Plots.xlabel!(p5, "Iteration")
Plots.ylabel!(p5, "Effective compression ratio")
Plots.title!(p5, "Effective H-matrix compression during matvec (2D cavity)")
Plots.savefig(p5, "helmholtz_cavity_compression.png")
