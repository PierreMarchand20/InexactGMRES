using LinearAlgebra
using StaticArrays
using HMatrices
using Plots
using InexactGMRES

## Problem setup: Laplace single-layer kernel on a circle (same setup as the
## HMatrix testset in test/runtests.jl)
n = 10_000
tol = sqrt(eps())

Point2D = SVector{2,Float64}
X = Y = [Point2D(sin(i * 2π / n), cos(i * 2π / n)) for i in 0:(n-1)]

struct LaplaceMatrix <: AbstractMatrix{Float64}
    X::Vector{Point2D}
    Y::Vector{Point2D}
end
Base.getindex(K::LaplaceMatrix, i::Int, j::Int) = -1 / 2π * log(norm(K.X[i] - K.Y[j]) + 1e-10)
Base.size(K::LaplaceMatrix) = length(K.X), length(K.Y)

K = LaplaceMatrix(X, Y)

Xclt = Yclt = ClusterTree(X)
adm = StrongAdmissibilityStd()
comp = PartialACA(; rtol=tol)
H = assemble_hmatrix(K, Xclt, Yclt; adm, comp, threads=false, distributed=false)

println(H) # includes number of leaves, rank range, and compression ratio

T = eltype(H)
b = rand(T, n)

## Run exact_gmres (for sigma_m) and both igmres precision strategies
bound_factor = 1.
study = InexactGMRES.igmres_precision_study(H, b, tol; bound_factor)
(; sigma_m, y_exact, y_sigma, residuals_sigma, true_residuals_sigma, residual_gap_sigma, bound_right4_sigma, sigma_heuristic, it_sigma,
    y_constant_factor, residuals_constant_factor, true_residuals_constant_factor, residual_gap_constant_factor,
    bound_right4_constant_factor, constant_factor_heuristic, it_constant_factor) = study

# residual_gap_* and bound_right4_* are in absolute units (matching the
# paper); normalize by ||b|| so they sit on the same relative scale as the
# other (already-relative) curves on these plots
bheta = norm(b)
gap_sigma = residual_gap_sigma ./ bheta
gap_constant_factor = residual_gap_constant_factor ./ bheta
bound_sigma = bound_right4_sigma ./ bheta
bound_constant_factor = bound_right4_constant_factor ./ bheta

## Plot 1: sigma(H_m) heuristic, residual decrease and heuristic value
p1 = Plots.plot(1:it_sigma, residuals_sigma; label="igmres residual (internal)", yaxis=:log, marker=:diamond)
Plots.plot!(p1, 1:it_sigma, true_residuals_sigma; label="true residual", marker=:circle)
Plots.plot!(p1, 1:it_sigma, sigma_heuristic; label="heuristic value (matvec rtol)", marker=:utriangle, linestyle=:dash)
Plots.plot!(p1, 1:it_sigma, gap_sigma; label="||true - internal||", marker=:star5, linestyle=:dot)
Plots.plot!(p1, 1:it_sigma, bound_sigma; label="Simoncini-Szyld bound (4.4)", marker=:rect, linestyle=:dashdot)
Plots.xlabel!(p1, "Iteration")
Plots.ylabel!(p1, "Relative residual / matvec rtol")
Plots.title!(p1, "igmres with σ(H_m) heuristic")
Plots.savefig(p1, "laplace_disk_sigma_heuristic.png")

## Plot 2: constant bound factor heuristic, residual decrease and heuristic value
p2 = Plots.plot(1:it_constant_factor, residuals_constant_factor; label="igmres residual (internal)", yaxis=:log, marker=:diamond)
Plots.plot!(p2, 1:it_constant_factor, true_residuals_constant_factor; label="true residual", marker=:circle)
Plots.plot!(p2, 1:it_constant_factor, constant_factor_heuristic; label="heuristic value (matvec rtol)", marker=:utriangle, linestyle=:dash)
Plots.plot!(p2, 1:it_constant_factor, gap_constant_factor; label="||true - internal||", marker=:star5, linestyle=:dot)
Plots.plot!(p2, 1:it_constant_factor, bound_constant_factor; label="Simoncini-Szyld bound (4.4)", marker=:rect, linestyle=:dashdot)
Plots.xlabel!(p2, "Iteration")
Plots.ylabel!(p2, "Relative residual / matvec rtol")
Plots.title!(p2, "igmres with constant bound factor ($bound_factor) heuristic")
Plots.savefig(p2, "laplace_disk_constant_factor_heuristic.png")

## Plot 3: solution comparison
p3 = Plots.plot(1:n, y_exact; label="exact", linewidth=2)
Plots.plot!(p3, 1:n, y_sigma; label="σ(H_m) heuristic", linestyle=:dash)
Plots.plot!(p3, 1:n, y_constant_factor; label="constant bound factor", linestyle=:dot)
Plots.xlabel!(p3, "Point index")
Plots.ylabel!(p3, "Solution")
Plots.title!(p3, "Solution (Laplace disk)")
Plots.savefig(p3, "laplace_disk_solution.png")
