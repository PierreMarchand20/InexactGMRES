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
X = Y = [Point2D(sin(i * 2π / n), cos(i * 2π / n)) for i in 0:n-1]

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

T = eltype(H)
b = rand(T, n)

## Reference: exact GMRES, also exposing its Hessenberg matrix at convergence
y_exact, residuals_exact, m, H_m = InexactGMRES.exact_gmres(H, b; tol, return_H=true)

sigma_m = svd(H_m).S[end] # smallest singular value of H_m

## igmres with a constant bound factor sigma_m plugged into rel_to_eps
y_sigma, residuals_sigma, it_sigma = igmres(H, b; tol,
    precision_strategy=(res, t) -> InexactGMRES.rel_to_eps(sigma_m, res, t))

## igmres with a flat, iteration-independent matvec tolerance
flat_rtol = 1e-3
y_flat, residuals_flat, it_flat = igmres(H, b; tol,
    precision_strategy=(res, t) -> flat_rtol)

## Plot residual decrease for all three
p = Plots.plot(1:m, residuals_exact; label="exact GMRES", yaxis=:log, marker=:circle)
Plots.plot!(p, 1:it_sigma, residuals_sigma; label="igmres, σ(H_m) heuristic", marker=:diamond)
Plots.plot!(p, 1:it_flat, residuals_flat; label="igmres, flat rtol=$flat_rtol", marker=:rect)
Plots.xlabel!(p, "Iteration")
Plots.ylabel!(p, "Relative residual")
Plots.title!(p, "Residual decrease: exact GMRES vs. igmres precision strategies")
display(p)
Plots.savefig(p, "laplace_precision_strategy_comparison.png")
