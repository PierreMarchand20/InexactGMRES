using Inti
using StaticArrays
using LinearAlgebra
using Gmsh
using SparseArrays
using Plots
using InexactGMRES

## Problem setup: Helmholtz scattering by a 2D cavity, Dirichlet CFIE
## (combined field integral equation, D - ik*S) on a Gmsh-imported mesh
## (same kernel/geometry as cavity2d_scattering_fixsize.jl, with a coarser
## mesh so this runs quickly)
λ = 0.25
k = 2π / λ
θ = π / 4
tol = sqrt(eps())

meshsize = λ / 10
gorder = 2
qorder = 4

filename = joinpath(Inti.PROJECT_ROOT, "docs", "assets", "elliptic_cavity_2D.geo")
gmsh.initialize()
gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
gmsh.open(filename)
gmsh.model.mesh.generate(1)
gmsh.model.mesh.setOrder(gorder)
msh = Inti.import_mesh(; dim=2)
gmsh.finalize()

ents = Inti.entities(msh)
Ω = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, ents)
Γ = Inti.boundary(Ω)
Γ_msh = view(msh, Γ)

d = SVector(cos(θ), sin(θ)) # incident direction
uᵢ = (x) -> exp(im * k * dot(x, d)) # incident plane wave

## Dirichlet CFIE kernel D - ik*S, built from Inti's own single/double layer
## kernels for the Helmholtz operator (no need to hand-derive the Hankel
## function formulas ourselves)
pde = Inti.Helmholtz(; k, dim=2)
SL = Inti.SingleLayerKernel(pde)
DL = Inti.DoubleLayerKernel(pde)
K = let SL = SL, DL = DL, k = k
    (t, q) -> DL(t, q) - im * k * SL(t, q)
end

Q = Inti.Quadrature(Γ_msh; qorder)
println("Number of quadrature points: ", length(Q))

## Right-hand side given by Dirichlet trace of plane wave
g = map(Q) do q
    return -uᵢ(q.coords)
end

Lop = Inti.IntegralOperator(K, Q, Q)
L = Inti.assemble_hmatrix(Lop; rtol=tol)
Id = sparse((0.5 + 0 * im)I, size(L))
axpy!(1.0, Id, L)

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
