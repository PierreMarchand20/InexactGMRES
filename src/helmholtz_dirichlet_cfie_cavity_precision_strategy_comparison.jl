using InexactGMRES

include(joinpath(pkgdir(InexactGMRES), "src", "experiment_utils.jl"))

## Helmholtz scattering by the 2D cavity (data/elliptic_cavity_2D.geo),
## Dirichlet CFIE (D - ik*S). k matches the tol sweeps.
tol = sqrt(eps())
k = 100.0

prob = cavity_problem(k; ε=tol)
println("Number of quadrature points: ", length(prob.Q))
precision_strategy_comparison(prob, k, tol; name="helmholtz_cavity", label="2D cavity, k=$k")
