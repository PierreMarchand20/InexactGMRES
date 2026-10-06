using InexactGMRES

include(joinpath(pkgdir(InexactGMRES), "src", "experiment_utils.jl"))

## Helmholtz scattering by a sound-soft disk, Dirichlet CFIE (combined field
## integral equation, D - ik*S) on a circle
λ = 0.005
k = 2π / λ
tol = sqrt(eps())

prob = disk_problem(k; ε=tol)
precision_strategy_comparison(prob, k, tol; name="helmholtz_disk", label="Helmholtz disk")
