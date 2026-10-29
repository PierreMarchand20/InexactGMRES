using InexactGMRES

include(joinpath(pkgdir(InexactGMRES), "src", "experiment_utils.jl"))
default(size=(800, 500), titlefontsize=9)

## `--npplo=N`: points per wavelength (default 10), part of the output names.
i = findfirst(startswith("--npplo="), ARGS)
npplo = i === nothing ? 10 : parse(Int, split(ARGS[i], "=")[2])
suffix = "_npplo$npplo"
op_label = ", npplo=$npplo"

## Helmholtz scattering by the bigger (more closed, narrower-mouth) 2D cavity
## (data/elliptic_cavity_bigger_2D.geo), Dirichlet CFIE (D - ik*S). k matches
## the tol sweeps.
tol = sqrt(eps())
k = 100.0

prob = cavity_problem(k; geo="elliptic_cavity_bigger_2D", ε=tol, npplo)
println("Number of quadrature points: ", length(prob.Q))
precision_strategy_comparison(prob, k, tol; name="helmholtz_bigger_cavity$suffix", label="bigger cavity, k=$k$op_label")
