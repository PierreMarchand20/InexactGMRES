using InexactGMRES

include(joinpath(pkgdir(InexactGMRES), "src", "experiment_utils.jl"))
default(size=(800, 500), titlefontsize=9)

## `--corrected`: singular-corrected operator, outputs suffixed `_corrected`.
## Without it: the uncorrected operator, as in the first runs.
## `--npplo=N`: points per wavelength (default 10), part of the output names.
i = findfirst(startswith("--npplo="), ARGS)
npplo = i === nothing ? 10 : parse(Int, split(ARGS[i], "=")[2])
correction = "--corrected" in ARGS
suffix = "_npplo$npplo" * (correction ? "_corrected" : "")
op_label = ", npplo=$npplo" * (correction ? ", corrected operator" : "")

## Helmholtz scattering by the 2D cavity (data/elliptic_cavity_2D.geo),
## Dirichlet CFIE (D - ik*S). k matches the tol sweeps.
tol = sqrt(eps())
k = 100.0

prob = cavity_problem(k; ε=tol, npplo, correction)
println("Number of quadrature points: ", length(prob.Q))
precision_strategy_comparison(prob, k, tol; name="helmholtz_cavity$suffix", label="2D cavity, k=$k$op_label")
