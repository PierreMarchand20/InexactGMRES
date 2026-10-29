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

## Helmholtz scattering by a sound-soft disk, Dirichlet CFIE (combined field
## integral equation, D - ik*S) on a circle
λ = 0.005
k = 2π / λ
tol = sqrt(eps())

prob = disk_problem(k; ε=tol, npplo, correction)
precision_strategy_comparison(prob, k, tol; name="helmholtz_disk$suffix", label="Helmholtz disk$op_label")
