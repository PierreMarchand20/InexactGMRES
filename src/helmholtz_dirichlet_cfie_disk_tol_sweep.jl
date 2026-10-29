using InexactGMRES
using Plots
using CSV

include(joinpath(pkgdir(InexactGMRES), "src", "experiment_utils.jl"))
default(size=(800, 500), titlefontsize=9)

## `--corrected`: singular-corrected operator, outputs suffixed `_corrected`.
## Without it: the uncorrected operator, as in the first runs.
## `--replot`: rebuild the figures from the CSV of a previous run, without solving.
## `--npplo=N`: points per wavelength (default 10), part of the output names.
i = findfirst(startswith("--npplo="), ARGS)
npplo = i === nothing ? 10 : parse(Int, split(ARGS[i], "=")[2])
correction = "--corrected" in ARGS
suffix = "_npplo$npplo" * (correction ? "_corrected" : "")
op_label = ", npplo=$npplo" * (correction ? ", corrected operator" : "")

## Fixed geometry and frequency, sweep the overall igmres/exact_gmres
## tolerance for the disk (baseline, no interior resonances expected).
## k is a representative frequency -- change it to probe a known mode from
## the frequency sweep instead.
k = 100.0
tols = [1e-2, 1e-4, 1e-6, 1e-8]

replot = "--replot" in ARGS
df = replot ? CSV.read("tol_sweep_disk$suffix.csv", DataFrame) : tol_sweep(k -> disk_problem(k; npplo, correction), k, tols)
replot || CSV.write("tol_sweep_disk$suffix.csv", df)

p1 = plot(df.tol, df.speedup; label="disk", marker=:diamond, xaxis=:log)
xlabel!(p1, "Overall tolerance")
ylabel!(p1, "Speed-up (exact_gmres / igmres)")
title!(p1, "igmres speed-up at k=$k (disk$op_label)")
savefig(p1, "tol_sweep_disk_speedup$suffix.png")

p2 = plot(df.tol, df.rel_error; label="disk", marker=:diamond, xaxis=:log, yaxis=:log)
plot!(p2, df.tol, df.tol; label="tol", linestyle=:dash, color=:black)
xlabel!(p2, "Overall tolerance")
ylabel!(p2, "Relative residual ||L*y - g||/||g||")
title!(p2, "igmres accuracy at k=$k (disk$op_label)")
savefig(p2, "tol_sweep_disk_accuracy$suffix.png")

p3 = plot(df.tol, df.igmres_it; label="igmres", marker=:diamond, xaxis=:log)
plot!(p3, df.tol, df.exact_it; label="exact_gmres", marker=:diamond, linestyle=:dash)
xlabel!(p3, "Overall tolerance")
ylabel!(p3, "Iterations")
title!(p3, "Iteration count at k=$k (disk$op_label)")
savefig(p3, "tol_sweep_disk_iterations$suffix.png")
