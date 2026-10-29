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
## `--direct`: direct CFIE (unknown ∂ₙu). `--deflated`: deflate with the Mathieu
## quasimodes selected at each k (needs `--direct`, whose null vectors they approximate).
direct = "--direct" in ARGS
deflated = "--deflated" in ARGS
deflated && !direct && error("--deflated needs --direct")
suffix = "_npplo$npplo" * (correction ? "_corrected" : "") * (direct ? "_direct" : "") * (deflated ? "_deflated" : "")
op_label = ", npplo=$npplo" * join([", corrected operator", ", direct", ", deflated"][[correction, direct, deflated]])
θ_mouth = 9π / 10 # mouth arc |θ| > θ_mouth of the inner ellipse
deflation = deflated ? ((prob, k) -> select_quasimodes(prob.Q, k; θ_mouth)) : nothing

bigger_cavity_problem(k) = cavity_problem(k; geo="elliptic_cavity_bigger_2D", npplo, correction, direct)

## Fixed geometry and frequency, sweep the overall igmres/exact_gmres
## tolerance for the bigger (more closed, narrower-mouth) cavity.
## k is a representative frequency -- change it to probe a known mode from
## the frequency sweep instead.
k = 100.0
tols = [1e-2, 1e-4, 1e-6, 1e-8]

replot = "--replot" in ARGS
df = replot ? CSV.read("tol_sweep_bigger_cavity$suffix.csv", DataFrame) : tol_sweep(bigger_cavity_problem, k, tols; deflation)
replot || CSV.write("tol_sweep_bigger_cavity$suffix.csv", df)

p1 = plot(df.tol, df.speedup; label="bigger cavity", marker=:circle, xaxis=:log)
xlabel!(p1, "Overall tolerance")
ylabel!(p1, "Speed-up (exact_gmres / igmres)")
title!(p1, "igmres speed-up at k=$k (bigger cavity$op_label)")
savefig(p1, "tol_sweep_bigger_cavity_speedup$suffix.png")

p2 = plot(df.tol, df.rel_error; label="bigger cavity", marker=:circle, xaxis=:log, yaxis=:log)
plot!(p2, df.tol, df.tol; label="tol", linestyle=:dash, color=:black)
xlabel!(p2, "Overall tolerance")
ylabel!(p2, "Relative residual ||L*y - g||/||g||")
title!(p2, "igmres accuracy at k=$k (bigger cavity$op_label)")
savefig(p2, "tol_sweep_bigger_cavity_accuracy$suffix.png")

p3 = plot(df.tol, df.igmres_it; label="igmres", marker=:circle, xaxis=:log)
plot!(p3, df.tol, df.exact_it; label="exact_gmres", marker=:circle, linestyle=:dash)
xlabel!(p3, "Overall tolerance")
ylabel!(p3, "Iterations")
title!(p3, "Iteration count at k=$k (bigger cavity$op_label)")
savefig(p3, "tol_sweep_bigger_cavity_iterations$suffix.png")
