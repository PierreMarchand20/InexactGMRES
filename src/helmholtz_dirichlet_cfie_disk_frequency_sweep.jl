using InexactGMRES
using Plots
using CSV

include(joinpath(pkgdir(InexactGMRES), "src", "experiment_utils.jl"))
default(size=(800, 500), titlefontsize=9)

## `--replot`: rebuild the figures from the CSV of a previous run, without solving.
## `--npplo=N`: points per wavelength (default 10), part of the output names.
i = findfirst(startswith("--npplo="), ARGS)
npplo = i === nothing ? 10 : parse(Int, split(ARGS[i], "=")[2])
suffix = "_npplo$npplo"
op_label = ", npplo=$npplo"

## Fixed overall tolerance, sweep frequency k for the disk -- baseline with
## no interior resonances expected, for comparison against the cavity sweep.
## Range follows https://arxiv.org/pdf/2102.05367's 2D experiments
## (k in [50, 300]).
kmin, kmax = 50.0, 300.0
tol = 1e-6

ks = collect(range(kmin, kmax; length=26))

## Cavity's known resonances, overlaid here purely as a baseline contrast --
## the disk has no interior cavity and shouldn't respond at these k's.
resonances = read_resonances(
    joinpath(pkgdir(InexactGMRES), "data", "bad_frequencies_ellipsis_half_even_modes_dir_first.csv");
    kmin, kmax,
)

replot = "--replot" in ARGS
df = replot ? CSV.read("frequency_sweep_disk$suffix.csv", DataFrame) : frequency_sweep(k -> disk_problem(k; npplo), ks; tol)
replot || CSV.write("frequency_sweep_disk$suffix.csv", df)

p1 = plot(df.k, df.exact_it; label="disk exact_gmres", marker=:diamond)
plot!(p1, df.k, df.igmres_it_sigma; label="disk igmres (σ(H_m) heuristic)", marker=:diamond, linestyle=:dash)
plot!(p1, df.k, df.igmres_it_constant_factor; label="disk igmres (constant bound factor)", marker=:diamond, linestyle=:dot)
vline!(p1, resonances; label="cavity bouncing-ball modes", linestyle=:dot, color=:gray, alpha=0.4)
xlabel!(p1, "k")
ylabel!(p1, "Iterations")
title!(p1, "Iteration count vs frequency, disk (tol=$tol$op_label)")
savefig(p1, "frequency_sweep_disk_iterations$suffix.png")

p2 = plot(df.k, df.sigma_m; label="disk", marker=:diamond, yaxis=:log)
vline!(p2, resonances; label="cavity bouncing-ball modes", linestyle=:dot, color=:gray, alpha=0.4)
xlabel!(p2, "k")
ylabel!(p2, "sigma_m(H_m)")
title!(p2, "Hessenberg smallest singular value vs frequency, disk (tol=$tol$op_label)")
savefig(p2, "frequency_sweep_disk_sigma_m$suffix.png")

p3 = plot(df.k, df.speedup_sigma; label="disk (σ(H_m) heuristic)", marker=:diamond)
plot!(p3, df.k, df.speedup_constant_factor; label="disk (constant bound factor)", marker=:diamond, linestyle=:dash)
vline!(p3, resonances; label="cavity bouncing-ball modes", linestyle=:dot, color=:gray, alpha=0.4)
xlabel!(p3, "k")
ylabel!(p3, "Speed-up (exact_gmres / igmres)")
title!(p3, "igmres speed-up vs frequency, disk (tol=$tol$op_label)")
savefig(p3, "frequency_sweep_disk_speedup$suffix.png")

p4 = plot(df.k, df.true_res_sigma; label="true residual (σ(H_m))", marker=:diamond, yaxis=:log)
plot!(p4, df.k, df.true_res_constant_factor; label="true residual (constant bound factor)", marker=:diamond, linestyle=:dash)
plot!(p4, df.k, df.gap_sigma; label="||r - r̃|| (σ(H_m))", marker=:star5, linestyle=:dot)
plot!(p4, df.k, df.gap_constant_factor; label="||r - r̃|| (constant bound factor)", marker=:star5, linestyle=:dashdot)
hline!(p4, [tol]; label="tol", color=:black)
vline!(p4, resonances; label="cavity bouncing-ball modes", linestyle=:dot, color=:gray, alpha=0.4)
xlabel!(p4, "k")
ylabel!(p4, "Relative to ||g||, at convergence")
title!(p4, "igmres final true residual and gap, disk (tol=$tol$op_label)")
savefig(p4, "frequency_sweep_disk_residual$suffix.png")

plot_frequency_extras(df, "disk", "disk", suffix, op_label, resonances; tol)
