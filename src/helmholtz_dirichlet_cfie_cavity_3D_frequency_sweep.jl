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

## Fixed overall tolerance, sweep frequency k on a fine grid for the bigger 3D
## elliptic cavity (prolate spheroidal shell with an opening, a trapping 3D
## obstacle), looking for resonances as dips in σ_m. n grows like k² (about
## 38k unknowns at k = 12). Same strategies as the sphere sweep, all with
## relaxed recycling.
ks = collect(4.0:0.25:12.0)
tol = 1e-6
resonances = Float64[] # no list of 3D cavity modes: found from the σ_m dips

replot = "--replot" in ARGS
df = replot ? CSV.read("frequency_sweep_cavity_3D$suffix.csv", DataFrame) :
     frequency_sweep(k -> cavity_problem(k; geo="elliptic_cavity_bigger_3D", npplo), ks; tol, igmres_kwargs=(; relaxed_recycling=true), estimated=true)
replot || CSV.write("frequency_sweep_cavity_3D$suffix.csv", df)

p1 = plot(df.k, df.exact_it; label="cavity exact_gmres", marker=:diamond)
plot!(p1, df.k, df.igmres_it_sigma; label="cavity igmres (σ(H_m) heuristic)", marker=:diamond, linestyle=:dash)
plot!(p1, df.k, df.igmres_it_constant_factor; label="cavity igmres (constant bound factor)", marker=:diamond, linestyle=:dot)
plot!(p1, df.k, df.igmres_it_estimated; label="cavity igmres (estimated σ_m/m, calibrated)", marker=:diamond, linestyle=:dashdot)
xlabel!(p1, "k")
ylabel!(p1, "Iterations")
title!(p1, "Iteration count vs frequency, bigger 3D cavity (tol=$tol$op_label)")
savefig(p1, "frequency_sweep_cavity_3D_iterations$suffix.png")

p2 = plot(df.k, df.sigma_m; label="cavity", marker=:diamond, yaxis=:log)
xlabel!(p2, "k")
ylabel!(p2, "sigma_m(H_m)")
title!(p2, "Hessenberg smallest singular value vs frequency, bigger 3D cavity (tol=$tol$op_label)")
savefig(p2, "frequency_sweep_cavity_3D_sigma_m$suffix.png")

p3 = plot(df.k, df.speedup_sigma; label="cavity (σ(H_m) heuristic)", marker=:diamond)
plot!(p3, df.k, df.speedup_constant_factor; label="cavity (constant bound factor)", marker=:diamond, linestyle=:dash)
plot!(p3, df.k, df.speedup_estimated; label="cavity (estimated σ_m/m, calibrated)", marker=:diamond, linestyle=:dot)
xlabel!(p3, "k")
ylabel!(p3, "Speed-up (exact_gmres / igmres)")
title!(p3, "igmres speed-up vs frequency, bigger 3D cavity (tol=$tol$op_label)")
savefig(p3, "frequency_sweep_cavity_3D_speedup$suffix.png")

p4 = plot(df.k, df.true_res_sigma; label="true residual (σ(H_m))", marker=:diamond, yaxis=:log)
plot!(p4, df.k, df.true_res_constant_factor; label="true residual (constant bound factor)", marker=:diamond, linestyle=:dash)
plot!(p4, df.k, df.true_res_estimated; label="true residual (estimated σ_m/m, calibrated)", marker=:diamond, linestyle=:dot)
plot!(p4, df.k, df.gap_sigma; label="||r - r̃|| (σ(H_m))", marker=:star5, linestyle=:dot)
plot!(p4, df.k, df.gap_constant_factor; label="||r - r̃|| (constant bound factor)", marker=:star5, linestyle=:dashdot)
hline!(p4, [tol]; label="tol", color=:black)
xlabel!(p4, "k")
ylabel!(p4, "Relative to ||g||, at convergence")
title!(p4, "igmres final true residual and gap, bigger 3D cavity (tol=$tol$op_label)")
savefig(p4, "frequency_sweep_cavity_3D_residual$suffix.png")

plot_frequency_extras(df, "cavity_3D", "bigger 3D cavity", suffix, op_label, resonances; tol)

