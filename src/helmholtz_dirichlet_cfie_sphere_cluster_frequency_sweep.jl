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

## Fixed overall tolerance, sweep frequency k for a 3 x 3 x 2 cluster of
## spheres of radius 0.25 with gaps of 0.1 (`data/sphere_cluster_3D.geo`): no
## cavity, but multiple scattering between the spheres raises the iteration
## count. Same strategies as the sphere sweep, all with relaxed recycling.
ks = collect(4.0:1.0:14.0)
tol = 1e-6
resonances = Float64[]

replot = "--replot" in ARGS
df = replot ? CSV.read("frequency_sweep_sphere_cluster$suffix.csv", DataFrame) :
     frequency_sweep(k -> cavity_problem(k; geo="sphere_cluster_3D", npplo), ks; tol, igmres_kwargs=(; relaxed_recycling=true), estimated=true)
replot || CSV.write("frequency_sweep_sphere_cluster$suffix.csv", df)

p1 = plot(df.k, df.exact_it; label="cluster exact_gmres", marker=:diamond)
plot!(p1, df.k, df.igmres_it_sigma; label="cluster igmres (σ(H_m) heuristic)", marker=:diamond, linestyle=:dash)
plot!(p1, df.k, df.igmres_it_constant_factor; label="cluster igmres (constant bound factor)", marker=:diamond, linestyle=:dot)
plot!(p1, df.k, df.igmres_it_estimated; label="cluster igmres (estimated σ_m/m, calibrated)", marker=:diamond, linestyle=:dashdot)
xlabel!(p1, "k")
ylabel!(p1, "Iterations")
title!(p1, "Iteration count vs frequency, sphere cluster (tol=$tol$op_label)")
savefig(p1, "frequency_sweep_sphere_cluster_iterations$suffix.png")

p2 = plot(df.k, df.sigma_m; label="cluster", marker=:diamond, yaxis=:log)
xlabel!(p2, "k")
ylabel!(p2, "sigma_m(H_m)")
title!(p2, "Hessenberg smallest singular value vs frequency, sphere cluster (tol=$tol$op_label)")
savefig(p2, "frequency_sweep_sphere_cluster_sigma_m$suffix.png")

p3 = plot(df.k, df.speedup_sigma; label="cluster (σ(H_m) heuristic)", marker=:diamond)
plot!(p3, df.k, df.speedup_constant_factor; label="cluster (constant bound factor)", marker=:diamond, linestyle=:dash)
plot!(p3, df.k, df.speedup_estimated; label="cluster (estimated σ_m/m, calibrated)", marker=:diamond, linestyle=:dot)
xlabel!(p3, "k")
ylabel!(p3, "Speed-up (exact_gmres / igmres)")
title!(p3, "igmres speed-up vs frequency, sphere cluster (tol=$tol$op_label)")
savefig(p3, "frequency_sweep_sphere_cluster_speedup$suffix.png")

p4 = plot(df.k, df.true_res_sigma; label="true residual (σ(H_m))", marker=:diamond, yaxis=:log)
plot!(p4, df.k, df.true_res_constant_factor; label="true residual (constant bound factor)", marker=:diamond, linestyle=:dash)
plot!(p4, df.k, df.true_res_estimated; label="true residual (estimated σ_m/m, calibrated)", marker=:diamond, linestyle=:dot)
plot!(p4, df.k, df.gap_sigma; label="||r - r̃|| (σ(H_m))", marker=:star5, linestyle=:dot)
plot!(p4, df.k, df.gap_constant_factor; label="||r - r̃|| (constant bound factor)", marker=:star5, linestyle=:dashdot)
hline!(p4, [tol]; label="tol", color=:black)
xlabel!(p4, "k")
ylabel!(p4, "Relative to ||g||, at convergence")
title!(p4, "igmres final true residual and gap, sphere cluster (tol=$tol$op_label)")
savefig(p4, "frequency_sweep_sphere_cluster_residual$suffix.png")

plot_frequency_extras(df, "sphere_cluster", "sphere cluster", suffix, op_label, resonances; tol)

