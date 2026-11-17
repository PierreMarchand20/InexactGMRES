using InexactGMRES
using Plots
using CSV

include(joinpath(pkgdir(InexactGMRES), "src", "experiment_utils.jl"))
default(size=(800, 500), titlefontsize=9)

## `--replot`: rebuild the figures from the CSV of a previous run, without solving.
## `--npplo=N`: points per wavelength (default 10), part of the output names.
i = findfirst(startswith("--npplo="), ARGS)
npplo = i === nothing ? 10 : parse(Int, split(ARGS[i], "=")[2])
## `--direct`: direct CFIE (unknown ∂ₙu). `--deflated`: deflate with the Mathieu
## quasimodes selected at each k (needs `--direct`, whose null vectors they approximate).
direct = "--direct" in ARGS
deflated = "--deflated" in ARGS
deflated && !direct && error("--deflated needs --direct")
suffix = "_npplo$npplo" * (direct ? "_direct" : "") * (deflated ? "_deflated" : "")
op_label = ", npplo=$npplo" * join([", direct", ", deflated"][[direct, deflated]])
θ_mouth = 9π / 10 # mouth arc |θ| > θ_mouth of the inner ellipse
deflation = deflated ? ((prob, k) -> select_quasimodes(prob.Q, k; θ_mouth)) : nothing

bigger_cavity_problem(k) = cavity_problem(k; geo="elliptic_cavity_bigger_2D", npplo, direct)

## Fixed overall tolerance, sweep frequency k for the bigger (more closed, narrower-mouth) cavity: looking for
## scattering resonances as spikes in exact_gmres's iteration count / dips
## in sigma_m. Range and bouncing-ball mode frequencies follow
## https://arxiv.org/pdf/2102.05367's 2D experiments (k in [50, 300]).
kmin, kmax = 50.0, 300.0
tol = 1e-6

ks_uniform = collect(range(kmin, kmax; length=26))
resonances = read_resonances(
    joinpath(pkgdir(InexactGMRES), "data", "bad_frequencies_ellipsis_half_even_modes_dir_first.csv");
    kmin, kmax,
)
ks = sort(unique(vcat(ks_uniform, resonances)))

replot = "--replot" in ARGS
df = replot ? CSV.read("frequency_sweep_bigger_cavity$suffix.csv", DataFrame) : frequency_sweep(bigger_cavity_problem, ks; tol, deflation, igmres_kwargs=(; relaxed_recycling=true), estimated=true)
replot || CSV.write("frequency_sweep_bigger_cavity$suffix.csv", df)

p1 = plot(df.k, df.exact_it; label="bigger cavity exact_gmres", marker=:circle)
plot!(p1, df.k, df.igmres_it_sigma; label="bigger cavity igmres (σ(H_m) heuristic)", marker=:circle, linestyle=:dash)
plot!(p1, df.k, df.igmres_it_constant_factor; label="bigger cavity igmres (constant bound factor)", marker=:circle, linestyle=:dot)
hasproperty(df, :igmres_it_estimated) && plot!(p1, df.k, df.igmres_it_estimated; label="bigger cavity igmres (estimated σ_m/m, calibrated)", marker=:circle, linestyle=:dashdot)
vline!(p1, resonances; label="bouncing-ball modes", linestyle=:dot, color=:gray, alpha=0.4)
xlabel!(p1, "k")
ylabel!(p1, "Iterations")
title!(p1, "Iteration count vs frequency, bigger cavity (tol=$tol$op_label)")
savefig(p1, "frequency_sweep_bigger_cavity_iterations$suffix.png")

p2 = plot(df.k, df.sigma_m; label="bigger cavity", marker=:circle, yaxis=:log)
vline!(p2, resonances; label="bouncing-ball modes", linestyle=:dot, color=:gray, alpha=0.4)
xlabel!(p2, "k")
ylabel!(p2, "sigma_m(H_m)")
title!(p2, "Hessenberg smallest singular value vs frequency, bigger cavity (tol=$tol$op_label)")
savefig(p2, "frequency_sweep_bigger_cavity_sigma_m$suffix.png")

p3 = plot(df.k, df.speedup_sigma; label="bigger cavity (σ(H_m) heuristic)", marker=:circle)
plot!(p3, df.k, df.speedup_constant_factor; label="bigger cavity (constant bound factor)", marker=:circle, linestyle=:dash)
hasproperty(df, :igmres_it_estimated) && plot!(p3, df.k, df.speedup_estimated; label="bigger cavity (estimated σ_m/m, calibrated)", marker=:circle, linestyle=:dot)
vline!(p3, resonances; label="bouncing-ball modes", linestyle=:dot, color=:gray, alpha=0.4)
xlabel!(p3, "k")
ylabel!(p3, "Speed-up (exact_gmres / igmres)")
title!(p3, "igmres speed-up vs frequency, bigger cavity (tol=$tol$op_label)")
savefig(p3, "frequency_sweep_bigger_cavity_speedup$suffix.png")

p4 = plot(df.k, df.true_res_sigma; label="true residual (σ(H_m))", marker=:circle, yaxis=:log)
plot!(p4, df.k, df.true_res_constant_factor; label="true residual (constant bound factor)", marker=:circle, linestyle=:dash)
hasproperty(df, :igmres_it_estimated) && plot!(p4, df.k, df.true_res_estimated; label="true residual (estimated σ_m/m, calibrated)", marker=:circle, linestyle=:dot)
plot!(p4, df.k, df.gap_sigma; label="||r - r̃|| (σ(H_m))", marker=:star5, linestyle=:dot)
plot!(p4, df.k, df.gap_constant_factor; label="||r - r̃|| (constant bound factor)", marker=:star5, linestyle=:dashdot)
hline!(p4, [tol]; label="tol", color=:black)
vline!(p4, resonances; label="bouncing-ball modes", linestyle=:dot, color=:gray, alpha=0.4)
xlabel!(p4, "k")
ylabel!(p4, "Relative to ||g||, at convergence")
title!(p4, "igmres final true residual and gap, bigger cavity (tol=$tol$op_label)")
savefig(p4, "frequency_sweep_bigger_cavity_residual$suffix.png")

plot_frequency_extras(df, "bigger_cavity", "bigger cavity", suffix, op_label, resonances; tol)
