using InexactGMRES
using Plots
using CSV

include(joinpath(pkgdir(InexactGMRES), "src", "experiment_utils.jl"))

bigger_cavity_problem(k) = cavity_problem(k; geo="elliptic_cavity_bigger_2D")

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

df = frequency_sweep(bigger_cavity_problem, ks; tol)
CSV.write("frequency_sweep_bigger_cavity.csv", df)

p1 = plot(df.k, df.exact_it; label="bigger cavity exact_gmres", marker=:circle)
plot!(p1, df.k, df.igmres_it_sigma; label="bigger cavity igmres (σ(H_m) heuristic)", marker=:circle, linestyle=:dash)
plot!(p1, df.k, df.igmres_it_constant_factor; label="bigger cavity igmres (constant bound factor)", marker=:circle, linestyle=:dot)
vline!(p1, resonances; label="bouncing-ball modes", linestyle=:dot, color=:gray, alpha=0.4)
xlabel!(p1, "k")
ylabel!(p1, "Iterations")
title!(p1, "Iteration count vs frequency, bigger cavity (tol=$tol)")
savefig(p1, "frequency_sweep_bigger_cavity_iterations.png")

p2 = plot(df.k, df.sigma_m; label="bigger cavity", marker=:circle, yaxis=:log)
vline!(p2, resonances; label="bouncing-ball modes", linestyle=:dot, color=:gray, alpha=0.4)
xlabel!(p2, "k")
ylabel!(p2, "sigma_m(H_m)")
title!(p2, "Hessenberg smallest singular value vs frequency, bigger cavity (tol=$tol)")
savefig(p2, "frequency_sweep_bigger_cavity_sigma_m.png")

p3 = plot(df.k, df.speedup_sigma; label="bigger cavity (σ(H_m) heuristic)", marker=:circle)
plot!(p3, df.k, df.speedup_constant_factor; label="bigger cavity (constant bound factor)", marker=:circle, linestyle=:dash)
vline!(p3, resonances; label="bouncing-ball modes", linestyle=:dot, color=:gray, alpha=0.4)
xlabel!(p3, "k")
ylabel!(p3, "Speed-up (exact_gmres / igmres)")
title!(p3, "igmres speed-up vs frequency, bigger cavity (tol=$tol)")
savefig(p3, "frequency_sweep_bigger_cavity_speedup.png")

p4 = plot(df.k, df.true_res_sigma; label="true residual (σ(H_m))", marker=:circle, yaxis=:log)
plot!(p4, df.k, df.true_res_constant_factor; label="true residual (constant bound factor)", marker=:circle, linestyle=:dash)
plot!(p4, df.k, df.gap_sigma; label="||r - r̃|| (σ(H_m))", marker=:star5, linestyle=:dot)
plot!(p4, df.k, df.gap_constant_factor; label="||r - r̃|| (constant bound factor)", marker=:star5, linestyle=:dashdot)
hline!(p4, [tol]; label="tol", color=:black)
vline!(p4, resonances; label="bouncing-ball modes", linestyle=:dot, color=:gray, alpha=0.4)
xlabel!(p4, "k")
ylabel!(p4, "Relative to ||g||, at convergence")
title!(p4, "igmres final true residual and gap, bigger cavity (tol=$tol)")
savefig(p4, "frequency_sweep_bigger_cavity_residual.png")
