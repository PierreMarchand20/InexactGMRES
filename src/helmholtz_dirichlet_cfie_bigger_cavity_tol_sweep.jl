using InexactGMRES
using Plots
using CSV

include(joinpath(pkgdir(InexactGMRES), "src", "experiment_utils.jl"))

bigger_cavity_problem(k) = cavity_problem(k; geo="elliptic_cavity_bigger_2D")

## Fixed geometry and frequency, sweep the overall igmres/exact_gmres
## tolerance for the bigger (more closed, narrower-mouth) cavity.
## k is a representative frequency -- change it to probe a known mode from
## the frequency sweep instead.
k = 100.0
tols = [1e-2, 1e-4, 1e-6, 1e-8]

df = tol_sweep(bigger_cavity_problem, k, tols)
CSV.write("tol_sweep_bigger_cavity.csv", df)

p1 = plot(df.tol, df.speedup; label="bigger cavity", marker=:circle, xaxis=:log)
xlabel!(p1, "Overall tolerance")
ylabel!(p1, "Speed-up (exact_gmres / igmres)")
title!(p1, "igmres speed-up at k=$k (bigger cavity)")
savefig(p1, "tol_sweep_bigger_cavity_speedup.png")

p2 = plot(df.tol, df.rel_error; label="bigger cavity", marker=:circle, xaxis=:log, yaxis=:log)
plot!(p2, df.tol, df.tol; label="tol", linestyle=:dash, color=:black)
xlabel!(p2, "Overall tolerance")
ylabel!(p2, "Relative residual ||L*y - g||/||g||")
title!(p2, "igmres accuracy at k=$k (bigger cavity)")
savefig(p2, "tol_sweep_bigger_cavity_accuracy.png")

p3 = plot(df.tol, df.igmres_it; label="igmres", marker=:circle, xaxis=:log)
plot!(p3, df.tol, df.exact_it; label="exact_gmres", marker=:circle, linestyle=:dash)
xlabel!(p3, "Overall tolerance")
ylabel!(p3, "Iterations")
title!(p3, "Iteration count at k=$k (bigger cavity)")
savefig(p3, "tol_sweep_bigger_cavity_iterations.png")
