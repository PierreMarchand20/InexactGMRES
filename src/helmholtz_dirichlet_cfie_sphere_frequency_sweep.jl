using InexactGMRES
using Plots
using CSV
using Bessels

include(joinpath(pkgdir(InexactGMRES), "src", "experiment_utils.jl"))
default(size=(800, 500), titlefontsize=9)

## `--replot`: rebuild the figures from the CSV of a previous run, without solving.
## `--npplo=N`: points per wavelength (default 10), part of the output names.
i = findfirst(startswith("--npplo="), ARGS)
npplo = i === nothing ? 10 : parse(Int, split(ARGS[i], "=")[2])
suffix = "_npplo$npplo"
op_label = ", npplo=$npplo"

## Fixed overall tolerance, sweep frequency k for the unit sphere: a 3D
## obstacle with no trapping. n grows like k² (about 78k unknowns at k = 18).
## Besides the σ(H_m) oracle and ℓ = 1, runs the practical safe strategy
## (estimated σ_m/m, calibrated products), all with relaxed recycling.
ks = collect(4.0:2.0:18.0)
tol = 1e-6
resonances = Float64[] # none for a convex obstacle

## Scattered field of a sound-soft unit sphere for the plane wave
## exp(ik x·d) (Mie series), at the point x with |x| > 1.
function mie_scattered(k, x, d)
    r, c = norm(x), dot(x, d) / norm(x)
    h(n, z) = sphericalbesselj(n, z) + im * sphericalbessely(n, z)
    P0, P1 = 1.0, c # Legendre polynomials P_n(c), by recurrence
    u = zero(ComplexF64)
    for n in 0:ceil(Int, k + 20)
        Pn = n == 0 ? P0 : P1
        u -= im^n * (2n + 1) * sphericalbesselj(n, k) / h(n, k) * h(n, k * r) * Pn
        n >= 1 && ((P0, P1) = (P1, ((2n + 1) * c * P1 - n * P0) / (n + 1)))
    end
    return u
end

## Relative error of the scattered field D[σ] - ik S[σ] from the exact solve,
## against Mie, on points of the sphere of radius 2 (plain quadrature is
## accurate that far from the boundary).
function mie_error(k)
    (; L, g, Q) = sphere_problem(k; npplo)
    σ, _, _ = InexactGMRES.exact_gmres(L, g; tol)
    d = SVector(cos(π / 4), sin(π / 4), 0.0)
    pts = [2 * normalize(SVector(randn(3)...)) for _ in 1:20]
    num = map(pts) do x
        sum(zip(Q, σ)) do (q, s)
            rv = x - q.coords; ρ = norm(rv)
            G = exp(im * k * ρ) / (4π * ρ)
            dG = -G * (im * k - 1 / ρ) * dot(rv, q.normal) / ρ # ∂G/∂n_y
            (dG - im * k * G) * s * q.weight
        end
    end
    ref = mie_scattered.(k, pts, Ref(d))
    return norm(num - ref) / norm(ref)
end

replot = "--replot" in ARGS
df = replot ? CSV.read("frequency_sweep_sphere$suffix.csv", DataFrame) :
     frequency_sweep(k -> sphere_problem(k; npplo), ks; tol, igmres_kwargs=(; relaxed_recycling=true), estimated=true)
replot || CSV.write("frequency_sweep_sphere$suffix.csv", df)

p1 = plot(df.k, df.exact_it; label="sphere exact_gmres", marker=:diamond)
plot!(p1, df.k, df.igmres_it_sigma; label="sphere igmres (σ(H_m) heuristic)", marker=:diamond, linestyle=:dash)
plot!(p1, df.k, df.igmres_it_constant_factor; label="sphere igmres (constant bound factor)", marker=:diamond, linestyle=:dot)
plot!(p1, df.k, df.igmres_it_estimated; label="sphere igmres (estimated σ_m/m, calibrated)", marker=:diamond, linestyle=:dashdot)
xlabel!(p1, "k")
ylabel!(p1, "Iterations")
title!(p1, "Iteration count vs frequency, sphere (tol=$tol$op_label)")
savefig(p1, "frequency_sweep_sphere_iterations$suffix.png")

p2 = plot(df.k, df.sigma_m; label="sphere", marker=:diamond, yaxis=:log)
xlabel!(p2, "k")
ylabel!(p2, "sigma_m(H_m)")
title!(p2, "Hessenberg smallest singular value vs frequency, sphere (tol=$tol$op_label)")
savefig(p2, "frequency_sweep_sphere_sigma_m$suffix.png")

p3 = plot(df.k, df.speedup_sigma; label="sphere (σ(H_m) heuristic)", marker=:diamond)
plot!(p3, df.k, df.speedup_constant_factor; label="sphere (constant bound factor)", marker=:diamond, linestyle=:dash)
plot!(p3, df.k, df.speedup_estimated; label="sphere (estimated σ_m/m, calibrated)", marker=:diamond, linestyle=:dot)
xlabel!(p3, "k")
ylabel!(p3, "Speed-up (exact_gmres / igmres)")
title!(p3, "igmres speed-up vs frequency, sphere (tol=$tol$op_label)")
savefig(p3, "frequency_sweep_sphere_speedup$suffix.png")

p4 = plot(df.k, df.true_res_sigma; label="true residual (σ(H_m))", marker=:diamond, yaxis=:log)
plot!(p4, df.k, df.true_res_constant_factor; label="true residual (constant bound factor)", marker=:diamond, linestyle=:dash)
plot!(p4, df.k, df.true_res_estimated; label="true residual (estimated σ_m/m, calibrated)", marker=:diamond, linestyle=:dot)
plot!(p4, df.k, df.gap_sigma; label="||r - r̃|| (σ(H_m))", marker=:star5, linestyle=:dot)
plot!(p4, df.k, df.gap_constant_factor; label="||r - r̃|| (constant bound factor)", marker=:star5, linestyle=:dashdot)
hline!(p4, [tol]; label="tol", color=:black)
xlabel!(p4, "k")
ylabel!(p4, "Relative to ||g||, at convergence")
title!(p4, "igmres final true residual and gap, sphere (tol=$tol$op_label)")
savefig(p4, "frequency_sweep_sphere_residual$suffix.png")

plot_frequency_extras(df, "sphere", "sphere", suffix, op_label, resonances; tol)

## Discretization check against Mie at the two ends of the sweep
replot || for k in (first(ks), last(ks))
    println("k = $k: relative error of the scattered field vs Mie = ", mie_error(k))
end
