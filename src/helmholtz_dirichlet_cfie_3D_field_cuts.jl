using InexactGMRES
using Plots

include(joinpath(pkgdir(InexactGMRES), "src", "experiment_utils.jl"))

## Total field u = uᵢ + D[σ] - ik S[σ] in the plane z = 0 (which contains the
## incident direction and the cavities' axis), for the 3D obstacles. The
## scattered field uses the plain quadrature rule, so it is slightly off within
## about a mesh size of the surface: points inside the material, or closer to
## the surface than half a mesh size, are left blank.
## Writes field_cut_<name>_k<k>.png.

const SHELL = cos(9π / 10) # the cavity is cut at x < cos(9π/10)
inside = Dict(
    "sphere" => (x, y) -> x^2 + y^2 < 1,
    "spheroid_3D" => (x, y) -> (x / 1.3)^2 + (y / 0.6)^2 < 1,
    "elliptic_cavity_bigger_3D" => (x, y) -> (x / 1.3)^2 + (y / 0.6)^2 < 1 && x^2 + (y / 0.5)^2 > 1 && x > SHELL,
)

function field_cut(name, k; tol=1e-6, xs=range(-2.5, 2.5; length=201), ys=range(-1.6, 1.6; length=129))
    (; L, g, Q, uᵢ, meshsize) = name == "sphere" ? sphere_problem(k) : cavity_problem(k; geo=name)
    σ, _, its = InexactGMRES.exact_gmres(L, g; tol)
    X = [q.coords for q in Q]; N = [q.normal for q in Q]; W = σ .* [q.weight for q in Q]
    u = fill(NaN + 0im, length(ys), length(xs))
    Threads.@threads for i in eachindex(xs)
        for j in eachindex(ys)
            inside[name](xs[i], ys[j]) && continue
            x = SVector(xs[i], ys[j], 0.0)
            us = zero(ComplexF64); ρmin = Inf
            @inbounds for l in eachindex(X)
                rv = x - X[l]; ρ = norm(rv); ρmin = min(ρmin, ρ)
                G = cis(k * ρ) / (4π * ρ)
                dG = -G * (im * k - 1 / ρ) * dot(rv, N[l]) / ρ # ∂G/∂n_y
                us += (dG - im * k * G) * W[l]
            end
            u[j, i] = ρmin < meshsize / 2 ? NaN : uᵢ(x) + us
        end
    end
    p1 = heatmap(xs, ys, real.(u); aspect_ratio=:equal, color=:balance, clims=(-2, 2), title="Re u", xlabel="x", ylabel="y")
    p2 = heatmap(xs, ys, abs.(u); aspect_ratio=:equal, color=:viridis, title="|u|", xlabel="x")
    p = plot(p1, p2; layout=(1, 2), size=(1300, 450),
             plot_title="Total field in z = 0, $name, k = $k (n = $(length(Q)), $its its)", plot_titlefontsize=10)
    savefig(p, "field_cut_$(name)_k$(k).png")
    println("$name k=$k: n=$(length(Q)), $its its, max |u| = $(round(maximum(abs, filter(!isnan, u)), digits=2))")
end

cases = isempty(ARGS) ? [("sphere", 8.0), ("spheroid_3D", 8.0), ("elliptic_cavity_bigger_3D", 8.0), ("elliptic_cavity_bigger_3D", 10.5)] :
        [(ARGS[1], parse(Float64, ARGS[2]))]
for (name, k) in cases
    field_cut(name, k)
end
