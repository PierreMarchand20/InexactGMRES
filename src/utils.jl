
"""
    triangularsquares!(y,A,b)

Solves the system Ax = b with backwards substitution, assuming A is always upper triangular.
A must be a vector of vectors, where each A[i] is a column vector added through push!(A,v).
"""

function triangularsquares!(y, A, b)
    #y is required to have same length as b
    #A is supposed to be upper triangular, and square
    #Here, A is also an vector of vectors, so the expression need to be adapted
    #b is the RHS, same length as A
    m = length(b)
    #x = zeros(ComplexF64,m)#always in complex values
    for k = m:-1:1
        y[k] = b[k]
        for j = m:-1:(k+1)
            y[k] = y[k] - A[j][k] * y[j]
        end
        y[k] = y[k] / A[k][k]
    end
end


"""
    my_arnoldi(Q,H,A,current_it)

Calculates the current_it'th iteration of the Arnoldi's Method.
Q and H are assumed as vectors of vectors, where Q[i] stores the i-th orthonormal vector we'll be using as a basis for Km(A,b)
and H[i] stores de i+1 first values of H's i-th column.
"""
function my_arnoldi!(Q, H, A::HMatrices.ITerm, current_it)
    dummy_v = similar(Q[current_it])
    mul!(dummy_v, A, Q[current_it], 1, 0; threads=false)
    push!(Q, dummy_v) # heavy part should be here
    push!(H, zeros(current_it+1))
    for j = 1:current_it
        H[current_it][j] = (Q[j]') * Q[current_it+1]
        #Q[current_it+1] -= H[current_it][j] * Q[j]

        #mul!(C,A,B,alpha,bheta) : C <- (A*B)alpha + bhetaC
        mul!(Q[current_it+1], I, Q[j], -H[current_it][j], 1)
    end
    H[current_it][current_it+1] = norm(Q[current_it+1])

    #Q[current_it+1] /= H[current_it][current_it+1]
    lmul!(1/H[current_it][current_it+1], Q[current_it+1])
end

# Accurate product. HMatrix products default to HMatrices' threaded code path,
# which is ~20% slower even on one thread: use the same serial path as ITerm.
accurate_mul(A, x) = A * x
function accurate_mul(A::HMatrices.HMatrix, x)
    y = similar(x, promote_type(eltype(A), eltype(x)), size(A, 1))
    return mul!(y, A, x, 1, 0; threads=false)
end

function my_arnoldi!(Q, H, A::AbstractMatrix, current_it)
    dummy_v = accurate_mul(A, Q[current_it])
    push!(Q, dummy_v) # heavy part should be here
    push!(H, zeros(current_it+1))
    for j = 1:current_it
        H[current_it][j] = (Q[j]') * Q[current_it+1]
        #Q[current_it+1] -= H[current_it][j] * Q[j]

        #mul!(C,A,B,alpha,bheta) : C <- (A*B)alpha + bhetaC
        mul!(Q[current_it+1], I, Q[j], -H[current_it][j], 1)
    end
    H[current_it][current_it+1] = norm(Q[current_it+1])

    #Q[current_it+1] /= H[current_it][current_it+1]
    lmul!(1/H[current_it][current_it+1], Q[current_it+1])
end

"""
    my_rotation!(H,J,rhs,current_it)

Applies Givens's Operator to rotate H and transform it in a upper triangular matrix.
The sequence of operators is assumed to be stored in J.
It also applies the transformations to the Right Hand Side(RHS).
"""
function my_rotation!(H, J, rhs, current_it)
    #given two integers k and k+1, it will return an object G with 4 numbers:(k,k+1,s,c)
    #b=G*a with a column vector a of length k+1 will return a new vector b which has
    #b[k+1] = 0 and b[k] changed by the rotation
    for j = 1:(current_it-1)

        #H[current_it] = J[j] * H[current_it]
        lmul!(J[j], H[current_it])
    end

    J[current_it], = givens(H[current_it][current_it], H[current_it][current_it+1], current_it, current_it + 1)

    #H[current_it] = J[current_it] * H[current_it]
    lmul!(J[current_it], H[current_it])


    #rhs[:] = J[current_it] * rhs
    lmul!(J[current_it], rhs)

end



"""
    rel_to_eps(res,tol)

Matvec tolerance `min(tol/min(res, 1), 1)` from the relative residual
`res = ‖r̃_{k-1}‖/‖b‖` of iteration k-1 and the relative target `tol`.

Relaxation heuristic from Simoncini & Szyld 2003 (SIAM J. Sci. Comput.):
bounding ‖r_m - r̃_m‖/‖b‖ ≤ Σ_k ‖E_k‖ (‖r̃_{k-1}‖/‖b‖) / σ_m(H_m) by `tol`
gives ‖E_k‖ ≤ ℓ tol / (‖r̃_{k-1}‖/‖b‖) with ℓ = σ_m(H_m)/m; this is ℓ = 1
(the 3-arg method takes ℓ as `bound_factor`).
"""
function rel_to_eps(res::Float64, tol::Float64)
    return min((tol/min(res, 1)), 1)
end

#next one is a test, mainly to test different IGmres'es versions and iteration modification

function rel_to_eps(bound_factor::Float64, res::Float64, tol::Float64)
    return min(bound_factor*(tol/min(res, 1)), 1)
end

"""
    igmres_precision_study(A, b, tol; bound_factor=1.0)

Run `exact_gmres` once (to get the smallest singular value `sigma_m` of its
final Hessenberg matrix) and `igmres` twice on `(A, b, tol)`, both using the
3-arg `rel_to_eps(factor, res, t)` relaxation formula with a *constant*
`factor`: once with `factor = sigma_m/m`, once with `factor = bound_factor`
(1.0, i.e. the same schedule as the default 2-arg `rel_to_eps`, by
definition). In both cases the matvec tolerance itself still varies with
`res` every iteration — only the scaling factor is held constant. Both
igmres runs track the true residual (`track_true_residual=true`) and the
Simoncini & Szyld bound (`track_bound=true`) at each iteration, and record
the matvec rtol actually used by their precision strategy.

Returns a NamedTuple with, for each strategy, the solution `y`, the internal
residuals, true residuals, residual_gap (`norm(r_k - r̃_k)`, in absolute
units), bound_right4, recorded heuristic values, and iteration count, plus
`sigma_m` and the exact solution `y_exact` (from `exact_gmres`) themselves.
"""
function igmres_precision_study(A, b, tol; bound_factor=1.0)
    y_exact, _, m, H_m = exact_gmres(A, b; tol, return_H=true)
    sigma_m = svd(H_m).S[end]

    sigma_heuristic = Float64[]
    y_sigma, residuals_sigma, it_sigma, true_residuals_sigma, residual_gap_sigma, bound_right4_sigma = igmres(A, b; tol,
        track_true_residual=true, track_bound=true,
        precision_strategy=(res, t) -> begin
            v = rel_to_eps(sigma_m/m, res, t)
            push!(sigma_heuristic, v)
            v
        end)

    constant_factor_heuristic = Float64[]
    y_constant_factor, residuals_constant_factor, it_constant_factor, true_residuals_constant_factor, residual_gap_constant_factor, bound_right4_constant_factor = igmres(A, b; tol,
        track_true_residual=true, track_bound=true,
        precision_strategy=(res, t) -> begin
            v = rel_to_eps(bound_factor, res, t)
            push!(constant_factor_heuristic, v)
            v
        end)

    return (; sigma_m, y_exact,
        y_sigma, residuals_sigma, true_residuals_sigma, residual_gap_sigma, bound_right4_sigma, sigma_heuristic, it_sigma,
        y_constant_factor, residuals_constant_factor, true_residuals_constant_factor, residual_gap_constant_factor,
        bound_right4_constant_factor, constant_factor_heuristic, it_constant_factor)
end

"""
    effective_compression_ratio(H::HMatrices.HMatrix, rtol)

Like `HMatrices.compression_ratio(H)`, but counting only the rank-1 terms
`HMatrices.ITerm`'s inexact matvec would actually use at the given `rtol`
for each admissible (low-rank) leaf -- up to the term after the last one
with `R.rel_er > rtol`, plus any terms beyond `R.rel_er`, the same truncation
`ITerm`'s own `mul!` applies -- instead of
each leaf's full stored rank. Full (non-admissible) leaves always cost
their dense size, same as the exact product. A value higher than
`HMatrices.compression_ratio(H)` means the inexact product at this `rtol`
is, in effect, using less of the matrix than its own (static) compression
already keeps.

    effective_compression_ratio(H, rtols::AbstractVector)

Broadcasts over a sequence of per-iteration `rtol` values (e.g. the
`sigma_heuristic`/`constant_factor_heuristic` arrays from
[`igmres_precision_study`](@ref)), returning one ratio per iteration.
"""
function effective_compression_ratio(H::HMatrices.HMatrix, rtol::Float64)
    total = length(H)
    used = 0
    for leaf in HMatrices.leaves(H)
        m, n = size(leaf)
        if HMatrices.isadmissible(leaf)
            R = HMatrices.data(leaf)
            r = length(R.rel_er) # columns beyond it (sparse corrections added by axpy!) are always used
            k = findlast(x -> x > rtol, R.rel_er)
            k = min(isnothing(k) ? 1 : k + 1, r)
            used += (k + size(R.A, 2) - r) * (m + n)
        else
            used += m * n
        end
    end
    return total / used
end

function effective_compression_ratio(H::HMatrices.HMatrix, rtols::AbstractVector{Float64})
    return effective_compression_ratio.(Ref(H), rtols)
end

"""
    dropped_term_estimate(H, x, η)

Estimate of `‖E x‖`, the error of the product by `H` truncated at `η`
(`HMatrices.ITerm`): the first dropped ACA term of each low-rank block,
applied to `x`. Costs one rank-one term per low-rank block.
"""
function dropped_term_estimate(H::HMatrices.HMatrix, x, η)
    xp = x[HMatrices.colperm(H)]
    y = zeros(promote_type(eltype(H), eltype(x)), size(H, 1)) # permuted, which leaves the norm unchanged
    off = HMatrices.pivot(H) .- 1
    for leaf in HMatrices.leaves(H)
        HMatrices.isadmissible(leaf) || continue
        R = HMatrices.data(leaf)
        r = length(R.rel_er) # terms beyond it are never dropped
        k = findlast(>(η), R.rel_er)
        k = min(isnothing(k) ? 1 : k + 1, r)
        k == r && continue
        I, J = HMatrices.rowrange(leaf) .- off[1], HMatrices.colrange(leaf) .- off[2]
        y[I] .+= view(R.A, :, k + 1) .* dot(view(R.B, :, k + 1), view(xp, J))
    end
    return norm(y)
end

"""
    sigma_over_m(H_raw, k, residuals, tol; window=10)

On-the-fly estimate of the Simoncini-Szyld factor `σ_m(H̄_m)/m` after `k`
iterations of the cycle: `σ_min(H̄_k)` over `m̂ = k + ` the iterations still
needed to reach `tol` at the mean convergence rate of the last `window`
relative residuals.
"""
function sigma_over_m(H_raw, k, residuals, tol; window=10)
    Hbar = zeros(eltype(H_raw[1]), k + 1, k)
    for j in 1:k
        Hbar[1:(j+1), j] = H_raw[j]
    end
    σ = svdvals(Hbar)[end] # ponytail: full SVD each iteration, O(k³); an incremental condition estimator if k gets large
    w = min(window, length(residuals) - 1)
    ρ = w > 0 ? (residuals[end] / residuals[end-w])^(1 / w) : 1.0
    remaining = ρ < 1 ? max(log(tol / residuals[end]) / log(ρ), 0.0) : float(k) # no rate yet: assume as many again
    return σ / (k + remaining)
end

"""
    gcro_solution(x0, Q, y, U, B, k)

Iterate `x0 + V_k y - U B y` of a GCRO cycle: with `C = A U`, `B = Cᴴ A V_k` and
`r0 ⊥ C`, choosing `-B y` as the coefficients on `U` cancels the `C` part of
the residual, so `y` is the usual GMRES least squares solution on `H̄_k`.
"""
function gcro_solution(x0, Q, y, U, B, k)
    k == 0 && return x0
    return x0 + sum(Q[n] * y[n] for n in 1:k) - U * sum(B[n] * y[n] for n in 1:k)
end

"""
    harmonic_ritz_vectors(U, C, Q, H_raw, B, k, p)

The `p` harmonic Ritz vectors of `A` with the smallest harmonic Ritz values
in `span([U V_k])`, and their harmonic Ritz values, using `A [U V_k] = [C V_{k+1}] G` with
`G = [I B; 0 H̄_k]` (GCRO-DR, Parks et al. 2006): `Gᴴ G z = θ Gᴴ [C V_{k+1}]ᴴ [U V_k] z`.
"""
function harmonic_ritz_vectors(U, C, Q, H_raw, B, k, p)
    T = eltype(U)
    q = size(U, 2)
    Hbar = zeros(T, k + 1, k)
    for j in 1:k
        Hbar[1:(j+1), j] = H_raw[j]
    end
    G = [Matrix{T}(I, q, q) reduce(hcat, B[1:k]); zeros(T, k + 1, q) Hbar]
    W = hcat(U, Q[1:k]...)
    Wp = hcat(C, Q[1:(k+1)]...)
    θ, Z = eigen(G' * G, G' * (Wp' * W))
    idx = sortperm(abs.(θ))[1:min(p, q + k)]
    return W * Z[:, idx], θ[idx]
end
