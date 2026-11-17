module InexactGMRES

using LinearAlgebra
using HMatrices

export igmres

include("utils.jl")



"""
    igmres(A, b; maxiter, restart, see_r, tol, precision_strategy=rel_to_eps, track_true_residual=false, track_bound=false,
           max_corrections=10, correction_log=nothing, recycle=10, relaxed_recycling=false, eta_log=nothing, deflation=nothing)

`precision_strategy(res, tol)` computes the relative tolerance used for the
approximate matrix-vector product at each iteration, given the current
relative residual `res = ‖r̃_{k-1}‖/‖b‖` and the overall relative target
`tol`. Pass a custom function to use a different schedule than the default
(see [`rel_to_eps`](@ref)).

When `track_true_residual=true`, also return, as 4th and 5th return values:
- `true_residuals`: `norm(A*x_k - b)/norm(b)` at each iteration (an exact
  matvec against the raw `A`, not the approximate `A_iterable`).
- `residual_gap`: `norm(r_k - r̃_k)` at each iteration, the true residual
  vector `r_k = b - A*x_k` minus the internal ("tilde") residual vector
  `r̃_k` reconstructed from the (unrotated) Hessenberg matrix and the
  current Krylov coefficients, in absolute (not relative) units, matching
  `bound_right4`. This is the quantity the Simoncini-Szyld bound (4.4)
  actually bounds — note it is *not* `abs(norm(r_k) - norm(r̃_k))`, which
  can be much smaller by the reverse triangle inequality.
Both are extra work done only when requested, for diagnostics/plots, not
for the default solve path.

When `track_bound=true`, also return `bound_right4` as an extra value
(after `true_residuals, residual_gap` if `track_true_residual` is also
requested): the running evaluation, at each iteration, of formula (4.4)
from Simoncini & Szyld's "Theory of Inexact Krylov Subspace Methods and
Applications to Scientific Computing" (SIAM J. Sci. Comput., 2003) — an
upper bound on the gap between the true and internal ("tilde") residual.
It is estimated as `Σ_n η_n |y_n|` over the current cycle, taking
`‖E_n v_n‖ ≈ η_n`; a restart recomputes the residual exactly, so each cycle
starts from a zero gap.

When the projection residual reaches `tol`, the true residual `b - A*x` is
computed with one accurate matvec (the raw `A`). If it is above `tol` (the
relaxation was too coarse), the solve restarts from `x` with that exact
residual, and all later tolerances are multiplied by
`0.5 * tol * ‖b‖ / ‖b - A*x‖`. This happens at most `max_corrections`
times (`0` disables the check); after that, `x` is returned as is. The
iteration counts at which corrections happen are pushed to `correction_log`
when a vector is given, and the tolerance of every product of the solve to
`eta_log` (`0.0` for the accurate ones).
A cycle also ends after `restart` iterations, and
the next one starts from the current iterate.

At each restart, `recycle` harmonic Ritz vectors `U` (smallest harmonic Ritz
values) are kept from the last cycle, as in GCRO-DR (Parks, de Sturler,
Mackey, Johnson & Maiti, SIAM J. Sci. Comput., 2006). `C = A*U` is
recomputed with `recycle` accurate matvecs (the relaxed products of the last
cycle are not accurate enough), and the next cycle minimizes over
`span(U) + K((I - CCᴴ)A, (I - CCᴴ)r0)`. `0` disables it; it is only done for
complex problems. With `relaxed_recycling=true`, these products are relaxed
like the Arnoldi ones, with the tolerance scaled by `|θ_min|/‖A‖` (smallest
harmonic Ritz value kept, `‖A‖` estimated from the Hessenberg columns): the
normalized `U` has norm ≈ `1/|θ_min|`, so the error of `A*U` is amplified by
`‖A‖/|θ_min|` in the residual gap. Near-singular recycled directions thus
still get nearly accurate products.

`deflation`, a matrix whose columns span an approximate invariant subspace
(e.g. quasimodes), deflates the solve from the start the same way (one
accurate matvec per column).
"""
function igmres(A, b; maxiter=size(A, 2), restart=min(length(b), size(A, 2)), see_r=false, tol=sqrt(eps()),
                precision_strategy=rel_to_eps, track_true_residual=false, track_bound=false,
                max_corrections=10, correction_log=nothing, recycle=10, relaxed_recycling=false, eta_log=nothing, deflation=nothing)
    T = promote_type(eltype(A), eltype(b))
    bheta = norm(b)
    residuals = Float64[]
    true_residuals = Float64[]
    residual_gap = Float64[]
    bound_right4 = Float64[]
    x0 = zeros(T, length(b)) # solution accumulated over cycles
    r0 = Vector{T}(b)        # exact residual at the start of the cycle
    A_iterable = A isa HMatrices.HMatrix ? HMatrices.ITerm(A, 1.0) : A
    it = 0
    corrections = 0
    scale = 1.0 # shrinks the requested tolerances after each correction
    normA = 0.0 # lower bound on ‖A‖ from the Hessenberg columns
    log_accurate(n=1) = eta_log === nothing || append!(eta_log, zeros(n))
    U = zeros(T, length(b), 0) # recycled space, with C = A*U orthonormal
    C = zeros(T, length(b), 0)
    if deflation !== nothing
        F = qr(reduce(hcat, [accurate_mul(A, u) for u in eachcol(deflation)])) # accurate C = A*U
        C = Matrix(F.Q)
        U = deflation / F.R
        log_accurate(size(U, 2))
    end

    while it < maxiter
        x0 += U * (C' * r0) # minimize over span(U) first
        r0 -= C * (C' * r0)
        β = norm(r0)
        Q = [r0 / β]
        H = Vector{Vector{T}}()     # Hessenberg columns, rotated in place
        H_raw = Vector{Vector{T}}() # the same columns before rotation
        J = Vector{Any}(undef, restart)
        e1 = zeros(T, restart + 1)
        e1[1] = β
        eta = Float64[]             # matvec tolerances used in this cycle
        y = zeros(T, restart)       # Krylov coefficients
        B = Vector{Vector{T}}()     # columns of Cᴴ A V
        res = β
        k = 0
        corrected = false
        while k < restart && it < maxiter
            k += 1
            η = min(scale * precision_strategy(res / bheta, tol), 1.0)
            A_iterable isa HMatrices.ITerm && (A_iterable.rtol = η)
            push!(eta, η)
            eta_log === nothing || push!(eta_log, η)
            my_arnoldi!(Q, H, A_iterable, k)
            # project the new vector out of span(C); since V ⊥ C, H[k][1:k] is unchanged
            w = Q[k+1] * H[k][k+1]
            push!(B, C' * w)
            w -= C * B[k]
            H[k][k+1] = norm(w)
            Q[k+1] = w / H[k][k+1]
            push!(H_raw, copy(H[k]))
            normA = max(normA, sqrt(norm(H_raw[k])^2 + norm(B[k])^2)) # ‖A v_k‖
            my_rotation!(H, J, e1, k)
            triangularsquares!(y, H, e1[1:k])
            res = abs(e1[k+1])
            it += 1
            push!(residuals, res / bheta)

            # Simoncini-Szyld (4.4) with ||E_n v_n|| ≈ η_n: estimated ||r_k - r̃_k||
            # accumulated in this cycle (earlier cycles start from an exact residual)
            est = sum(eta[n] * abs(y[n]) for n in 1:k)
            track_bound && push!(bound_right4, est)

            if track_true_residual
                xk = gcro_solution(x0, Q, y, U, B, k)
                true_res_vec = b - accurate_mul(A, xk)
                push!(true_residuals, norm(true_res_vec) / bheta)
                Hy = zeros(T, k + 1) # H̄_k y_k, from the unrotated columns
                for i in 1:k, j in eachindex(H_raw[i])
                    Hy[j] += H_raw[i][j] * y[i]
                end
                tilde_res_vec = r0 - sum(Q[i] * Hy[i] for i in 1:(k+1))
                push!(residual_gap, norm(true_res_vec - tilde_res_vec))
            end

            see_r && println("Iteration: ", it, " Current residual: ", res / bheta, " estimated gap: ", est / bheta)

            if res / bheta < tol
                sol = gcro_solution(x0, Q, y, U, B, k)
                r = corrections < max_corrections ? b - accurate_mul(A, sol) : nothing
                r === nothing || log_accurate()
                if r === nothing || norm(r) < tol * bheta
                    track_true_residual && track_bound && return sol, residuals, it, true_residuals, residual_gap, bound_right4
                    track_true_residual && return sol, residuals, it, true_residuals, residual_gap
                    track_bound && return sol, residuals, it, bound_right4
                    return sol, residuals, it
                end
                # the relaxation was too coarse: restart from sol with its exact residual
                corrections += 1
                correction_log === nothing || push!(correction_log, it)
                scale *= 0.5 * tol * bheta / norm(r)
                x0, r0, corrected = sol, r, true
                break
            end
        end
        # new cycle from the current iterate, with an exact residual
        corrected || (x0 = gcro_solution(x0, Q, y, U, B, k); r0 = b - accurate_mul(A, x0); log_accurate())
        # ponytail: real problems would need real Schur vectors, skipped (complex conjugate pairs)
        if recycle > 0 && T <: Complex && it < maxiter
            U, θ = harmonic_ritz_vectors(U, C, Q, H_raw, B, k, recycle)
            ηU = 0.0
            if relaxed_recycling && A_iterable isa HMatrices.ITerm
                ηU = min(scale * minimum(abs, θ) / normA * precision_strategy(norm(r0) / bheta, tol), 1.0)
                A_iterable.rtol = ηU
            end
            mulU(u) = ηU == 0 ? accurate_mul(A, u) : mul!(similar(u), A_iterable, u, 1, 0; threads=false)
            F = qr(reduce(hcat, [mulU(u) for u in eachcol(U)])) # C = A*U
            eta_log === nothing || append!(eta_log, fill(ηU, size(U, 2)))
            C = Matrix(F.Q)
            U = U / F.R
        end
    end
    throw("Maximum iteration reached")
end


###exact implementation, for comparing reasons

function exact_gmres(A, b; maxiter=size(A, 2), restart=min(length(b), size(A, 2)), see_r=false, tol=sqrt(eps()), return_H=false)
    #choose type to create vectors and matrices
    TA = eltype(A)
    Tb = eltype(b)
    T = promote_type(TA, Tb)


    x = zeros(T, size(b))#will hold answer
    residuals = Vector{Float64}()
    it = 0
    bheta = norm(b)
    m = restart
    res = bheta

    while it < maxiter
        Q = Vector{Vector{T}}()
        H = Vector{Vector{T}}()
        J = Vector{Any}(undef, m)#

        #resduals =
        e1 = zeros(T, m + 1, 1)
        e1[1] = bheta
        v = b / bheta
        push!(Q, v)
        for k = 1:m
            if it >= maxiter
                break
            end

            ###Arnold's iteration inside GMRES to use Q,H from past iterations
            #----------------------------------------------
            my_arnoldi!(Q, H, A, k)
            #---------------------------#

            ###Givens rotation
            #-----------------------------------
            #Rotations on H to make it triangular
            my_rotation!(H, J, e1, k)
            #-------------------------------------

            triangularsquares!(x, H, e1[1:k])

            #Residuals are always stored in the last element of e1
            res = norm(e1[k+1])
            push!(residuals, res/bheta)

            it += 1
            if see_r
                println("Iteration: ", it, " Current residual: ", res)
            end

            if res/bheta < tol # stop on the relative (not absolute) projection residual
                y = zero(x)
                for n = 1:k
                    y += Q[n] * x[n]
                end
                # println("Finished at iteration: ", it + 1, " Final residual: ", res)
                if return_H
                    Hmat = zeros(T, k+1, k)
                    for j = 1:k
                        Hmat[1:(j+1), j] = H[j]
                    end
                    return y, residuals, it, Hmat
                end
                return y, residuals, it
            end
        end
    end #main while loop
    # y = zero(x)
    # for n = 1:length(x)
    #     y += Q[n] * x[n]
    # end


    # println("Maximum iteration reached")
    throw("Maximum iteration reached")
end

end # module InexactGMRES
