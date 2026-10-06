module InexactGMRES

using LinearAlgebra
using HMatrices

export igmres

include("utils.jl")



"""
    igmres(A, b; maxiter, restart, see_r, tol, precision_strategy=rel_to_eps, track_true_residual=false, track_bound=false)

`precision_strategy(res, tol)` computes the relative tolerance used for the
approximate matrix-vector product at each iteration, given the current
residual `res` and the overall target `tol`. Pass a custom function to use a
different schedule than the default (see [`rel_to_eps`](@ref)).

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
"""
function igmres(A, b; maxiter=size(A, 2), restart=min(length(b), size(A, 2)), see_r=false, tol=sqrt(eps()), precision_strategy=rel_to_eps, track_true_residual=false, track_bound=false)
    #choose type to create vectors and matrices
    TA = eltype(A)
    Tb = eltype(b)
    T = promote_type(TA, Tb)


    x = zeros(T, size(b))#will hold answer
    #residuals = zeros(real(T), maxiter) # will hold residuals
    residuals = Vector{Float64}()
    true_residuals = Vector{Float64}()
    residual_gap = Vector{Float64}()
    bound_right4 = Vector{Float64}()
    eta_history = Vector{Float64}()
    it = 0
    bheta = norm(b)
    m = restart
    res = bheta
    current_perror = Float64
    A_iterable = A isa HMatrices.HMatrix ? HMatrices.ITerm(A, res) : A
    while it < maxiter
        Q = Vector{Vector{T}}()
        H = Vector{Vector{T}}()
        H_raw = Vector{Vector{T}}() # unrotated Hessenberg columns, for tilde-residual reconstruction
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


            ###Transformation of current residue and overall tolerance in the new error we'll use
            current_perror = precision_strategy(res, tol)
            A_iterable isa HMatrices.ITerm && (A_iterable.rtol = current_perror)
            track_bound && push!(eta_history, current_perror)
            ###Arnold's iteration inside GMRES to use Q,H from past iterations
            #----------------------------------------------
            my_arnoldi!(Q, H, A_iterable, k)#no new vector is created, everything is done directly in H and Q
            #---------------------------#

            track_true_residual && push!(H_raw, copy(H[k])) # save before my_rotation! mutates H[k] in place

            ###Givens rotation
            #-----------------------------------
            #Rotations on H to make it triangular
            my_rotation!(H, J, e1, k)
            #-------------------------------------

            triangularsquares!(x, H, e1[1:k])

            #Residuals are always stored in the last element of e1
            res = norm(e1[k+1])
            #residuals[it] = res/bheta
            push!(residuals, res/bheta)

            if track_true_residual
                y_k = zero(x)
                for n = 1:k
                    y_k += Q[n] * x[n]
                end
                true_res_vec = b - A*y_k
                push!(true_residuals, norm(true_res_vec)/bheta)

                # reconstruct the tilde residual vector r̃_k = b - V_{k+1}*(H̄_k*x_k),
                # using the raw (unrotated) Hessenberg columns H_raw
                Hx = zeros(T, k+1)
                for i = 1:k
                    col = H_raw[i]
                    for j = 1:length(col)
                        Hx[j] += col[j]*x[i]
                    end
                end
                tilde_res_vec = copy(b)
                for i = 1:(k+1)
                    tilde_res_vec -= Q[i]*Hx[i]
                end
                push!(residual_gap, norm(true_res_vec - tilde_res_vec))
            end

            if track_bound
                dummy_right = 0.0
                for n = 1:k
                    dummy_right += eta_history[n] * abs(x[n])
                end
                push!(bound_right4, dummy_right)
            end

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
                if track_true_residual && track_bound
                    return y, residuals, it, true_residuals, residual_gap, bound_right4
                elseif track_true_residual
                    return y, residuals, it, true_residuals, residual_gap
                elseif track_bound
                    return y, residuals, it, bound_right4
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



"""
    trefethen_fast(m)

Create the matrix A from Trefethen and Bau's book, formula 35.17, in examples 35.1 and 35.2
"""
function trefethen_fast(m)
    A = Matrix((2.0 + 0im) * I, m, m) + 0.5 * randn(m, m) / sqrt(m)
    D = Matrix((1.0 + 0im) * I, m, m)
    for i = 0:(m-1)
        D[i+1, i+1] = (-2 + 2 * sin((i * pi) / (m - 1))) + cos((i * pi) / (m - 1))im
    end
    return A
end

function trefethen_slow(m)
    A = trefethen_fast(m)
    D = Matrix((1.0 + 0im) * I, m, m)
    for i = 0:(m-1)
        D[i+1, i+1] = (-2 + 2 * sin((i * pi) / (m - 1))) + cos((i * pi) / (m - 1))im
    end
    return A + D
end



end # module InexactGMRES
