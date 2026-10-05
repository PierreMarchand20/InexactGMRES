
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

function my_arnoldi!(Q, H, A::AbstractMatrix, current_it)
    dummy_v = similar(Q[current_it])
    mul!(dummy_v, A, Q[current_it], 1, 0)
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

Converts the residue from iteration k-1 and overall desired tolerance into and eps we'll use to approximate the original problem's matrix A.

Relaxation heuristic from Simoncini & Szyld 2003 (SIAM J. Sci. Comput.).
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

Returns a NamedTuple with, for each strategy, the internal residuals, true
residuals, residual_gap (`norm(r_k - r̃_k)`, in absolute units), bound_right4,
recorded heuristic values, and iteration count, plus `sigma_m` itself.
"""
function igmres_precision_study(A, b, tol; bound_factor=1.0)
    _, _, m, H_m = exact_gmres(A, b; tol, return_H=true)
    sigma_m = svd(H_m).S[end]

    sigma_heuristic = Float64[]
    _, residuals_sigma, it_sigma, true_residuals_sigma, residual_gap_sigma, bound_right4_sigma = igmres(A, b; tol,
        track_true_residual=true, track_bound=true,
        precision_strategy=(res, t) -> begin
            v = rel_to_eps(sigma_m/m, res, t)
            push!(sigma_heuristic, v)
            v
        end)

    constant_factor_heuristic = Float64[]
    _, residuals_constant_factor, it_constant_factor, true_residuals_constant_factor, residual_gap_constant_factor, bound_right4_constant_factor = igmres(A, b; tol,
        track_true_residual=true, track_bound=true,
        precision_strategy=(res, t) -> begin
            v = rel_to_eps(bound_factor, res, t)
            push!(constant_factor_heuristic, v)
            v
        end)

    return (; sigma_m,
        residuals_sigma, true_residuals_sigma, residual_gap_sigma, bound_right4_sigma, sigma_heuristic, it_sigma,
        residuals_constant_factor, true_residuals_constant_factor, residual_gap_constant_factor,
        bound_right4_constant_factor, constant_factor_heuristic, it_constant_factor)
end

