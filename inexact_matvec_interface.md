# A general interface for inexact matrix-vector products

## The problem

`igmres` only relaxes the matrix-vector product when `A` is literally an
`HMatrices.HMatrix`:

```julia
# src/InexactGMRES.jl:59
A_iterable = A isa HMatrices.HMatrix ? HMatrices.ITerm(A, res) : A
```

Anything else — a dense matrix, a `LinearMap`, a sum of several operators —
falls through to an exact matvec every iteration, regardless of what
`precision_strategy` asks for. `exact_gmres` is unaffected (it always wants
an exact product), but `igmres` silently loses its whole reason for existing
on any operator that isn't a single, physical `HMatrices.HMatrix`.

This is a real limitation, not a hypothetical one: it's why
`helmholtz_dirichlet_cfie_disk_precision_strategy_comparison.jl` and the
cavity equivalent hand-fuse the Dirichlet CFIE kernel `D - ik*S` into one
kernel function assembled as a single H-matrix, instead of using Inti.jl's
own documented pattern

```julia
S, D = Inti.single_double_layer(; op, target, source, compression = (method = :hmatrix, tol), correction = (method = :dim,))
L = I/2 + LinearMap(D) - im*k*LinearMap(S)
```

(see the [Helmholtz scattering
example](https://integralequations.github.io/Inti.jl/stable/pluto-examples/helmholtz_scattering/)).
That pattern keeps `S` and `D` as two separately-compressed H-matrices,
composed lazily via `LinearMap` arithmetic. `igmres` has no way to look
inside a `LinearMap` composition and relax each underlying H-matrix's own
product — it just sees "not an `HMatrices.HMatrix`" and gives up on
inexactness entirely. The kernel-fusion workaround works for a single
difference of two layer potentials, but it stops being practical for
anything with more pieces (Calderón-type preconditioners, volume+boundary
coupling, block systems), and it means every new combined operator needs a
hand-written fused kernel instead of just composing already-assembled
pieces.

## The idea

Replace the `isa HMatrices.HMatrix` check with a small, duck-typed protocol
that any operator can opt into, and give `igmres` a generic combinator that
implements it for linear combinations of inexact-capable pieces.

```julia
# in utils.jl, generalizing what HMatrices.ITerm already does for one HMatrix

supports_inexact_matvec(A) = false
supports_inexact_matvec(::HMatrices.HMatrix) = true
supports_inexact_matvec(::AbstractInexactOperator) = true

as_inexact(A::HMatrices.HMatrix, rtol) = HMatrices.ITerm(A, rtol)

"""
    InexactLinearCombination(terms)

`terms` is a vector of `(coefficient, operator)` pairs, where each
`operator` satisfies `supports_inexact_matvec`. `mul!` applies the
*same* `rtol` to every term's own inexact product and accumulates the
weighted sum, so a `LinearMap`-style composition like `D - ik*S` can be
built from two already-compressed H-matrices without fusing them into
a single assembled kernel.
"""
struct InexactLinearCombination{T} <: AbstractInexactOperator
    terms::Vector{Tuple{T, Any}}  # (coefficient, inexact-capable operator)
end

function LinearAlgebra.mul!(y, A::InexactLinearCombination, x, a, b; rtol)
    iszero(b) ? fill!(y, zero(eltype(y))) : rmul!(y, b)
    tmp = similar(y)
    for (c, op) in A.terms
        mul!(tmp, as_inexact(op, rtol), x, a * c, 0)
        y .+= tmp
    end
    return y
end
```

Then `igmres` (and `exact_gmres`'s `return_H` machinery, unaffected) changes
from a hard-coded type check to:

```julia
A_iterable = supports_inexact_matvec(A) ? as_inexact(A, res) : A
```

and `helmholtz_dirichlet_cfie_*_precision_strategy_comparison.jl` could then
follow Inti's own documented pattern directly:

```julia
S, D = Inti.single_double_layer(; op, target = Q, source = Q,
    compression = (method = :hmatrix, tol), correction = (method = :dim,))
L = InexactLinearCombination([(0.5, I), (1.0, D), (-im*k, S)])
y, residuals, it = igmres(L, g; tol)
```

with `S` and `D` each individually, adaptively truncated every iteration —
not one fused operator whose blocks mix both kernels' low-rank structure
together at assembly time.

## Scope

This is a design note, not a plan to implement yet. Before building it, worth
checking:

- Does relaxing `S` and `D` to the *same* `rtol` per iteration actually
  behave the same as relaxing a single fused kernel to that `rtol`? They're
  different matrices with different rank structure, so the achieved error
  for a given `rtol` won't be identical, only comparable in spirit — plotting
  `igmres_precision_study` on both versions of the disk problem would be the
  natural way to check before assuming this is a drop-in replacement.
- Whether `HMatrices.ITerm`'s `mul!` signature (positional `rtol` as a field
  mutated before the call, not a keyword) should change to match, or whether
  `as_inexact` should keep adapting it.
