using LinearAlgebra
using Random

if !isdefined(@__MODULE__, :LandauChebyshevKappaMLE)
    include("landau_mle_inverse_chebyshev_kappa_ad.jl")
end
using .LandauChebyshevKappaMLE

# Dependencies used internally by the supplied hodlr.jl.
include("../../src/dyadic_idx.jl")
include("../../src/rsvd.jl")
include("../../src/hodlr.jl")

"""
Store the HODLR approximation of the observation covariance and its one-way
factorization.

Only

    Sigma(theta) = H*Ku(theta)*H' + sigma_noise^2*I + jitter*I

is factorized. The Chebyshev kappa field, conservative diffusion operator,
implicit mean, and `dKu/dtheta[j]` are constructed in
`landau_mle_inverse_chebyshev_kappa_ad.jl`.
"""
struct NativeSigmaHODLRFactor{TH, TF}
    matrix_hodlr::TH
    factor_hodlr::TF
end

function _check_hodlr_partition(n::Int, max_level::Int)
    max_level >= 1 ||
        throw(ArgumentError("sigma_max_level must be at least one."))
    divisor = 2^max_level
    n % divisor == 0 || throw(ArgumentError(
        "HODLR dimension n=$n must be divisible by 2^sigma_max_level=$divisor.",
    ))
    return div(n, divisor)
end

function _resolve_local_rank(
    n::Int,
    max_level::Int,
    local_rank::Union{Nothing, Int},
    rank_divisor::Int,
    oversampling::Int,
)
    rank_divisor >= 1 ||
        throw(ArgumentError("sigma_rank_divisor must be positive."))
    oversampling >= 0 ||
        throw(ArgumentError("sigma_oversampling must be nonnegative."))
    leaf_size = _check_hodlr_partition(n, max_level)
    rank = isnothing(local_rank) ? max(1, div(n, rank_divisor)) : local_rank
    rank >= 1 || throw(ArgumentError("sigma_local_rank must be positive."))
    rank + oversampling <= leaf_size || throw(ArgumentError(
        "sigma_local_rank + sigma_oversampling = $(rank + oversampling) " *
        "exceeds the finest HODLR block size $leaf_size.",
    ))
    return rank
end

function _compress_native_hodlr(
    A::AbstractMatrix,
    max_level::Int,
    local_rank::Union{Nothing, Int},
    rank_divisor::Int,
    oversampling::Int,
    seed::Union{Nothing, Int},
)
    n, m = size(A)
    n == m || throw(DimensionMismatch("HODLR input must be square."))
    rank = _resolve_local_rank(
        n,
        max_level,
        local_rank,
        rank_divisor,
        oversampling,
    )

    symmetric_matrix = Matrix{Float64}(A)
    symmetric_matrix .=
        0.5 .* (symmetric_matrix .+ transpose(symmetric_matrix))

    # The supplied randomized constructor uses the default RNG. Re-seeding
    # makes objective/score evaluations deterministic for L-BFGS.
    isnothing(seed) || Random.seed!(seed)
    matrix_vector_query = vector -> symmetric_matrix * vector
    return hodlr(
        matrix_vector_query,
        Int64(n),
        Int64(max_level),
        Int64(rank),
        Int64(oversampling),
    )
end

function _factorize_native_sigma(
    Sigma::AbstractMatrix,
    max_level::Int,
    local_rank::Union{Nothing, Int},
    rank_divisor::Int,
    oversampling::Int,
    seed::Union{Nothing, Int},
)
    Sigma_hodlr = _compress_native_hodlr(
        Sigma,
        max_level,
        local_rank,
        rank_divisor,
        oversampling,
        seed,
    )
    return NativeSigmaHODLRFactor(
        Sigma_hodlr,
        hodlr_factorize(Sigma_hodlr),
    )
end

function _solve_native_sigma(
    wrapper::NativeSigmaHODLRFactor,
    rhs::AbstractVector,
)
    matrix_rhs = reshape(Float64.(collect(rhs)), :, 1)
    return vec(hodlr_solve(wrapper.factor_hodlr, matrix_rhs))
end

function _solve_native_sigma(
    wrapper::NativeSigmaHODLRFactor,
    rhs::AbstractMatrix,
)
    return Matrix{Float64}(
        hodlr_solve(wrapper.factor_hodlr, Matrix{Float64}(rhs)),
    )
end

_logdet_native_sigma(wrapper::NativeSigmaHODLRFactor) =
    Float64(hodlr_logdet(wrapper.factor_hodlr))

_multiply_native_derivative(derivative_hodlr, rhs::AbstractMatrix) =
    Matrix{Float64}(
        hodlr_prod(derivative_hodlr, Matrix{Float64}(rhs)),
    )

function _trace_inverse_native_derivative(
    wrapper::NativeSigmaHODLRFactor,
    derivative_hodlr,
)
    inverse_product = hodlr_invmult(
        wrapper.factor_hodlr,
        derivative_hodlr,
    )
    return Float64(hodlr_tr(inverse_product))
end

"""
    make_native_hodlr_backend(; kwargs...)

Create the native HODLR observation-covariance backend. For every Chebyshev
coefficient, `dSigma/dtheta[j]` is compressed as a symmetric (generally
indefinite) HODLR matrix. With `trace_mode=:exact`, the inverse-product and
trace are evaluated by `hodlr_invmult` and `hodlr_tr`; the dense trace formula
is not used by this backend.

Keyword defaults match the earlier Landau backend:

- `sigma_max_level=2`
- `sigma_local_rank=nothing`
- `sigma_rank_divisor=32`
- `sigma_oversampling=10`
- `sigma_random_seed=314159`
"""
function make_native_hodlr_backend(;
    sigma_max_level::Int=2,
    sigma_local_rank::Union{Nothing, Int}=nothing,
    sigma_rank_divisor::Int=32,
    sigma_oversampling::Int=10,
    sigma_random_seed::Union{Nothing, Int}=314159,
)
    factorize_callback = Sigma -> _factorize_native_sigma(
        Sigma,
        sigma_max_level,
        sigma_local_rank,
        sigma_rank_divisor,
        sigma_oversampling,
        sigma_random_seed,
    )

    compress_derivative_callback = derivative -> _compress_native_hodlr(
        derivative,
        sigma_max_level,
        sigma_local_rank,
        sigma_rank_divisor,
        sigma_oversampling,
        sigma_random_seed,
    )

    return LandauChebyshevKappaMLE.HODLRBackend(
        factorize_sigma=factorize_callback,
        solve_sigma=_solve_native_sigma,
        logdet_sigma=_logdet_native_sigma,
        compress_derivative=compress_derivative_callback,
        multiply_derivative=_multiply_native_derivative,
        trace_inverse_derivative=_trace_inverse_native_derivative,
    )
end
