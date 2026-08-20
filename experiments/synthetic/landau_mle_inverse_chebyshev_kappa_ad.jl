module LandauChebyshevKappaMLE

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using Optim
using ForwardDiff

export AbstractCovarianceBackend,
       DenseBackend,
       HODLRBackend,
       LandauKappaMLEOptions,
       LandauKappaMLEProblem,
       chebyshev_basis_pairs,
       constant_kappa_theta,
       chebyshev_kappa_field,
       diffusion_operator_and_ad,
       landau_nll_score_theta,
       estimate_kappa_theta,
       estimate_kappa_from_observations,
       check_kappa_score

# -----------------------------------------------------------------------------
# Observation-covariance backends
# -----------------------------------------------------------------------------

abstract type AbstractCovarianceBackend end

"""Exact dense reference backend for the observation covariance."""
struct DenseBackend <: AbstractCovarianceBackend end

"""
Adapter for a user-supplied HODLR implementation.

HODLR is used only for the observed covariance `Sigma(theta)` and its
derivatives. The nonlinear Landau closure and its implicit sensitivities are
computed by the ordinary dense/sparse algebra in this module.
"""
struct HODLRBackend{FF, FS, FL, FC, FM, FT} <: AbstractCovarianceBackend
    factorize_sigma::FF
    solve_sigma::FS
    logdet_sigma::FL
    compress_derivative::FC
    multiply_derivative::FM
    trace_inverse_derivative::FT
end

function HODLRBackend(;
    factorize_sigma,
    solve_sigma,
    logdet_sigma,
    compress_derivative,
    multiply_derivative,
    trace_inverse_derivative,
)
    return HODLRBackend(
        factorize_sigma,
        solve_sigma,
        logdet_sigma,
        compress_derivative,
        multiply_derivative,
        trace_inverse_derivative,
    )
end

struct SigmaFactor{B<:AbstractCovarianceBackend, F}
    backend::B
    factor::F
    logdet_value::Float64
end

# -----------------------------------------------------------------------------
# Options and inverse-problem definition
# -----------------------------------------------------------------------------

Base.@kwdef mutable struct LandauKappaMLEOptions
    backend::AbstractCovarianceBackend = DenseBackend()

    observation_noise_std::Float64 = 0.0
    covariance_jitter::Float64 = 1.0e-8

    # Damped Newton solve for the implicit Gaussian-closure mean m(theta).
    mean_tol::Float64 = 1.0e-8
    mean_maxiter::Int = 30
    mean_max_backtracking::Int = 15
    mean_armijo::Float64 = 1.0e-4
    mean_branch::Symbol = :positive
    mean_initial::Union{Nothing, Vector{Float64}} = nothing

    # The reference velocity code bounds every Chebyshev coefficient in
    # [-4,4]. Here theta parameterizes log(kappa), not kappa itself.
    lower_theta::Float64 = -4.0
    upper_theta::Float64 = 4.0
    box_outer_iterations::Int = 20

    spatial_ordering::Bool = true

    # :exact means an exact dense trace for DenseBackend and the native HODLR
    # inverse-product/trace calculation for HODLRBackend. :hutchinson uses
    # fixed Rademacher probes for either backend.
    trace_mode::Symbol = :exact
    trace_samples::Int = 32
    trace_seed::Int = 314159

    lbfgs_memory::Int = 10
    optimizer_iterations::Int = 50
    optimizer_time_limit_seconds::Float64 = 600.0
    optimizer_g_tol::Float64 = 1.0e-6
    optimizer_f_tol::Float64 = 1.0e-10
    optimizer_x_tol::Float64 = 1.0e-10
    show_optimizer_trace::Bool = true
    show_evaluations::Bool = true

    throw_on_evaluation_failure::Bool = false
    failure_penalty::Float64 = 1.0e20
    failure_quadratic_scale::Float64 = 1.0e8
end

mutable struct LandauKappaMLEProblem
    observations::Matrix{Float64}          # M x P
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    Kphi::Matrix{Float64}
    mean_phi::Vector{Float64}
    fixed_a::Float64
    fixed_b::Float64
    total_degree::Int
    basis_pairs::Vector{Tuple{Int, Int}}
    basis_matrix::Matrix{Float64}          # N x ntheta
    chebyshev_domain::NTuple{4, Float64}
    face_average::Symbol
    stencil_rows::Vector{Int}
    stencil_cols::Vector{Int}
    Isparse::SparseMatrixCSC{Float64, Int}
    obs_indices::Vector{Int}
    permutation::Vector{Int}
    trace_probes::Matrix{Float64}
    options::LandauKappaMLEOptions
    evaluation_count::Int
    last_theta::Union{Nothing, Vector{Float64}}
    last_theta_value::Any
end

"""
    chebyshev_basis_pairs(total_degree)

Return the total-degree two-dimensional Chebyshev ordering used by
`velocity_basis_2d_mat_exp`:

    [(i,j) for i in 0:r for j in 0:(r-i)].
"""
function chebyshev_basis_pairs(total_degree::Integer)
    total_degree >= 0 ||
        throw(ArgumentError("total_degree must be nonnegative."))
    return [(i, j) for i in 0:Int(total_degree) for j in 0:(Int(total_degree) - i)]
end

function _chebyshev_values(z, degree::Int)
    values = Vector{typeof(z)}(undef, degree + 1)
    values[1] = one(z)
    if degree >= 1
        values[2] = z
        @inbounds for k in 2:degree
            values[k + 1] = 2 * z * values[k] - values[k - 1]
        end
    end
    return values
end

_scale_to_chebyshev(z, lower, upper) =
    2.0 * (z - lower) / (upper - lower) - 1.0

function _build_basis_matrix(
    xgrid::Vector{Float64},
    ygrid::Vector{Float64},
    pairs::Vector{Tuple{Int, Int}},
    domain::NTuple{4, Float64},
    degree::Int,
)
    xmin, xmax, ymin, ymax = domain

    N = length(xgrid) * length(ygrid)
    basis = Matrix{Float64}(undef, N, length(pairs))
    row = 0
    # y outer, x inner: the same column-major vec(U) ordering as the solver.
    for y in ygrid, x in xgrid
        row += 1
        Tx = _chebyshev_values(_scale_to_chebyshev(x, xmin, xmax), degree)
        Ty = _chebyshev_values(_scale_to_chebyshev(y, ymin, ymax), degree)
        @inbounds for (column, (i, j)) in enumerate(pairs)
            basis[row, column] = Tx[i + 1] * Ty[j + 1]
        end
    end
    return basis
end

function _uniform_spacing(grid::Vector{Float64}, name::AbstractString)
    deltas = diff(grid)
    h = deltas[1]
    return h
end

function _stencil_pattern(nx::Int, ny::Int)
    rows = Vector{Int}(undef, 5 * nx * ny)
    cols = similar(rows)
    cursor = 0
    linear(i, j) = i + (j - 1) * nx
    for j in 1:ny, i in 1:nx
        row = linear(i, j)
        east = linear(i == nx ? 1 : i + 1, j)
        west = linear(i == 1 ? nx : i - 1, j)
        north = linear(i, j == ny ? 1 : j + 1)
        south = linear(i, j == 1 ? ny : j - 1)
        @inbounds for column in (row, east, west, north, south)
            cursor += 1
            rows[cursor] = row
            cols[cursor] = column
        end
    end
    return rows, cols
end

"""
    LandauKappaMLEProblem(observations, xgrid, ygrid, Kphi;
                         fixed_a, fixed_b, total_degree=5, ...)

Construct an inverse problem for

    div(kappa_theta * grad(u)) + fixed_a*u - fixed_b*u^3 + phi = 0,

where

    log(kappa_theta(x,y)) = sum_{i+j<=r} theta_ij T_i(xhat)T_j(yhat),
    kappa_theta(x,y) = exp(log(kappa_theta(x,y))).

Thus every optimization variable is a Chebyshev coefficient and `fixed_a`,
`fixed_b` are never optimized. By default, the Chebyshev domain is the full
periodic interval, including the wrap endpoint omitted from `xgrid`/`ygrid`.
It can be overridden with `(xmin,xmax,ymin,ymax)` through
`chebyshev_domain`.
"""
function LandauKappaMLEProblem(
    observations::AbstractMatrix,
    xgrid::AbstractVector{<:Real},
    ygrid::AbstractVector{<:Real},
    Kphi::AbstractMatrix;
    fixed_a::Real,
    fixed_b::Real,
    total_degree::Integer=5,
    mean_phi::Union{Nothing, AbstractVector}=nothing,
    obs_indices::Union{Nothing, AbstractVector{<:Integer}}=nothing,
    chebyshev_domain=nothing,
    face_average::Symbol=:harmonic,
    options::LandauKappaMLEOptions=LandauKappaMLEOptions(),
)
    face_average in (:harmonic, :arithmetic) ||
        throw(ArgumentError("face_average must be :harmonic or :arithmetic."))

    x = Float64.(collect(xgrid))
    y = Float64.(collect(ygrid))
    nx, ny = length(x), length(y)
    N = nx * ny
    hx = _uniform_spacing(x, "xgrid")
    hy = _uniform_spacing(y, "ygrid")

    a = Float64(fixed_a)
    b = Float64(fixed_b)

    size(Kphi) == (N, N) ||
        throw(DimensionMismatch("Kphi must have size ($N,$N)."))
    covariance = Matrix{Float64}(Kphi)
    all(isfinite, covariance) || throw(ArgumentError("Kphi contains NaN or Inf."))
    covariance .= 0.5 .* (covariance .+ transpose(covariance))

    idx = isnothing(obs_indices) ? collect(1:N) : Int.(collect(obs_indices))
    isempty(idx) && throw(ArgumentError("obs_indices cannot be empty."))
    all(i -> 1 <= i <= N, idx) ||
        throw(ArgumentError("Every observation index must lie in 1:$N."))
    length(unique(idx)) == length(idx) ||
        throw(ArgumentError("obs_indices must not contain duplicates."))

    input = Matrix{Float64}(observations)
    Y = if size(input, 2) == N
        input[:, idx]
    elseif size(input, 2) == length(idx)
        input
    else
        throw(DimensionMismatch(
            "observations must have $N columns or $(length(idx)) selected columns.",
        ))
    end
    all(isfinite, Y) || throw(ArgumentError("observations contain NaN or Inf."))

    forcing_mean = isnothing(mean_phi) ? zeros(Float64, N) : Float64.(collect(mean_phi))
    all(isfinite, forcing_mean) ||
        throw(ArgumentError("mean_phi contains NaN or Inf."))

    domain = if isnothing(chebyshev_domain)
        (first(x), last(x) + hx, first(y), last(y) + hy)
    else
        length(chebyshev_domain) == 4 ||
            throw(DimensionMismatch("chebyshev_domain must contain four values."))
        Tuple(Float64.(collect(chebyshev_domain)))
    end
    domain_typed = (domain[1], domain[2], domain[3], domain[4])
    all(isfinite, domain_typed) ||
        throw(ArgumentError("chebyshev_domain contains NaN or Inf."))

    degree = Int(total_degree)
    pairs_list = chebyshev_basis_pairs(degree)
    basis = _build_basis_matrix(x, y, pairs_list, domain_typed, degree)
    stencil_rows, stencil_cols = _stencil_pattern(nx, ny)

    permutation = options.spatial_ordering ?
        _morton_permutation(nx, ny, idx) : collect(1:length(idx))
    P = length(idx)
    probes = zeros(Float64, P, max(options.trace_samples, 0))
    if size(probes, 2) > 0
        rng = MersenneTwister(options.trace_seed)
        probes .= ifelse.(rand(rng, Bool, size(probes)), 1.0, -1.0)
    end

    return LandauKappaMLEProblem(
        Y,
        x,
        y,
        covariance,
        forcing_mean,
        a,
        b,
        degree,
        pairs_list,
        basis,
        domain_typed,
        face_average,
        stencil_rows,
        stencil_cols,
        spdiagm(0 => ones(Float64, N)),
        idx,
        permutation,
        probes,
        options,
        0,
        nothing,
        nothing,
    )
end

function _morton_permutation(nx::Int, ny::Int, obs_indices::Vector{Int})
    bits = max(1, ceil(Int, log2(max(nx, ny))))
    bits <= 31 || throw(ArgumentError("Morton ordering supports dimensions below 2^31."))
    codes = Vector{UInt64}(undef, length(obs_indices))
    for (k, linear_index) in pairs(obs_indices)
        ix = (linear_index - 1) % nx
        iy = (linear_index - 1) ÷ nx
        code = UInt64(0)
        for bit in 0:(bits - 1)
            code |= UInt64((ix >> bit) & 1) << (2 * bit)
            code |= UInt64((iy >> bit) & 1) << (2 * bit + 1)
        end
        codes[k] = code
    end
    return sortperm(codes)
end

"""Coefficient vector representing a constant positive kappa field."""
function constant_kappa_theta(problem::LandauKappaMLEProblem, kappa::Real)
    kappa_float = Float64(kappa)
    isfinite(kappa_float) && kappa_float > 0.0 ||
        throw(ArgumentError("kappa must be positive and finite."))
    theta = zeros(Float64, length(problem.basis_pairs))
    theta[1] = log(kappa_float)
    return theta
end

function _validate_theta(problem::LandauKappaMLEProblem, theta)
    length(theta) == length(problem.basis_pairs) ||
        throw(DimensionMismatch(
            "theta must contain $(length(problem.basis_pairs)) coefficients.",
        ))
    values = Float64.(collect(theta))
    all(isfinite, values) || throw(ArgumentError("theta contains NaN or Inf."))
    return values
end

"""Evaluate the positive `nx`-by-`ny` Chebyshev kappa field."""
function chebyshev_kappa_field(problem::LandauKappaMLEProblem, theta)
    theta_float = _validate_theta(problem, theta)
    values = exp.(problem.basis_matrix * theta_float)
    all(isfinite, values) || error("exp(Psi*theta) overflowed.")
    return reshape(values, length(problem.xgrid), length(problem.ygrid))
end

_face_value(left, right, ::Val{:arithmetic}) = (left + right) / 2
_face_value(left, right, ::Val{:harmonic}) = 2 * left * right / (left + right)

function _diffusion_stencil_values(
    problem::LandauKappaMLEProblem,
    theta::AbstractVector,
)
    nx, ny = length(problem.xgrid), length(problem.ygrid)
    N = nx * ny
    hx = _uniform_spacing(problem.xgrid, "xgrid")
    hy = _uniform_spacing(problem.ygrid, "ygrid")
    inv_hx2 = inv(hx * hx)
    inv_hy2 = inv(hy * hy)
    kappa = reshape(exp.(problem.basis_matrix * theta), nx, ny)
    T = eltype(kappa)
    values = Vector{T}(undef, 5 * N)
    face_rule = Val(problem.face_average)

    cursor = 0
    for j in 1:ny, i in 1:nx
        ip = i == nx ? 1 : i + 1
        im = i == 1 ? nx : i - 1
        jp = j == ny ? 1 : j + 1
        jm = j == 1 ? ny : j - 1

        east = _face_value(kappa[i, j], kappa[ip, j], face_rule) * inv_hx2
        west = _face_value(kappa[i, j], kappa[im, j], face_rule) * inv_hx2
        north = _face_value(kappa[i, j], kappa[i, jp], face_rule) * inv_hy2
        south = _face_value(kappa[i, j], kappa[i, jm], face_rule) * inv_hy2

        @inbounds begin
            values[cursor + 1] = -(east + west + north + south)
            values[cursor + 2] = east
            values[cursor + 3] = west
            values[cursor + 4] = north
            values[cursor + 5] = south
        end
        cursor += 5
    end
    return values
end

"""
    diffusion_operator_and_ad(problem, theta)

Return `(kappa, operator, derivatives)`. `derivatives[j]` is
`d operator / d theta[j]`.

`ForwardDiff.jacobian` differentiates only the nonzero conservative-stencil
values. This is scheme-consistent and avoids an `N^2`-row AD tape. Sparse
operators are assembled after AD, so no sparse factorization is differentiated.
"""
function diffusion_operator_and_ad(problem::LandauKappaMLEProblem, theta)
    theta_float = _validate_theta(problem, theta)
    value_function = t -> _diffusion_stencil_values(problem, t)
    stencil_values = value_function(theta_float)
    stencil_jacobian = ForwardDiff.jacobian(value_function, theta_float)

    N = length(problem.xgrid) * length(problem.ygrid)
    operator = sparse(
        problem.stencil_rows,
        problem.stencil_cols,
        Float64.(stencil_values),
        N,
        N,
    )
    derivatives = Vector{SparseMatrixCSC{Float64, Int}}(undef, length(theta_float))
    for j in eachindex(theta_float)
        derivatives[j] = sparse(
            problem.stencil_rows,
            problem.stencil_cols,
            Float64.(view(stencil_jacobian, :, j)),
            N,
            N,
        )
    end

    kappa = reshape(
        exp.(problem.basis_matrix * theta_float),
        length(problem.xgrid),
        length(problem.ygrid),
    )
    return (kappa=kappa, operator=operator, derivatives=derivatives)
end

# -----------------------------------------------------------------------------
# Implicit Gaussian closure and theta sensitivities
# -----------------------------------------------------------------------------

function _dense_closure_matrices(B::AbstractMatrix, Kphi::Matrix{Float64})
    N = size(B, 1)
    factor = lu(B)
    A = Matrix{Float64}(factor \ Matrix{Float64}(I, N, N))
    Ku = A * Kphi * transpose(A)
    A .= 0.5 .* (A .+ transpose(A))
    Ku .= 0.5 .* (Ku .+ transpose(Ku))
    return A, Ku
end

function _initial_mean(
    problem::LandauKappaMLEProblem,
    D::SparseMatrixCSC{Float64, Int},
)
    options = problem.options
    N = length(problem.mean_phi)
    if !isnothing(options.mean_initial)
        length(options.mean_initial) == N ||
            throw(DimensionMismatch("mean_initial must have length $N."))
        return copy(options.mean_initial)
    elseif options.mean_branch == :zero
        return zeros(Float64, N)
    elseif options.mean_branch == :linear
        return -((D + problem.fixed_a .* problem.Isparse) \ problem.mean_phi)
    end

    amplitude = problem.fixed_a > 0.0 ? sqrt(problem.fixed_a / problem.fixed_b) : 0.0
    sign = options.mean_branch == :positive ? 1.0 : -1.0
    return fill(sign * amplitude, N)
end

function _closure_state(
    problem::LandauKappaMLEProblem,
    m::Vector{Float64},
    D::SparseMatrixCSC{Float64, Int},
)
    a, b = problem.fixed_a, problem.fixed_b
    diagonal_B = fill(a, length(m)) .- 3.0 .* b .* (m .^ 2)
    B = D + spdiagm(0 => diagonal_B)
    A, Ku = _dense_closure_matrices(B, problem.Kphi)
    q = diag(Ku)
    residual =
        D * m .+ a .* m .-
        b .* (m .^ 3 .+ 3.0 .* m .* q) .+
        problem.mean_phi
    return (m=m, B=B, A=A, Ku=Ku, q=q, residual=residual)
end

function _closure_jacobian(
    problem::LandauKappaMLEProblem,
    state,
    Ddense::Matrix{Float64},
)
    a, b = problem.fixed_a, problem.fixed_b
    m, q = state.m, state.q
    N = length(m)
    G = state.A .* state.Ku

    J = copy(Ddense)
    diagonal_J = fill(a, N) .- 3.0 .* b .* (m .^ 2 .+ q)
    @inbounds for i in 1:N
        J[i, i] += diagonal_J[i]
    end
    correction = copy(G)
    correction .*= reshape(m, N, 1)
    correction .*= reshape(m, 1, N)
    J .-= 36.0 .* b^2 .* correction
    J .= 0.5 .* (J .+ transpose(J))
    return J
end

function _solve_implicit_mean(
    problem::LandauKappaMLEProblem,
    D::SparseMatrixCSC{Float64, Int},
)
    options = problem.options
    N = length(problem.mean_phi)
    Ddense = Matrix{Float64}(D)
    state = _closure_state(problem, _initial_mean(problem, D), D)

    for iteration in 0:options.mean_maxiter
        residual_norm = norm(state.residual)
        residual_rms = residual_norm / sqrt(N)
        if residual_rms <= options.mean_tol
            return merge(state, (
                iterations=iteration,
                residual_rms=residual_rms,
                D=D,
                Ddense=Ddense,
            ))
        end
        iteration == options.mean_maxiter && break

        J = _closure_jacobian(problem, state, Ddense)
        direction = -(lu(J) \ state.residual)
        all(isfinite, direction) ||
            error("The implicit-mean Newton direction contains NaN or Inf.")

        step = 1.0
        accepted_state = nothing
        for _ in 1:options.mean_max_backtracking
            trial_state = try
                _closure_state(problem, state.m .+ step .* direction, D)
            catch
                nothing
            end
            if !isnothing(trial_state) &&
               norm(trial_state.residual) <=
               (1.0 - options.mean_armijo * step) * residual_norm
                accepted_state = trial_state
                break
            end
            step *= 0.5
        end
        isnothing(accepted_state) && error(
            "The damped Newton solve for m(theta) could not reduce the closure residual.",
        )
        state = accepted_state
    end

    final_rms = norm(state.residual) / sqrt(N)
    error(
        "The implicit mean did not converge in $(options.mean_maxiter) iterations; " *
        "final RMS residual=$final_rms.",
    )
end

function _state_and_mean_sensitivities(
    problem::LandauKappaMLEProblem,
    theta::Vector{Float64},
)
    diffusion = diffusion_operator_and_ad(problem, theta)
    state = _solve_implicit_mean(problem, diffusion.operator)
    m, A, Ku = state.m, state.A, state.Ku
    b = problem.fixed_b
    N = length(m)
    p = length(theta)

    J = _closure_jacobian(problem, state, state.Ddense)
    partial_F = Matrix{Float64}(undef, N, p)

    # At fixed m, dB/dtheta_j = dD/dtheta_j and
    # dKu = -A*dD*Ku - (A*dD*Ku)'. ForwardDiff supplies dD/dtheta.
    for j in 1:p
        Dtheta = diffusion.derivatives[j]
        partial_T = A * (Dtheta * Ku)
        q_partial = -2.0 .* diag(partial_T)
        partial_F[:, j] .= Dtheta * m .- 3.0 .* b .* m .* q_partial
    end

    mean_sensitivities = lu(J) \ (-partial_F)
    return (
        state=state,
        kappa=diffusion.kappa,
        Dtheta=diffusion.derivatives,
        mean_sensitivities=mean_sensitivities,
    )
end

function _covariance_sensitivity(
    problem::LandauKappaMLEProblem,
    sensitivities,
    j::Int,
)
    state = sensitivities.state
    mtheta = view(sensitivities.mean_sensitivities, :, j)
    diagonal_total = -6.0 .* problem.fixed_b .* state.m .* mtheta
    Btheta_Ku =
        sensitivities.Dtheta[j] * state.Ku .+
        reshape(diagonal_total, :, 1) .* state.Ku
    T = state.A * Btheta_Ku
    Ku_theta = -(T .+ transpose(T))
    Ku_theta .= 0.5 .* (Ku_theta .+ transpose(Ku_theta))
    return Ku_theta
end

# -----------------------------------------------------------------------------
# Likelihood, native HODLR trace, and score
# -----------------------------------------------------------------------------

function _factor_sigma(backend::DenseBackend, Sigma::Matrix{Float64})
    factor = cholesky(Symmetric(Sigma); check=true)
    return SigmaFactor(backend, factor, 2.0 * sum(log, diag(factor.L)))
end

function _factor_sigma(backend::HODLRBackend, Sigma::Matrix{Float64})
    factor = backend.factorize_sigma(Sigma)
    logdet_value = Float64(backend.logdet_sigma(factor))
    isfinite(logdet_value) || error("HODLR logdet(Sigma) is not finite.")
    return SigmaFactor(backend, factor, logdet_value)
end

function _solve_sigma(factor::SigmaFactor, rhs::AbstractVector)
    raw = factor.backend isa DenseBackend ?
        factor.factor \ rhs : factor.backend.solve_sigma(factor.factor, rhs)
    result = Vector{Float64}(raw)
    all(isfinite, result) || error("Sigma solve returned NaN or Inf.")
    return result
end

function _solve_sigma(factor::SigmaFactor, rhs::AbstractMatrix)
    raw = try
        factor.backend isa DenseBackend ?
            factor.factor \ rhs : factor.backend.solve_sigma(factor.factor, rhs)
    catch
        columns = [_solve_sigma(factor, view(rhs, :, j)) for j in axes(rhs, 2)]
        return isempty(columns) ? zeros(Float64, size(rhs, 1), 0) : hcat(columns...)
    end
    result = Matrix{Float64}(raw)
    size(result) == size(rhs) ||
        throw(DimensionMismatch("Sigma solve returned the wrong size."))
    all(isfinite, result) || error("Sigma solve returned NaN or Inf.")
    return result
end

function _compress_derivative(factor::SigmaFactor, derivative::Matrix{Float64})
    return factor.backend isa DenseBackend ?
        derivative : factor.backend.compress_derivative(derivative)
end

function _multiply_derivative(factor::SigmaFactor, derivative, rhs::AbstractMatrix)
    raw = factor.backend isa DenseBackend ?
        derivative * rhs : factor.backend.multiply_derivative(derivative, rhs)
    result = Matrix{Float64}(raw)
    size(result) == size(rhs) ||
        throw(DimensionMismatch("Derivative product returned the wrong size."))
    all(isfinite, result) || error("Derivative product returned NaN or Inf.")
    return result
end

function _trace_inverse_product(
    problem::LandauKappaMLEProblem,
    factor::SigmaFactor,
    derivative_dense::Matrix{Float64},
    derivative_representation,
)
    if problem.options.trace_mode == :exact
        if factor.backend isa DenseBackend
            return sum(diag(_solve_sigma(factor, derivative_dense)))
        end
        value = Float64(factor.backend.trace_inverse_derivative(
            factor.factor,
            derivative_representation,
        ))
        isfinite(value) || error("HODLR trace is not finite.")
        return value
    end

    probes = problem.trace_probes
    size(probes, 2) > 0 ||
        error("trace_mode=:hutchinson requires trace_samples > 0.")
    products = _multiply_derivative(
        factor,
        derivative_representation,
        probes,
    )
    solved = _solve_sigma(factor, products)
    return mean(vec(sum(probes .* solved; dims=1)))
end

"""
    landau_nll_score_theta(problem, theta)

Return `(negative_loglikelihood, score_theta, diagnostics)` with `a` and `b`
fixed. ForwardDiff obtains the derivative of the conservative diffusion
stencil. The total `dm/dtheta` and `dKu/dtheta` include the implicit response of
the nonlinear Gaussian-closure mean.
"""
function landau_nll_score_theta(problem::LandauKappaMLEProblem, theta)
    theta_float = _validate_theta(problem, theta)
    sensitivities = _state_and_mean_sensitivities(problem, theta_float)
    state = sensitivities.state
    idx = problem.obs_indices
    permutation = problem.permutation
    P = length(idx)
    M = size(problem.observations, 1)
    p = length(theta_float)

    mean_observed = state.m[idx]
    Sigma = Matrix{Float64}(state.Ku[idx, idx])
    diagonal_noise =
        problem.options.observation_noise_std^2 +
        problem.options.covariance_jitter
    @inbounds for i in 1:P
        Sigma[i, i] += diagonal_noise
    end
    Sigma .= 0.5 .* (Sigma .+ transpose(Sigma))

    Sigma_p = Sigma[permutation, permutation]
    residuals =
        Matrix(transpose(problem.observations)) .-
        reshape(mean_observed, P, 1)
    residuals_p = residuals[permutation, :]

    sigma_factor = _factor_sigma(problem.options.backend, Sigma_p)
    alpha = _solve_sigma(sigma_factor, residuals_p)
    quadratic = sum(residuals_p .* alpha)
    nll = 0.5 * (
        M * (P * log(2.0 * pi) + sigma_factor.logdet_value) + quadratic
    )
    alpha_sum = vec(sum(alpha; dims=2))

    score = Vector{Float64}(undef, p)
    trace_terms = similar(score)
    covariance_quadratics = similar(score)
    mean_terms = similar(score)

    for j in 1:p
        Ku_theta = _covariance_sensitivity(problem, sensitivities, j)
        dSigma = Matrix{Float64}(Ku_theta[idx, idx])
        dSigma .= 0.5 .* (dSigma .+ transpose(dSigma))
        dSigma_p = dSigma[permutation, permutation]
        dmean_p = sensitivities.mean_sensitivities[idx, j][permutation]

        representation = _compress_derivative(sigma_factor, dSigma_p)
        trace_term = _trace_inverse_product(
            problem,
            sigma_factor,
            dSigma_p,
            representation,
        )
        derivative_alpha = _multiply_derivative(
            sigma_factor,
            representation,
            alpha,
        )
        covariance_quadratic = sum(alpha .* derivative_alpha)
        mean_term = -dot(dmean_p, alpha_sum)

        score[j] = 0.5 * (M * trace_term - covariance_quadratic) + mean_term
        trace_terms[j] = trace_term
        covariance_quadratics[j] = covariance_quadratic
        mean_terms[j] = mean_term
    end

    diagnostics = (
        theta=copy(theta_float),
        fixed_a=problem.fixed_a,
        fixed_b=problem.fixed_b,
        total_degree=problem.total_degree,
        basis_pairs=hcat(first.(problem.basis_pairs), last.(problem.basis_pairs)),
        kappa=sensitivities.kappa,
        kappa_min=minimum(sensitivities.kappa),
        kappa_max=maximum(sensitivities.kappa),
        face_average=problem.face_average,
        mean=state.m,
        mean_iterations=state.iterations,
        mean_residual_rms=state.residual_rms,
        backend=typeof(problem.options.backend),
        logdet_sigma=sigma_factor.logdet_value,
        quadratic=quadratic,
        trace_terms=trace_terms,
        covariance_quadratics=covariance_quadratics,
        mean_score_terms=mean_terms,
    )
    return nll, score, diagnostics
end

# -----------------------------------------------------------------------------
# Bounded L-BFGS optimization
# -----------------------------------------------------------------------------

function _same_vector(left::Vector{Float64}, right::AbstractVector)
    length(left) == length(right) || return false
    return all(isequal(left[i], right[i]) for i in eachindex(left, right))
end

function _evaluate_theta(problem::LandauKappaMLEProblem, theta::AbstractVector)
    theta_float = _validate_theta(problem, theta)
    if !isnothing(problem.last_theta) && _same_vector(problem.last_theta, theta_float)
        return problem.last_theta_value
    end

    problem.evaluation_count += 1
    options = problem.options
    evaluation = try
        nll, gradient, diagnostics = landau_nll_score_theta(problem, theta_float)
        (
            value=nll,
            gradient=gradient,
            theta=copy(theta_float),
            diagnostics=diagnostics,
            failed=false,
            error=nothing,
        )
    catch err
        options.throw_on_evaluation_failure && rethrow(err)
        scale = options.failure_quadratic_scale
        (
            value=options.failure_penalty + scale * sum(abs2, theta_float),
            gradient=2.0 .* scale .* theta_float,
            theta=copy(theta_float),
            diagnostics=nothing,
            failed=true,
            error=sprint(showerror, err),
        )
    end

    if options.show_evaluations
        if evaluation.failed
            @printf(
                "eval %4d | penalty=% .6e | |theta|=% .3e | %s\n",
                problem.evaluation_count,
                evaluation.value,
                norm(theta_float),
                evaluation.error,
            )
        else
            @printf(
                "eval %4d | nll=% .10e | |g_theta|=% .3e | kappa=[%.3e, %.3e] | mean rms=% .3e\n",
                problem.evaluation_count,
                evaluation.value,
                norm(evaluation.gradient),
                evaluation.diagnostics.kappa_min,
                evaluation.diagnostics.kappa_max,
                evaluation.diagnostics.mean_residual_rms,
            )
        end
    end

    problem.last_theta = copy(theta_float)
    problem.last_theta_value = evaluation
    return evaluation
end

function _optimizer_options(options::LandauKappaMLEOptions)
    try
        return Optim.Options(
            iterations=options.optimizer_iterations,
            outer_iterations=options.box_outer_iterations,
            time_limit=options.optimizer_time_limit_seconds,
            g_abstol=options.optimizer_g_tol,
            f_reltol=options.optimizer_f_tol,
            x_abstol=options.optimizer_x_tol,
            show_trace=options.show_optimizer_trace,
            store_trace=true,
        )
    catch err
        if err isa MethodError || err isa UndefKeywordError
            return Optim.Options(
                iterations=options.optimizer_iterations,
                outer_iterations=options.box_outer_iterations,
                time_limit=options.optimizer_time_limit_seconds,
                g_tol=options.optimizer_g_tol,
                f_tol=options.optimizer_f_tol,
                x_tol=options.optimizer_x_tol,
                show_trace=options.show_optimizer_trace,
                store_trace=true,
            )
        end
        rethrow(err)
    end
end

"""Estimate only the Chebyshev coefficient vector theta; a and b stay fixed."""
function estimate_kappa_theta(problem::LandauKappaMLEProblem, initial_theta)
    theta0 = _validate_theta(problem, initial_theta)
    options = problem.options
    lower = fill(options.lower_theta, length(theta0))
    upper = fill(options.upper_theta, length(theta0))
    all(lower .< theta0) && all(theta0 .< upper) ||
        throw(ArgumentError("initial_theta must be strictly inside the Fminbox bounds."))

    problem.evaluation_count = 0
    problem.last_theta = nothing
    problem.last_theta_value = nothing

    objective = theta -> _evaluate_theta(problem, theta).value
    function gradient!(G, theta)
        G .= _evaluate_theta(problem, theta).gradient
        return G
    end

    result = Optim.optimize(
        objective,
        gradient!,
        lower,
        upper,
        theta0,
        Optim.Fminbox(Optim.LBFGS(m=options.lbfgs_memory)),
        _optimizer_options(options),
    )

    theta_hat = Float64.(Optim.minimizer(result))
    final_evaluation = _evaluate_theta(problem, theta_hat)
    final_evaluation.failed && error(
        "The final kappa evaluation failed: $(final_evaluation.error)",
    )
    return (
        theta_hat=theta_hat,
        kappa_hat=final_evaluation.diagnostics.kappa,
        fixed_a=problem.fixed_a,
        fixed_b=problem.fixed_b,
        basis_pairs=hcat(first.(problem.basis_pairs), last.(problem.basis_pairs)),
        chebyshev_domain=problem.chebyshev_domain,
        lower_theta=lower,
        upper_theta=upper,
        minimum=Optim.minimum(result),
        converged=Optim.converged(result),
        iterations=Optim.iterations(result),
        evaluations=problem.evaluation_count,
        diagnostics=final_evaluation.diagnostics,
        result=result,
    )
end

function estimate_kappa_from_observations(
    observations::AbstractMatrix,
    xgrid::AbstractVector{<:Real},
    ygrid::AbstractVector{<:Real},
    Kphi::AbstractMatrix,
    initial_theta;
    fixed_a::Real,
    fixed_b::Real,
    total_degree::Integer=5,
    mean_phi::Union{Nothing, AbstractVector}=nothing,
    obs_indices::Union{Nothing, AbstractVector{<:Integer}}=nothing,
    chebyshev_domain=nothing,
    face_average::Symbol=:harmonic,
    options::LandauKappaMLEOptions=LandauKappaMLEOptions(),
)
    problem = LandauKappaMLEProblem(
        observations,
        xgrid,
        ygrid,
        Kphi;
        fixed_a=fixed_a,
        fixed_b=fixed_b,
        total_degree=total_degree,
        mean_phi=mean_phi,
        obs_indices=obs_indices,
        chebyshev_domain=chebyshev_domain,
        face_average=face_average,
        options=options,
    )
    return (fit=estimate_kappa_theta(problem, initial_theta), problem=problem)
end

"""
Compare the theta score with centered finite differences. Use DenseBackend and
`trace_mode=:exact` for this validation; HODLR compression is approximate.
"""
function check_kappa_score(
    problem::LandauKappaMLEProblem,
    theta;
    relative_step::Float64=1.0e-5,
)
    relative_step > 0.0 ||
        throw(ArgumentError("relative_step must be positive."))
    theta0 = _validate_theta(problem, theta)
    value, analytic, diagnostics = landau_nll_score_theta(problem, theta0)
    finite_difference = similar(theta0)
    steps = relative_step .* max.(abs.(theta0), 1.0)

    for j in eachindex(theta0)
        plus = copy(theta0)
        minus = copy(theta0)
        plus[j] += steps[j]
        minus[j] -= steps[j]
        fplus = first(landau_nll_score_theta(problem, plus))
        fminus = first(landau_nll_score_theta(problem, minus))
        finite_difference[j] = (fplus - fminus) / (2.0 * steps[j])
    end

    absolute_error = abs.(analytic .- finite_difference)
    relative_error = absolute_error ./ max.(1.0, abs.(finite_difference))
    return (
        value=value,
        analytic=analytic,
        finite_difference=finite_difference,
        absolute_error=absolute_error,
        relative_error=relative_error,
        diagnostics=diagnostics,
    )
end

end # module LandauChebyshevKappaMLE
