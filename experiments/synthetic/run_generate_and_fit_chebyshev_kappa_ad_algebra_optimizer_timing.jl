using Random
using LinearAlgebra
using Statistics
using Printf
using MAT

include("utils/landau_sample_heterogeneous_kappa.jl")
include("native_hodlr_backend_chebyshev_kappa_ad.jl")
using .LandauChebyshevKappaMLE

include("general_positive_kappa.jl")

const BACKEND_MODE = :hodlr  # Set to :exact or :hodlr.
const TRUE_KAPPA_FUNCTION = GeneralPositiveKappa.kappa_true

@eval LandauChebyshevKappaMLE begin
    mutable struct KappaAlgebraOptimizerTiming
        dense_K_seconds::Float64
        dense_dK_seconds::Float64
        observation_K_dK_seconds::Float64
        dense_cholesky_seconds::Float64
        hodlr_sigma_construction_seconds::Float64
        hodlr_dsigma_construction_seconds::Float64
    end

    const _kappa_algebra_optimizer_timing =
        Ref{Union{Nothing, KappaAlgebraOptimizerTiming}}(nothing)

    function reset_kappa_algebra_optimizer_timing!()
        _kappa_algebra_optimizer_timing[] = KappaAlgebraOptimizerTiming(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        return nothing
    end

    function kappa_algebra_optimizer_timing_snapshot()
        timing = _kappa_algebra_optimizer_timing[]
        isnothing(timing) && return (
            dense_K_seconds=0.0,
            dense_dK_seconds=0.0,
            observation_K_dK_seconds=0.0,
            dense_cholesky_seconds=0.0,
            hodlr_sigma_construction_seconds=0.0,
            hodlr_dsigma_construction_seconds=0.0,
        )
        return (
            dense_K_seconds=timing.dense_K_seconds,
            dense_dK_seconds=timing.dense_dK_seconds,
            observation_K_dK_seconds=timing.observation_K_dK_seconds,
            dense_cholesky_seconds=timing.dense_cholesky_seconds,
            hodlr_sigma_construction_seconds=
                timing.hodlr_sigma_construction_seconds,
            hodlr_dsigma_construction_seconds=
                timing.hodlr_dsigma_construction_seconds,
        )
    end

    @inline function _record_kappa_timing!(field::Symbol, start_ns::UInt64)
        timing = _kappa_algebra_optimizer_timing[]
        isnothing(timing) && return nothing
        elapsed = (time_ns() - start_ns) * 1.0e-9
        setfield!(timing, field, getfield(timing, field) + elapsed)
        return nothing
    end

    # Dense construction of A=B^{-1} and Ku=A*Kphi*A'.
    function _dense_closure_matrices(
        B::AbstractMatrix,
        Kphi::Matrix{Float64},
    )
        started = time_ns()
        N = size(B, 1)
        factor = lu(B)
        A = Matrix{Float64}(factor \ Matrix{Float64}(I, N, N))
        Ku = A * Kphi * transpose(A)
        A .= 0.5 .* (A .+ transpose(A))
        Ku .= 0.5 .* (Ku .+ transpose(Ku))
        _record_kappa_timing!(:dense_K_seconds, started)
        return A, Ku
    end

    # Total dKu/dtheta_j, including the implicit dm/dtheta_j contribution.
    function _covariance_sensitivity(
        problem::LandauKappaMLEProblem,
        sensitivities,
        j::Int,
    )
        started = time_ns()
        state = sensitivities.state
        mean_theta = view(sensitivities.mean_sensitivities, :, j)
        diagonal_total =
            -6.0 .* problem.fixed_b .* state.m .* mean_theta
        Btheta_Ku =
            sensitivities.Dtheta[j] * state.Ku .+
            reshape(diagonal_total, :, 1) .* state.Ku
        product = state.A * Btheta_Ku
        Ku_theta = -(product .+ transpose(product))
        Ku_theta .= 0.5 .* (Ku_theta .+ transpose(Ku_theta))
        _record_kappa_timing!(:dense_dK_seconds, started)
        return Ku_theta
    end

    function _factor_sigma(backend::DenseBackend, Sigma::Matrix{Float64})
        started = time_ns()
        factor = cholesky(Symmetric(Sigma); check=true)
        _record_kappa_timing!(:dense_cholesky_seconds, started)
        # The log-determinant is retained in the headline algebra time.
        logdet_value = 2.0 * sum(log, diag(factor.L))
        return SigmaFactor(backend, factor, logdet_value)
    end

    function _factor_sigma(backend::HODLRBackend, Sigma::Matrix{Float64})
        started = time_ns()
        factor = backend.factorize_sigma(Sigma)
        _record_kappa_timing!(:hodlr_sigma_construction_seconds, started)
        # The native HODLR log-determinant is retained.
        logdet_value = Float64(backend.logdet_sigma(factor))
        isfinite(logdet_value) || error("HODLR logdet(Sigma) is not finite.")
        return SigmaFactor(backend, factor, logdet_value)
    end

    function _compress_derivative(
        factor::SigmaFactor,
        derivative::Matrix{Float64},
    )
        factor.backend isa DenseBackend && return derivative
        started = time_ns()
        representation = factor.backend.compress_derivative(derivative)
        _record_kappa_timing!(:hodlr_dsigma_construction_seconds, started)
        return representation
    end

    # Same likelihood/score as the dependency, with timing around only the
    # observed Ku/dKu extraction and permutation.
    function landau_nll_score_theta(
        problem::LandauKappaMLEProblem,
        theta,
    )
        theta_float = _validate_theta(problem, theta)
        sensitivities = _state_and_mean_sensitivities(problem, theta_float)
        state = sensitivities.state
        idx = problem.obs_indices
        permutation = problem.permutation
        P = length(idx)
        M = size(problem.observations, 1)
        parameter_count = length(theta_float)

        mean_observed = state.m[idx]
        started = time_ns()
        Sigma = Matrix{Float64}(state.Ku[idx, idx])
        diagonal_noise =
            problem.options.observation_noise_std^2 +
            problem.options.covariance_jitter
        @inbounds for i in 1:P
            Sigma[i, i] += diagonal_noise
        end
        Sigma .= 0.5 .* (Sigma .+ transpose(Sigma))
        Sigma_permuted = Sigma[permutation, permutation]
        _record_kappa_timing!(:observation_K_dK_seconds, started)

        residuals =
            Matrix(transpose(problem.observations)) .-
            reshape(mean_observed, P, 1)
        residuals_permuted = residuals[permutation, :]

        sigma_factor = _factor_sigma(
            problem.options.backend,
            Sigma_permuted,
        )
        alpha = _solve_sigma(sigma_factor, residuals_permuted)
        quadratic = sum(residuals_permuted .* alpha)
        nll = 0.5 * (
            M * (P * log(2.0 * pi) + sigma_factor.logdet_value) +
            quadratic
        )
        alpha_sum = vec(sum(alpha; dims=2))

        score = Vector{Float64}(undef, parameter_count)
        trace_terms = similar(score)
        covariance_quadratics = similar(score)
        mean_terms = similar(score)

        for j in 1:parameter_count
            Ku_theta = _covariance_sensitivity(problem, sensitivities, j)

            started = time_ns()
            derivative_sigma = Matrix{Float64}(Ku_theta[idx, idx])
            derivative_sigma .=
                0.5 .* (derivative_sigma .+ transpose(derivative_sigma))
            derivative_sigma_permuted =
                derivative_sigma[permutation, permutation]
            derivative_mean_permuted =
                sensitivities.mean_sensitivities[idx, j][permutation]
            _record_kappa_timing!(:observation_K_dK_seconds, started)

            representation = _compress_derivative(
                sigma_factor,
                derivative_sigma_permuted,
            )
            trace_term = _trace_inverse_product(
                problem,
                sigma_factor,
                derivative_sigma_permuted,
                representation,
            )
            derivative_alpha = _multiply_derivative(
                sigma_factor,
                representation,
                alpha,
            )
            covariance_quadratic = sum(alpha .* derivative_alpha)
            mean_term = -dot(derivative_mean_permuted, alpha_sum)

            score[j] =
                0.5 * (M * trace_term - covariance_quadratic) + mean_term
            trace_terms[j] = trace_term
            covariance_quadratics[j] = covariance_quadratic
            mean_terms[j] = mean_term
        end

        diagnostics = (
            theta=copy(theta_float),
            fixed_a=problem.fixed_a,
            fixed_b=problem.fixed_b,
            total_degree=problem.total_degree,
            basis_pairs=hcat(
                first.(problem.basis_pairs),
                last.(problem.basis_pairs),
            ),
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
end

const A_TRUE = 2.0
const B_TRUE = 3.0
const CHEBYSHEV_TOTAL_DEGREE = 5
const CHEBYSHEV_PROJECTION_GRID_SIZE = 32
const CHEBYSHEV_DOMAIN = (0.0, 1.0, 0.0, 1.0)
const BASIS_PAIRS = chebyshev_basis_pairs(CHEBYSHEV_TOTAL_DEGREE)
const BASIS_PAIRS_MATRIX = hcat(first.(BASIS_PAIRS), last.(BASIS_PAIRS))
const NTHETA = length(BASIS_PAIRS)
const KAPPA_INITIAL = 1.0
const THETA_INITIAL = let theta = zeros(Float64, NTHETA)
    theta[1] = log(KAPPA_INITIAL)
    theta
end

function periodic_grid(nx::Int, ny::Int)
    xgrid = collect(range(0.0, 1.0; length=nx + 1))[1:end-1]
    ygrid = collect(range(0.0, 1.0; length=ny + 1))[1:end-1]
    return xgrid, ygrid
end

function chebyshev_values(z::Real, degree::Int)
    values = Vector{Float64}(undef, degree + 1)
    values[1] = 1.0
    if degree >= 1
        values[2] = Float64(z)
        for k in 2:degree
            values[k + 1] = 2.0 * z * values[k] - values[k - 1]
        end
    end
    return values
end

function periodic_coordinate(z::Real, lower::Real, upper::Real)
    width = Float64(upper) - Float64(lower)
    width > 0.0 || throw(ArgumentError("periodic domain width must be positive."))
    return Float64(lower) + mod(Float64(z) - Float64(lower), width)
end

function chebyshev_kappa(x::Real, y::Real, theta::AbstractVector)
    xmin, xmax, ymin, ymax = CHEBYSHEV_DOMAIN
    x_periodic = periodic_coordinate(x, xmin, xmax)
    y_periodic = periodic_coordinate(y, ymin, ymax)
    xhat = 2.0 * (x_periodic - xmin) / (xmax - xmin) - 1.0
    yhat = 2.0 * (y_periodic - ymin) / (ymax - ymin) - 1.0
    Tx = chebyshev_values(xhat, CHEBYSHEV_TOTAL_DEGREE)
    Ty = chebyshev_values(yhat, CHEBYSHEV_TOTAL_DEGREE)
    log_kappa = 0.0
    for (coefficient, (i, j)) in zip(theta, BASIS_PAIRS)
        log_kappa += coefficient * Tx[i + 1] * Ty[j + 1]
    end
    return exp(log_kappa)
end

function project_true_log_kappa()
    xmin, xmax, ymin, ymax = CHEBYSHEV_DOMAIN
    xgrid, ygrid = periodic_grid(
        CHEBYSHEV_PROJECTION_GRID_SIZE,
        CHEBYSHEV_PROJECTION_GRID_SIZE,
    )
    sample_count = length(xgrid) * length(ygrid)
    design = Matrix{Float64}(undef, sample_count, NTHETA)
    target = Vector{Float64}(undef, sample_count)

    row = 0
    for y in ygrid, x in xgrid
        row += 1
        xhat = 2.0 * (x - xmin) / (xmax - xmin) - 1.0
        yhat = 2.0 * (y - ymin) / (ymax - ymin) - 1.0
        Tx = chebyshev_values(xhat, CHEBYSHEV_TOTAL_DEGREE)
        Ty = chebyshev_values(yhat, CHEBYSHEV_TOTAL_DEGREE)
        for (column, (i, j)) in pairs(BASIS_PAIRS)
            design[row, column] = Tx[i + 1] * Ty[j + 1]
        end
        target[row] = log(TRUE_KAPPA_FUNCTION(x, y))
    end
    return design \ target
end

const THETA_REFERENCE = project_true_log_kappa()

const RBF_MEAN_CENTERS = [
    0.25 0.25
    0.75 0.30
    0.50 0.75
]
const RBF_MEAN_COEFFICIENTS = [0.9, 1.1, 1.0]
const RBF_MEAN_LENGTH_SCALE = 0.22

function rbf_mean_phi(x::Real, y::Real)
    ell_squared = RBF_MEAN_LENGTH_SCALE^2
    value = 0.0
    @inbounds for j in axes(RBF_MEAN_CENTERS, 1)
        dx = Float64(x) - RBF_MEAN_CENTERS[j, 1]
        dy = Float64(y) - RBF_MEAN_CENTERS[j, 2]
        value += RBF_MEAN_COEFFICIENTS[j] *
                 exp(-(dx^2 + dy^2) / (2.0 * ell_squared))
    end
    return value
end

const NU_PHI = 1.5
const ELL_PHI = 0.20
const SIGMA_PHI = 0.15
const OBSERVATION_NOISE_STD = 0.01
const COVARIANCE_JITTER = 1.0e-8
const FACE_AVERAGE = :harmonic

const M_VALUES = 2 .^ (8:12)
const PRODUCTION_SEED = 1234
const PRODUCTION_NX = 32
const PRODUCTION_NY = PRODUCTION_NX

const LOWER_THETA = -1.0
const UPPER_THETA = 1.0
const OPTIMIZER_MAX_ITERATIONS = 50
const OPTIMIZER_TIME_LIMIT_SECONDS = 600.0
const BOX_OUTER_MAX_ITERATIONS = 20
const OPTIMIZER_G_TOL = 1.0e-4
const OPTIMIZER_F_TOL = 1.0e-8
const OPTIMIZER_X_TOL = 1.0e-8
const LBFGS_MEMORY = 10

const HODLR_MAX_LEVEL = 2
const HODLR_RANK_DIVISOR = 16
const HODLR_OVERSAMPLING = 10
const HODLR_RANDOM_SEED = 123456
# Large enough that rank + oversampling fits in the finest HODLR leaves.
const WARMUP_NX = 16
const WARMUP_NY = 16

function generate_data(
    xgrid::AbstractVector,
    ygrid::AbstractVector,
    M::Int;
    seed::Int,
    progress::Bool,
)
    nx, ny = length(xgrid), length(ygrid)
    initial_solution = fill(sqrt(A_TRUE / B_TRUE), nx, ny)
    return generate_landau_observations(
        xgrid,
        ygrid,
        M,
        NU_PHI,
        ELL_PHI,
        SIGMA_PHI,
        A_TRUE,
        B_TRUE,
        TRUE_KAPPA_FUNCTION;
        mean_fn=rbf_mean_phi,
        rng=MersenneTwister(seed),
        periodic_kernel=false,
        observation_noise_std=OBSERVATION_NOISE_STD,
        u0=initial_solution,
        warm_start=false,
        solver_face_average=FACE_AVERAGE,
        solver_verbose=false,
        progress=progress,
        return_forcing=false,
    )
end

function make_hodlr_backend()
    return make_native_hodlr_backend(
        sigma_max_level=HODLR_MAX_LEVEL,
        sigma_local_rank=nothing,
        sigma_rank_divisor=HODLR_RANK_DIVISOR,
        sigma_oversampling=HODLR_OVERSAMPLING,
        sigma_random_seed=HODLR_RANDOM_SEED,
    )
end

function fit_options(backend)
    return LandauKappaMLEOptions(
        backend=backend,
        observation_noise_std=OBSERVATION_NOISE_STD,
        covariance_jitter=COVARIANCE_JITTER,
        mean_branch=:positive,
        lower_theta=LOWER_THETA,
        upper_theta=UPPER_THETA,
        box_outer_iterations=BOX_OUTER_MAX_ITERATIONS,
        spatial_ordering=true,
        trace_mode=:exact,
        mean_tol=1.0e-8,
        mean_maxiter=30,
        lbfgs_memory=LBFGS_MEMORY,
        optimizer_iterations=OPTIMIZER_MAX_ITERATIONS,
        optimizer_time_limit_seconds=OPTIMIZER_TIME_LIMIT_SECONDS,
        optimizer_g_tol=OPTIMIZER_G_TOL,
        optimizer_f_tol=OPTIMIZER_F_TOL,
        optimizer_x_tol=OPTIMIZER_X_TOL,
        show_optimizer_trace=true,
        show_evaluations=true,
    )
end

function fit_one_backend(
    observations_M::AbstractMatrix,
    data,
    backend,
)
    problem = LandauKappaMLEProblem(
        observations_M,
        data.xgrid,
        data.ygrid,
        data.covariance_phi;
        fixed_a=A_TRUE,
        fixed_b=B_TRUE,
        total_degree=CHEBYSHEV_TOTAL_DEGREE,
        mean_phi=data.mean_phi,
        chebyshev_domain=CHEBYSHEV_DOMAIN,
        face_average=FACE_AVERAGE,
        options=fit_options(backend),
    )

    local fit
    GC.gc()
    LandauChebyshevKappaMLE.reset_kappa_algebra_optimizer_timing!()
    raw_optimizer_wall_seconds = @elapsed begin
        fit = estimate_kappa_theta(problem, THETA_INITIAL)
    end

    excluded =
        LandauChebyshevKappaMLE.kappa_algebra_optimizer_timing_snapshot()
    excluded_total_seconds =
        excluded.dense_K_seconds +
        excluded.dense_dK_seconds +
        excluded.observation_K_dK_seconds +
        excluded.dense_cholesky_seconds +
        excluded.hodlr_sigma_construction_seconds +
        excluded.hodlr_dsigma_construction_seconds

    adjusted_seconds = max(
        0.0,
        raw_optimizer_wall_seconds - excluded_total_seconds,
    )
    timing = (
        algebra_optimizer_elapsed_seconds=adjusted_seconds,
        raw_optimizer_wall_seconds=raw_optimizer_wall_seconds,
        excluded_total_seconds=excluded_total_seconds,
        dense_K_seconds=excluded.dense_K_seconds,
        dense_dK_seconds=excluded.dense_dK_seconds,
        observation_K_dK_seconds=excluded.observation_K_dK_seconds,
        dense_cholesky_seconds=excluded.dense_cholesky_seconds,
        hodlr_sigma_construction_seconds=
            excluded.hodlr_sigma_construction_seconds,
        hodlr_dsigma_construction_seconds=
            excluded.hodlr_dsigma_construction_seconds,
    )
    return fit, timing
end

function empty_backend_results(nx::Int, ny::Int)
    count = length(M_VALUES)
    return (
        theta_hat=zeros(Float64, NTHETA, count),
        kappa_hat=zeros(Float64, nx * ny, count),
        minimum=zeros(Float64, count),
        converged=zeros(Int, count),
        iterations=zeros(Int, count),
        evaluations=zeros(Int, count),
        elapsed_seconds=zeros(Float64, count),
        raw_optimizer_wall_seconds=zeros(Float64, count),
        excluded_total_seconds=zeros(Float64, count),
        dense_K_seconds=zeros(Float64, count),
        dense_dK_seconds=zeros(Float64, count),
        observation_K_dK_seconds=zeros(Float64, count),
        dense_cholesky_seconds=zeros(Float64, count),
        hodlr_sigma_construction_seconds=zeros(Float64, count),
        hodlr_dsigma_construction_seconds=zeros(Float64, count),
        theta_reference_relative_error=zeros(Float64, count),
        kappa_relative_error=zeros(Float64, count),
        kappa_maximum_error=zeros(Float64, count),
        implicit_mean=zeros(Float64, nx * ny, count),
        empirical_mean=zeros(Float64, nx * ny, count),
        mean_relative_error=zeros(Float64, count),
        mean_maximum_error=zeros(Float64, count),
    )
end

function record_fit!(
    storage,
    index::Int,
    fit,
    timing,
    observations_M::AbstractMatrix,
    true_kappa::AbstractVector,
)
    isnothing(fit.diagnostics) &&
        error("The final likelihood evaluation failed and cannot be exported.")

    theta_hat = Vector{Float64}(fit.theta_hat)
    kappa_hat = vec(Matrix{Float64}(fit.kappa_hat))
    implicit_mean = Vector{Float64}(fit.diagnostics.mean)
    empirical_mean = vec(mean(observations_M; dims=1))

    storage.theta_hat[:, index] .= theta_hat
    storage.kappa_hat[:, index] .= kappa_hat
    storage.minimum[index] = fit.minimum
    storage.converged[index] = Int(fit.converged)
    storage.iterations[index] = fit.iterations
    storage.evaluations[index] = fit.evaluations
    storage.elapsed_seconds[index] = timing.algebra_optimizer_elapsed_seconds
    storage.raw_optimizer_wall_seconds[index] =
        timing.raw_optimizer_wall_seconds
    storage.excluded_total_seconds[index] = timing.excluded_total_seconds
    storage.dense_K_seconds[index] = timing.dense_K_seconds
    storage.dense_dK_seconds[index] = timing.dense_dK_seconds
    storage.observation_K_dK_seconds[index] =
        timing.observation_K_dK_seconds
    storage.dense_cholesky_seconds[index] = timing.dense_cholesky_seconds
    storage.hodlr_sigma_construction_seconds[index] =
        timing.hodlr_sigma_construction_seconds
    storage.hodlr_dsigma_construction_seconds[index] =
        timing.hodlr_dsigma_construction_seconds

    storage.theta_reference_relative_error[index] =
        norm(theta_hat .- THETA_REFERENCE) /
        max(norm(THETA_REFERENCE), eps(Float64))
    kappa_error = kappa_hat .- true_kappa
    storage.kappa_relative_error[index] =
        norm(kappa_error) / max(norm(true_kappa), eps(Float64))
    storage.kappa_maximum_error[index] = maximum(abs, kappa_error)

    storage.implicit_mean[:, index] .= implicit_mean
    storage.empirical_mean[:, index] .= empirical_mean
    mean_error = implicit_mean .- empirical_mean
    storage.mean_relative_error[index] =
        norm(mean_error) / max(norm(empirical_mean), eps(Float64))
    storage.mean_maximum_error[index] = maximum(abs, mean_error)
    return nothing
end

function common_export_dictionary(data, nx::Int, ny::Int)
    reference_kappa_grid = [
        chebyshev_kappa(x, y, THETA_REFERENCE)
        for x in data.xgrid, y in data.ygrid
    ]
    reference_relative_error =
        norm(reference_kappa_grid .- data.kappa_field) /
        max(norm(data.kappa_field), eps(Float64))

    return Dict(
        "fixed_a" => A_TRUE,
        "fixed_b" => B_TRUE,
        "chebyshev_total_degree" => CHEBYSHEV_TOTAL_DEGREE,
        "chebyshev_basis_pairs" => BASIS_PAIRS_MATRIX,
        "chebyshev_domain" => collect(CHEBYSHEV_DOMAIN),
        "theta_reference" => THETA_REFERENCE,
        "chebyshev_projection_grid_size" => CHEBYSHEV_PROJECTION_GRID_SIZE,
        "theta_initial" => THETA_INITIAL,
        "theta_lower_bound" => LOWER_THETA,
        "theta_upper_bound" => UPPER_THETA,
        "true_kappa_vector" => data.kappa_vector,
        "true_kappa_grid" => data.kappa_field,
        "chebyshev_reference_kappa_vector" => vec(reference_kappa_grid),
        "chebyshev_reference_kappa_grid" => reference_kappa_grid,
        "chebyshev_reference_relative_l2_error" =>
            reference_relative_error,
        "face_average" => String(FACE_AVERAGE),
        "nu_phi" => NU_PHI,
        "ell_phi" => ELL_PHI,
        "sigma_phi" => SIGMA_PHI,
        "observation_noise_std" => OBSERVATION_NOISE_STD,
        "covariance_jitter" => COVARIANCE_JITTER,
        "optimizer_time_limit_seconds" => OPTIMIZER_TIME_LIMIT_SECONDS,
        "optimizer_max_iterations" => OPTIMIZER_MAX_ITERATIONS,
        "box_outer_max_iterations" => BOX_OUTER_MAX_ITERATIONS,
        "optimizer_g_tol" => OPTIMIZER_G_TOL,
        "optimizer_f_tol" => OPTIMIZER_F_TOL,
        "optimizer_x_tol" => OPTIMIZER_X_TOL,
        "lbfgs_memory" => LBFGS_MEMORY,
        "latent_forcing_mean_vector" => data.mean_phi,
        "latent_forcing_mean_grid" => reshape(data.mean_phi, nx, ny),
        "rbf_mean_centers_xy" => RBF_MEAN_CENTERS,
        "rbf_mean_coefficients" => RBF_MEAN_COEFFICIENTS,
        "rbf_mean_length_scale" => RBF_MEAN_LENGTH_SCALE,
        "xgrid" => data.xgrid,
        "ygrid" => data.ygrid,
        "nx" => nx,
        "ny" => ny,
        "production_seed" => PRODUCTION_SEED,
        "hodlr_max_level" => HODLR_MAX_LEVEL,
        "hodlr_rank_divisor" => HODLR_RANK_DIVISOR,
        "hodlr_oversampling" => HODLR_OVERSAMPLING,
        "hodlr_random_seed" => HODLR_RANDOM_SEED,
    )
end

function fit_dictionary(
    backend_label::AbstractString,
    M::Int,
    nx::Int,
    ny::Int,
    index::Int,
    storage,
    data,
)
    output = common_export_dictionary(data, nx, ny)
    merge!(
        output,
        Dict(
            "backend" => String(backend_label),
            "M" => M,
            "theta_hat" => storage.theta_hat[:, index],
            "kappa_hat_vector" => storage.kappa_hat[:, index],
            "kappa_hat_grid" => reshape(storage.kappa_hat[:, index], nx, ny),
            "theta_reference_relative_l2_error" =>
                storage.theta_reference_relative_error[index],
            "kappa_relative_l2_error" => storage.kappa_relative_error[index],
            "kappa_maximum_absolute_error" =>
                storage.kappa_maximum_error[index],
            "negative_loglikelihood" => storage.minimum[index],
            "converged" => storage.converged[index],
            "iterations" => storage.iterations[index],
            "evaluations" => storage.evaluations[index],
            "optimization_elapsed_seconds" => storage.elapsed_seconds[index],
            "algebra_optimizer_elapsed_seconds" =>
                storage.elapsed_seconds[index],
            "raw_optimizer_wall_seconds" =>
                storage.raw_optimizer_wall_seconds[index],
            "excluded_total_seconds" => storage.excluded_total_seconds[index],
            "excluded_dense_K_seconds" => storage.dense_K_seconds[index],
            "excluded_dense_dK_seconds" => storage.dense_dK_seconds[index],
            "excluded_observation_K_dK_seconds" =>
                storage.observation_K_dK_seconds[index],
            "excluded_dense_cholesky_seconds" =>
                storage.dense_cholesky_seconds[index],
            "excluded_hodlr_sigma_construction_seconds" =>
                storage.hodlr_sigma_construction_seconds[index],
            "excluded_hodlr_dsigma_construction_seconds" =>
                storage.hodlr_dsigma_construction_seconds[index],
            "implicit_solution_mean_vector" => storage.implicit_mean[:, index],
            "implicit_solution_mean_grid" =>
                reshape(storage.implicit_mean[:, index], nx, ny),
            "empirical_solution_mean_vector" =>
                storage.empirical_mean[:, index],
            "empirical_solution_mean_grid" =>
                reshape(storage.empirical_mean[:, index], nx, ny),
            "implicit_mean_relative_l2_error" =>
                storage.mean_relative_error[index],
            "implicit_mean_maximum_absolute_error" =>
                storage.mean_maximum_error[index],
        ),
    )
    return output
end

function write_backend_summary(
    output_dir::AbstractString,
    backend_label::AbstractString,
    filename::AbstractString,
    storage,
    data,
    nx::Int,
    ny::Int,
)
    output = common_export_dictionary(data, nx, ny)
    merge!(
        output,
        Dict(
            "backend" => String(backend_label),
            "M_values" => collect(M_VALUES),
            "theta_hat" => storage.theta_hat,
            "kappa_hat" => storage.kappa_hat,
            "theta_reference_relative_l2_error" =>
                storage.theta_reference_relative_error,
            "kappa_relative_l2_error" => storage.kappa_relative_error,
            "kappa_maximum_absolute_error" => storage.kappa_maximum_error,
            "negative_loglikelihood" => storage.minimum,
            "converged" => storage.converged,
            "iterations" => storage.iterations,
            "evaluations" => storage.evaluations,
            "optimization_elapsed_seconds" => storage.elapsed_seconds,
            "algebra_optimizer_elapsed_seconds" => storage.elapsed_seconds,
            "total_optimization_elapsed_seconds" =>
                sum(storage.elapsed_seconds),
            "total_algebra_optimizer_elapsed_seconds" =>
                sum(storage.elapsed_seconds),
            "raw_optimizer_wall_seconds" =>
                storage.raw_optimizer_wall_seconds,
            "total_raw_optimizer_wall_seconds" =>
                sum(storage.raw_optimizer_wall_seconds),
            "excluded_total_seconds" => storage.excluded_total_seconds,
            "total_excluded_seconds" => sum(storage.excluded_total_seconds),
            "excluded_dense_K_seconds" => storage.dense_K_seconds,
            "excluded_dense_dK_seconds" => storage.dense_dK_seconds,
            "excluded_observation_K_dK_seconds" =>
                storage.observation_K_dK_seconds,
            "excluded_dense_cholesky_seconds" =>
                storage.dense_cholesky_seconds,
            "excluded_hodlr_sigma_construction_seconds" =>
                storage.hodlr_sigma_construction_seconds,
            "excluded_hodlr_dsigma_construction_seconds" =>
                storage.hodlr_dsigma_construction_seconds,
            "timing_definition" =>
                "raw optimizer wall time minus dense Ku/dKu construction, observed K/dK assembly, dense Cholesky, and HODLR covariance/derivative construction",
            "implicit_solution_mean" => storage.implicit_mean,
            "empirical_solution_mean" => storage.empirical_mean,
            "implicit_mean_relative_l2_error" => storage.mean_relative_error,
            "implicit_mean_maximum_absolute_error" =>
                storage.mean_maximum_error,
        ),
    )
    matwrite(joinpath(output_dir, filename), output; compress=true)
end

function selected_backend()
    BACKEND_MODE === :exact && return ("exact", DenseBackend())
    BACKEND_MODE === :hodlr && return ("hodlr", make_hodlr_backend())
end

function warm_up_timing_path()
    backend_label, backend = selected_backend()
    println("\nWarming up the $backend_label path (not reported) ...")
    xgrid, ygrid = periodic_grid(WARMUP_NX, WARMUP_NY)
    data = generate_data(
        xgrid,
        ygrid,
        16;
        seed=PRODUCTION_SEED + 1,
        progress=false,
    )

    options = fit_options(backend)
    options.optimizer_iterations = 1
    options.box_outer_iterations = 1
    options.optimizer_time_limit_seconds =
        min(60.0, OPTIMIZER_TIME_LIMIT_SECONDS)
    options.show_optimizer_trace = false
    options.show_evaluations = false
    problem = LandauKappaMLEProblem(
        data.observations,
        data.xgrid,
        data.ygrid,
        data.covariance_phi;
        fixed_a=A_TRUE,
        fixed_b=B_TRUE,
        total_degree=CHEBYSHEV_TOTAL_DEGREE,
        mean_phi=data.mean_phi,
        chebyshev_domain=CHEBYSHEV_DOMAIN,
        face_average=FACE_AVERAGE,
        options=options,
    )
    estimate_kappa_theta(problem, THETA_INITIAL)
    return nothing
end

function run_production_fits(output_dir::AbstractString)
    backend_label, _ = selected_backend()
    nx, ny = PRODUCTION_NX, PRODUCTION_NY
    xgrid, ygrid = periodic_grid(nx, ny)
    M_max = maximum(M_VALUES)

    println("\nGenerating one nested heterogeneous-kappa ensemble with M_max=$M_max ...")
    data = generate_data(
        xgrid,
        ygrid,
        M_max;
        seed=PRODUCTION_SEED,
        progress=true,
    )
    @printf(
        "True kappa range on the production grid: [%.6f, %.6f]\n",
        minimum(data.kappa_field),
        maximum(data.kappa_field),
    )

    reference_kappa = [
        chebyshev_kappa(x, y, THETA_REFERENCE) for x in xgrid, y in ygrid
    ]
    reference_error =
        norm(reference_kappa .- data.kappa_field) /
        max(norm(data.kappa_field), eps(Float64))
    @printf(
        "Degree-%d Chebyshev-exp reference relative error: %.3e\n",
        CHEBYSHEV_TOTAL_DEGREE,
        reference_error,
    )

    storage = empty_backend_results(nx, ny)

    for (index, M) in pairs(M_VALUES)
        println("\n", repeat("=", 72))
        println("Fitting the same M=$M fields with fixed a,b and unknown kappa")
        println(repeat("=", 72))
        observations_M = Matrix(@view data.observations[1:M, :])

        println("\n--- $backend_label backend ---")
        _, backend = selected_backend()
        fit, timing = fit_one_backend(observations_M, data, backend)
        record_fit!(
            storage,
            index,
            fit,
            timing,
            observations_M,
            data.kappa_vector,
        )
        @printf(
            "%s: kappa relative error=%.3e, algebra/optimizer=%.3f s, iterations=%d\n",
            backend_label,
            storage.kappa_relative_error[index],
            timing.algebra_optimizer_elapsed_seconds,
            fit.iterations,
        )
        filename = @sprintf(
            "landau_kappa_chebyshev_ad_%s_algebra_optimizer_M%04d.mat",
            backend_label,
            M,
        )
        matwrite(
            joinpath(output_dir, filename),
            fit_dictionary(
                backend_label,
                M,
                nx,
                ny,
                index,
                storage,
                data,
            );
            compress=true,
        )
    end

    summary_filename = @sprintf(
        "landau_kappa_chebyshev_ad_%s_algebra_optimizer_timing.mat",
        backend_label,
    )
    write_backend_summary(
        output_dir,
        backend_label,
        summary_filename,
        storage,
        data,
        nx,
        ny,
    )

    @printf(
        "\nTotal %s algebra/optimizer time: %.3f seconds\n",
        backend_label,
        sum(storage.elapsed_seconds),
    )
    println("Summary: ", joinpath(output_dir, summary_filename))
    return (
        M_values=collect(M_VALUES),
        backend=BACKEND_MODE,
        results=storage,
    )
end

function main()
    output_dir = "landau_kappa_chebyshev_ad_algebra_optimizer_timing_results"
    mkpath(output_dir)
    selected_backend()

    if "--no-warmup" in ARGS
        println(
            "\nWarning: production timing may include Julia compilation overhead.",
        )
    else
        warm_up_timing_path()
    end
    return run_production_fits(output_dir)
end

main()
