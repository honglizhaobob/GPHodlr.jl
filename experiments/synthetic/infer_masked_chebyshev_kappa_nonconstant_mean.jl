using Random
using LinearAlgebra
using Statistics
using Printf
using MAT

# Edit the RUN_* constants below, then run this script from the directory
# containing the project files:
#   julia infer_masked_chebyshev_kappa_nonconstant_mean.jl
include("utils/landau_sample_heterogeneous_kappa.jl")
include("native_hodlr_backend_chebyshev_kappa_ad.jl")
using .LandauChebyshevKappaMLE

include("general_positive_kappa.jl")

# =============================================================================
# Model and experiment configuration
# =============================================================================

const A_TRUE = 2.0
const B_TRUE = 3.0
const TRUE_KAPPA_FUNCTION = GeneralPositiveKappa.kappa_true

const CHEBYSHEV_TOTAL_DEGREE = 5
const CHEBYSHEV_PROJECTION_GRID_SIZE = 129
const CHEBYSHEV_DOMAIN = (0.0, 1.0, 0.0, 1.0)
const BASIS_PAIRS = chebyshev_basis_pairs(CHEBYSHEV_TOTAL_DEGREE)
const BASIS_PAIRS_MATRIX = hcat(first.(BASIS_PAIRS), last.(BASIS_PAIRS))
const NTHETA = length(BASIS_PAIRS)

const THETA_INITIAL = let theta = zeros(Float64, NTHETA)
    theta[1] = log(1.0)
    theta
end

function chebyshev_values(z::Real, degree::Int)
    values = Vector{Float64}(undef, degree + 1)
    values[1] = 1.0
    if degree >= 1
        values[2] = Float64(z)
        for k in 2:degree
            values[k + 1] = 2.0 * Float64(z) * values[k] - values[k - 1]
        end
    end
    return values
end

function chebyshev_kappa(x::Real, y::Real, theta::AbstractVector)
    xmin, xmax, ymin, ymax = CHEBYSHEV_DOMAIN
    xhat = 2.0 * (Float64(x) - xmin) / (xmax - xmin) - 1.0
    yhat = 2.0 * (Float64(y) - ymin) / (ymax - ymin) - 1.0
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
    xgrid = range(xmin, xmax; length=CHEBYSHEV_PROJECTION_GRID_SIZE)
    ygrid = range(ymin, ymax; length=CHEBYSHEV_PROJECTION_GRID_SIZE)
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

# Non-constant latent-GP mean, matching the timing/fitting script.
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

const GRID_NX = 32
const GRID_NY = 32

# =============================================================================
# User-editable run configuration
# =============================================================================

# Percentages of spatial locations to hide. Values must be between 30 and 70.
const RUN_MASK_PERCENTS = 30.0:10.0:70.0

# Number of independent solution realizations used to estimate kappa (NMC).
const RUN_REALIZATIONS = 1

# Select :dense for exact dense fitting or :hodlr for the HODLR backend.
const RUN_BACKEND = :dense

# Directory in which the MATLAB result file will be written.
const RUN_OUTPUT_DIR =
    "landau_masked_chebyshev_kappa_nonconstant_mean_results"

const MIN_MASK_PERCENT = 30.0
const MAX_MASK_PERCENT = 70.0

const GENERATION_SEED = 1234
const MASK_SEED = 424242
const OBSERVATION_NOISE_SEED = 161803

const LOWER_THETA = -4.0
const UPPER_THETA = 4.0
const OPTIMIZER_MAX_ITERATIONS = 50
const OPTIMIZER_TIME_LIMIT_SECONDS = 600.0
const BOX_OUTER_MAX_ITERATIONS = 20
const OPTIMIZER_G_TOL = 1.0e-6
const OPTIMIZER_F_TOL = 1.0e-8
const OPTIMIZER_X_TOL = 1.0e-8
const LBFGS_MEMORY = 10

const HODLR_MAX_LEVEL = 2
const HODLR_RANK_DIVISOR = 16
const HODLR_OVERSAMPLING = 10
const HODLR_RANDOM_SEED = 314159


function periodic_grid(nx::Int, ny::Int)
    xgrid = collect(range(0.0, 1.0; length=nx + 1))[1:end-1]
    ygrid = collect(range(0.0, 1.0; length=ny + 1))[1:end-1]
    return xgrid, ygrid
end


"""
    validate_mask_percent(mask_percent)

Validate and return the percentage of grid locations to hide. This experiment
deliberately restricts the adjustable range to 30--70 percent, inclusive.
"""
function validate_mask_percent(mask_percent::Real)
    percent = Float64(mask_percent)
    isfinite(percent) || throw(ArgumentError("mask_percent must be finite."))
    MIN_MASK_PERCENT <= percent <= MAX_MASK_PERCENT || throw(ArgumentError(
        "mask_percent must be between $(MIN_MASK_PERCENT)% and " *
        "$(MAX_MASK_PERCENT)% (inclusive); received $percent%.",
    ))
    return percent
end


"""
    random_observation_mask(N, mask_percent; seed=MASK_SEED)

Return sorted observed and hidden indices in the x-fast `vec(U)` ordering.
Exactly one spatial mask is shared by all realizations.
"""
function random_observation_mask(
    N::Int,
    mask_percent::Real;
    seed::Int=MASK_SEED,
)
    percent = validate_mask_percent(mask_percent)
    number_hidden = clamp(round(Int, percent * N / 100.0), 1, N - 1)
    permutation = randperm(MersenneTwister(seed), N)
    hidden_indices = sort(permutation[1:number_hidden])
    observed_indices = sort(permutation[(number_hidden + 1):end])
    return observed_indices, hidden_indices
end


"""
    generate_latent_ensemble(xgrid, ygrid, M; seed=GENERATION_SEED)

Generate noise-free solution fields with heterogeneous kappa and a non-constant
RBF mean for the latent forcing GP. Observation noise is added later, only at
locations that are retained by the mask. Thus hidden-location errors are
measured against the latent solution, not an unobserved noise realization.
"""
function generate_latent_ensemble(
    xgrid::AbstractVector,
    ygrid::AbstractVector,
    M::Int;
    seed::Int=GENERATION_SEED,
    progress::Bool=true,
)
    M >= 1 || throw(ArgumentError("M must be at least one."))
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
        observation_noise_std=0.0,
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


function inference_options(backend)
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


function backend_from_symbol(backend_name::Symbol)
    backend_name === :dense && return DenseBackend()
    backend_name === :hodlr && return make_hodlr_backend()
    throw(ArgumentError("backend must be :dense or :hodlr."))
end


"""
    conditional_hidden_solutions(problem, observed_values, observed_indices,
                                 hidden_indices, theta_hat)

Condition the implicit Gaussian closure on all observed fields. If
`observed_values` is `M x P`, the returned hidden posterior means are `M x H`.
The conditional covariance is common to all `M` independent realizations.

This calls `_state_and_mean_sensitivities`, the same internal closure routine
used by `landau_nll_score_theta` in the supplied timing script. It ensures that
the mean and covariance used for prediction correspond exactly to `theta_hat`.
"""
function conditional_hidden_solutions(
    problem::LandauChebyshevKappaMLE.LandauKappaMLEProblem,
    observed_values::AbstractMatrix,
    observed_indices::AbstractVector{<:Integer},
    hidden_indices::AbstractVector{<:Integer},
    theta_hat::AbstractVector,
)
    size(observed_values, 2) == length(observed_indices) ||
        throw(DimensionMismatch(
            "observed_values columns must match observed_indices.",
        ))

    theta = LandauChebyshevKappaMLE._validate_theta(problem, theta_hat)
    sensitivities =
        LandauChebyshevKappaMLE._state_and_mean_sensitivities(problem, theta)
    state = sensitivities.state
    # `state.m` and `sensitivities.kappa` may be returned either as vectors or
    # as grid-shaped matrices. Broadcasting followed by `vec` handles both
    # representations and preserves Julia's x-fast column-major ordering.
    implicit_mean = vec(Float64.(state.m))
    solution_covariance = Matrix{Float64}(state.Ku)

    Koo = Matrix{Float64}(
        solution_covariance[observed_indices, observed_indices],
    )
    Kho = Matrix{Float64}(
        solution_covariance[hidden_indices, observed_indices],
    )
    Khh = Matrix{Float64}(
        solution_covariance[hidden_indices, hidden_indices],
    )

    observation_variance =
        problem.options.observation_noise_std^2 +
        problem.options.covariance_jitter
    @inbounds for i in axes(Koo, 1)
        Koo[i, i] += observation_variance
    end
    Koo .= 0.5 .* (Koo .+ transpose(Koo))

    factor = cholesky(Symmetric(Koo); check=true)
    # Convert M x P observations to P x M residual columns.
    residuals =
        Matrix(transpose(observed_values)) .-
        reshape(implicit_mean[observed_indices], :, 1)
    alpha = factor \ residuals
    hidden_means_by_column =
        reshape(implicit_mean[hidden_indices], :, 1) .+
        Kho * alpha

    conditional_covariance = Khh .- Kho * (factor \ transpose(Kho))
    conditional_covariance .= 0.5 .* (
        conditional_covariance .+ transpose(conditional_covariance)
    )
    conditional_variance = max.(diag(conditional_covariance), 0.0)

    return (
        hidden_mean=Matrix(transpose(hidden_means_by_column)),
        hidden_covariance=conditional_covariance,
        hidden_variance=conditional_variance,
        implicit_mean=implicit_mean,
        solution_covariance=solution_covariance,
        kappa=vec(Float64.(sensitivities.kappa)),
        mean_iterations=state.iterations,
        mean_residual_rms=state.residual_rms,
    )
end


"""
    run_masked_inference(; mask_percent=first(RUN_MASK_PERCENTS),
                         M=RUN_REALIZATIONS,
                         backend_name=RUN_BACKEND,
                         output_dir=RUN_OUTPUT_DIR)

Fit heterogeneous kappa from the observed locations only and infer all masked
locations by latent-GP conditioning. `mask_percent` may be any value from 30 to
70, inclusive.
"""
function run_masked_inference(;
    mask_percent::Real=first(RUN_MASK_PERCENTS),
    M::Int=RUN_REALIZATIONS,
    backend_name::Symbol=RUN_BACKEND,
    output_dir::AbstractString=RUN_OUTPUT_DIR,
)
    percent = validate_mask_percent(mask_percent)
    M >= 1 || throw(ArgumentError("M must be at least one."))
    backend = backend_from_symbol(backend_name)
    mkpath(output_dir)

    xgrid, ygrid = periodic_grid(GRID_NX, GRID_NY)
    data = generate_latent_ensemble(xgrid, ygrid, M)
    latent_solutions = Matrix{Float64}(data.observations)
    N = size(latent_solutions, 2)

    observed_indices, hidden_indices =
        random_observation_mask(N, percent)

    noise_rng = MersenneTwister(OBSERVATION_NOISE_SEED)
    observed_values =
        latent_solutions[:, observed_indices] .+
        OBSERVATION_NOISE_STD .* randn(
            noise_rng,
            M,
            length(observed_indices),
        )

    # Only available columns enter the likelihood. Hidden values are not
    # filled with zeros and never leak into the kappa fit.
    options = inference_options(backend)
    problem = LandauKappaMLEProblem(
        observed_values,
        data.xgrid,
        data.ygrid,
        data.covariance_phi;
        fixed_a=A_TRUE,
        fixed_b=B_TRUE,
        total_degree=CHEBYSHEV_TOTAL_DEGREE,
        mean_phi=data.mean_phi,
        obs_indices=observed_indices,
        chebyshev_domain=CHEBYSHEV_DOMAIN,
        face_average=FACE_AVERAGE,
        options=options,
    )

    local fit
    GC.gc()
    fit_elapsed_seconds = @elapsed begin
        fit = estimate_kappa_theta(problem, THETA_INITIAL)
    end
    isnothing(fit.diagnostics) && error(
        "The heterogeneous-kappa fit failed at its final evaluation.",
    )
    if !fit.converged
        warning_message =
            "The MLE did not satisfy the optimizer convergence test; " *
            "prediction will use the final iterate."
        @warn warning_message iterations=fit.iterations evaluations=fit.evaluations
    end

    prediction = conditional_hidden_solutions(
        problem,
        observed_values,
        observed_indices,
        hidden_indices,
        fit.theta_hat,
    )

    hidden_truth = latent_solutions[:, hidden_indices]
    hidden_error = prediction.hidden_mean .- hidden_truth
    hidden_rmse = sqrt(mean(abs2, hidden_error))
    hidden_relative_l2 =
        norm(hidden_error) / max(norm(hidden_truth), eps(Float64))
    hidden_maximum_absolute_error = maximum(abs, hidden_error)
    hidden_rmse_by_realization =
        vec(sqrt.(mean(abs2, hidden_error; dims=2)))
    hidden_rmse_by_location =
        vec(sqrt.(mean(abs2, hidden_error; dims=1)))

    posterior_standard_deviation = sqrt.(prediction.hidden_variance)
    coverage_matrix = abs.(hidden_error) .<=
                      1.96 .* reshape(posterior_standard_deviation, 1, :)
    coverage_95 = mean(coverage_matrix)
    coverage_95_by_realization = vec(mean(coverage_matrix; dims=2))
    coverage_95_by_location = vec(mean(coverage_matrix; dims=1))

    theta_hat = vec(Float64.(fit.theta_hat))
    kappa_hat = vec(Float64.(prediction.kappa))
    kappa_true_vector = vec(Float64.(data.kappa_vector))
    theta_reference_relative_l2 =
        norm(theta_hat .- THETA_REFERENCE) /
        max(norm(THETA_REFERENCE), eps(Float64))
    reference_kappa_grid = [
        chebyshev_kappa(x, y, THETA_REFERENCE)
        for x in data.xgrid, y in data.ygrid
    ]
    reference_relative_l2 =
        norm(reference_kappa_grid .- data.kappa_field) /
        max(norm(data.kappa_field), eps(Float64))
    kappa_error = kappa_hat .- kappa_true_vector
    kappa_relative_l2 =
        norm(kappa_error) / max(norm(kappa_true_vector), eps(Float64))

    # Full M x N arrays are convenient for plots and downstream comparisons.
    masked_observations = fill(NaN, M, N)
    masked_observations[:, observed_indices] .= observed_values
    reconstructed_solutions = repeat(
        reshape(prediction.implicit_mean, 1, :),
        M,
        1,
    )
    reconstructed_solutions[:, observed_indices] .= observed_values
    reconstructed_solutions[:, hidden_indices] .= prediction.hidden_mean
    observed_mask = zeros(Int, N)
    observed_mask[observed_indices] .= 1

    @printf("\nMasked heterogeneous-kappa latent-GP inference\n")
    @printf("  realizations       = %d\n", M)
    @printf("  backend            = %s\n", String(backend_name))
    @printf("  masked percentage  = %.2f%%\n", percent)
    @printf("  observed locations = %d\n", length(observed_indices))
    @printf("  hidden locations   = %d\n", length(hidden_indices))
    println("  MLE converged      = ", fit.converged)
    @printf("  fit elapsed        = %.3f s\n", fit_elapsed_seconds)
    @printf(
        "  theta reference relative L2 = %.6e\n",
        theta_reference_relative_l2,
    )
    @printf(
        "  Chebyshev reference kappa relative L2 = %.6e\n",
        reference_relative_l2,
    )
    @printf("  kappa relative L2  = %.6e\n", kappa_relative_l2)
    @printf("  hidden RMSE        = %.6e\n", hidden_rmse)
    @printf("  hidden relative L2 = %.6e\n", hidden_relative_l2)
    @printf("  hidden max error   = %.6e\n", hidden_maximum_absolute_error)
    @printf("  empirical 95%% coverage = %.4f\n", coverage_95)

    output_path = joinpath(
        output_dir,
        @sprintf(
            "landau_masked_chebyshev_kappa_%02dpct_%s.mat",
            round(Int, percent),
            String(backend_name),
        ),
    )
    matwrite(
        output_path,
        Dict(
            "xgrid" => data.xgrid,
            "ygrid" => data.ygrid,
            "nx" => GRID_NX,
            "ny" => GRID_NY,
            "N" => N,
            "M" => M,
            "mask_percent_requested" => percent,
            "mask_fraction_realized" => length(hidden_indices) / N,
            "observed_indices" => observed_indices,
            "hidden_indices" => hidden_indices,
            "observed_mask_grid" => reshape(observed_mask, GRID_NX, GRID_NY),
            "masked_observations" => masked_observations,
            "latent_solutions" => latent_solutions,
            "reconstructed_solutions" => reconstructed_solutions,
            "hidden_truth" => hidden_truth,
            "hidden_posterior_mean" => prediction.hidden_mean,
            "hidden_posterior_covariance" =>
                prediction.hidden_covariance,
            "hidden_posterior_variance" => prediction.hidden_variance,
            "hidden_posterior_std" => posterior_standard_deviation,
            "hidden_error" => hidden_error,
            "hidden_rmse" => hidden_rmse,
            "hidden_relative_l2_error" => hidden_relative_l2,
            "hidden_maximum_absolute_error" =>
                hidden_maximum_absolute_error,
            "hidden_rmse_by_realization" => hidden_rmse_by_realization,
            "hidden_rmse_by_location" => hidden_rmse_by_location,
            "hidden_coverage_95" => coverage_95,
            "hidden_coverage_95_by_realization" =>
                coverage_95_by_realization,
            "hidden_coverage_95_by_location" => coverage_95_by_location,
            "implicit_solution_mean_vector" => prediction.implicit_mean,
            "implicit_solution_mean_grid" =>
                reshape(prediction.implicit_mean, GRID_NX, GRID_NY),
            "latent_forcing_mean_vector" => data.mean_phi,
            "latent_forcing_mean_grid" =>
                reshape(data.mean_phi, GRID_NX, GRID_NY),
            "rbf_mean_centers_xy" => RBF_MEAN_CENTERS,
            "rbf_mean_coefficients" => RBF_MEAN_COEFFICIENTS,
            "rbf_mean_length_scale" => RBF_MEAN_LENGTH_SCALE,
            "fixed_a" => A_TRUE,
            "fixed_b" => B_TRUE,
            "chebyshev_total_degree" => CHEBYSHEV_TOTAL_DEGREE,
            "chebyshev_basis_pairs" => BASIS_PAIRS_MATRIX,
            "chebyshev_domain" => collect(CHEBYSHEV_DOMAIN),
            "theta_reference" => THETA_REFERENCE,
            "chebyshev_projection_grid_size" =>
                CHEBYSHEV_PROJECTION_GRID_SIZE,
            "theta_initial" => THETA_INITIAL,
            "theta_hat" => theta_hat,
            "theta_reference_relative_l2_error" =>
                theta_reference_relative_l2,
            "kappa_true_vector" => kappa_true_vector,
            "kappa_true_grid" => data.kappa_field,
            "chebyshev_reference_kappa_vector" => vec(reference_kappa_grid),
            "chebyshev_reference_kappa_grid" => reference_kappa_grid,
            "chebyshev_reference_relative_l2_error" =>
                reference_relative_l2,
            "kappa_hat_vector" => kappa_hat,
            "kappa_hat_grid" => reshape(kappa_hat, GRID_NX, GRID_NY),
            "kappa_relative_l2_error" => kappa_relative_l2,
            "kappa_maximum_absolute_error" => maximum(abs, kappa_error),
            "face_average" => String(FACE_AVERAGE),
            "backend" => String(backend_name),
            "mle_converged" => Int(fit.converged),
            "mle_iterations" => fit.iterations,
            "mle_evaluations" => fit.evaluations,
            "mle_negative_loglikelihood" => fit.minimum,
            "fit_elapsed_seconds" => fit_elapsed_seconds,
            "observation_noise_std" => OBSERVATION_NOISE_STD,
            "covariance_jitter" => COVARIANCE_JITTER,
            "nu_phi" => NU_PHI,
            "ell_phi" => ELL_PHI,
            "sigma_phi" => SIGMA_PHI,
            "mean_iterations" => prediction.mean_iterations,
            "mean_residual_rms" => prediction.mean_residual_rms,
            "generation_seed" => GENERATION_SEED,
            "mask_seed" => MASK_SEED,
            "observation_noise_seed" => OBSERVATION_NOISE_SEED,
        );
        compress=true,
    )

    println("  wrote ", output_path)
    return (
        output_path=output_path,
        observed_indices=observed_indices,
        hidden_indices=hidden_indices,
        fit=fit,
        prediction=prediction,
        hidden_rmse=hidden_rmse,
        hidden_relative_l2=hidden_relative_l2,
        coverage_95=coverage_95,
    )
end


# =============================================================================
# Script entry point
# =============================================================================

function main()
    results = []
    for mask_percent in RUN_MASK_PERCENTS
        push!(
            results,
            run_masked_inference(
                mask_percent=mask_percent,
                M=RUN_REALIZATIONS,
                backend_name=RUN_BACKEND,
                output_dir=RUN_OUTPUT_DIR,
            ),
        )
    end
    return results
end

main()
