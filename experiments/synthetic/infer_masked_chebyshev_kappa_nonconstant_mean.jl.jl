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

include("fixed_matern_kappa_landau_observations.jl")
using .FixedMaternKappaLandauData

# =============================================================================
# Model and experiment configuration
# =============================================================================

const A_TRUE = 2.0
const B_TRUE = 3.0
const TRUE_LANDAU_OBSERVATIONS_PATH =
    FixedMaternKappaLandauData.LANDAU_DATA_PATH

const CHEBYSHEV_TOTAL_DEGREE = 5
const CHEBYSHEV_PROJECTION_GRID_SIZE = FixedMaternKappaLandauData.MAX_NX
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
    xgrid, ygrid = FixedMaternKappaLandauData.periodic_grid(
        CHEBYSHEV_PROJECTION_GRID_SIZE,
        CHEBYSHEV_PROJECTION_GRID_SIZE,
    )
    true_log_kappa = log.(vec(FixedMaternKappaLandauData.kappa_grid(
        CHEBYSHEV_PROJECTION_GRID_SIZE,
        CHEBYSHEV_PROJECTION_GRID_SIZE,
    )))
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
        target[row] = true_log_kappa[row]
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

# Any divisor of FixedMaternKappaLandauData.MAX_NX gives a nested subset of the
# fixed maximum-grid realization. For example, use 2^3 for an 8 x 8 solve.
const RUN_GRID_SIZES = 2 .^ (3:6)
const GRID_NX = first(RUN_GRID_SIZES)
const GRID_NY = GRID_NX

# =============================================================================
# User-editable run configuration
# =============================================================================

# Percentages of spatial locations to hide. Values must be between 30 and 70.
const RUN_MASK_PERCENTS = 40.0:10.0:50.0

# Number of independent solution realizations used to estimate kappa (NMC).
const RUN_REALIZATIONS = FixedMaternKappaLandauData.DEFAULT_REALIZATIONS

# Select :dense for exact dense fitting or :hodlr for the HODLR backend.
const RUN_BACKEND = :hodlr

# Directory in which the MATLAB result file will be written.
const RUN_OUTPUT_DIR =
    "landau_masked_chebyshev_kappa_matern_nonconstant_mean_results"

const MIN_MASK_PERCENT = 40.0
const MAX_MASK_PERCENT = 50.0

const GENERATION_SEED = 1234
const MASK_SEED = 424242
const OBSERVATION_NOISE_SEED = 161803

const LOWER_THETA = -4.0
const UPPER_THETA = 4.0
const OPTIMIZER_MAX_ITERATIONS = 50
const OPTIMIZER_TIME_LIMIT_SECONDS = 3600.0
const BOX_OUTER_MAX_ITERATIONS = 20
const OPTIMIZER_G_TOL = 1.0e-6
const OPTIMIZER_F_TOL = 1.0e-8
const OPTIMIZER_X_TOL = 1.0e-8
const LBFGS_MEMORY = 10

const HODLR_MAX_LEVEL = 2
const HODLR_RANK_DIVISOR = 16
const HODLR_OVERSAMPLING = 2
const HODLR_RANDOM_SEED = 314159

hodlr_observation_divisor() = 2^HODLR_MAX_LEVEL
hodlr_local_rank(n::Integer) = max(1, div(Int(n), HODLR_RANK_DIVISOR))

function effective_hodlr_oversampling(n::Integer)
    divisor = hodlr_observation_divisor()
    n % divisor == 0 || throw(ArgumentError(
        "HODLR observed location count $n must be divisible by $divisor.",
    ))
    leaf_size = div(Int(n), divisor)
    return max(0, min(HODLR_OVERSAMPLING, leaf_size - hodlr_local_rank(n)))
end

function observed_count_divisor(backend_name::Symbol)
    backend_name in (:dense, :hodlr) || throw(ArgumentError(
        "backend must be :dense or :hodlr.",
    ))
    # Dense exact runs use the same split as HODLR so the two modes are
    # directly comparable at every requested mask percentage.
    return hodlr_observation_divisor()
end


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
    random_observation_mask(N, mask_percent; seed=MASK_SEED,
                            observed_divisor=1)

Return sorted observed and hidden indices in the x-fast `vec(U)` ordering.
Exactly one spatial mask is shared by all realizations. If `observed_divisor`
is greater than one, the number of observed locations is rounded to the nearest
legal multiple, with ties keeping more observations for the fit.
"""
function random_observation_mask(
    N::Int,
    mask_percent::Real;
    seed::Int=MASK_SEED,
    observed_divisor::Int=1,
)
    percent = validate_mask_percent(mask_percent)
    observed_divisor >= 1 ||
        throw(ArgumentError("observed_divisor must be positive."))
    requested_hidden = clamp(round(Int, percent * N / 100.0), 1, N - 1)
    requested_observed = N - requested_hidden

    max_observed =
        observed_divisor * floor(Int, (N - 1) / observed_divisor)
    max_observed >= observed_divisor || throw(ArgumentError(
        "N=$N has no non-empty train/mask split with observed_divisor=$observed_divisor.",
    ))
    lower_observed =
        observed_divisor * floor(Int, requested_observed / observed_divisor)
    upper_observed =
        observed_divisor * ceil(Int, requested_observed / observed_divisor)
    candidate_observed_counts = Int[]
    for observed_count in (lower_observed, upper_observed)
        if observed_divisor <= observed_count <= max_observed
            push!(candidate_observed_counts, observed_count)
        end
    end
    isempty(candidate_observed_counts) && error(
        "Could not choose a valid observed count divisible by $observed_divisor.",
    )
    number_observed = first(candidate_observed_counts)
    best_distance = abs(number_observed - requested_observed)
    for observed_count in Iterators.drop(candidate_observed_counts, 1)
        distance = abs(observed_count - requested_observed)
        if distance < best_distance ||
           (distance == best_distance && observed_count > number_observed)
            number_observed = observed_count
            best_distance = distance
        end
    end
    number_hidden = N - number_observed

    permutation = randperm(MersenneTwister(seed), N)
    hidden_indices = sort(permutation[1:number_hidden])
    observed_indices = sort(permutation[(number_hidden + 1):end])
    return observed_indices, hidden_indices
end


"""
    load_latent_ensemble(xgrid, ygrid, M; seed=GENERATION_SEED)

Load fixed maximum-grid Landau samples and return the nested subset for the
requested grid. `observations` are latent/noise-free; `noisy_observations`
contains the corresponding fixed noisy observations for likelihood input.
"""
function load_latent_ensemble(
    xgrid::AbstractVector,
    ygrid::AbstractVector,
    M::Int;
    seed::Int=GENERATION_SEED,
    progress::Bool=true,
)
    M >= 1 || throw(ArgumentError("M must be at least one."))
    nx, ny = length(xgrid), length(ygrid)
    progress && println(
        "Loading fixed nested latent Landau observations for grid ",
        nx,
        " x ",
        ny,
        " and M = ",
        M,
    )
    return FixedMaternKappaLandauData.load_landau_subset(
        nx,
        ny,
        M;
        use_noisy_observations=false,
    )
end


function make_hodlr_backend(observed_count::Integer)
    sigma_oversampling = effective_hodlr_oversampling(observed_count)
    if sigma_oversampling < HODLR_OVERSAMPLING
        @warn "Reducing HODLR oversampling for the observed problem size." observed_count requested=HODLR_OVERSAMPLING effective=sigma_oversampling
    end
    return make_native_hodlr_backend(
        sigma_max_level=HODLR_MAX_LEVEL,
        sigma_local_rank=nothing,
        sigma_rank_divisor=HODLR_RANK_DIVISOR,
        sigma_oversampling=sigma_oversampling,
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


function backend_from_symbol(
    backend_name::Symbol;
    observed_count::Union{Nothing, Integer}=nothing,
)
    backend_name === :dense && return DenseBackend()
    if backend_name === :hodlr
        isnothing(observed_count) && throw(ArgumentError(
            "observed_count is required for the HODLR backend.",
        ))
        return make_hodlr_backend(observed_count)
    end
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
    run_masked_inference(; nx=GRID_NX,
                         ny=GRID_NY,
                         mask_percent=first(RUN_MASK_PERCENTS),
                         M=RUN_REALIZATIONS,
                         backend_name=RUN_BACKEND,
                         output_dir=RUN_OUTPUT_DIR)

Fit heterogeneous kappa from the observed locations only and infer all masked
locations by latent-GP conditioning. `mask_percent` may be any value from 30 to
70, inclusive.
"""
function run_masked_inference(;
    nx::Int=GRID_NX,
    ny::Int=GRID_NY,
    mask_percent::Real=first(RUN_MASK_PERCENTS),
    M::Int=RUN_REALIZATIONS,
    backend_name::Symbol=RUN_BACKEND,
    output_dir::AbstractString=RUN_OUTPUT_DIR,
)
    percent = validate_mask_percent(mask_percent)
    M >= 1 || throw(ArgumentError("M must be at least one."))
    grid_output_dir = joinpath(output_dir, @sprintf("grid_%04dx%04d", nx, ny))
    mkpath(grid_output_dir)

    xgrid, ygrid = periodic_grid(nx, ny)
    data = load_latent_ensemble(xgrid, ygrid, M)
    latent_solutions = Matrix{Float64}(data.latent_observations)
    N = size(latent_solutions, 2)

    observation_divisor = observed_count_divisor(backend_name)
    observed_indices, hidden_indices =
        random_observation_mask(
            N,
            percent;
            observed_divisor=observation_divisor,
        )
    backend = backend_from_symbol(
        backend_name;
        observed_count=length(observed_indices),
    )
    hodlr_oversampling_effective = backend_name === :hodlr ?
        effective_hodlr_oversampling(length(observed_indices)) : 0
    hodlr_local_rank_effective = backend_name === :hodlr ?
        hodlr_local_rank(length(observed_indices)) : 0

    observed_values = Matrix{Float64}(
        @view(data.noisy_observations[:, observed_indices])
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
    @printf("  grid               = %d x %d\n", nx, ny)
    @printf("  realizations       = %d\n", M)
    @printf("  backend            = %s\n", String(backend_name))
    @printf("  requested mask     = %.2f%%\n", percent)
    @printf(
        "  realized mask      = %.2f%%\n",
        100.0 * length(hidden_indices) / N,
    )
    @printf("  observed locations = %d\n", length(observed_indices))
    @printf("  hidden locations   = %d\n", length(hidden_indices))
    if backend_name === :hodlr
        @printf("  HODLR divisor      = %d\n", observation_divisor)
        @printf("  HODLR local rank   = %d\n", hodlr_local_rank_effective)
        @printf(
            "  HODLR oversampling = %d\n",
            hodlr_oversampling_effective,
        )
    end
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
        grid_output_dir,
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
            "nx" => nx,
            "ny" => ny,
            "N" => N,
            "run_grid_sizes" => collect(RUN_GRID_SIZES),
            "M" => M,
            "mask_percent_requested" => percent,
            "mask_percent_realized" => 100.0 * length(hidden_indices) / N,
            "mask_fraction_realized" => length(hidden_indices) / N,
            "observation_count_divisor" => observation_divisor,
            "observed_indices" => observed_indices,
            "hidden_indices" => hidden_indices,
            "observed_mask_grid" => reshape(observed_mask, nx, ny),
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
                reshape(prediction.implicit_mean, nx, ny),
            "latent_forcing_mean_vector" => data.mean_phi,
            "latent_forcing_mean_grid" =>
                reshape(data.mean_phi, nx, ny),
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
            "kappa_hat_grid" => reshape(kappa_hat, nx, ny),
            "kappa_relative_l2_error" => kappa_relative_l2,
            "kappa_maximum_absolute_error" => maximum(abs, kappa_error),
            "face_average" => String(FACE_AVERAGE),
            "backend" => String(backend_name),
            "mle_converged" => Int(fit.converged),
            "mle_iterations" => fit.iterations,
            "mle_evaluations" => fit.evaluations,
            "mle_negative_loglikelihood" => fit.minimum,
            "fit_elapsed_seconds" => fit_elapsed_seconds,
            "hodlr_max_level" => HODLR_MAX_LEVEL,
            "hodlr_observation_divisor" => observation_divisor,
            "hodlr_rank_divisor" => HODLR_RANK_DIVISOR,
            "hodlr_local_rank" => hodlr_local_rank_effective,
            "hodlr_oversampling_requested" => HODLR_OVERSAMPLING,
            "hodlr_oversampling" => hodlr_oversampling_effective,
            "hodlr_random_seed" => HODLR_RANDOM_SEED,
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
            "kappa_realization" => "fixed Matern GP log-kappa",
            "kappa_source_path" => TRUE_LANDAU_OBSERVATIONS_PATH,
            "landau_observations_path" => TRUE_LANDAU_OBSERVATIONS_PATH,
            "landau_observation_source" => data.observation_source,
            "observed_values_source" => "fixed_noisy_observations",
            "hidden_truth_source" => "fixed_latent_observations",
            "landau_observations_max_nx" => data.max_nx,
            "landau_observations_max_ny" => data.max_ny,
            "landau_observations_max_realizations" => data.max_M,
            "landau_nested_spatial_indices" => data.spatial_indices,
            "landau_nested_x_indices" => data.x_indices,
            "landau_nested_y_indices" => data.y_indices,
            "landau_generation_seed" => data.generation_seed,
            "landau_observation_noise_seed" => data.observation_noise_seed,
            "kappa_realization_seed" => FixedMaternKappaLandauData.GP_SEED,
            "kappa_realization_max_nx" => FixedMaternKappaLandauData.MAX_NX,
            "kappa_realization_max_ny" => FixedMaternKappaLandauData.MAX_NY,
            "kappa_log_mean" => FixedMaternKappaLandauData.LOG_KAPPA_MEAN,
            "kappa_log_std" => FixedMaternKappaLandauData.LOG_KAPPA_STD,
            "kappa_matern_nu" => FixedMaternKappaLandauData.MATERN_NU,
            "kappa_matern_length_scale" =>
                FixedMaternKappaLandauData.MATERN_LENGTH_SCALE,
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
    for grid_size in RUN_GRID_SIZES
        for mask_percent in RUN_MASK_PERCENTS
            push!(
                results,
                run_masked_inference(
                    nx=Int(grid_size),
                    ny=Int(grid_size),
                    mask_percent=mask_percent,
                    M=RUN_REALIZATIONS,
                    backend_name=RUN_BACKEND,
                    output_dir=RUN_OUTPUT_DIR,
                ),
            )
        end
    end
    return results
end

main()
