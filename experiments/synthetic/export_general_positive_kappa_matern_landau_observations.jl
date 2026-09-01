using Random
using Printf
using MAT
using GaussianRandomFields

include("utils/landau_sample_heterogeneous_kappa.jl")

const A_TRUE = 2.0
const B_TRUE = 3.0

const NU_PHI = 1.5
const ELL_PHI = 0.20
const SIGMA_PHI = 0.15
const OBSERVATION_NOISE_STD = 0.01
const COVARIANCE_JITTER = 1.0e-8
const FACE_AVERAGE = :harmonic

const MAX_NX = 2^6
const MAX_NY = 2^6
const MAX_REALIZATIONS = 2^4

const XMIN = 0.0
const XMAX = 1.0
const YMIN = 0.0
const YMAX = 1.0

const KAPPA_GP_SEED = 1729
const LOG_KAPPA_MEAN = log(0.8)
const LOG_KAPPA_STD = 0.35
const KAPPA_MATERN_NU = 1.5
const KAPPA_MATERN_LENGTH_SCALE = 0.20
const KAPPA_CIRCULANT_MIN_PADDING = (MAX_NX, MAX_NY)

const GENERATION_SEED = 1234
const OBSERVATION_NOISE_SEED = 161803

const DATA_DIR = joinpath(@__DIR__, "fixed_kappa")
const LANDAU_DATA_FILENAME =
    "general_positive_kappa_matern_landau_observations.mat"
const LANDAU_DATA_PATH = joinpath(DATA_DIR, LANDAU_DATA_FILENAME)

const RBF_MEAN_CENTERS = [
    0.25 0.25
    0.75 0.30
    0.50 0.75
]
const RBF_MEAN_COEFFICIENTS = [0.9, 1.1, 1.0]
const RBF_MEAN_LENGTH_SCALE = 0.22

function _mat_scalar(data, key::AbstractString, default)
    haskey(data, key) || return default
    value = data[key]
    value isa AbstractArray && return first(value)
    return value
end

function _existing_data_matches_current_settings(path::AbstractString)
    required_keys = (
        "latent_observations",
        "noisy_observations",
        "covariance_phi",
        "mean_phi",
        "kappa_matrix",
        "log_kappa_matrix",
        "xgrid",
        "ygrid",
        "max_nx",
        "max_ny",
        "max_realizations",
        "kappa_source",
        "kappa_gp_seed",
        "generation_seed",
        "observation_noise_seed",
    )

    data = try
        matread(path)
    catch
        return false
    end
    all(haskey(data, key) for key in required_keys) || return false

    values = try
        (
            latent_observations=Matrix{Float64}(data["latent_observations"]),
            noisy_observations=Matrix{Float64}(data["noisy_observations"]),
            kappa_matrix=Matrix{Float64}(data["kappa_matrix"]),
            max_nx=Int(round(Float64(_mat_scalar(data, "max_nx", -1)))),
            max_ny=Int(round(Float64(_mat_scalar(data, "max_ny", -1)))),
            max_realizations=Int(round(Float64(
                _mat_scalar(data, "max_realizations", -1),
            ))),
            kappa_seed=Int(round(Float64(
                _mat_scalar(data, "kappa_gp_seed", -1),
            ))),
            generation_seed=Int(round(Float64(
                _mat_scalar(data, "generation_seed", -1),
            ))),
            noise_seed=Int(round(Float64(
                _mat_scalar(data, "observation_noise_seed", -1),
            ))),
            noise_std=Float64(
                _mat_scalar(data, "observation_noise_std", NaN),
            ),
        )
    catch
        return false
    end

    return (
        (values.max_nx, values.max_ny) == (MAX_NX, MAX_NY) &&
        values.max_realizations == MAX_REALIZATIONS &&
        values.kappa_seed == KAPPA_GP_SEED &&
        values.generation_seed == GENERATION_SEED &&
        values.noise_seed == OBSERVATION_NOISE_SEED &&
        isapprox(
            values.noise_std,
            OBSERVATION_NOISE_STD;
            rtol=1.0e-12,
            atol=1.0e-12,
        ) &&
        size(values.kappa_matrix) == (MAX_NX, MAX_NY) &&
        size(values.latent_observations) ==
            (MAX_REALIZATIONS, MAX_NX * MAX_NY) &&
        size(values.noisy_observations) == size(values.latent_observations)
    )
end

function periodic_grid(n::Integer, lower::Real, upper::Real)
    n >= 1 || throw(ArgumentError("grid size must be positive."))
    lower_float = Float64(lower)
    upper_float = Float64(upper)
    width = upper_float - lower_float
    width > 0.0 ||
        throw(ArgumentError("grid upper bound must exceed lower bound."))
    return range(lower_float; step=width / Int(n), length=Int(n))
end

function periodic_grid(nx::Integer, ny::Integer)
    return (
        periodic_grid(nx, XMIN, XMAX),
        periodic_grid(ny, YMIN, YMAX),
    )
end

function sample_log_kappa_field()
    xgrid, ygrid = periodic_grid(MAX_NX, MAX_NY)
    mean_grid = zeros(Float64, MAX_NX, MAX_NY)
    covariance = CovarianceFunction(
        2,
        Matern(KAPPA_MATERN_LENGTH_SCALE, KAPPA_MATERN_NU; σ=LOG_KAPPA_STD),
    )
    grf = GaussianRandomField(
        mean_grid,
        covariance,
        CirculantEmbedding(),
        xgrid,
        ygrid;
        minpadding=KAPPA_CIRCULANT_MIN_PADDING,
        measure=false,
    )
    realization = GaussianRandomFields.sample(MersenneTwister(KAPPA_GP_SEED), grf)
    return LOG_KAPPA_MEAN .+ Matrix{Float64}(realization), xgrid, ygrid
end

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

function export_landau_observations(; force::Bool=false)
    mkpath(DATA_DIR)
    if isfile(LANDAU_DATA_PATH) && !force
        if _existing_data_matches_current_settings(LANDAU_DATA_PATH)
            println(
                "Fixed integrated Matern kappa and Landau observation data " *
                "already exists: ",
                LANDAU_DATA_PATH,
            )
            println("Use --force to overwrite it deliberately.")
            return LANDAU_DATA_PATH
        end

        error(
            "Existing fixed Landau observation data at $(LANDAU_DATA_PATH) " *
            "does not match the current integrated settings. Run " *
            "`julia export_general_positive_kappa_matern_landau_observations.jl " *
            "--force` to regenerate the M=$(MAX_REALIZATIONS) data.",
        )
    end

    log_kappa_matrix, xgrid, ygrid = sample_log_kappa_field()
    kappa_matrix = exp.(log_kappa_matrix)
    initial_solution = fill(sqrt(A_TRUE / B_TRUE), MAX_NX, MAX_NY)

    println("Generating fixed maximum-grid Landau observations")
    println("  grid            = ", MAX_NX, " x ", MAX_NY)
    println("  realizations    = ", MAX_REALIZATIONS)
    println("  kappa source    = embedded fixed Matern GP realization")
    println("  kappa seed      = ", KAPPA_GP_SEED)
    println("  generation seed = ", GENERATION_SEED)

    data = generate_landau_observations(
        xgrid,
        ygrid,
        MAX_REALIZATIONS,
        NU_PHI,
        ELL_PHI,
        SIGMA_PHI,
        A_TRUE,
        B_TRUE,
        kappa_matrix;
        mean_fn=rbf_mean_phi,
        rng=MersenneTwister(GENERATION_SEED),
        covariance_jitter=COVARIANCE_JITTER,
        periodic_kernel=false,
        observation_noise_std=0.0,
        u0=initial_solution,
        warm_start=false,
        solver_face_average=FACE_AVERAGE,
        solver_verbose=false,
        progress=true,
        return_forcing=false,
    )

    latent_observations = Matrix{Float64}(data.observations)
    noise_rng = MersenneTwister(OBSERVATION_NOISE_SEED)
    noisy_observations =
        latent_observations .+
        OBSERVATION_NOISE_STD .*
        randn(noise_rng, size(latent_observations)...)

    matwrite(
        LANDAU_DATA_PATH,
        Dict(
            "latent_observations" => latent_observations,
            "noisy_observations" => noisy_observations,
            "covariance_phi" => data.covariance_phi,
            "mean_phi" => data.mean_phi,
            "converged" => Int.(data.converged),
            "iterations" => data.iterations,
            "relative_residuals" => data.relative_residuals,
            "kappa_matrix" => kappa_matrix,
            "log_kappa_matrix" => log_kappa_matrix,
            "kappa_vector" => vec(kappa_matrix),
            "xgrid" => collect(xgrid),
            "ygrid" => collect(ygrid),
            "max_nx" => MAX_NX,
            "max_ny" => MAX_NY,
            "max_realizations" => MAX_REALIZATIONS,
            "xmin" => XMIN,
            "xmax" => XMAX,
            "ymin" => YMIN,
            "ymax" => YMAX,
            "fixed_a" => A_TRUE,
            "fixed_b" => B_TRUE,
            "nu_phi" => NU_PHI,
            "ell_phi" => ELL_PHI,
            "sigma_phi" => SIGMA_PHI,
            "covariance_jitter" => COVARIANCE_JITTER,
            "observation_noise_std" => OBSERVATION_NOISE_STD,
            "face_average" => String(FACE_AVERAGE),
            "kappa_source" => "embedded fixed Matern GP log-kappa",
            "kappa_gp_seed" => KAPPA_GP_SEED,
            "log_kappa_mean" => LOG_KAPPA_MEAN,
            "log_kappa_std" => LOG_KAPPA_STD,
            "kappa_matern_nu" => KAPPA_MATERN_NU,
            "kappa_matern_length_scale" => KAPPA_MATERN_LENGTH_SCALE,
            "kappa_circulant_min_padding" =>
                collect(KAPPA_CIRCULANT_MIN_PADDING),
            "generation_seed" => GENERATION_SEED,
            "observation_noise_seed" => OBSERVATION_NOISE_SEED,
            "grid_convention" =>
                "periodic endpoint-excluded max grid; coarser grids are x-fast nested column subsets",
        );
        compress=true,
    )

    @printf(
        "Wrote fixed Landau data: %s\nlatent range = [%.6e, %.6e]\nnoisy range  = [%.6e, %.6e]\n",
        LANDAU_DATA_PATH,
        minimum(latent_observations),
        maximum(latent_observations),
        minimum(noisy_observations),
        maximum(noisy_observations),
    )
    return LANDAU_DATA_PATH
end

export_landau_observations(force=("--force" in ARGS))
