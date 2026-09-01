module FixedMaternKappaLandauData

using MAT

export LANDAU_DATA_PATH,
       DEFAULT_REALIZATIONS,
       MAX_NX,
       MAX_NY,
       GP_SEED,
       LOG_KAPPA_MEAN,
       LOG_KAPPA_STD,
       MATERN_NU,
       MATERN_LENGTH_SCALE,
       periodic_grid,
       kappa_grid,
       kappa_vector,
       load_landau_subset,
       nested_spatial_indices,
       nested_axis_indices

const DATA_DIR = joinpath(@__DIR__, "fixed_kappa")
const LANDAU_DATA_FILENAME =
    "general_positive_kappa_matern_landau_observations.mat"
const LANDAU_DATA_PATH = joinpath(DATA_DIR, LANDAU_DATA_FILENAME)
const MAX_NX = 2^6
const MAX_NY = 2^6
const DEFAULT_REALIZATIONS = 2^4
const XMIN = 0.0
const XMAX = 1.0
const YMIN = 0.0
const YMAX = 1.0
const GP_SEED = 1729
const LOG_KAPPA_MEAN = log(0.8)
const LOG_KAPPA_STD = 0.35
const MATERN_NU = 1.5
const MATERN_LENGTH_SCALE = 0.20

const _DATA_CACHE = Ref{Union{Nothing, Dict{String, Any}}}(nothing)

function _mat_scalar(data, key::AbstractString, default)
    haskey(data, key) || return default
    value = data[key]
    value isa AbstractArray && return first(value)
    return value
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

function _load_data()
    cached = _DATA_CACHE[]
    cached !== nothing && return cached

    isfile(LANDAU_DATA_PATH) || error(
        "Fixed Landau observation data not found at $(LANDAU_DATA_PATH). " *
        "Run export_general_positive_kappa_matern_landau_observations.jl " *
        "once before running fitting, inference, or verification scripts.",
    )

    data = matread(LANDAU_DATA_PATH)
    for key in (
        "latent_observations",
        "noisy_observations",
        "covariance_phi",
        "mean_phi",
        "kappa_matrix",
        "xgrid",
        "ygrid",
    )
        haskey(data, key) || error(
            "Fixed Landau observation MAT file is missing variable $key.",
        )
    end

    _DATA_CACHE[] = data
    return data
end

function nested_axis_indices(
    n::Integer,
    nmax::Integer,
    axis_name::AbstractString,
)
    n_int = Int(n)
    nmax_int = Int(nmax)
    n_int >= 1 || throw(ArgumentError("$axis_name must be positive."))
    nmax_int % n_int == 0 || throw(ArgumentError(
        "$axis_name=$n_int is not nested in the maximum grid size " *
        "$nmax_int. Use a divisor of $nmax_int.",
    ))
    step = div(nmax_int, n_int)
    return collect(1:step:nmax_int)
end

function nested_spatial_indices(
    nx::Integer,
    ny::Integer;
    max_nx::Integer,
    max_ny::Integer,
)
    ix = nested_axis_indices(nx, max_nx, "nx")
    iy = nested_axis_indices(ny, max_ny, "ny")
    indices = [i + (j - 1) * Int(max_nx) for j in iy for i in ix]
    return indices, ix, iy
end

function load_landau_subset(
    nx::Integer,
    ny::Integer,
    M::Integer=DEFAULT_REALIZATIONS;
    use_noisy_observations::Bool=true,
)
    data = _load_data()

    latent_all = Matrix{Float64}(data["latent_observations"])
    noisy_all = Matrix{Float64}(data["noisy_observations"])
    covariance_all = Matrix{Float64}(data["covariance_phi"])
    mean_all = vec(Float64.(data["mean_phi"]))
    kappa_all = Matrix{Float64}(data["kappa_matrix"])
    xgrid_all = vec(Float64.(data["xgrid"]))
    ygrid_all = vec(Float64.(data["ygrid"]))

    max_nx = Int(round(Float64(_mat_scalar(data, "max_nx", size(kappa_all, 1)))))
    max_ny = Int(round(Float64(_mat_scalar(data, "max_ny", size(kappa_all, 2)))))
    max_M = size(latent_all, 1)

    size(kappa_all) == (max_nx, max_ny) || throw(DimensionMismatch(
        "Saved kappa_matrix size $(size(kappa_all)) does not match " *
        "saved maximum grid size ($max_nx, $max_ny).",
    ))
    size(noisy_all) == size(latent_all) || throw(DimensionMismatch(
        "noisy_observations and latent_observations must have the same size.",
    ))
    size(latent_all, 2) == max_nx * max_ny || throw(DimensionMismatch(
        "Observation columns must equal max_nx * max_ny.",
    ))
    size(covariance_all) == (max_nx * max_ny, max_nx * max_ny) ||
        throw(DimensionMismatch(
            "covariance_phi must have size Nmax x Nmax.",
        ))
    length(mean_all) == max_nx * max_ny ||
        throw(DimensionMismatch("mean_phi must have length Nmax."))
    length(xgrid_all) == max_nx ||
        throw(DimensionMismatch("xgrid must have length max_nx."))
    length(ygrid_all) == max_ny ||
        throw(DimensionMismatch("ygrid must have length max_ny."))

    M_int = Int(M)
    1 <= M_int <= max_M || throw(ArgumentError(
        "Requested M=$M_int, but fixed data contains $max_M realizations.",
    ))

    spatial_indices, ix, iy = nested_spatial_indices(
        nx,
        ny;
        max_nx=max_nx,
        max_ny=max_ny,
    )

    source_observations =
        use_noisy_observations ? noisy_all : latent_all
    observations = Matrix{Float64}(
        @view(source_observations[1:M_int, spatial_indices])
    )
    latent_observations = Matrix{Float64}(
        @view(latent_all[1:M_int, spatial_indices])
    )
    noisy_observations = Matrix{Float64}(
        @view(noisy_all[1:M_int, spatial_indices])
    )
    covariance_phi = Matrix{Float64}(
        @view(covariance_all[spatial_indices, spatial_indices])
    )
    covariance_phi .=
        0.5 .* (covariance_phi .+ transpose(covariance_phi))

    kappa_field = copy(@view(kappa_all[ix, iy]))

    return (
        observations=observations,
        latent_observations=latent_observations,
        noisy_observations=noisy_observations,
        covariance_phi=covariance_phi,
        mean_phi=mean_all[spatial_indices],
        kappa_field=kappa_field,
        kappa_vector=vec(kappa_field),
        xgrid=xgrid_all[ix],
        ygrid=ygrid_all[iy],
        spatial_indices=spatial_indices,
        x_indices=ix,
        y_indices=iy,
        max_nx=max_nx,
        max_ny=max_ny,
        max_M=max_M,
        source_path=LANDAU_DATA_PATH,
        observation_source=use_noisy_observations ? "noisy" : "latent",
        observation_noise_std=Float64(
            _mat_scalar(data, "observation_noise_std", 0.0),
        ),
        generation_seed=Int(round(Float64(
            _mat_scalar(data, "generation_seed", 0),
        ))),
        observation_noise_seed=Int(round(Float64(
            _mat_scalar(data, "observation_noise_seed", 0),
        ))),
    )
end

function kappa_grid(nx::Integer=MAX_NX, ny::Integer=MAX_NY)
    data = _load_data()
    kappa_all = Matrix{Float64}(data["kappa_matrix"])
    max_nx = Int(round(Float64(_mat_scalar(data, "max_nx", size(kappa_all, 1)))))
    max_ny = Int(round(Float64(_mat_scalar(data, "max_ny", size(kappa_all, 2)))))
    _, ix, iy = nested_spatial_indices(
        nx,
        ny;
        max_nx=max_nx,
        max_ny=max_ny,
    )
    return copy(@view(kappa_all[ix, iy]))
end

kappa_vector(nx::Integer=MAX_NX, ny::Integer=MAX_NY) = vec(kappa_grid(nx, ny))

end
