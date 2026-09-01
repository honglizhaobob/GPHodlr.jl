using LinearAlgebra
using Random
using SpecialFunctions
using Statistics

# Load the conservative heterogeneous-diffusion Landau solver.
include("Landau_solver_heterogeneous_kappa.jl")


"""
    grid_points_2d(xgrid, ygrid)

Return the tensor-product grid points in Julia column-major ordering:

    (x[1], y[1]), ..., (x[nx], y[1]),
    (x[1], y[2]), ..., (x[nx], y[ny]).

This ordering is consistent with `vec(U)` for an `nx × ny` matrix `U`.
"""
function grid_points_2d(
    xgrid::AbstractVector{<:Real},
    ygrid::AbstractVector{<:Real},
)
    return [
        (Float64(x), Float64(y))
        for y in ygrid
        for x in xgrid
    ]
end


"""
    evaluate_gp_mean(mean_fn, xgrid, ygrid)

Evaluate the Gaussian-process mean on the tensor-product grid.

Supported definitions of `mean_fn`:

1. `mean_fn(xgrid, ygrid)` returns an `nx × ny` matrix or a vector
   of length `nx * ny`.

2. `mean_fn(x, y)` returns a scalar at a single grid point.

3. `mean_fn([x, y])` returns a scalar at a single grid point.
"""
function evaluate_gp_mean(
    mean_fn::Function,
    xgrid::AbstractVector{<:Real},
    ygrid::AbstractVector{<:Real},
)
    nx = length(xgrid)
    ny = length(ygrid)
    N = nx * ny

    # Try whole-grid evaluation.
    try
        values = mean_fn(xgrid, ygrid)

        if values isa AbstractMatrix && size(values) == (nx, ny)
            return Float64.(vec(values))
        elseif values isa AbstractVector && length(values) == N
            return Float64.(values)
        end
    catch error
        if !(error isa MethodError || error isa DimensionMismatch)
            rethrow(error)
        end
    end

    # Try scalar mean_fn(x, y).
    if applicable(mean_fn, first(xgrid), first(ygrid))
        return [
            Float64(mean_fn(x, y))
            for y in ygrid
            for x in xgrid
        ]
    end

    # Try scalar mean_fn([x, y]).
    point = [Float64(first(xgrid)), Float64(first(ygrid))]

    applicable(mean_fn, point) ||
        throw(ArgumentError(
            "mean_fn must accept (xgrid, ygrid), (x, y), or [x, y]."
        ))

    return [
        Float64(mean_fn([x, y]))
        for y in ygrid
        for x in xgrid
    ]
end


"""
    matern_covariance_2d(xgrid, ygrid, nu, ell, sigma;
                         jitter=1e-10,
                         periodic_distance=false)

Construct the dense Matérn covariance matrix

    K(r) = sigma² * 2^(1-nu) / Gamma(nu)
           * z^nu * K_nu(z),

where

    z = sqrt(2nu) * r / ell.

Arguments
---------
- `xgrid`, `ygrid`: coordinate vectors.
- `nu`: Matérn smoothness parameter.
- `ell`: correlation length.
- `sigma`: marginal standard deviation of the Gaussian process.
- `jitter`: nonnegative diagonal stabilization.
- `periodic_distance`: use wrapped distances consistent with a periodic
  rectangular domain.

Returns
-------
An `N × N` covariance matrix, where `N = length(xgrid)*length(ygrid)`.
"""
function matern_covariance_2d(
    xgrid::AbstractVector{<:Real},
    ygrid::AbstractVector{<:Real},
    nu::Real,
    ell::Real,
    sigma::Real;
    jitter::Real = 1e-10,
    periodic_distance::Bool = false,
)

    nx = length(xgrid)
    ny = length(ygrid)

    points = grid_points_2d(xgrid, ygrid)
    N = length(points)

    covariance = Matrix{Float64}(undef, N, N)

    nu_float = Float64(nu)
    ell_float = Float64(ell)
    sigma_float = Float64(sigma)

    prefactor =
        sigma_float^2 *
        2.0^(1.0 - nu_float) /
        gamma(nu_float)

    # For a grid that does not repeat its periodic endpoint, infer
    # the period as last point - first point + one mesh spacing.
    if periodic_distance
        hx = Float64(xgrid[2] - xgrid[1])
        hy = Float64(ygrid[2] - ygrid[1])

        Lx = Float64(xgrid[end] - xgrid[1] + hx)
        Ly = Float64(ygrid[end] - ygrid[1] + hy)
    else
        Lx = 0.0
        Ly = 0.0
    end

    for j in 1:N
        xj, yj = points[j]
        covariance[j, j] = sigma_float^2 + Float64(jitter)

        for i in (j + 1):N
            xi, yi = points[i]

            dx = abs(xi - xj)
            dy = abs(yi - yj)

            if periodic_distance
                dx = min(dx, Lx - dx)
                dy = min(dy, Ly - dy)
            end

            distance = hypot(dx, dy)

            # The r = 0 limit is sigma^2 and is handled on the diagonal.
            z = sqrt(2.0 * nu_float) * distance / ell_float

            covariance_value =
                prefactor *
                z^nu_float *
                besselk(nu_float, z)

            covariance[i, j] = covariance_value
            covariance[j, i] = covariance_value
        end
    end

    return covariance
end


"""
    generate_landau_observations(
        xgrid,
        ygrid,
        M,
        nu,
        ell,
        sigma,
        a,
        b,
        kappa;
        kwargs...
    )

Generate `M` realizations of a Matérn Gaussian forcing field `phi` and
solve the nonlinear periodic Landau equation

    div(kappa(x,y)*grad(u)) + a*u - b*u^3 + phi = 0.

The returned observation matrix has size `M × N`, where

    N = length(xgrid) * length(ygrid).

Each row stores one solution in the same ordering as

    vec(u_matrix),

so the x-index varies fastest.

Keyword arguments
-----------------
- `mean_fn`: GP mean function. Default: zero mean.
- `rng`: random-number generator.
- `covariance_jitter`: diagonal stabilization used in Cholesky sampling.
- `periodic_kernel`: if true, use wrapped distances in the Matérn kernel.
- `observation_noise_std`: additive observation-noise standard deviation.
- `u0`: common Newton initial guess, as an `nx × ny` matrix or length-N vector.
- `warm_start`: use the previous converged solution as the next initial guess.
- `solver_tol`: nonlinear solver tolerance.
- `solver_maxiter`: maximum Newton iterations.
- `solver_max_backtracking`: maximum line-search reductions.
- `solver_face_average`: `:harmonic` (default) or `:arithmetic`.
- `solver_verbose`: print Newton iteration information.
- `progress`: print ensemble progress.
- `require_convergence`: throw an error if any realization fails to converge.
- `return_forcing`: include the sampled forcing matrix in the output.

Returns
-------
A named tuple containing:

- `observations`: `M × N` matrix of Landau solutions.
- `covariance_phi`: `N × N` Matérn covariance matrix.
- `mean_phi`: GP mean vector of length `N`.
- `converged`: length-M convergence flags.
- `iterations`: Newton iterations for each realization.
- `relative_residuals`: final residuals.
- `kappa_field`: evaluated positive `nx × ny` diffusion field.
- `diffusion_operator`: precomputed heterogeneous diffusion operator.
- `xgrid`, `ygrid`: copies of the grids.
- `forcing`: `M × N` forcing matrix when `return_forcing=true`.
"""
function generate_landau_observations(
    xgrid::AbstractVector{<:Real},
    ygrid::AbstractVector{<:Real},
    M::Integer,
    nu::Real,
    ell::Real,
    sigma::Real,
    a::Real,
    b::Real,
    kappa;
    mean_fn::Function = (x, y) -> 0.0,
    rng::AbstractRNG = Random.default_rng(),
    covariance_jitter::Real = 1e-8,
    periodic_kernel::Bool = false,
    observation_noise_std::Real = 0.0,
    u0::Union{Nothing, AbstractVector, AbstractMatrix} = nothing,
    warm_start::Bool = false,
    solver_tol::Real = 1e-10,
    solver_maxiter::Integer = 50,
    solver_max_backtracking::Integer = 15,
    solver_face_average::Symbol = :harmonic,
    solver_verbose::Bool = false,
    progress::Bool = true,
    require_convergence::Bool = true,
    return_forcing::Bool = false,
)

    nx = length(xgrid)
    ny = length(ygrid)
    N = nx * ny

    # Convert grids once to the floating-point representation expected
    # by the PDE solver.
    x = Float64.(collect(xgrid))
    y = Float64.(collect(ygrid))

    # Build the heterogeneous diffusion operator once and reuse it for all M
    # nonlinear solves.
    diffusion = periodic_diffusion_operator(
        x,
        y,
        kappa;
        face_average=solver_face_average,
    )

    mean_phi = evaluate_gp_mean(mean_fn, x, y)

    covariance_phi = matern_covariance_2d(
        x,
        y,
        nu,
        ell,
        sigma;
        jitter=covariance_jitter,
        periodic_distance=periodic_kernel,
    )

    # Symmetric wrapper prevents small floating-point asymmetries from
    # affecting the Cholesky factorization.
    cholesky_factor = cholesky(
        Symmetric(covariance_phi);
        check=true,
    ).L

    # phi_samples[m, :] and observations[m, :] correspond to ensemble m.
    phi_samples = Matrix{Float64}(undef, M, N)
    observations = Matrix{Float64}(undef, M, N)

    converged = falses(M)
    iterations = zeros(Int, M)
    relative_residuals = fill(Inf, M)

    current_initial_guess = u0

    for ensemble_index in 1:M
        if progress &&
           (
               ensemble_index == 1 ||
               ensemble_index == M ||
               ensemble_index % 10 == 0
           )
            println(
                "Solving Landau realization ",
                ensemble_index,
                " / ",
                M,
            )
        end

        standard_normal_sample = randn(rng, N)

        phi_vector =
            mean_phi +
            cholesky_factor * standard_normal_sample

        phi_samples[ensemble_index, :] .= phi_vector

        result = solve_landau_periodic(
            kappa,
            a,
            b,
            x,
            y,
            phi_vector;
            u0=current_initial_guess,
            face_average=solver_face_average,
            precomputed_diffusion=diffusion,
            tol=solver_tol,
            maxiter=Int(solver_maxiter),
            max_backtracking=Int(solver_max_backtracking),
            verbose=solver_verbose,
        )

        observations[ensemble_index, :] .= result.u_vector

        converged[ensemble_index] = result.converged
        iterations[ensemble_index] = result.iterations
        relative_residuals[ensemble_index] =
            result.relative_residual

        if require_convergence && !result.converged
            error(
                "Landau solver failed for realization " *
                "$ensemble_index of $M. Final RMS residual = " *
                "$(result.relative_residual)."
            )
        end

        if warm_start && result.converged
            current_initial_guess = copy(result.u_vector)
        else
            current_initial_guess = u0
        end
    end

    if observation_noise_std > 0
        observations .+=
            Float64(observation_noise_std) .* randn(rng, M, N)
    end

    common_output = (
        observations=observations,
        covariance_phi=covariance_phi,
        mean_phi=mean_phi,
        converged=converged,
        iterations=iterations,
        relative_residuals=relative_residuals,
        kappa_field=diffusion.kappa_matrix,
        kappa_vector=vec(diffusion.kappa_matrix),
        diffusion_operator=diffusion.operator,
        face_average=solver_face_average,
        xgrid=x,
        ygrid=y,
    )

    if return_forcing
        return merge(
            common_output,
            (forcing=phi_samples,),
        )
    end

    return common_output
end
