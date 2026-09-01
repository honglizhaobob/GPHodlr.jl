using LinearAlgebra
using SparseArrays

# This explicitly named revision supports
# `solve_landau_periodic(...; precomputed_diffusion=diffusion)` so ensemble
# sampling reuses one conservative heterogeneous-diffusion operator.

"""
    periodic_second_derivative(n, h)

Construct the one-dimensional central finite-difference matrix

    d²u/dx² ≈ (u[i+1] - 2u[i] + u[i-1]) / h²

with periodic boundary conditions.

The grid must not repeat the periodic endpoint. For example, use
x = range(0.0, Lx; length=nx+1)[1:end-1].
"""
function periodic_second_derivative(n::Int, h::Real)

    main_diag = fill(-2.0, n)
    off_diag  = ones(n - 1)

    D = spdiagm(
        -1 => off_diag,
         0 => main_diag,
         1 => off_diag,
    )

    # Periodic wrap-around entries:
    # u[0] = u[n] and u[n+1] = u[1]
    D[1, n] = 1.0
    D[n, 1] = 1.0

    return D / h^2
end


"""
    periodic_laplacian(x, y)

Construct the sparse two-dimensional periodic Laplacian

    L = I_y ⊗ Dxx + Dyy ⊗ I_x.

The solution is stored as an `nx × ny` matrix U, and `vec(U)` is used
internally. Therefore, the x-index varies fastest in the vector ordering.

The coordinate vectors must describe uniform periodic grids and must not
contain repeated endpoints.
"""
function periodic_laplacian(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    nx = length(x)
    ny = length(y)

    hx = Float64(x[2] - x[1])
    hy = Float64(y[2] - y[1])

    Dxx = periodic_second_derivative(nx, hx)
    Dyy = periodic_second_derivative(ny, hy)

    Ix = sparse(I, nx, nx)
    Iy = sparse(I, ny, ny)

    return kron(Iy, Dxx) + kron(Dyy, Ix)
end


"""
    evaluate_kappa_field(kappa, x, y)

Convert kappa into an nx-by-ny Float64 matrix. The input may be a positive
scalar, a callable kappa(x,y), an nx-by-ny matrix, or a vector of length
nx*ny in column-major vec(U) ordering.
"""
function evaluate_kappa_field(
    kappa,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
)
    nx = length(x)
    ny = length(y)
    N = nx * ny

    kappa_matrix = if kappa isa Real
        fill(Float64(kappa), nx, ny)
    elseif kappa isa AbstractMatrix
        size(kappa) == (nx, ny) || throw(DimensionMismatch(
            "kappa matrix must have size ($nx, $ny), " *
            "but has size $(size(kappa)).",
        ))
        Float64.(kappa)
    elseif kappa isa AbstractVector
        length(kappa) == N || throw(DimensionMismatch(
            "kappa vector must have length $N, " *
            "but has length $(length(kappa)).",
        ))
        reshape(Float64.(collect(kappa)), nx, ny)
    elseif applicable(kappa, x[1], y[1])
        values = Matrix{Float64}(undef, nx, ny)
        @inbounds for j in 1:ny, i in 1:nx
            values[i, j] = Float64(kappa(x[i], y[j]))
        end
        values
    end

    all(isfinite, kappa_matrix) ||
        throw(ArgumentError("All kappa values must be finite."))

    return kappa_matrix
end


@inline function face_kappa(
    left::Float64,
    right::Float64,
    face_average::Symbol,
)
    if face_average == :harmonic
        return 2.0 * left * right / (left + right)
    elseif face_average == :arithmetic
        return 0.5 * (left + right)
    end
    throw(ArgumentError(
        "face_average must be :harmonic or :arithmetic.",
    ))
end


"""
    periodic_diffusion_operator(x, y, kappa; face_average=:harmonic)

Construct the conservative periodic finite-difference operator for

    div(kappa(x,y) * grad(u)).

The face fluxes use harmonic averaging by default, which is robust for
high-contrast positive fields. Arithmetic averaging is also available.

Returns a named tuple with fields operator, kappa_matrix, and face_average.
"""
function periodic_diffusion_operator(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    kappa;
    face_average::Symbol=:harmonic,
)
    nx = length(x)
    ny = length(y)
    N = nx * ny

    # Reuse the grid validation performed by periodic_laplacian.
    hx = Float64(x[2] - x[1])
    hy = Float64(y[2] - y[1])

    Kappa = evaluate_kappa_field(kappa, x, y)

    # Five nonzeros per row: center, west, east, south, and north.
    rows = Vector{Int}(undef, 5 * N)
    cols = Vector{Int}(undef, 5 * N)
    vals = Vector{Float64}(undef, 5 * N)
    position = 1

    @inbounds for j in 1:ny
        jm = j == 1  ? ny : j - 1
        jp = j == ny ? 1  : j + 1

        for i in 1:nx
            im = i == 1  ? nx : i - 1
            ip = i == nx ? 1  : i + 1

            center = i + (j - 1) * nx
            west   = im + (j - 1) * nx
            east   = ip + (j - 1) * nx
            south  = i + (jm - 1) * nx
            north  = i + (jp - 1) * nx

            kw = face_kappa(Kappa[i, j], Kappa[im, j], face_average)
            ke = face_kappa(Kappa[i, j], Kappa[ip, j], face_average)
            ks = face_kappa(Kappa[i, j], Kappa[i, jm], face_average)
            kn = face_kappa(Kappa[i, j], Kappa[i, jp], face_average)

            cw = kw / hx^2
            ce = ke / hx^2
            cs = ks / hy^2
            cn = kn / hy^2

            rows[position] = center
            cols[position] = center
            vals[position] = -(cw + ce + cs + cn)
            position += 1

            rows[position] = center
            cols[position] = west
            vals[position] = cw
            position += 1

            rows[position] = center
            cols[position] = east
            vals[position] = ce
            position += 1

            rows[position] = center
            cols[position] = south
            vals[position] = cs
            position += 1

            rows[position] = center
            cols[position] = north
            vals[position] = cn
            position += 1
        end
    end

    Dkappa = sparse(rows, cols, vals, N, N)
    return (
        operator=Dkappa,
        kappa_matrix=Kappa,
        face_average=face_average,
    )
end


"""
    landau_residual(u, diffusion_operator, phi, a, b)

Evaluate the residual

    F(u) = Dkappa*u + a*u - b*u.^3 + phi,

where Dkappa discretizes div(kappa*grad(u)).
"""
function landau_residual(
    u::AbstractVector,
    diffusion_operator::SparseMatrixCSC,
    phi::AbstractVector,
    a::Real,
    b::Real,
)
    return diffusion_operator * u .+
           a .* u .-
           b .* (u .^ 3) .+
           phi
end


"""
    landau_residual(u, L, phi, kappa, a, b)

Backward-compatible residual for constant scalar kappa.
"""
function landau_residual(
    u::AbstractVector,
    L::SparseMatrixCSC,
    phi::AbstractVector,
    kappa::Real,
    a::Real,
    b::Real,
)
    return kappa .* (L * u) .+ a .* u .- b .* (u .^ 3) .+ phi
end


"""
    solve_landau_periodic(kappa, a, b, x, y, phi; kwargs...)

Solve

    div(kappa(x,y)*grad(u)) + a*u - b*u^3 + phi = 0

on a rectangular periodic grid using a conservative finite-difference
operator and a damped Newton method.

Inputs
------
- `kappa`: positive scalar, callable `kappa(x,y)`, `nx × ny` matrix,
  or length-`nx*ny` vector in column-major ordering
- `a`, `b`: Landau-equation parameters
- `x`, `y`: uniform periodic mesh coordinate vectors
- `phi`: forcing values, either:
    - an `nx × ny` matrix, or
    - a vector of length `nx*ny`

Keyword arguments
-----------------
- `u0`: initial guess, matrix or vector; default is zero
- `face_average`: `:harmonic` (default) or `:arithmetic`
- `precomputed_diffusion`: optional output from
  `periodic_diffusion_operator`; useful for many forcing realizations
- `tol`: RMS nonlinear residual tolerance
- `maxiter`: maximum Newton iterations
- `max_backtracking`: maximum line-search reductions
- `verbose`: print convergence information

Returns
-------
A named tuple containing:
- `u`: numerical solution as an `nx × ny` matrix
- `u_vector`: vectorized solution
- `kappa_field`: evaluated positive diffusion field
- `diffusion_operator`: sparse operator for `div(kappa*grad(u))`
- `laplacian`: backward-compatible alias of `diffusion_operator`
- `converged`: convergence flag
- `iterations`: number of Newton iterations
- `relative_residual`: final RMS residual (name retained for compatibility)
"""
function solve_landau_periodic(
    kappa,
    a::Real,
    b::Real,
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    phi::Union{AbstractVector, AbstractMatrix};
    u0::Union{Nothing, AbstractVector, AbstractMatrix}=nothing,
    face_average::Symbol=:harmonic,
    precomputed_diffusion=nothing,
    tol::Real=1e-10,
    maxiter::Int=50,
    max_backtracking::Int=15,
    verbose::Bool=true,
)

    nx = length(x)
    ny = length(y)
    N = nx * ny

    # Convert forcing to the vector ordering associated with vec(U).
    phi_vector = if phi isa AbstractMatrix
        size(phi) == (nx, ny) ||
            throw(DimensionMismatch(
                "phi matrix must have size ($nx, $ny), but has size $(size(phi))."
            ))
        Float64.(vec(phi))
    else
        length(phi) == N ||
            throw(DimensionMismatch(
                "phi vector must have length $N, but has length $(length(phi))."
            ))
        Float64.(phi)
    end

    # Initial solution guess.
    u = if isnothing(u0)
        zeros(Float64, N)
    elseif u0 isa AbstractMatrix
        size(u0) == (nx, ny) ||
            throw(DimensionMismatch(
                "u0 matrix must have size ($nx, $ny), but has size $(size(u0))."
            ))
        Float64.(vec(u0))
    else
        length(u0) == N ||
            throw(DimensionMismatch(
                "u0 vector must have length $N, but has length $(length(u0))."
            ))
        Float64.(u0)
    end

    diffusion = if isnothing(precomputed_diffusion)
        periodic_diffusion_operator(
            x,
            y,
            kappa;
            face_average=face_average,
        )
    else
        precomputed_diffusion
    end

    Dkappa = diffusion.operator
    kappa_matrix = diffusion.kappa_matrix
    identity_matrix = sparse(I, N, N)

    converged = false
    relative_residual = Inf
    completed_iterations = 0

    for iteration in 0:maxiter
        F = landau_residual(u, Dkappa, phi_vector, a, b)
        relative_residual = norm(F) / sqrt(N)

        verbose && println(
            "Iteration $(lpad(iteration, 3)): RMS residual = ",
            relative_residual,
        )

        if relative_residual <= tol
            converged = true
            completed_iterations = iteration
            break
        end

        if iteration == maxiter
            completed_iterations = iteration
            break
        end

        # Jacobian:
        #
        # J(u) = Dkappa + a*I - 3*b*diag(u.^2)
        J = Dkappa +
            a .* identity_matrix -
            spdiagm(0 => 3.0 .* b .* (u .^ 2))

        # Newton direction: J*delta = -F
        delta = try
            -(J \ F)
        catch error
            throw(ErrorException(
                "The Newton Jacobian could not be solved at iteration " *
                "$iteration. The parameters or initial guess may place the " *
                "problem near a singular solution branch.\nOriginal error: $error"
            ))
        end

        all(isfinite, delta) ||
            throw(ErrorException(
                "The Newton update contains NaN or Inf values."
            ))

        # Backtracking line search to reduce the nonlinear residual.
        current_norm = norm(F)
        step = 1.0
        accepted = false

        for _ in 1:max_backtracking
            trial_u = u .+ step .* delta
            trial_F = landau_residual(
                trial_u,
                Dkappa,
                phi_vector,
                a,
                b,
            )

            if norm(trial_F) < current_norm
                u = trial_u
                accepted = true
                break
            end

            step *= 0.5
        end

        if !accepted
            @warn(
                "The line search did not reduce the residual at iteration " *
                "$iteration. Applying the smallest trial step."
            )
            u .+= step .* delta
        end

        completed_iterations = iteration + 1
    end

    if !converged
        @warn(
            "Newton's method did not reach the requested tolerance. " *
            "Final RMS residual = $relative_residual."
        )
    end

    return (
        u=reshape(u, nx, ny),
        u_vector=u,
        kappa_field=kappa_matrix,
        kappa_vector=vec(kappa_matrix),
        diffusion_operator=Dkappa,
        # Retained for compatibility. For heterogeneous kappa this is the
        # variable-coefficient diffusion operator, not the ordinary Laplacian.
        laplacian=Dkappa,
        face_average=face_average,
        converged=converged,
        iterations=completed_iterations,
        relative_residual=relative_residual,
    )
end
