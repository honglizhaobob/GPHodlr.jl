"""Timing driver for cached HODLR solves and inverse-times-matrix products.

The dense `inv(M)*A` comparison is retained for LU and Cholesky. The HODLR
matrix comparison uses `inv(M)*hodlr(A)`. Trace and final HODLR matvec timings
are intentionally excluded. Statistics are averaged over independent random
matrix realizations; each realization is timed once.
"""

using Dates
using LinearAlgebra
using Printf
using Random
using Statistics

include("../../src/hodlr_optimized.jl")

const FIRST_POWER = parse(Int, get(ENV, "HODLR_TIMING_FIRST_POWER", "6"))
const LAST_POWER = parse(Int, get(ENV, "HODLR_TIMING_LAST_POWER", "12"))
const REALIZATIONS = parse(Int, get(ENV, "HODLR_TIMING_REALIZATIONS", "5"))
const LEAF_SIZE = parse(Int, get(ENV, "HODLR_TIMING_LEAF_SIZE", "16"))
const HODLR_EPS = parse(Float64, get(ENV, "HODLR_TIMING_HODLR_EPS", "1e-8"))
const SYNTHETIC_RANK = parse(Int, get(ENV, "HODLR_TIMING_SYNTHETIC_RANK", "16"))
const RNG_SEED = parse(Int, get(ENV, "HODLR_TIMING_SEED", "20260828"))
const OUTPUT_CSV = get(
    ENV,
    "HODLR_TIMING_OUTPUT_CSV",
    joinpath(@__DIR__, "decomposition_inverse_matmul_timing_optimized_results.csv"),
)

function random_spd(n::Int, rng::AbstractRNG; rank::Int=SYNTHETIC_RANK)::Matrix{Float64}
    r = max(1, min(rank, n))
    X = randn(rng, n, r)
    M = (X * X') / r
    d = 1.0 .+ rand(rng, n)
    @inbounds for i in 1:n
        M[i, i] += d[i]
    end
    return Matrix(Symmetric(M))
end

hodlr_level(n::Int) = Int(log2(n)) - Int(log2(LEAF_SIZE))

hodlr_solve_vector(F, v::Vector{Float64}) =
    vec(hodlr_solve_cached(F, reshape(v, length(v), 1)))

elapsed_seconds(t0::UInt64) = Float64(time_ns() - t0) * 1.0e-9

function time_once(f::Function)::Float64
    GC.gc()
    t0 = time_ns()
    result = f()
    result = nothing
    return elapsed_seconds(t0)
end

function csv_escape(x)
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function benchmark_size(n::Int, rng::AbstractRNG)
    level = hodlr_level(n)
    timestamp = Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")

    # One independent M, A, and v realization is generated per iteration.
    # Each operation is timed exactly once for that realization.
    operation_names = [
        ("inv(M)*v", "lu_solve"),
        ("inv(M)*v", "cholesky_solve"),
        ("inv(M)*v", "hodlr_solve_cached"),
        ("inv(M)*A", "lu_solve_dense_rhs"),
        ("inv(M)*A", "cholesky_solve_dense_rhs"),
        ("inv(M)*hodlr(A)", "hodlr_invmult"),
    ]
    timing = Dict{Tuple{String,String},Vector{Float64}}(
        key => Float64[] for key in operation_names
    )

    for realization in 1:REALIZATIONS
        @printf("Preparing n = %d, realization = %d/%d\n", n, realization, REALIZATIONS)
        M = random_spd(n, rng)
        A = random_spd(n, rng)
        v = randn(rng, n)

        # Matrix representations are built outside the timed operation.
        M_lu = lu(M)
        M_chol = cholesky(Symmetric(M))
        M_hodlr = hodlr(M, level, HODLR_EPS)
        A_hodlr = hodlr(A, level, HODLR_EPS)
        M_cached_fact = hodlr_factorize_cached(M_hodlr)
        M_hodlr_fact = M_cached_fact.base

        operations = [
            ("inv(M)*v", "lu_solve", () -> M_lu \ v),
            ("inv(M)*v", "cholesky_solve", () -> M_chol \ v),
            ("inv(M)*v", "hodlr_solve_cached", () -> hodlr_solve_vector(M_cached_fact, v)),
            ("inv(M)*A", "lu_solve_dense_rhs", () -> M_lu \ A),
            ("inv(M)*A", "cholesky_solve_dense_rhs", () -> M_chol \ A),
            ("inv(M)*hodlr(A)", "hodlr_invmult", () -> hodlr_invmult(M_hodlr_fact, A_hodlr)),
        ]

        # Exclude compilation from the statistics once, without repeating a
        # realization in the reported averages.
        if realization == 1
            for (_, _, f) in operations
                f()
            end
        end

        for (operation, method, f) in operations
            push!(timing[(operation, method)], time_once(f))
        end
    end

    rows = NamedTuple[]
    for (operation, method) in operation_names
        times = timing[(operation, method)]
        push!(rows, (
            timestamp=timestamp,
            n=n,
            operation=operation,
            method=method,
            status="ok",
            min_seconds=minimum(times),
            median_seconds=median(times),
            mean_seconds=mean(times),
            samples=length(times),
            reps=1,
            hodlr_level=level,
            hodlr_eps=HODLR_EPS,
            leaf_size=LEAF_SIZE,
            synthetic_rank=SYNTHETIC_RANK,
            error="",
        ))
    end
    return rows
end

function write_results(path::String, rows::Vector{NamedTuple})
    headers = (
        :timestamp, :n, :operation, :method, :status,
        :min_seconds, :median_seconds, :mean_seconds, :samples, :reps,
        :hodlr_level, :hodlr_eps, :leaf_size, :synthetic_rank, :error,
    )
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(headers), ","))
        for row in rows
            println(io, join((csv_escape(getfield(row, h)) for h in headers), ","))
        end
    end
end

rng = MersenneTwister(RNG_SEED)
rows = NamedTuple[]
for power in FIRST_POWER:LAST_POWER
    append!(rows, benchmark_size(2^power, rng))
end
write_results(OUTPUT_CSV, rows)
@printf("Wrote %d rows to %s\n", length(rows), OUTPUT_CSV)
