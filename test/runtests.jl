using GPHodlr
using Test
# fix random seed
Random.seed!(10)

# include helper functions
include("basic_hodlr_accuracy_helpers.jl");
######################################################################
# Utility function test suites
######################################################################


######################################################################
# Gaussian process tests
######################################################################
@testset "basic hodlr accuracy" begin
#----------
# Test 1: Peeling 2-level HODLR matrix (full access, i.e. exact matrix 
# is stored in memory).
#----------
n = 1024;
exact_level = 2;
idx = GPHodlr.dyadic_idx(n, exact_level);
# fill the diagonals
A = zeros(n, n);
A[idx[1], idx[1]] = randn(Int(n / 4), Int(n / 4));
A[idx[2], idx[2]] = randn(Int(n / 4), Int(n / 4));
A[idx[3], idx[3]] = randn(Int(n / 4), Int(n / 4));
A[idx[4], idx[4]] = randn(Int(n / 4), Int(n / 4));
# fill the off-diagonals
A[idx[1], idx[2]] = randn(Int(n / 4), Int(n / 4));
A[idx[2], idx[1]] = randn(Int(n / 4), Int(n / 4));
A[idx[3], idx[4]] = randn(Int(n / 4), Int(n / 4));
A[idx[4], idx[3]] = randn(Int(n / 4), Int(n / 4));
# make symmetric
A = transpose(A) * A;
# construct HODLR
peel_levels = 2;
local_rank = Int(n / 2^2);
hodlr_mat = GPHodlr.hodlr(A, peel_levels, local_rank);
# reconstruct
A_hodlr = GPHodlr.hodlr_to_full(hodlr_mat);
@test norm(A - A_hodlr) <= 1e-10;

# ----------
# Test 2: computing matrix-vector product in HODLR format
# ----------
n = 1024;
exact_level = 2;
idx = GPHodlr.dyadic_idx(n, exact_level);
# fill the diagonals
A = zeros(n, n);
A[idx[1], idx[1]] = randn(Int(n / 4), Int(n / 4));
A[idx[2], idx[2]] = randn(Int(n / 4), Int(n / 4));
A[idx[3], idx[3]] = randn(Int(n / 4), Int(n / 4));
A[idx[4], idx[4]] = randn(Int(n / 4), Int(n / 4));
# fill the off-diagonals
A[idx[1], idx[2]] = randn(Int(n / 4), Int(n / 4));
A[idx[2], idx[1]] = randn(Int(n / 4), Int(n / 4));
A[idx[3], idx[4]] = randn(Int(n / 4), Int(n / 4));
A[idx[4], idx[3]] = randn(Int(n / 4), Int(n / 4));

# make s.p.d.
A = transpose(A) * A;

# vector
x = randn(n, 1);
# exact product
exact_prod = A*x;
# peel twice and put in HODLR form
# construct HODLR
peel_levels = 2;
local_rank = Int(n / 2^2);
hodlr_mat = GPHodlr.hodlr(A, peel_levels, local_rank);

# compute matvec in HODLR format
approx_prod = GPHodlr.hodlr_prod(hodlr_mat, x);
@test norm(GPHodlr.hodlr_to_full(hodlr_mat)*x - exact_prod) < 1e-6
@test norm(exact_prod - approx_prod) < 1e-6


#----------
# Test 3: recover exact matrix from matrix-vector multiply routine
#----------
n = 1024;
exact_level = 2;
idx = GPHodlr.dyadic_idx(n, exact_level);
# fill the diagonals (all symmetric)
A = create_low_rank_two_level_matrix(n);
# HODLR 
local_rank = Int(n/8);
Aₕ= GPHodlr.hodlr(v->dummy_matvec_query(A, v), n, 2, local_rank, 0);
# error should be close to 0
@test norm(A-GPHodlr.hodlr_to_full(Aₕ)) <= 1e-6           

#----------
# Test 4: test taking gradient directly in HODLR format
#----------
# helper functions for this test
function create_low_rank_two_level_matrix_two_params(θ :: AbstractVector{T}, n :: Int) where T
    """ 
        For testing HODLR grad, creates a parametrized 
        (with 2 parameters) matrix that can be recovered 
        exactly with a 2-level HODLR. Each off-diagonal 
        factor has low-rank (rank=n/2^3).

        The off-diagonal blocks are simply:
        randn_symmetric(n/2^i, n/2^i) * θi^2 
        i.e. first level off-diag blocks depend on θ1,
        second level off-diag blocks depend on θ2

        Warning: Input θ must be of type Float64, otherwise 
        will get error conflicting with type T.
        https://discourse.julialang.org/t/error-with-forwarddiff-no-method-matching-float64/41905
    """
    @assert(length(θ) == 2)
    k = Int(n / 2^3);               # exact rank
    exact_level = 2;
    idx = dyadic_idx(n, exact_level);
    # fill the diagonals (all symmetric)
    A = zeros(T, n, n);
    θ1, θ2 = θ;
    A[idx[1], idx[1]] = randn_symmetric(Int(n / 4));
    A[idx[2], idx[2]] = randn_symmetric(Int(n / 4));
    A[idx[3], idx[3]] = randn_symmetric(Int(n / 4));
    A[idx[4], idx[4]] = randn_symmetric(Int(n / 4));
    # fill the off-diagonals
    tmp = randn_symmetric(Int(n / 4));
    A[idx[2], idx[1]] = tmp;
    A[idx[1], idx[2]] = tmp';
    tmp = randn_symmetric(Int(n / 4));
    A[idx[4], idx[3]] = tmp;
    A[idx[3], idx[4]] = tmp';

    # fill the off-diagonals of the prior level
    idx = dyadic_merge(idx, 1);
    tmp = randn_symmetric(Int(n / 2));
    A[idx[2], idx[1]] = tmp;
    A[idx[1], idx[2]] = tmp';
    @assert(issymmetric(A))
    # modify blocks such that off-diagonals are low rank
    idx = dyadic_idx(n, exact_level);
    tmp = A[idx[2], idx[1]];
    tmp[:, Int(n/8)+1:end] = tmp[:, 1:Int(n/8)];
    # add parametric dependence
    tmp = tmp .* (θ2^2);

    A[idx[2], idx[1]] .= tmp;
    A[idx[1], idx[2]] .= tmp';
    tmp = A[idx[4], idx[3]];
    tmp[:, Int(n/8)+1:end] = tmp[:, 1:Int(n/8)];
    # add parametric dependence
    tmp = tmp .* (θ2^2);

    A[idx[4], idx[3]] .= tmp;
    A[idx[3], idx[4]] .= tmp';
    # modify blocks from prior level
    idx = dyadic_merge(idx, 1);
    tmp = A[idx[2], idx[1]];
    tmp[:, Int(n/8)+1:end] = [tmp[:, 1:Int(n/8)] tmp[:, 1:Int(n/8)] tmp[:, 1:Int(n/8)]];
    # add parametric dependence
    tmp = tmp .* (θ1^2);

    A[idx[2], idx[1]] .= tmp;
    A[idx[1], idx[2]] .= tmp';
    @assert(issymmetric(A))
    return A
end

function two_level_matvec_grad_dummy(θ, n, v)
    """
        A dummy query procedure that returns garbage
        gradients / gradients disabled. Meant to test 
        the HODLR accuracy of `hodlr_grad`.
    """
    p = length(θ);
    grad = Array{Matrix}(undef, p);
    for k = 1:p
        grad[k] = zeros(size(v));
    end
    return A*v, grad;
end

# (Dev note) creating function handle
# A(θ::Vector) = create_low_rank_two_level_matrix_two_params(θ, n);
# # differentiate the matrix A(θ)
# dAdθ = ForwardDiff.jacobian(A, θ);
# dAdθ1 = reshape(dAdθ[:, 1], n, n);
# dAdθ2 = reshape(dAdθ[:, 2], n, n);

# matrix size
n = 1024;
# parameter values
θ = [0.5, 0.1];
# exact matrix
A = create_low_rank_two_level_matrix_two_params(θ, n);

# compute HODLR along with its derivative
local_rank = Int(n / 2^3);
peel_level = 2;
c = 10;
# build both A_hodlr and A_hodlr_grad with `hodlr_grad``
A_hodlr, dAdθ_hodlr = GPHodlr.hodlr_grad(two_level_matvec_grad_dummy, n, peel_level, local_rank, c, θ);
# build A_hodlr with `hodlr` and compare
A_hodlr2 = GPHodlr.hodlr(v->dummy_matvec_query(A, v), n, peel_level, local_rank, c);
@test norm(GPHodlr.hodlr_to_full(A_hodlr) - GPHodlr.hodlr_to_full(A_hodlr2)) < 1e-6
#----------
# Test 5: test inverting in HODLR format 
#----------
n = 1024;
# create two level exact matrix
A_exact = create_low_rank_two_level_matrix(n);
# create random rhs
b = randn(n, 1);
# exact inverse
Ainvb_exact = inv(A_exact)*b;
# build HODLR 
peel_level = 2;
local_rank = Int(n/8);
c = 0;
A_hodlr = GPHodlr.hodlr(v->dummy_matvec_query(A_exact, v), n, peel_level, local_rank, c);
# check HODLR solve (factorize first then solve)
A_hodlr_fact = GPHodlr.hodlr_factorize(A_hodlr);
Ainvb_hodlr_solve = GPHodlr.hodlr_solve(A_hodlr_fact, b);
# compare with coverting to full matrix and inversing directly
Ainvb_direct_hodlr = GPHodlr.hodlr_to_full(A_hodlr)\b;
# check error
@test norm(Ainvb_direct_hodlr - Ainvb_hodlr_solve) <= 1e-6

end
