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


end
