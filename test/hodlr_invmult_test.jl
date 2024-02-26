using GPHodlr
include("basic_hodlr_accuracy_helpers.jl");
using BlockDiagonals
using LinearAlgebra
using Test


@testset begin
#-----------
# set up example (see `dev_test.m`) in MATLAB
n = 1024;
local_rank = Int(n/8);
exact_level = 2;
v = exp.(-sin.(1:n));
theta = [0.1, -0.5];
# exact matrices
A, Agrad, Av, Agradv = create_two_level_matvec(theta, v);
# build HODLR
function Av_wrapper(v)
    _, _, Av, _ = create_two_level_matvec(theta, v);
    return Av
end
A_hodlr = GPHodlr.hodlr(Av_wrapper, n, exact_level, local_rank, 0);
# build one-way factorization
A_hodlr_fact = GPHodlr.hodlr_factorize(A_hodlr);

# build another identical matrix in base HODLR form
B, Bgrad, Bv, Bgradv = create_two_level_matvec(theta, v);
B_hodlr = GPHodlr.hodlr(Av_wrapper, n, exact_level, local_rank, 0);

# compute A_inv_B with B = A, expect identity
A_inv_B = GPHodlr.hodlr_invmult(A_hodlr_fact, B_hodlr);
A_inv_B_recovered = GPHodlr.hodlr_to_full(A_inv_B); # should be close to identity

@test norm(A_inv_B_recovered-I(n)) <= 1e-6
#----------
end
