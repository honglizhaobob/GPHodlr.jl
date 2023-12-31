using GPHodlr
include("basic_hodlr_accuracy_helpers.jl");
using BlockDiagonals
using LinearAlgebra

##########
# !!! (12/31/2023) Runs for a long time, potentially inefficient 
error("runs for a long time. ")
@testset begin
#----------
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
B, Bgrad, Bv, Bgradv = create_two_level_matvec(theta, v);
B_hodlr = GPHodlr.hodlr(Av_wrapper, n, exact_level, local_rank, 0);

# exact Hadamard product
C = A .* B;
# HODLR Hadamard
C_hodlr = GPHodlr.hodlr_hadamard(A_hodlr, B_hodlr);

# recover matrix from HODLR 
C_recovered = GPHodlr.hodlr_to_full(C_hodlr);
@test norm(C_recovered - C) <= 1e-6;
#----------
end