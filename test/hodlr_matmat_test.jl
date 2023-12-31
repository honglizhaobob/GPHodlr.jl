using GPHodlr
using BlockDiagonals
using LinearAlgebra


# include helper functions
include("basic_hodlr_accuracy_helpers.jl");

@testset begin

#----------
# create HODLR matrix
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

# multiply with a prespecified matrix (compare to MATLAB)
V = reshape(1.0./(1:2n), n,2);
res = GPHodlr.hodlr_prod(A_hodlr, V);
# compare with exact
res_exact = A*V;
@test norm(res-res_exact) <= 1e-6
#-----------
end