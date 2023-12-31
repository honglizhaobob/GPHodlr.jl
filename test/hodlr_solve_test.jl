using GPHodlr
using BlockDiagonals
using LinearAlgebra


# include helper functions
include("basic_hodlr_accuracy_helpers.jl");

@testset begin
#----------
# Test 1: A\b
#----------

# create two level matrix with parameters
n = 1024;
local_rank = Int(n/8);
exact_level = 2;
theta = [0.1, -0.5];

# create vector: exp(-sin(1:n))
v = reshape(exp.(-sin.(1:n)), n, 1);

# create two level matrix
A, Agrad, Av, Agradv = create_two_level_matvec(theta, v);



# build HODLR with matvec
function Av_wrapper(v)
    _, _, Av, _ = create_two_level_matvec(theta, v);
    return Av
end

A_hodlr = GPHodlr.hodlr(Av_wrapper, n, exact_level, local_rank, 10);
# build one-way factorization
A_hodlr_fact = GPHodlr.hodlr_factorize(A_hodlr);
# solve directly
x = A\v;
# solve with factorization
x_approx = GPHodlr.hodlr_solve(A_hodlr_fact, v);

@test norm(x-x_approx) <= 1e-6;

end


