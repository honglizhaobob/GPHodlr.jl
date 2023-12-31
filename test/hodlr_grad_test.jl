using GPHodlr
include("basic_hodlr_accuracy_helpers.jl");
using LinearAlgebra
using ForwardDiff
using BlockDiagonals
using Test

## !!! (12/31/2023) has problems, derivatives do not match

@testset "hodlr grad matvec" begin
#----------
# Test 1: build gradient directly from matvec query
#----------
# matrix size
n = 1024;
# parameters
θ = [0.5, 2.1];
max_level = 2;
# exact matrix (before adding θ dependence)
A = create_low_rank_two_level_matrix(n);
A_param(θ) = dummy_param_mat(A, θ);
# take exact jacobian
tmp = ForwardDiff.jacobian(A_param, θ);
dAdθ1 = reshape(tmp[:, 1], n, n);
dAdθ2 = reshape(tmp[:, 2], n, n);

function dummy_param_mat_gradquery(θ, n, v)
    """
        A dummy query procedure that wraps
        `dummy_param_mat` and returns the 
        gradient of matvec.
    """
    p = length(θ);
    # matvec
    Av(θ_param) = A_param(θ_param)*v;
    res = Av(θ);
    res_grad = ForwardDiff.jacobian(Av, θ);
    grad = Array{Matrix}(undef, p);
    for k = 1:p
        grad[k] = reshape(res_grad[:, k], n, size(v, 2));
    end
    return res, grad;
end

_, dA_hodlr = GPHodlr.hodlr_grad(dummy_param_mat_gradquery, n, 2, Int(n/8), 0, θ);

# block diagonals should be zeros
@test norm(dAdθ1[1:Int(n/4), 1:Int(n/4)] - zeros(Int(n/4), Int(n/4))) == 0.0
@test norm(dA_hodlr[1].leaves[1] - zeros(Int(n/4), Int(n/4))) == 0.0


#----------
end

@testset "hodlr grad analytic gradient" begin
#----------
# Test 2: build HODLR grad with analytic gradient 
#----------
n = 1024;
local_rank = Int(n/8);
exact_level = 2;
# vector
#v = exp.(-(1:n)./200);
#v = -ones(n,1);
v = exp.(-sin.(1:n));
theta = [0.1, -0.5];
A, Agrad, Av, Agradv = create_two_level_matvec(theta, v);

function Agradv_wrapper(theta, n, rhs)
    _, _, Av, Agradv = create_two_level_matvec(theta, rhs);
    return Av, Agradv
end

_, A_grad_hodlr = GPHodlr.hodlr_grad(Agradv_wrapper, n, exact_level, local_rank, 0, theta);
# check error of gradients
A_grad1_recovered = GPHodlr.hodlr_to_full(A_grad_hodlr[1]);
A_grad2_recovered = GPHodlr.hodlr_to_full(A_grad_hodlr[2]);
@test norm(Agrad[1]-A_grad1_recovered) <= 1e-6
@test norm(Agrad[2]-A_grad2_recovered) <= 1e-6
#----------
end