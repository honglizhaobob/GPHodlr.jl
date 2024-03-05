# Testing reimplementation in PhysicsMLE
using GPHodlr
# fix random seed
using Random
Random.seed!(10);
using GaussianRandomFields
using Plots
using ForwardDiff
# adr solving
include("utils/finitediff_ops.jl");
include("utils/mle_helpers.jl");
include("../../src/hodlr.jl");
##

#############################################################################
#############################################################################
# # test operator matrix match for 1D implementation
# n = 2^8;
# @assert ispow2(n);
# # left and right boundaries
# xmin, xmax = -1., 1.;
# pts = range(xmin, stop=xmax, length=n);
# dx = pts[2]-pts[1];
# # sample Gaussian random field
# m(x) = 20. .* exp.(-(x.^2) ./2 ./0.2 ./ 0.2);
# # latent parameters
# l, nu = 0.05, 1.0;
# sigma_phi = 1.0;
# # nugget level 
# nugget = 0.1;

# # physical parameters
# kappa = 1e-2; 
# c = 0.1;
# # true PDE velocity model (taken to be cosine)
# _vtrue_function = x -> 2.0.*cos.(2.0.*x);
# vtrue = GPHodlr.PhysicsMLE1d.CustomVelocity1d(xmin, xmax, _vtrue_function);

# # PhysicsMLE1d implementation 
# L = GPHodlr.PhysicsMLE1d.advection_diffusion_reaction_homogeneous_neumann1d(
#     pts, 
#     kappa,
#     vtrue,
#     c
# );

# # plain implementation 
# v_true(x, θ) = 2.0.*cos.(2.0.*x);
# Lold = adv_mat1d(pts, c, v_true, [], kappa);

#############################################################################
#############################################################################
# # test Chebyshev implementation 

# # custom implementation 
# theta = ones(5);
# pts = collect(range(-1.0, 1.0, 100));
# v_modelold(x, θ) = ChebyshevT(θ).(x);

# # PhysicsMLE1d implementation 
# v_modelnew = GPHodlr.PhysicsMLE1d.ChebyshevVelocity1d(
#     -1.0, 1.0, 
#     theta 
# );
# # evaluate and compare 
# testold = v_modelold(pts, theta);
# testnew = v_modelnew.(pts);

#############################################################################
#############################################################################
# test operator derivative matrix (wrt velocity) match for 1D implementation
n = 2^8;
@assert ispow2(n);
# left and right boundaries
xmin, xmax = -1., 1.;
pts = range(xmin, stop=xmax, length=n);
dx = pts[2]-pts[1];
# sample Gaussian random field
m(x) = zeros(length(x));
# latent parameters
l, nu = 0.05, 1.0;
sigma_phi = 1.0;
# nugget level 
nugget = 0.1;

# physical parameters
kappa = 1e-2; 
c = 0.1;

# basis PDE model 
theta = randn(5);
v_modelnew = GPHodlr.PhysicsMLE1d.ChebyshevVelocity1d(
    -1.0, 1.0, 
    theta 
);

# generate observations
nmc = 1;
u, Mᵥ, σᵤ = GPHodlr.PhysicsMLE1d.generate_samples(n, xmin, xmax, m,
    [sigma_phi, nu, l], [kappa, c], v_modelnew, [],
    nmc, "CirculantEmbedding", nugget, true);
u = u';
u = u[:]; 

# check gradient of velocity 
test_x_points = [0.1, -0.1, -0.12312, 0.5345, 0.767, -0.89922];
all_vgrad_new = zeros(length(test_x_points), length(theta));
for i = eachindex(test_x_points)
    all_vgrad_new[i, :] .= GPHodlr.PhysicsMLE1d.∂v∂θ(v_modelnew, test_x_points[i]);
end

# test old implementation 
v_modelold(x, theta) = ChebyshevT(theta).(x);
all_vgrad_old = zeros(length(test_x_points), length(theta));
for i = eachindex(test_x_points)
    # this mimics the old implementation
    xpt = test_x_points[i];
    veval(theta) = v_modelold(xpt, theta);
    # take gradient 
    all_vgrad_old[i, :] = ForwardDiff.gradient(veval, theta);
end
println(norm(all_vgrad_new-all_vgrad_old));

#check derivative matrices match

#derivative of operator wrt velocity 
# dLdv_old = adv_mat1d_grad(
#     pts, c, v_modelold, theta, kappa
# );
# dLdv_new = GPHodlr.PhysicsMLE1d.advection_diffusion_reaction_homogeneous_neumann1d_∂v(
#     pts, v_modelnew
# );
# error_old_new = 0.0;
# for i = 1:length(dLdv_old)
#     global error_old_new = error_old_new + norm(dLdv_old[i]-dLdv_new[i]);
# end

# check other derivatives 

# sample some observations

prob = GPHodlr.PhysicsMLE1d.MLEProblem1d(
    collect(pts), u, collect(1:length(u)), nugget, 
    kappa, v_modelnew, c, sigma_phi, nu, l
);

# evaluate operator matrix 
Leval_new = GPHodlr.PhysicsMLE1d.L(prob);
Leval_old = adv_mat1d(pts, c, v_modelold, theta, kappa);

# check ∂L∂v
dLdv_new = GPHodlr.PhysicsMLE1d.∂L∂v(prob);
dLdv_old = adv_mat1d_grad(pts, c, v_modelold, theta, kappa);
error_old_new = 0.0;
for i = eachindex(dLdv_new)
    global error_old_new = error_old_new + norm(dLdv_new[i] - dLdv_old[i]);
end
println(error_old_new);

# check ∂Linv∂v
L(θ) = adv_mat1d(pts, c, v_modelold, θ, kappa);
∂L∂v(θ) = adv_mat1d_grad(pts, c, v_modelold, θ, kappa);
function ∂L⁻∂v(θ)
    """ returns ∂L⁻ derivatives w.r.t. all θ, as an array. """
    p = length(θ);
    res = Array{Any}(undef, p);
    ∂L∂v_eval = ∂L∂v(θ);
    _L = L(θ);
    for i = eachindex(res)
        res[i] = -_L\(∂L∂v_eval[i]*(_L\I(n)));
    end
    return res;
end
dLinvdv_new = GPHodlr.PhysicsMLE1d.∂Linv∂v(Leval_new, dLdv_new);
dLinvdv_old = ∂L⁻∂v(theta);
error_old_new = 0.0;
for i = eachindex(dLinvdv_new)
    global error_old_new = error_old_new + norm(dLinvdv_new[i]-dLinvdv_old[i]);
end
println(error_old_new);

Meval_new = GPHodlr.PhysicsMLE1d.M(prob);
Keval_new = GPHodlr.PhysicsMLE1d.K(Meval_new, Leval_new, prob);
function K(θ)
    _L = L(θ);
    return _L\(Mᵥ*(_L'\I(n)));
end
K_noisy(θ) = K(θ) + (prob.sigma_u^2)*I(n);
Keval_old = K_noisy(theta);
error_old_new = norm(Keval_old-Keval_new);
println(error_old_new);

# compare dKdv 
function K_dv(θ)
    p = length(θ);
    res = Array{Any}(undef, p);
    ∂L⁻∂v_eval = ∂L⁻∂v(θ);
    for i = eachindex(res)
        res[i] = ∂L⁻∂v_eval[i]*Mᵥ*((L(θ)')\I(n))+(L(θ)\Mᵥ)*(∂L⁻∂v_eval[i]');
    end
    return res;
end
K_noisy_dv(θ) = K_dv(θ);
dKdveval_old = K_noisy_dv(theta);
dKdveval_new = GPHodlr.PhysicsMLE1d.∂K∂v(
    Leval_new\Meval_new,
    dLinvdv_new,
    prob
);
error_old_new = 0.0;
for i = eachindex(dKdveval_new)
    error_old_new = error_old_new + norm(dKdveval_old[i]-dKdveval_new[i]);
end
println(error_old_new);

# compare log likelihood 
function likelihood(θ, u)
    """ The data `u` is constant throughout optimization. """
    nmc = size(u, 1);
    _K = K_noisy(θ);
    # center all data points
    y = u'; # (n x nmc)
    # term 1 and term 2 are constants with respect to u
    #term1 = nmc * (-0.5*n*log(2π));
    term1 = 0.0;
    term2 = nmc * (-0.5*logdet(_K));
    term3 = -0.5*y'*(_K\y);
    return term1 + term2 + term3;
end
likeeval_old = likelihood(theta, u');
likeeval_new = GPHodlr.PhysicsMLE1d.log_likelihood(
    Keval_new, Keval_new\prob.u_observed, prob.u_observed
);

# check if score matches
function score(θ, u)
    """ 
        evaluates the score function for each θ, returns a vector. 
    """
    p = length(θ);
    nmc = size(u, 1);
    _K = K_noisy(θ);
    _K_grad = K_noisy_dv(θ);
    _K_inv_K_grad = Array{Any}(undef, p);
    for i = eachindex(_K_inv_K_grad)
        _K_inv_K_grad[i] = _K\_K_grad[i];
    end
    # center all data points (n x nmc)
    y = u';
    # loop over all parameters
    res = zeros(Float64, p);
    # precompute K_inv_u_centered
    tmp = _K\y;
    for i = eachindex(res)
        # term 1 is constant with respect to u
        term1 = nmc * (-0.5*LinearAlgebra.tr(_K_inv_K_grad[i]));
        term2 = 0.5*tmp'*_K_grad[i]*tmp;
        res[i] = term1 + term2;
    end
    return res;
end
scoreeval_old = score(theta, u');
scoreeval_new = GPHodlr.PhysicsMLE1d.score∂v(
    Keval_new,
    dKdveval_new,
    Keval_new\prob.u_observed
);
