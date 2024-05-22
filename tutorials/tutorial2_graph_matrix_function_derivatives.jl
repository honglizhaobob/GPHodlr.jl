# In this tutorial, we implement and test (with finite difference)
# associated matrix-valued derivatives with respect to the parameters.
# We do this because we would like to use gradient-based methods.

# Remark:
#   Implemented all parameter matrix-valued derivatives, and all 
#   matches well with what finite difference approximates. All 
#   derivative matrices should be implemented efficiently. Especially
#   for the one with smoothness parameter. It is worth comparing 
#   whether analytic implementation is as efficient as finite difference.

using Statistics, Random, StatsBase
Random.seed!(10);
using LinearAlgebra, BlockArrays, BlockDiagonals
using Plots
using LaTeXStrings

include("utils.jl");

# create a graph matrix with clusters 
n = 500;
r = 2;
p, q = 0.8, 0.1;
# returns graph adjacency 
A, idx = sbm(
    n, r, p, q
); A = Matrix(A);
# plot(Gray.(A))
# form Laplacian 
L = laplacian(A, true);

# fixed Matern parameters
sigma_k = 1.0;
kappa = 3.0;
nu = 2.0;

# external noise
sigma_e = 1e-4;


############################################################
# test derivative with respect to marginal variance


# compute exact matrix function with eigendecomposition 
lam, Q = eigen(L); 
@assert norm(Q*diagm(lam)*Q'-L) < 1e-6
# exponential kernel plus error
#lam_exponential = exp.(-(kappa^2/2).*lam);
lam_matern = (lam.+2nu/(kappa^2)).^(-nu);
K = sigma_k * sigma_k * (Q*diagm(lam_matern)*Q') + sigma_e^2 * I(n);
K = (K + K')./2;

function M(kappa, nu)
    return (2nu/kappa/kappa)*I(n)+L
end

function K_(sigma_k, kappa, nu, sigma_e)
    """
        exact evaluation of covariance matrix.
    """
    lam_matern = (lam.+2nu/(kappa^2)).^(-nu);
    K = sigma_k * sigma_k * (Q*diagm(lam_matern)*Q') + sigma_e^2 * I(n);
    return (K' + K)./2;
end

function dKdsigma_k(sigma_k, kappa, nu, sigma_e)
    """
        Analytic implementation.
    """
    K = K_(sigma_k, kappa, nu, sigma_e);
    K = K - sigma_e * sigma_e * I(n);
    K = (2/sigma_k)*K;
end

function dKdsigma_k_finite_difference(sigma_k, kappa, nu, sigma_e, eps=1e-8)
    """
        Finite difference implementation.
    """
    Kplus = K_(sigma_k+eps, kappa, nu, sigma_e);
    K = K_(sigma_k, kappa, nu, sigma_e);
    dK = (Kplus-K)./eps;
    return dK;
end

dK_exact = dKdsigma_k(sigma_k, kappa, nu, sigma_e);
dK_finite_difference = dKdsigma_k_finite_difference(sigma_k, kappa, nu, sigma_e);
@assert norm(dK_exact-dK_finite_difference)<1e-4;

############################################################
# test derivative with respect to noise variance 


function dKdsigma_e(sigma_k, kappa, nu, sigma_e)
    """
        Analytic implementation.
    """
    return 2*sigma_e*I(n);
end


function dKdsigma_e_finite_difference(sigma_k, kappa, nu, sigma_e, eps=1e-8)
    """
        Finite difference implementation.
    """
    Kplus = K_(sigma_k, kappa, nu, sigma_e+eps);
    K = K_(sigma_k, kappa, nu, sigma_e);
    dK = (Kplus-K)./eps;
    return dK;
end

dK_exact = dKdsigma_e(sigma_k, kappa, nu, sigma_e);
dK_finite_difference = dKdsigma_e_finite_difference(sigma_k, kappa, nu, sigma_e);
@assert norm(dK_exact-dK_finite_difference)<1e-4;

############################################################
# test derivative with respect to length scale 

function M(sigma_k, kappa, nu, sigma_e)
    """
        Helper matrix 
    """ 
    return (2nu/kappa/kappa)*I(n)+L;
end
function dKdkappa(sigma_k, kappa, nu, sigma_e)
    """
        Analytic implementation.
    """
    K = K_(sigma_k, kappa, nu, sigma_e);
    return (4*nu*nu/kappa/kappa/kappa)*K*inv(M(sigma_k, kappa, nu, sigma_e));
end


function dKdkappa_finite_difference(sigma_k, kappa, nu, sigma_e, eps=1e-8)
    """
        Finite difference implementation.
    """
    Kplus = K_(sigma_k, kappa+eps, nu, sigma_e);
    K = K_(sigma_k, kappa, nu, sigma_e);
    dK = (Kplus-K)./eps;
    return dK;
end

dK_exact = dKdkappa(sigma_k, kappa, nu, sigma_e);
dK_finite_difference = dKdkappa_finite_difference(sigma_k, kappa, nu, sigma_e);
@assert norm(dK_exact-dK_finite_difference)<1e-4;

############################################################
# test derivative with respect to smoothness

# the below code log on a matrix, which is inefficient. 
function dKdnu(sigma_k, kappa, nu, sigma_e)
    K = K_(sigma_k, kappa, nu, sigma_e);
    Meval = M(sigma_k, kappa, nu, sigma_e);
    M_inv = Meval\I(n);
    return -(sigma_k^2)*K*(log(Meval)+(2nu/kappa/kappa)*M_inv);
end

function dKdnu_finite_difference(sigma_k, kappa, nu, sigma_e, eps=1e-8)
    Kplus = K_(sigma_k, kappa, nu+eps, sigma_e);
    K = K_(sigma_k, kappa, nu, sigma_e);
    dK = (Kplus-K)./eps;
    return dK;
end

dK_exact = dKdnu(sigma_k, kappa, nu, sigma_e);
dK_finite_difference = dKdnu_finite_difference(sigma_k, kappa, nu, sigma_e);
@assert norm(dK_exact-dK_finite_difference)<1e-4;
