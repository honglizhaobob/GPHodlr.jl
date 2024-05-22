# Simple tutorial for Gaussian process regression on a randomly 
# generated graph. The data model is:
#       y = f(x) + noise 

# Remark:
#   - parameters are not uniquely identifiable (one likelihood value can 
#   correspond to multiple parameter choices).
#   - While training error is small, prediction error on hidden nodes is 
#   large. 
#   - Observations are in line with: https://arxiv.org/abs/2010.15538
#

using Statistics, Random, StatsBase
Random.seed!(10);
using LinearAlgebra, BlockArrays, BlockDiagonals
using Plots
using LaTeXStrings
using Optim

include("utils.jl");

######################################################################
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

# form exact covariance matrix (Matern)
# - marginal variance 
# - length scale 
# - smoothness (integer)

sigma_k = 1.0;
kappa = 3.0;
nu = 2.0;

# external noise
sigma_e = 1e-4;

# compute exact matrix function with eigendecomposition 
lam, Q = eigen(L); 
@assert norm(Q*diagm(lam)*Q'-L) < 1e-6
# exponential kernel plus error
#lam_exponential = exp.(-(kappa^2/2).*lam);
lam_matern = (lam.+2nu/(kappa^2)).^(-nu);
K = sigma_k * sigma_k * (Q*diagm(lam_matern)*Q') + sigma_e^2 * I(n);
K = (K + K')./2;
# compute cholesky 
Khalf = Matrix(cholesky(K).L);
@assert norm(Khalf*Khalf'-K)<1e-6
# generate samples from exact process
nmc = 1;
f = Khalf*randn(n,nmc); 
# estimate covariance from samples 
#Kest = (f * f')/(nmc-1);
#error()
# generate data
y = f;

# true parameters
theta_true = [sigma_k,kappa,nu];

# use MLE to estimate parameters from data using maximum likelihood 
function Knew(theta)
    """
        theta = [sigma_k, kappa, nu]
    """
    sigma_k, kappa, nu = theta[1], theta[2], theta[3];
    # transform eigenvalues 
    lam_matern = (lam.+2nu/(kappa^2)).^(-nu);
    K = sigma_k * sigma_k * (Q*diagm(lam_matern)*Q')+ (sigma_e^2)*I(n);
    K = (K + K')./2; 
    return K;
end


function negative_log_likelihood(theta)
    # evaluate K 
    K = Knew(theta);
    
    # Calculate the quadratic term y^T * K^(-1) * y
    quadratic_term = 0.0;
    for i = 1:nmc 
        quadratic_term = quadratic_term + y[:, i]'*(K\y[:, i]);
    end
    
    # Calculate the logarithm of the determinant of the covariance matrix
    log_det = logdet(K)[1]
    
    # Calculate the constant term n * log(2π)
    constant_term = n * log(2π)
    
    # Calculate the negative log-likelihood
    nll = 0.5 * (quadratic_term + log_det + constant_term)
    
    return nll
end
# test if minimum is at true parameter 
# all_kappa = collect(0.001:0.001:1.0);
# all_nu = collect(0.01:0.01:2.0);
# #all_likelihood = zeros(length(all_kappa));
# all_likelihood = zeros(length(all_nu));
# for k = eachindex(all_nu)
#     println(k)
#     all_likelihood[k] = negative_log_likelihood([sigma_k, kappa, all_nu[k]]);
# end
# p = plot(all_nu, all_likelihood);
# # plot first derivative
# p2 = plot(all_nu[1:end-1], (all_likelihood[2:end]-all_likelihood[1:end-1])./(all_nu[2]-all_nu[1]));

# use optimization to identify 
#lower_bounds = [1e-6, 1e-6, 1e-6];  # Lower bounds for parameters
lower_bounds = (1e-5).*ones(3);  # Lower bounds for parameters
upper_bounds = 6.0*ones(3);    # Upper bounds for parameters
initial_params = 0.6 .* ones(3);

# Optimize hyperparameters using maximum likelihood estimation with bounds
result = optimize(
            params -> negative_log_likelihood(params), 
            lower_bounds, upper_bounds, 
            initial_params, Fminbox(NelderMead()),
            Optim.Options(iterations=100, show_trace=true)
        );

# Extract optimized hyperparameters
theta_mle = Optim.minimizer(result);

# compare true likelihood and fitted likelihood 
tmp1 = negative_log_likelihood(theta_true);
tmp2 = negative_log_likelihood(theta_mle);
println("True = $(tmp1), fitted = $(tmp2)");

# It seems that the Matern GP is not uniquely identifiable
# which is also reported in the Appendix of https://proceedings.mlr.press/v130/borovitskiy21a/borovitskiy21a-supp.pdf
# Next, we check MSE of predicting on unseen nodes of the graph.
Kmle = Knew(theta_mle)-(sigma_e^2)*I(n);
ypred = Kmle * ((Kmle + (sigma_e^2)*I(n))\y);
p = plot(ypred, label="True"); plot!(p, y, color=:red, label="Predict");
plot!(p, title="Training Error");
error()
# subselection matrix on a random subset of entries
train_ratio = 0.8;
# randomly select entries for training data 
ntrain = Int(ceil(n*train_ratio));
ntest = n-ntrain;
idx = sample(1:n, ntrain, replace=false);
idxtest = setdiff(1:n, idx);
ytrain = y[idx];
D = Matrix(I(n))[idx, :];
Dtest = Matrix(I(n))[idxtest, :];
function Ktrain(theta)
    """ 
        Subselect the portion of covariance used for training.
    """
    K = Knew(theta);
    return K[idx, idx];
end

function negative_log_likelihood_train(theta)
    """
        Redefine log likelihood only for training nodes. 
    """
    # evaluate K 
    K = Ktrain(theta);
    
    # Calculate the quadratic term y^T * K^(-1) * y
    quadratic_term = 0.0;
    for i = 1:nmc 
        quadratic_term = quadratic_term + ytrain[:, i]'*(K\ytrain[:, i]);
    end
    
    # Calculate the logarithm of the determinant of the covariance matrix
    log_det = logdet(K)[1]
    
    # Calculate the constant term n * log(2π)
    constant_term = ntrain * log(2π)
    
    # Calculate the negative log-likelihood
    nll = 0.5 * (quadratic_term + log_det + constant_term)
    
    return nll
end

lower_bounds = (1e-5).*ones(3);  # Lower bounds for parameters
upper_bounds = 6.0*ones(3);    # Upper bounds for parameters
initial_params = 0.6 .* ones(3);

# Optimize hyperparameters using maximum likelihood estimation with bounds
result = optimize(
            params -> negative_log_likelihood_train(params), 
            lower_bounds, upper_bounds, 
            initial_params, Fminbox(NelderMead()),
            Optim.Options(iterations=100, show_trace=true)
        );

# Extract optimized hyperparameters
theta_mle = Optim.minimizer(result);
Kmle = Knew(theta_mle)-(sigma_e^2)*I(n);
# predict on subset
ypred = Kmle[idxtest, idx] * ((Kmle[idx, idx] + (sigma_e^2)*I(ntrain))\y[idx]);