# Helper function for MLE related evaluations, specific to the
# ADR 1d example.

######################################################################
# Helper functions 
######################################################################
using GaussianRandomFields
using Plots
using Plots: savefig
using LaTeXStrings
# basis functions in L^2
using Polynomials
# optimization routines
using Optim, NLSolversBase, Random, LinearAlgebra, Statistics
# save simple files
using DelimitedFiles

# adr solving
include("finitediff_ops.jl");

######################################################################
# Advection diffsion reaction (1d) subroutines
######################################################################
function generate_samples(
        n :: Int, 
        x_min :: Float64, 
        x_max :: Float64, 
        mean_fn :: Function,
        matern_params :: Union{Vector, Matrix},
        pde_params :: Union{Vector, Matrix},
        v :: Function,
        θ_true :: Union{Matrix, Vector},
        nmc :: Int,
        sampling_strategy :: String,
        nugget_level :: Float64,
        verbose :: Bool
    )
    """ 
        Sample from the physical model given by 
        inverting the discretized ARD 1d operator
        on Matérn latent Gaussian process as forcing
        term.

        Inputs: 
            n               Number of spatial grid points.

            x_min, x_max    Boundary of the observed process

            mean_fn         A vectorized evaluation procedure 
                            that returns the mean function.

            matern_params   parameters for the Matern process generation.

            pde_params      static parameters for the ADR operator. Not including
                            advection.

            v               an evaluation procedure for advection velocity
                            at a grid point, with a set of parameters.
                            Should be of form `v(x, θ)`

            θ_true          An array of parametric inputs for the advection velocity.
                            Can either contain values or be empty. If empty,
                            the problem case does not have a ground truth (i.e.
                            model is being inferred). E.g. the true model can 
                            be 2*cos(x), but we are interpolating it 
                            using Chebyshev basis.

            nmc             number of observations.

            sampling_strategy 
                            A string to indicate how latent samples 
                            are generated. Currently supports:
                            `Cholesky` and `CirculantEmbedding`.

            nugget_level    A float representing the Tikhonov regularization
                            strength we are adding to our sampled observations.
                            The nugget level represents a proportion of 
                            observed standard deviation.
            
            verbose         If `true`, prints additional information 
                            such as sampling and solution progress.
    """
    @assert length(matern_params) == 3
    pts = range(x_min, x_max, length=n);
    # evaluate mean function
    mf = mean_fn(pts);
    # unpack Matern process parameters
    sigma_phi, nu, l = matern_params;
    # unpack differential parameters
    kappa, c = pde_params;
    # spatial grid size
    h = pts[2] - pts[1];
    # get Matern covariance matrix
    covfunc = CovarianceFunction(1, Matern(l, nu, σ=sigma_phi));
    M = (sigma_phi^2).*Matrix(apply(covfunc, pts));
    # generate latent process
    if sampling_strategy == "Cholesky"
        M_chol = LinearAlgebra.cholesky(M).U;
        phi = zeros(nmc, n);
        for i = 1:nmc
            phi[i, :] .= M_chol'*randn(n) .+ mf;
        end
    elseif sampling_strategy == "CirculantEmbedding"
        # Gaussian random field object
        grf = GaussianRandomField(mf, covfunc, CirculantEmbedding(), pts);
        phi = zeros(nmc, n);
        for i = 1:nmc
            phi[i, :] .= sample(grf);
        end
    else
        error("Not Implemented! ")
    end
    # get discretized operator
    L = adv_mat1d(pts, c, v, θ_true, kappa);
    # solve for observations
    u = zeros(nmc, n);
    for i = 1:nmc
        u[i, :] = L\phi[i, :];
    end
    # after generating samples, add nugget
    σᵤ = nugget_level * std(u);
    for i = 1:nmc
        if verbose
            if i % 10 == 0
                println("Generating sample with nugget: $i\n")
            end
        end
        u[i, :] .= u[i, :] .+ σᵤ.*randn(n);
    end
    return u, M, σᵤ;
end




##########
function likelihood_hodlr(θ, u)
    """ The data `u` is constant throughout optimization. """
    nmc = size(u, 1);
    n = size(u, 2);
    _K = K_noisy(θ);
    # build HODLR
    _K_hodlr = hodlr(v->_K*v, n, 2, Int(n/256), 10);
    # factorize HODLR
    _K_hodlr_factorized = hodlr_factorize(_K_hodlr);
    _m = mean_u(θ);
    # center all data points
    y = copy(u' .- _m); # (n x nmc)
    # term 1 and term 2 are constants with respect to u
    term1 = nmc * (-0.5*n*log(2π));
    term2 = nmc * (-0.5*hodlr_logdet(_K_hodlr_factorized));
    term3 = -0.5*sum(diag(y'*(hodlr_solve(_K_hodlr_factorized, y))));
    return term1 + term2 + term3;
end

function score_hodlr(θ, u)
    """ Compute score via HODLR format. """
    p = length(θ);
    nmc = size(u, 1);
    n = size(u, 2);
    # test accuracy of HODLR at θ_true
    _K = K_noisy(θ);
    _K_hodlr = hodlr(v->_K*v, n, 2, Int(n/256), 10);
    # HODLR grad
    _K_grad = K_noisy_dv(θ);
    _K_grad_hodlr = Array{Any}(undef, p);
    for i = 1:p
        # compute HODLR for all derivative matrices
        _K_grad_hodlr[i] = hodlr(v->_K_grad[i]*v, n, 2, Int(n/256), 10);
    end
    _m = mean_u(θ);
    _m_grad = mean_u_dv(θ);
    # center all data points (n x nmc)
    y = copy(u' .- _m);
    # compute one-way multiplicative factorization
    _K_hodlr_factorized = hodlr_factorize(_K_hodlr);
    # compute A⁻*Aⱼ
    _K_inv_K_grad_hodlr = Array{Any}(undef, p);
    for i = 1:p
        # !!! possible source of error, check if _K or _K_grad_hodlr are mutable
        _K_inv_K_grad_hodlr[i] = hodlr_invmult(_K_hodlr_factorized, _K_grad_hodlr[i]);
    end
    # precompute K_inv_u_centered
    tmp_hodlr = hodlr_solve(_K_hodlr_factorized, y);
    # compute score
    score_hodlr = zeros(p);
    for i = 1:p
        # term 1 is constant with respect to u
        term1 = nmc * (-0.5*hodlr_tr(_K_inv_K_grad_hodlr[i]));
        term2 = 0.5*( sum(diag(tmp_hodlr'*hodlr_prod(_K_grad_hodlr[i], tmp_hodlr))) 
            + 2*sum(_m_grad[i]'*tmp_hodlr));
        score_hodlr[i] = term1 + term2;
    end
    return score_hodlr;
end

function fisher(θ_mle)
    """ 
        Computes fisher information matrix, which can be 
        used for evaluating confidence interval.
    """
    p = length(θ_mle);
    fish = zeros(p, p);
    _K = K_noisy(θ_mle);
    _K_grad = K_noisy_dv(θ_mle);
    for i = 1:p
        for j = 1:p
            fish[i, j] = 0.5*LinearAlgebra.tr((_K\_K_grad[i])*(_K\_K_grad[j]));
        end
    end
    return fish;
end