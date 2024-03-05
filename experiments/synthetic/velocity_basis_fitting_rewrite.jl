# Driver script for interpolating an arbitrary
# advection velocity using different bases.
using GaussianRandomFields
using Plots
using Plots: savefig
using LaTeXStrings
# basis functions in L^2
using Polynomials, SpecialPolynomials
# optimization routines
using Optim, NLSolversBase, Random, LinearAlgebra
using FiniteDifferences 
using LineSearches
# save simple files
using DelimitedFiles

# adr solving
include("utils/finitediff_ops.jl");
include("utils/mle_helpers.jl");
include("../../src/hodlr.jl");

# fix random seed
import Random
Random.seed!(5)
# the observations are generated using function `v_true`, however,
# only `v_model` is used (a Chebyshev expansion) for MLE. Given 
# observations, the goal is to discover a set of coefficients such that
# `v_model(θ) ≈ v_true`.
############################################################
# Generating observations
############################################################
# here we take the domain to be [-1, 1] to be used with Chebyshev
n = 2^8;
@assert ispow2(n)
xmin = -1.; 
xmax = 1.;
pts = range(xmin, stop=xmax, length=n);
#mean_fn_func(x) = 20 .* exp.(-x.^2 ./2 ./0.2 ./ 0.2);
# zero mean 
mean_fn_func(x) = zeros(length(x));
mean_fn = mean_fn_func(pts);
sigma_phi = 1.;
l = 0.05;
nu = 1;
# true PDE velocity model
v_true(x, θ) = 2.0.*cos.(2.0.*x);
#v_true(x, θ) = 2.0.*(cos.(2.0.*x) .+ sin.(8.0.*x));
# check a Chebyshev polynomial can be exactly recovered 
theta = ones(5);
#v_true(x, θ) = ChebyshevT(theta).(x)

# nugget level
nugget_level = 0.1;
# differential parameters
c = 0.1;
kappa = 1e-2;

# generate Matern source
nmc = 1;
u, Mᵥ, σᵤ = generate_samples(n, xmin, xmax, mean_fn_func,
    [sigma_phi, nu, l], [kappa, c], v_true, [],
    nmc, "CirculantEmbedding", nugget_level, true);
############################################################
# MLE model specification and parameter estimation
############################################################
#all_degs = 4, 5;  # for cos (gives almost perfect fit)
#all_degs = 7;  # for cos (Runge's phenomenon)

all_degs = [5];  # can also be a list
for kk = eachindex(all_degs)
    # number of basis functions
    p = all_degs[kk];
    println("Running degree $p")
    # define model velocity function

    # (01/03/2023)
    # Use different kinds of bases
    all_basis_names = ["Chebyshev"];#, "Legendre"]; #Hermite
    for ll = eachindex(all_basis_names)
        basis_name = all_basis_names[ll];
        println("Using $basis_name basis ...")
        if basis_name == "Chebyshev"
            global v_model(x, θ) = ChebyshevT(θ)(x);
        elseif basis_name == "Legendre"
            global v_model(x, θ) = Legendre(θ)(x);
        elseif basis_name == "Hermite"
            global v_model(x, θ) = Hermite(θ)(x);
        else
            error("Basis not recognized. ")
        end

        # define factor matrices
        ∂L∂v(θ) = adv_mat1d_grad(pts, c, v_model, θ, kappa);
        L(θ) = adv_mat1d(pts, c, v_model, θ, kappa);
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
        # mean function (as a vector on grid, parameterized by θ)
        mean_u(θ) = L(θ)\mean_fn_func(pts);
        # derivative of mean 
        function mean_u_dv(θ)
            """ derivatives of the mean with respect to all θ, as an array. """
            p = length(θ);
            res = Array{Any}(undef, p);
            ∂L⁻∂v_eval = ∂L⁻∂v(θ);
            for i = eachindex(res)
                res[i] = ∂L⁻∂v_eval[i]*mean_fn_func(pts);
            end
            return res;
        end
        function K(θ)
            _L = L(θ);
            return _L\(Mᵥ*(_L'\I(n)));
        end
        K_noisy(θ) = K(θ) + (σᵤ^2)*I(n);

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
        ######################################################################
        # Exact likelihood computations
        ######################################################################
        function likelihood(θ, u)
            """ The data `u` is constant throughout optimization. """
            nmc = size(u, 1);
            _K = K_noisy(θ);
            _m = mean_u(θ);
            # center all data points
            y = u' .- _m; # (n x nmc)
            # term 1 and term 2 are constants with respect to u
            term1 = nmc * (-0.5*n*log(2π));
            term2 = nmc * (-0.5*logdet(_K));
            term3 = -0.5*sum(diag(y'*(_K\y)));
            return term1 + term2 + term3;
        end

        function neg_likelihood(θ, u)
            """ Negative log likelihood. """
            res = -likelihood(θ, u);
            return res;
        end

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
            _m = mean_u(θ);
            _m_grad = mean_u_dv(θ);
            # center all data points (n x nmc)
            y = u' .- _m;
            # loop over all parameters
            res = zeros(Float64, p);
            # precompute K_inv_u_centered
            tmp = _K\y;
            for i = eachindex(res)
                # term 1 is constant with respect to u
                term1 = nmc * (-0.5*LinearAlgebra.tr(_K_inv_K_grad[i]));
                term2 = 0.5*( sum(diag(tmp'*_K_grad[i]*tmp)) + 2*sum(_m_grad[i]'*tmp));
                res[i] = term1 + term2;
            end
            return res;
        end

        function neg_score(θ, u)
            res = -score(θ, u)
            return res;
        end
        # define HODLR MLE functions
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

        function neg_likelihood_hodlr(θ, u)
            """ Negative log likelihood. """
            res = -likelihood_hodlr(θ, u);
            return res;
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
            res = zeros(p);
            for i = 1:p
                # term 1 is constant with respect to u
                term1 = nmc * (-0.5*hodlr_tr(_K_inv_K_grad_hodlr[i]));
                term2 = 0.5*( sum(diag(tmp_hodlr'*hodlr_prod(_K_grad_hodlr[i], tmp_hodlr))) 
                    + 2*sum(_m_grad[i]'*tmp_hodlr));
                res[i] = term1 + term2;
            end
            return res;
        end

        function neg_score_hodlr(θ, u)
            println(θ)
            res = -score_hodlr(θ, u)
            return res;
        end

        ######################################################
        # Optimization
        ######################################################

        # starting point
        θ_init = ones(p+1);
        # set box to [-4, 4] for all coefficients.
        lower = -4.0.*ones(p+1);
        upper = 4.0.*ones(p+1);


        use_hodlr = false;
        if ~use_hodlr
            # inner optimizer
            #inner_optimizer = GradientDescent(linesearch=LineSearches.BackTracking(order=3));
            inner_optimizer = LBFGS();
            # no gradient
            global res = optimize(θ -> neg_likelihood(θ, u), #θ -> neg_score(θ, u),
                lower,
                upper,
                θ_init, 
                Fminbox(inner_optimizer),
                Optim.Options(
                    outer_iterations = 10000,
                    iterations=500, 
                    x_tol=-1e-8, f_tol=-1e-6,
                    f_reltol=-1e-8, f_abstol=-1e-6,
                    x_reltol=-1e-8, x_abstol=-1e-6,
                    g_abstol = 1e-4,
                    show_trace=true, 
                    show_every=10, 
                    time_limit=7200
                ),
                    inplace=false
            );
            
        else
            println("Using HODLR for optimization. ");
            inner_optimizer = LBFGS();
            # no gradient (try-catch)
            global res = optimize(
                θ -> neg_likelihood_hodlr(θ, u), 
                θ -> neg_score_hodlr(θ, u),
                lower,
                upper,
                θ_init, 
                Fminbox(inner_optimizer),
                Optim.Options(
                    outer_iterations = 10000,
                    iterations=500, 
                    x_tol=1e-8, f_tol=1e-6,
                    f_reltol=-1e-8, f_abstol=-1e-6,
                    x_reltol=-1e-8, x_abstol=-1e-6,
                    g_abstol = 1e-7,
                    show_trace=true, 
                    show_every=10, 
                    time_limit=7200
                ),
                    inplace=false
            )
        end
        print(res)
        θ_mle = Optim.minimizer(res);

        p = length(θ_mle);
    end # loop over basis
end # loop over degrees
