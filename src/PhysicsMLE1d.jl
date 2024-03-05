#=
    (05/10/23) A mini-version of the 2d `PhysicsMLE.jl` module for conducting
    1d experiments.

    The physics model is assumed to be:

        -κu_xx + v⋅u_x + c⋅u = 0
    
    With homogeneous Neumann boundaries. `v` is a parameterized model.

    In likelihood evaluations, the process is assumed to be zero-mean.

    In this mini-module, we are only concerned with optimizing PDE parameters 
    to verify success of the PDE model. In `Optim.jl`, we assume the ordering
    of parameters to be:
    [
        kappa,
        v_theta[:],
        c
    ]
=#
######################################################################
# Velocity models
######################################################################

# Define a separate module 
module PhysicsMLE1d

# import other packages 
using GaussianRandomFields
using SpecialFunctions, Polynomials, SpecialPolynomials
using LinearAlgebra
using Random, Statistics
using SparseArrays
using Optim

##########
abstract type VelocityModel1d end
mutable struct ChebyshevVelocity1d{T <: AbstractFloat} <: VelocityModel1d
    """
        Evaluable 1d velocity field with Chebyshev expansion.
    """
    # domain 
    xmin :: T
    xmax :: T
    # mutable parameterizations
    theta :: Vector{T}

end

mutable struct CustomVelocity1d{T <: AbstractFloat} <: VelocityModel1d
    """
        Model wrapper for an analytic function in 1d.
    """
    # domain 
    xmin :: T
    xmax :: T
    # evalable parameterization
    model :: Function
end

function (v :: ChebyshevVelocity1d)(x :: T) where T <: AbstractFloat
    """ 
        Evaluable method for `ChebyshevVelocity` model. 


        Input:
            x           1 dimensional input.
        Output:
            res         1 dimensional velocity evaluations.

    """
    p = length(v.theta);

    # rescale coefficients such that bases are orthonormal
    tmp = ones(T, p);
    # adjust for built-in normalization constant
    #tmp[1] = tmp[1] ./ sqrt(pi);
    #tmp[2:end] .= tmp[2:end] ./ sqrt(pi ./ 2);

    # adjust for domain shifting 
    tmp[:] .= tmp[:] .* sqrt(2 ./ (v.xmax .- v.xmin));

    # rescale input coefficients
    tmp[:] .= tmp[:] .* v.theta;

    # shifted Chebyshev polynomials
    y1 = (2 * x-(v.xmax + v.xmin)) / (v.xmax - v.xmin);

    res = Chebyshev(tmp)(y1);

    return res
end


function (v :: CustomVelocity1d)(x :: T) where T <: AbstractFloat
    """
        Wrapper for known velocity model. 
    """
    return v.model(x);
end

##

function ∂v∂θ(v_model :: ChebyshevVelocity1d, x :: T) where T <: AbstractFloat
    p = length(v_model.theta);

    # evalute gradient with respect to each parameters
    # ∂θᵢ∑θᵢTᵢ = Tᵢ

    res_grad = zeros(T, p);
    y1 = (2 * x -(v_model.xmax+v_model.xmin)) / (v_model.xmax - v_model.xmin);
    for i = 1:p
        if i == 1
            res_grad[i] = sqrt(2 ./ (v_model.xmax .- v_model.xmin)) .* basis(Chebyshev, i-1)(y1); # ./ sqrt(pi);
        else
            res_grad[i] = sqrt(2 ./ (v_model.xmax .- v_model.xmin)) .* basis(Chebyshev, i-1)(y1);# ./ sqrt(pi ./ 2);
        end
    end

    return res_grad;
end


######################################################################
# Differential operators evaluation
######################################################################
function advection_diffusion_reaction_homogeneous_neumann1d(
    gridpts :: Any, 
    kappa :: T, 
    v_model :: VelocityModel1d, 
    c :: T
) where T <: AbstractFloat
    """ 
        Underlying implementation of discretized PDE operator given 
        diffusivity, reaction and velocity parameters.
    """
    gridpts = convert(Vector, gridpts);
    n = length(gridpts);
    h = gridpts[2] - gridpts[1];
    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();
    for i = 1:n
        # grid location
        x = gridpts[i];
        # advection velocity
        v_val = v_model(x);
        # compute coefficients
        # form: a1U_i-1 + a2U_i + a3U_i+1 = f_i
        a1 = (-v_val/2h-kappa/h^2);
        a2 = (c+2kappa/h^2);
        a3 = (v_val/2h-kappa/h^2);
        # store coefficients
        if i == 1
            # U_i 
            push!(row_ind, i);
            push!(col_ind, i);
            push!(entry, a2);
            # U_i+1
            push!(row_ind, i);
            push!(col_ind, i+1);
            push!(entry, a1+a3);
        elseif i == n
            # U_i-1
            push!(row_ind, i);
            push!(col_ind, i-1);
            push!(entry, a1+a3);
            # U_i
            push!(row_ind, i);
            push!(col_ind, i);
            push!(entry, a2);
        else
            # U_i-1
            push!(row_ind, i);
            push!(col_ind, i-1);
            push!(entry, a1);
            # U_i 
            push!(row_ind, i);
            push!(col_ind, i);
            push!(entry, a2);
            # U_i+1
            push!(row_ind, i);
            push!(col_ind, i+1);
            push!(entry, a3);
        end
    end
    # create differential operator as a matrix
    L = sparse(row_ind, col_ind, entry);
    return L;
end

# function advection_diffusion_reaction_homogeneous_neumann1d(
#     gridpts, 
#     kappa :: T, 
#     v_model, 
#     v_params,
#     c :: T
# ) where T
#     """ 
#         Underlying implementation of discretized PDE operator given 
#         diffusivity, reaction and velocity parameters.
#     """
#     gridpts = convert(Vector, gridpts);
#     n = length(gridpts);
#     h = gridpts[2] - gridpts[1];
#     row_ind = Vector{Int64}();
#     col_ind = Vector{Int64}();
#     entry = Vector{Float64}();
#     for i = 1:n
#         # grid location
#         x = gridpts[i];
#         # advection velocity
#         v_val = v_model(x, v_params);
#         # compute coefficients
#         # form: a1U_i-1 + a2U_i + a3U_i+1 = f_i
#         a1 = (-v_val/2h-kappa/h^2);
#         a2 = (c+2kappa/h^2);
#         a3 = (v_val/2h-kappa/h^2);
#         # store coefficients
#         if i == 1
#             # U_i 
#             push!(row_ind, i);
#             push!(col_ind, i);
#             push!(entry, a2);
#             # U_i+1
#             push!(row_ind, i);
#             push!(col_ind, i+1);
#             push!(entry, a1+a3);
#         elseif i == n
#             # U_i-1
#             push!(row_ind, i);
#             push!(col_ind, i-1);
#             push!(entry, a1+a3);
#             # U_i
#             push!(row_ind, i);
#             push!(col_ind, i);
#             push!(entry, a2);
#         else
#             # U_i-1
#             push!(row_ind, i);
#             push!(col_ind, i-1);
#             push!(entry, a1);
#             # U_i 
#             push!(row_ind, i);
#             push!(col_ind, i);
#             push!(entry, a2);
#             # U_i+1
#             push!(row_ind, i);
#             push!(col_ind, i+1);
#             push!(entry, a3);
#         end
#     end
#     # create differential operator as a matrix
#     L = sparse(row_ind, col_ind, entry);
#     return L;
# end

## Matrix derivatives with respect to PDE parameters
function advection_diffusion_reaction_homogeneous_neumann1d_∂kappa(
    gridpts
)
    gridpts = convert(Vector, gridpts);
    n = length(gridpts);
    h = gridpts[2] - gridpts[1];
    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();
    # compute coefficients ∂kappa
    # form: a1U_i-1 + a2U_i + a3U_i+1 = f_i
    a1 = -1/h^2;
    a2 = 2/h^2;
    a3 = -1/h^2;
    for i = 1:n
        # store coefficients
        if i == 1
            # U_i 
            push!(row_ind, i);
            push!(col_ind, i);
            push!(entry, a2);
            # U_i+1
            push!(row_ind, i);
            push!(col_ind, i+1);
            push!(entry, a1+a3);
        elseif i == n
            # U_i-1
            push!(row_ind, i);
            push!(col_ind, i-1);
            push!(entry, a1+a3);
            # U_i
            push!(row_ind, i);
            push!(col_ind, i);
            push!(entry, a2);
        else
            # U_i-1
            push!(row_ind, i);
            push!(col_ind, i-1);
            push!(entry, a1);
            # U_i 
            push!(row_ind, i);
            push!(col_ind, i);
            push!(entry, a2);
            # U_i+1
            push!(row_ind, i);
            push!(col_ind, i+1);
            push!(entry, a3);
        end
    end
    # create differential operator as a matrix
    L = sparse(row_ind, col_ind, entry);
    return L;
end

# function advection_diffusion_reaction_homogeneous_neumann1d_∂kappa(
#     gridpts, 
#     kappa :: T, 
#     v_model, 
#     v_params, 
#     c :: T
# ) where T
#     """ 
#         Derivative of advection-diffusion-reaction model with respect to 
#         constant diffusivity `kappa`. 

#         The matrix derivative is:
#             -u_xx
#     """
#     gridpts = convert(Vector, gridpts);
#     n = length(gridpts);
#     h = gridpts[2] - gridpts[1];
#     row_ind = Vector{Int64}();
#     col_ind = Vector{Int64}();
#     entry = Vector{Float64}();

#     # compute coefficients ∂kappa
#     # form: a1U_i-1 + a2U_i + a3U_i+1 = f_i
#     a1 = -1/h^2;
#     a2 = 2/h^2;
#     a3 = -1/h^2;
#     for i = 1:n
#         # store coefficients
#         if i == 1
#             # U_i 
#             push!(row_ind, i);
#             push!(col_ind, i);
#             push!(entry, a2);
#             # U_i+1
#             push!(row_ind, i);
#             push!(col_ind, i+1);
#             push!(entry, a1+a3);
#         elseif i == n
#             # U_i-1
#             push!(row_ind, i);
#             push!(col_ind, i-1);
#             push!(entry, a1+a3);
#             # U_i
#             push!(row_ind, i);
#             push!(col_ind, i);
#             push!(entry, a2);
#         else
#             # U_i-1
#             push!(row_ind, i);
#             push!(col_ind, i-1);
#             push!(entry, a1);
#             # U_i 
#             push!(row_ind, i);
#             push!(col_ind, i);
#             push!(entry, a2);
#             # U_i+1
#             push!(row_ind, i);
#             push!(col_ind, i+1);
#             push!(entry, a3);
#         end
#     end
#     # create differential operator as a matrix
#     L = sparse(row_ind, col_ind, entry);
#     return L;
# end


function advection_diffusion_reaction_homogeneous_neumann1d_∂v(
    gridpts, 
    v_model :: VelocityModel1d
)
    gridpts = convert(Vector, gridpts);
    n = length(gridpts);
    h = gridpts[2] - gridpts[1];
    p = length(v_model.theta);
    Ldθ = Array{SparseMatrixCSC}(undef, p);
    for pp = 1:p
        #println("Parameter $pp")
        row_ind = Vector{Int64}();
        col_ind = Vector{Int64}();
        entry = Vector{Float64}();
        for i = 1:n
            # grid location
            x = gridpts[i];
            # compute coefficients ∂v
            # form: a1U_i-1 + a2U_i + a3U_i+1 = f_i

            # take derivative with respect to θ
            a1 = -∂v∂θ(v_model, x)[pp] / (2h);
            # a2 has no dependence on `v`
            a2 = 0.0;
            # a3 always has opposite signs as a1
            a3 = -a1;
            
            # store coefficients
            if i == 1
                # U_i 
                push!(row_ind, i);
                push!(col_ind, i);
                push!(entry, a2);
                # U_i+1
                push!(row_ind, i);
                push!(col_ind, i+1);
                push!(entry, a1+a3);
            elseif i == n
                # U_i-1
                push!(row_ind, i);
                push!(col_ind, i-1);
                push!(entry, a1+a3);
                # U_i
                push!(row_ind, i);
                push!(col_ind, i);
                push!(entry, a2);
            else
                # U_i-1
                push!(row_ind, i);
                push!(col_ind, i-1);
                push!(entry, a1);
                # U_i 
                push!(row_ind, i);
                push!(col_ind, i);
                push!(entry, a2);
                # U_i+1
                push!(row_ind, i);
                push!(col_ind, i+1);
                push!(entry, a3);
            end
        end
        Ldθ[pp] = sparse(row_ind, col_ind, entry);
    end
    return Ldθ;
end

# function advection_diffusion_reaction_homogeneous_neumann1d_∂v(
#     gridpts, 
#     kappa :: T, 
#     v_model, 
#     v_model∂θ, 
#     v_params, 
#     c :: T
# ) where T
#     """ 
#         Returns Jacobian of the finite difference 
#         matrix for 1d, in all parameters θ. 

#         The result is returned as an array of matrices.

#         `v` is assumed to have form v(x, θ) where
#         θ are parameterization. 

#     """
#     gridpts = convert(Vector, gridpts);
#     n = length(gridpts);
#     h = gridpts[2] - gridpts[1];
#     p = length(v_params);
#     Ldθ = Array{SparseMatrixCSC}(undef, p);
#     for pp = 1:p
#         #println("Parameter $pp")
#         row_ind = Vector{Int64}();
#         col_ind = Vector{Int64}();
#         entry = Vector{Float64}();
#         for i = 1:n
#             # grid location
#             x = gridpts[i];
#             # compute coefficients ∂v
#             # form: a1U_i-1 + a2U_i + a3U_i+1 = f_i

#             # take derivative with respect to θ
#             a1 = -ChebyshevVelocityModel∂θ(x, v_params)[pp] / (2h);
#             # a2 has no dependence on `v`
#             a2 = 0.0;
#             # a3 always has opposite signs as a1
#             a3 = -a1;
            
#             # store coefficients
#             if i == 1
#                 # U_i 
#                 push!(row_ind, i);
#                 push!(col_ind, i);
#                 push!(entry, a2);
#                 # U_i+1
#                 push!(row_ind, i);
#                 push!(col_ind, i+1);
#                 push!(entry, a1+a3);
#             elseif i == n
#                 # U_i-1
#                 push!(row_ind, i);
#                 push!(col_ind, i-1);
#                 push!(entry, a1+a3);
#                 # U_i
#                 push!(row_ind, i);
#                 push!(col_ind, i);
#                 push!(entry, a2);
#             else
#                 # U_i-1
#                 push!(row_ind, i);
#                 push!(col_ind, i-1);
#                 push!(entry, a1);
#                 # U_i 
#                 push!(row_ind, i);
#                 push!(col_ind, i);
#                 push!(entry, a2);
#                 # U_i+1
#                 push!(row_ind, i);
#                 push!(col_ind, i+1);
#                 push!(entry, a3);
#             end
#         end
#         Ldθ[pp] = sparse(row_ind, col_ind, entry);
#     end
#     return Ldθ;
# end

function advection_diffusion_reaction_homogeneous_neumann1d_∂c(
    gridpts, 
    kappa :: T, 
    v_model, 
    v_params, 
    c :: T
) where T
    """ 
        Derivative of advection-diffusion-reaction model with respect to 
        constant diffusivity `c`. 

        The matrix derivative is the identity matrix.

    """
    gridpts = convert(Vector, gridpts);
    n = length(gridpts);
    return sparse(I(n));
end

######################################################################
# MLE problem in 1d
######################################################################
mutable struct MLEProblem1d
    """
        A maximum likelihood problem defined on a set of data, 
        defined in 1d, working mostly with synthetic data.

        Only PDE parameters are being optimized and updated.
    """
    # spatial grid
    xgrid :: Vector{Float64}

    # observed values 
    u_observed :: Vector{Float64}

    # observed indices
    obs_local_inds :: Vector{Int64}

    # covariance parameters (latent)

    # nugget standard deviation
    sigma_u :: Float64

    # 1d Matern covariance function (latent parameters included)
    covfunc :: CovarianceFunction{1, Matern{Float64}}

    # covariance parameters (PDE)

    # diffusivity
    kappa :: Float64

    # velocity model 
    v_model :: VelocityModel1d

    # reaction 
    c :: Float64

    function MLEProblem1d(
        xgrid :: Vector{Float64}, 
        u_observed :: Vector{Float64},
        obs_local_inds :: Vector{Int64},
        nugget_level :: Float64,
        kappa :: Float64,
        v_model :: VelocityModel1d,
        c :: Float64,
        sigma_phi :: Float64,
        nu :: Float64, 
        l :: Float64
    )
        """
            Compute necessary preprocessing statistics and initialize MLE problem.
        """
        # compute nugget standard deviation
        @assert sum(isnan.(u_observed)) == 0 
        sigma_u = nugget_level * std(u_observed);
        covfunc = CovarianceFunction(1, Matern(l, nu, σ=sigma_phi));

        return new(
            xgrid, 
            u_observed,
            obs_local_inds,
            sigma_u,
            covfunc,
            kappa,
            v_model,
            c
        );
    end
end

## Wrapped covariance evaluation functions
function M(prob :: MLEProblem1d)
    # apply covariance function
    sigma_phi = prob.covfunc.cov.σ;
    M = (sigma_phi^2).*apply(prob.covfunc, prob.xgrid);
    return M;
end

function L(prob :: MLEProblem1d)
    return advection_diffusion_reaction_homogeneous_neumann1d(
        prob.xgrid, 
        prob.kappa,
        prob.v_model,
        prob.c
    );
end

function ∂L∂kappa(
    prob
)
    return advection_diffusion_reaction_homogeneous_neumann1d_∂kappa(
        prob.xgrid
    );
end

function ∂L∂v(
    prob
)
    return advection_diffusion_reaction_homogeneous_neumann1d_∂v(
        prob.xgrid,
        prob.v_model
    );
end

function ∂L∂c(
    prob
)
    n = length(prob.xgrid);
    return sparse(I(n));
end

function ∂Linv∂kappa(L_eval, ∂L∂kappa_eval)
    # ∂L∂θ * L^-1 => -L^-1 * ∂L∂θ * L^-1
    res = -L_eval\((L_eval')\(Matrix(∂L∂kappa_eval)'))';
    return res;
end

function ∂Linv∂v(
    L_eval, ∂L∂v_eval
)
    p = length(∂L∂v_eval);
    res = Array{Any}(undef, p);
    for i = 1:p
        res[i] = -L_eval\((L_eval')\(Matrix(∂L∂v_eval[i])'))';
    end
    return res;
end

function ∂Linv∂c(L_eval, ∂L∂c_eval)
    # ∂L∂θ * L^-1 => -L^-1 * ∂L∂θ * L^-1
    res = -L_eval\((L_eval')\(Matrix(∂L∂c_eval)'))';
    return res;
end

##
function K(M_eval, L_eval, prob)
    M_times_L_inv_T = (L_eval\M_eval)';
    L_inv_M_times_L_inv_T = L_eval\M_times_L_inv_T;
    # subselect => D_obs * (L_inv * M * L_inv^T) * D_obs^T
    K_obs_eval = L_inv_M_times_L_inv_T[prob.obs_local_inds, prob.obs_local_inds];
    # add perturbation => K_obs + (sigma_u^2) * I
    tmp = diag(K_obs_eval).+(prob.sigma_u^2);
    K_obs_eval[diagind(K_obs_eval)] .= tmp;
    return Symmetric(K_obs_eval);
end

function K_revamped(L_eval, L_inv_M_eval, prob)
    """
        Revamped version. L^-1 * M is precomputed.
    """
    M_times_L_inv_T = L_inv_M_eval';
    L_inv_M_times_L_inv_T = L_eval\M_times_L_inv_T;
    # subselect => D_obs * (L_inv * M * L_inv^T) * D_obs^T
    K_obs_eval = L_inv_M_times_L_inv_T[prob.obs_local_inds, prob.obs_local_inds];
    # add perturbation => K_obs + (sigma_u^2) * I
    tmp = diag(K_obs_eval).+(prob.sigma_u^2);
    K_obs_eval[diagind(K_obs_eval)] .= tmp;
    return Symmetric(K_obs_eval);
end

##
function ∂K∂kappa(
    M_eval, 
    L_eval,
    ∂Linv∂kappa_eval,
    prob
)
    M_times_L_inv_eval = (L_eval\M_eval)';
    L_inv_times_M_eval = L_eval\M_eval;
    res = ∂Linv∂kappa_eval * M_times_L_inv_eval + L_inv_times_M_eval * ∂Linv∂kappa_eval';
    res = res[prob.obs_local_inds, prob.obs_local_inds];
    return Symmetric(res);
end

function ∂K∂v(
    M_eval, 
    L_eval,
    ∂Linv∂v_eval,
    prob
)
    p = length(∂Linv∂v_eval);
    res = Array{Any}(undef, p);
    M_times_L_inv_eval = (L_eval\M_eval)';
    L_inv_times_M_eval = L_eval\M_eval;
    for i = 1:p
        res[i] = Symmetric((∂Linv∂v_eval[i] * M_times_L_inv_eval 
            + L_inv_times_M_eval * ∂Linv∂v_eval[i]')[prob.obs_local_inds, prob.obs_local_inds]
        );
    end
    return res;
end

function ∂K∂c(
    M_eval, 
    L_eval,
    ∂Linv∂c_eval,
    prob
)
    M_times_L_inv_eval = (L_eval\M_eval)';
    L_inv_times_M_eval = L_eval\M_eval;
    res = ∂Linv∂c_eval * M_times_L_inv_eval + L_inv_times_M_eval * ∂Linv∂c_eval';
    res = res[prob.obs_local_inds, prob.obs_local_inds];
    return Symmetric(res);
end

## revamped version of ∂K∂θ, only L^-1 * M needs to be precomputed
function ∂K∂kappa(
    L_inv_M_eval,
    ∂Linv∂kappa_eval,
    prob
)
    res = ∂Linv∂kappa_eval * (L_inv_M_eval') + L_inv_M_eval * ∂Linv∂kappa_eval';
    res = res[prob.obs_local_inds, prob.obs_local_inds];
    return Symmetric(res);
end

function ∂K∂v(
    L_inv_M_eval,
    ∂Linv∂v_eval,
    prob
)
    p = length(∂Linv∂v_eval);
    res = Array{Any}(undef, p);
    for i = 1:p
        res[i] = Symmetric((∂Linv∂v_eval[i] * (L_inv_M_eval') 
            + L_inv_M_eval * ∂Linv∂v_eval[i]')[prob.obs_local_inds, prob.obs_local_inds]
        );
    end
    return res;
end

function ∂K∂c(
    L_inv_M_eval,
    ∂Linv∂c_eval,
    prob
)
    res = ∂Linv∂c_eval * (L_inv_M_eval') + L_inv_M_eval * ∂Linv∂c_eval';
    res = res[prob.obs_local_inds, prob.obs_local_inds];
    return Symmetric(res);
end

##
function log_likelihood(
    K_eval,
    K_inv_u_eval,
    u
)
    """
        Evaluates log-likelihood on observed values.
    """
    logabsdet_K_eval, flag = logabsdet(K_eval);
    if flag < 0
        @warn("Negative determinant encountered, check for conditioning. ");
    end
    # K^-1u
    res = -0.5*logabsdet_K_eval - 0.5*u'*K_inv_u_eval;
    return res;
end

function score∂kappa(
    K_eval,
    ∂K∂kappa_eval,
    K_inv_u_eval
)
    res = -0.5*tr(K_eval\∂K∂kappa_eval)+0.5*K_inv_u_eval'*∂K∂kappa_eval*K_inv_u_eval;
    return res;
end

function score∂v(
    K_eval,
    ∂K∂v_eval,
    K_inv_u_eval
)
    p = length(∂K∂v_eval);
    res = zeros(Float64, p);
    for i = 1:p
        res[i] = -0.5*tr(K_eval\∂K∂v_eval[i])+0.5*K_inv_u_eval'*∂K∂v_eval[i]*K_inv_u_eval;
    end
    return res;
end

function score∂c(
    K_eval,
    ∂K∂c_eval,
    K_inv_u_eval
)
    res = -0.5*tr(K_eval\∂K∂c_eval)+0.5*K_inv_u_eval'*∂K∂c_eval*K_inv_u_eval;
    return res;
end


## solve MLE problem
function solve!(prob :: MLEProblem1d)
    # number of parameters
    p = length(prob.v_model.theta);
    # unpack lower and upper boundaries for optimization 
    lower, upper = -4.0*ones(p), 4.0.*ones(p);
    # initial parameters
    theta_init = copy(prob.v_model.theta[:]);
    # evaluate constant Matern kernel 
    M_eval = M(prob);
    function negloglike(theta)
        # update the problem velocity parameters 
        prob.v_model.theta[:] .= copy(theta);
        # evaluate PDE covariance
        L_eval = L(prob); 
        K_eval = K(M_eval, L_eval, prob);
        K_inv_u_eval = K_eval\prob.u_observed;
        return -log_likelihood(
            K_eval,
            K_inv_u_eval,
            prob.u_observed
        );
    end

    # function negscore(theta)
    #     # update the problem velocity parameters 
    #     prob.v_model.theta[:] .= copy(theta);
    #     # evaluate PDE covariance 
    #     L_eval = L(prob);
    #     K_eval = K(M_eval, L_eval, prob);
    #     dLdv_eval = ∂L∂v(prob);
    #     dLinvdv_eval = ∂Linv∂v(L_eval, dLdv_eval);
    #     dKdv_eval = ∂K∂v(M_eval, L_eval, dLinvdv_eval, prob);
    #     K_inv_u_eval = K_eval\prob.u_observed;
    #     return -score∂v(
    #         K_eval, dKdv_eval, K_inv_u_eval
    #     );
    # end

    # optimize 
    inner_optimizer = Optim.LBFGS();
    res = optimize(negloglike,
                lower,
                upper,
                theta_init, 
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
    return res;
end


function solve!(
    prob :: MLEProblem1d,
    param_constraints :: Matrix{Float64}
)
    """
        1d MLE optimization, applied only on PDE parameters. By default uses 
        gradient-based optimization.
    """
    # total number of PDE parameters
    n_vel_params = length(prob.v_model.theta);
    n_trainable = 1+n_vel_params+1;
    # initialize 
    _params_init = dump_pde_parameters(prob);
    # precompute 
    M_eval = M(prob);

    # function that computes both objective and gradient at once, reusing calculations
    function _fg!(F, G, theta)
        # first update problem state
        update!(prob, theta);
        # ----------------------------------------
        # Common computations
        # ----------------------------------------
        # evaluate operator
        L_eval = L(prob);
        # precompute 
        L_inv_M_eval = L_eval\M_eval;

        # evaluate observation covariance
        K_eval = K(L_eval, L_inv_M_eval, prob);
        K_inv_u_eval = K_eval\prob.u_observed;

        # compute score first
        # ----------------------------------------
        # Score 
        # ----------------------------------------
        if G !== nothing
            # result vector
            tmp = zeros(Float64, n_trainable);

            # compute score with respect to kappa
            ∂L∂kappa_eval = ∂L∂kappa(prob);
            ∂Linv∂kappa_eval = ∂Linv∂kappa(L_eval, ∂L∂kappa_eval);
            ∂K∂kappa_eval = ∂K∂kappa(L_inv_M_eval, ∂Linv∂kappa_eval, prob);
            tmp[1] = score∂kappa(K_eval, ∂K∂kappa_eval, K_inv_u_eval);

            # compute score with respect to velocity 
            ∂L∂v_eval = ∂L∂v(prob);
            ∂Linv∂v_eval = ∂Linv∂v(L_eval, ∂L∂v_eval);
            ∂K∂v_eval = ∂K∂v(L_inv_M_eval, ∂Linv∂v_eval, prob);
            tmp[2:2+n_vel_params-1] = score∂v(K_eval, ∂K∂v_eval, K_inv_u_eval);

            # compute score with respect to reaction c
            ∂L∂c_eval = ∂L∂c(prob);
            ∂Linv∂c_eval = ∂Linv∂c(L_eval, ∂L∂c_eval);
            ∂K∂c_eval = ∂K∂c(L_inv_M_eval, ∂Linv∂c_eval, prob);
            tmp[end] = score∂c(K_eval, ∂K∂c_eval, K_inv_u_eval);
            
            # filter and update G (negative score)
            G[:] .= -tmp[:];
        end

        # ----------------------------------------
        # Likelihood
        # ----------------------------------------
        if F !== nothing
            return -log_likelihood(K_eval, K_inv_u_eval, prob.u_observed);
        end
    end

    # optimization
    _lower = param_constraints[:, 1];
    _upper = param_constraints[:, 2];
    # initialize optimization algorithm
    _optimization_alg = Optim.LBFGS();

    # define function to be optimized
    _optimization_function = OnceDifferentiable(Optim.only_fg!(_fg!), _params_init);
    _optimizer_result = optimize(_optimization_function,
        _lower,
        _upper,
        _params_init, Fminbox(_optimization_alg),
        Optim.Options(
            outer_iterations = 50,
            iterations=50,
            f_tol=-1.0, f_reltol=-1.0, f_abstol=-1.0,
            x_tol=-1.0, x_reltol=-1.0, x_abstol=-1.0,
            g_abstol = 1e-3, # hard constraint on gradient converging, all others are set to -1.0 (never reachable)
            store_trace=true,
            show_trace=true, 
            extended_trace=true,
            show_every=1
        )
    );
    _params_final = Optim.minimizer(_optimizer_result);
    # check if converged, if not error
    _converged_status = Optim.converged(_optimizer_result);
    if !_converged_status
        @warn "Problem did not converge. ";
    end
    # update problem parameters
    update!(prob, _params_final);
    return _optimizer_result;
end

function dump_pde_parameters(prob :: MLEProblem1d)
    """
        Collects PDE parameters in a vector.
    """
    n_vel_params = length(prob.v_model.theta);
    n_params = 1+n_vel_params+1;
    theta = zeros(Float64, n_params);
    theta[1] = prob.kappa;
    theta[2:2+n_vel_params-1] .= prob.v_model.theta;
    theta[end] = prob.c;
    return theta;
end

function update!(prob :: MLEProblem1d, theta :: Vector)
    """
        Updates PDE parameters stored in MLE problem.
    """
    n_vel_params = length(prob.v_model.theta);
    kappa = theta[1];
    vel_params = theta[2:2+n_vel_params-1];
    c = theta[end];
    # update PDE parameters
    prob.kappa = kappa;
    prob.v_model.theta = vel_params;
    prob.c = c;
end

## Krigging helpers
function imputation_statistics(prob)
    M_eval = M(prob);
    L_eval = L(prob);
    # compute masked local indices
    obs_local_inds = prob.obs_local_inds;
    n_full = size(M_eval, 1);
    full_local_inds = collect(1:n_full);
    mask_local_inds = filter(x -> !in(x, obs_local_inds), full_local_inds);

    K_full_eval = L_eval\M_eval;
    K_full_eval = (L_eval\K_full_eval')';
    # projections
    K_hidhid = K_full_eval[mask_local_inds, mask_local_inds];
    K_hidobs = K_full_eval[mask_local_inds, obs_local_inds]; 
    K_obsobs = K_full_eval[obs_local_inds, obs_local_inds];
    tmp = K_obsobs[diagind(K_obsobs)];
    K_obsobs[diagind(K_obsobs)] .= tmp .+ (prob.sigma_u).^2;
    u_hid_mean = K_hidobs*(K_obsobs\u_train);
    u_hid_cov = Symmetric(
        K_hidhid - K_hidobs*(K_obsobs\K_hidobs')
    );
    return u_hid_mean, u_hid_cov;
end


## other helpers
function generate_samples(
        n :: Int, 
        x_min :: Float64, 
        x_max :: Float64, 
        mean_fn :: Function,
        matern_params :: Union{Vector, Matrix},
        pde_params :: Union{Vector, Matrix},
        v :: Any,
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
    L = advection_diffusion_reaction_homogeneous_neumann1d(collect(pts), kappa, v, c);
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

########## end module


end