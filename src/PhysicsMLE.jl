#=
    (03/16/2023) 
    Helper functions to perform (parallelized) maximum likelihood 
    estimation of a physics-based model for Gaussian processes. 

    Furthermore, missing value imputations and statistics (e.g. 
    information matrix) can be computed scalably using HODLR matrices.

    The problem is defined in 2 dimensions. In likelihood evaluations, 
    zero mean is assumed. 

    In terms of implementation, the module relies on the built-in 
    optimization routine in `Optim.jl`. When inputting parameters
    during each iteration, the following order:
        [
            sigma_phi, 
            nu, 
            l, 
            kappa, 
            v1_theta[:],
            v2_theta[:],
            c
        ]
    is assumed. 

    Optim.jl configurations:
        https://github.com/JuliaNLSolvers/NLSolversBase.jl
        https://julianlsolvers.github.io/Optim.jl/

    Performance tips:
        https://gist.github.com/flcong/2eba0189d7d3686ea9633a6d14398931
        https://docs.julialang.org/en/v1/manual/performance-tips/
        Closures: https://tomohiro-soejima.github.io/blog/julia/post1/
    
    Related Julia Discussions:
        https://discourse.julialang.org/t/alternative-to-function-as-field-in-struct/55094
        https://mmmlab.rbind.io/posts/2021-03-09-julia-mle/
        https://github.com/JuliaNLSolvers/Optim.jl/blob/master/docs/src/user/minimization.md
        https://github.com/JuliaNLSolvers/Optim.jl/blob/master/docs/src/user/tipsandtricks.md
    
    Developing packages:
        https://syl1.gitbook.io/julia-language-a-concise-tutorial/language-core/11-developing-julia-packages

    WARNINGS (03/22/2023)
    (1) Matrix derivative: Can be made more efficient by directly 
    noting: 
        -κΔu + v⋅∇u + cu = 0

        ∂/∂κ = -Δu
        ∂/∂c = u
        ∂/∂θ₁ = ∂/∂v₁⋅∂v₁/∂θ₁
        ∂/∂θ₂ = ∂/∂v₂⋅∂v₂/∂θ₂
    instead of looping and computing each value.
=#
########################################################################
# function solve(MLEProblem) => does optimization and returns best parameters
########################################################################
mutable struct MLEProblem 
    """
        A maximum likelihood problem defined on a set of data, which holds
        the observations, statistics of latent field for this problem, and
        physical parameters over which the MLE is to be optimized.
    """
    # nugget standard deviation
    sigma_u :: Float64

    # noisy observation
    u_noisy :: Vector{Float64}

    # SST data along with grids
    data :: SSTData

    # PDE parameters
    # diffusivity (assumed to be constant)
    kappa :: Float64

    # velocity parameterization
    v_model :: VelocityModel

    # reaction coefficient (assumed to be constant)
    c :: Float64

    # latent Gaussian field parameters
    # marginal variance
    sigma_phi :: Float64
    # smoothness parameter
    nu :: Float64
    # correlation length
    l :: Float64
    # 2d covariance function
    covfunc :: CovarianceFunction{2, Matern{Float64}}

    # update manual is a vector of strings that let the 
    # optimizing routine know which parameters are free
    # so that we selectively compute gradients and update.
    # the update manual should have the same length as 
    # the number of inputs to `Optim.optimize`

    # the possible strings are:
    #   `sigma_phi`, `l`, `nu`, `kappa`, `v1_theta`, `v2_theta`, `c`
    update_manual :: Vector{String}

    function MLEProblem(
        data :: SSTData, 
        nugget_level :: Float64,
        kappa :: Float64,
        v_model :: VelocityModel,
        c :: Float64,
        sigma_phi :: Float64,
        nu :: Float64,
        l :: Float64,
        update_manual :: Vector{String}
    )
        """ 
            Construct for the MLE problem to preprocess the data
            before running optimization.
        """
        # (05/03/2023) Update: nugget is a modeling assumption that is 
        # independent of data, should not change the observations. 

        # add noise to data as a proportion of observation std
        u_noisy = data.u_observed;
        sigma_u = nugget_level*std(u_noisy);

        # create covariance function for latent evaluation
        covfunc = CovarianceFunction(2, Matern(l, nu, σ=sigma_phi));

        return new(
            sigma_u, 
            u_noisy,
            data,
            kappa,
            v_model,
            c,
            sigma_phi,
            nu,
            l,
            covfunc,
            update_manual
        )
    end
end

#######################################################################
# Helper functions to compute likelihood functions at application level
#######################################################################
function M(prob :: MLEProblem)
    """
        Given a physics-based MLE problem instance in its
        current parameteric state, unwraps the latent statistical
        parameters and returns the latent covariance function.
    """
    # apply covariance function
    M = (prob.sigma_phi^2).*apply(prob.covfunc, prob.data.xgrid, prob.data.ygrid);
    return M;
end

function ∂M∂sigma(M :: Matrix{Float64}, prob :: MLEProblem)
    """
        Evaluates ∂M with respect to marginal standard deviation, which can 
        reuse the already evaluated covariance kernel. 
    """
    return 2.0 .* (M ./ prob.sigma_phi);
end


function ∂M∂l(M :: Matrix{Float64}, prob :: MLEProblem)
    """
        Evaluates ∂M with respect to correlation length, which can 
        reuse the already evaluated covariance kernel, but needs to 
        compute another kernel at (ν+1).
    """
    l_corr = prob.l;
    nu = prob.nu;
    sigma_phi = prob.sigma_phi
    # create new covariance function
    covfunc = CovarianceFunction(2, Matern(l_corr, nu+1, σ=sigma_phi));
    M_plus = (prob.sigma_phi^2).*apply(covfunc, prob.data.xgrid, prob.data.ygrid);
    return (-(nu/(l_corr^2))-(nu/l_corr)).*M .+ (2nu/l).*M_plus;
end

function ∂M∂nu(prob :: MLEProblem)
    """
        Evaluates ∂M with respect to correlation smoothness, which by default
        uses finite difference, due to complexity of computing analytic formula.

        @benchmark => 6.449s, 874.63 MiB, over 16374180 allocations.
    """
    # unpack parameters
    l_corr = prob.l;
    nu = prob.nu;
    sigma_phi = prob.sigma_phi
    # centered difference
    ϵ = eps();
    # M(nu+eps)
    covfunc_plus = CovarianceFunction(2, Matern(l_corr, nu+ϵ, σ=sigma_phi));
    M_plus = (prob.sigma_phi^2).*apply(covfunc_plus, prob.data.xgrid, prob.data.ygrid);
    # M(nu-eps)
    covfunc_minus = CovarianceFunction(2, Matern(l_corr, nu-ϵ, σ=sigma_phi));
    M_minus = (prob.sigma_phi^2).*apply(covfunc_minus, prob.data.xgrid, prob.data.ygrid);
    # M'(nu) ≈ (1/2*eps)*(M(nu+eps) + M(nu-eps))
    res = (0.5./ϵ).*(M_plus.+M_minus);
    return res;
end

function L(prob :: MLEProblem)
    """
        Given a physics-based MLE problem instance in its 
        current state, unwraps the PDE parameters and returns
        the discretized (FDM) differential operator.

        The `L` operator is assumed to have homogeneous(=0) 
        Neumann condition on all sides of the domain. 
    """
    return advection_diffusion_reaction_homogeneous_neumann(prob)
end

function ∂L∂kappa(prob :: MLEProblem)
    return advection_diffusion_reaction_homogeneous_neumann_∂kappa(prob);
end

function ∂L∂v(prob :: MLEProblem)
    return advection_diffusion_reaction_homogeneous_neumann_∂v(prob);
end

function ∂L∂c(prob :: MLEProblem)
    return advection_diffusion_reaction_homogeneous_neumann_∂c(prob);
end

## Special derivatives

# PDE
function ∂Linv∂kappa(
    L_inv_buffer :: Matrix{Float64}, 
    prob :: MLEProblem
)
    """
        Given the SST problem in its current parametric state, 
        computes ∂L⁻/∂κ, L⁻ evaluated at the same parameters is
        assumed to be pre-computed.
    """
    # evaluate ∂L/∂κ
    ∂L∂kappa_eval = ∂L∂kappa(prob);
    res = -L_inv_buffer*∂L∂kappa_eval*L_inv_buffer;
    return res;
end

function ∂Linv∂kappa(
    L_eval :: SparseMatrixCSC,
    prob :: MLEProblem
)
    """
        (05/04/2023) Sparse solve version since storing L^-1 is expensive.
    """
    # evaluate ∂L/∂κ
    ∂L∂kappa_eval = ∂L∂kappa(prob);
    res = -L_eval\∂L∂kappa_eval;
    res = Matrix(((L_eval')\(res'))');
    return res;
end

function ∂Linv∂v(
    L_inv_buffer :: Matrix{Float64},
    prob :: MLEProblem
)
    """
        Computes ∂L⁻/∂v with respect to all parameters of velocity
        field v, stored in the same format as ∂v/∂θ.
    """
    # evaluate all derivatives ∂L∂v
    ∂L∂v_eval = ∂L∂v(prob);
    v_model_p1 = length(prob.v_model.v1_theta);
    v_model_p2 = length(prob.v_model.v2_theta);
    # create sparse operator matrix 
    res = Dict{String, Vector{Matrix{Float64}}}(
        "grad1" => Vector{Matrix{Float64}}(undef, v_model_p1),
        "grad2" => Vector{Matrix{Float64}}(undef, v_model_p2)
    );
    for i = 1:v_model_p1
        res["grad1"][i] = -L_inv_buffer*∂L∂v_eval["grad1"][i]*L_inv_buffer;
    end
    for i = 1:v_model_p2
        res["grad2"][i] = -L_inv_buffer*∂L∂v_eval["grad2"][i]*L_inv_buffer;
    end
    return res;
end

function ∂Linv∂c(
    L_inv_buffer :: Matrix{Float64},
    prob :: MLEProblem
)
    # evaluate ∂L/∂c
    ∂L∂c_eval = ∂L∂c(prob);
    
    res = -L_inv_buffer*∂L∂c_eval*L_inv_buffer;
    return res;
end

##################################################################
# MLE likelihood helpers
##################################################################
function K!(
    M :: Matrix{Float64},
    L_inv_buffer :: Matrix{Float64},
    prob :: MLEProblem
)
    """
        Given a physics-based MLE problem instance in its
        current parameteric state, computes the observed 
        covariance matrix with masking and noise. Masking 
        and noise-perturbation are never explicitly represented
        as matrices. The discretized operator and latent covariance
        should be pre-computed.

    """
    # nugget std
    sigma_u = prob.sigma_u;
    # compute inverse of L and sub-select => DL^-1
    DL_inv = (L_inv_buffer)[prob.data.obs_local_inds, :];
    # multiply to obtain K_obs
    res = DL_inv*M*DL_inv';
    # add noise by only modifying diagonal
    tmp = diag(res).+(sigma_u^2);
    # ressign diagonal
    res[diagind(res)] .= tmp;
    # force symmetric
    return Symmetric(res);
end


function log_likelihood!(
    K :: Symmetric{Float64, Matrix{Float64}},
    K_inv_u_buffer :: Vector{Float64},
    prob :: MLEProblem
)
    """
        Given a physics-based MLE problem instance in its 
        current state, computes log-likelihood. This method
        is exact. This method assumes zero mean.

        The observed covariance should be precomputed.

        Assigns computed values to buffer to avoid recomputing in 
        other functions:
            K^-1*u          also used in score

        @benchmark => 606.940 ms ±  17.614 ms
        Memory estimate: 159.46 MiB, allocs estimate: 12.

        neg_tol                 A matrix with only 1 element counting
                                numbers of negative determinants encountered.
    """
    # unpack parameters (use noise perturbed version)
    u = prob.u_noisy;
    logabsdet_K, flag = logabsdet(K);
    if flag < 0
        # if negative eigenvalue encountered
        @warn("Negative determinant encountered, check for conditioning. ");
    end
    # K^-1u
    tmp = K_inv_u_buffer;
    res = -0.5*logabsdet_K-0.5*u'*tmp;
    return res;
end

##################################################################
# Score function helpers
##################################################################

## Latent
function ∂K∂sigma(
    L_inv_buffer :: Matrix{Float64},
    M :: Matrix{Float64},
    prob :: MLEProblem
)
    # subselect to compute D*L^-1
    DL_inv = (L_inv_buffer)[prob.data.obs_local_inds, :];
    # evaluate ∂M∂sigma
    ∂M∂sigma_eval = ∂M∂sigma(M, prob);
    
    res = DL_inv*∂M∂sigma_eval*DL_inv';
    return res;
end

function ∂K∂l(
    L_inv_buffer :: Matrix{Float64},
    M :: Matrix{Float64},
    prob :: MLEProblem
)
    
    # subselect to compute D*L^-1
    DL_inv = (L_inv_buffer)[prob.data.obs_local_inds, :];
    # evaluate ∂M∂sigma
    ∂M∂l_eval = ∂M∂l(M, prob);
    
    res = DL_inv*∂M∂l_eval*DL_inv';
    return res;
end

function ∂K∂nu(
    L_inv_buffer :: Matrix{Float64},
    prob :: MLEProblem
)
    # subselect to compute D*L^-1
    DL_inv = (L_inv_buffer)[prob.data.obs_local_inds, :];
    # evaluate ∂M∂sigma
    ∂M∂nu_eval = ∂M∂nu(prob);
    
    res = DL_inv*∂M∂nu_eval*DL_inv';
    return res;
end

## PDE
function ∂K∂kappa(
    ∂Linv∂kappa :: Matrix{Float64},
    MLinv_t_buffer :: Matrix{Float64},
    LinvM_buffer :: Matrix{Float64},
    prob :: MLEProblem
)

    # subselect to reduce computational overhead
    Dinds = prob.data.obs_local_inds;
    # D * ∂Linv/∂κ => select rows
    D∂Linv∂kappa = ∂Linv∂kappa[Dinds, :];
    # M * Linv^T * D^T => select columns
    MLinvDt = MLinv_t_buffer[:, Dinds];
    term1 = D∂Linv∂kappa*MLinvDt;

    # D * Linv * M => select rows
    DLinvM = LinvM_buffer[Dinds, :];
    # D∂Linv∂kappa' => ∂Linv/∂κ^T * D^T
    term2 = DLinvM*D∂Linv∂kappa';
    return term1 + term2;
end

function ∂K∂v(
    ∂Linv∂v :: Dict{String, Vector{Matrix{Float64}}},
    MLinv_t_buffer :: Matrix{Float64},
    LinvM_buffer :: Matrix{Float64},
    prob :: MLEProblem
)
    v_model_p1 = length(prob.v_model.v1_theta);
    v_model_p2 = length(prob.v_model.v2_theta);
    res = Dict{String, Vector{Matrix{Float64}}}(
        "grad1" => Vector{Matrix{Float64}}(undef, v_model_p1),
        "grad2" => Vector{Matrix{Float64}}(undef, v_model_p2)
    );
    # subselect to reduce computational overhead
    Dinds = prob.data.obs_local_inds;
    # M * Linv^T * D^T => select columns
    MLinvDt = MLinv_t_buffer[:, Dinds];
    # D * Linv * M => select rows
    DLinvM = LinvM_buffer[Dinds, :];
    for i = 1:v_model_p1
        # D * ∂Linv/∂v => select rows
        D∂Linv∂v = ∂Linv∂v["grad1"][i][Dinds, :];
        term1 = D∂Linv∂v*MLinvDt;
        # D∂Linv∂v' => ∂Linv/∂v^T * D^T
        term2 = DLinvM*D∂Linv∂v';
        res["grad1"][i] = term1 + term2;
    end
    for i = 1:v_model_p2
        # D * ∂Linv/∂v => select rows
        D∂Linv∂v = ∂Linv∂v["grad2"][i][Dinds, :];
        term1 = D∂Linv∂v*MLinvDt;
        # D∂Linv∂v' => ∂Linv/∂v^T * D^T
        term2 = DLinvM*D∂Linv∂v';
        res["grad2"][i] = term1 + term2;
    end
    return res;
end

function ∂K∂c(
    ∂Linv∂c :: Matrix{Float64},
    MLinv_t_buffer :: Matrix{Float64},
    LinvM_buffer :: Matrix{Float64},
    prob :: MLEProblem
)

    # subselect to reduce computational overhead
    Dinds = prob.data.obs_local_inds;
    # D * ∂Linv/∂c => select rows
    D∂Linv∂c = ∂Linv∂c[Dinds, :];
    # M * Linv * D^T => select columns
    MLinvDt = MLinv_t_buffer[:, Dinds];
    term1 = D∂Linv∂c*MLinvDt;

    # D * Linv * M => select rows
    DLinvM = LinvM_buffer[Dinds, :];
    # D∂Linv∂c' => ∂Linv/∂c^T * D^T
    term2 = DLinvM*D∂Linv∂c';
    return term1 + term2;
end

## Score functions
function score∂sigma(
    L_inv_buffer :: Matrix{Float64},
    M :: Matrix{Float64},
    K :: Symmetric{Float64, Matrix{Float64}},
    Kinvu_buffer :: Vector{Float64},
    prob :: MLEProblem
) 
    # evaluate ∂K∂sigma
    ∂K∂sigma_eval = ∂K∂sigma(L_inv_buffer, M, prob);
    score = -0.5*tr(K\∂K∂sigma_eval)+0.5*Kinvu_buffer'*∂K∂sigma_eval*Kinvu_buffer;
    return score;
end
function score∂l(
    L_inv_buffer :: Matrix{Float64},
    M :: Matrix{Float64},
    K :: Symmetric{Float64, Matrix{Float64}},
    Kinvu_buffer :: Vector{Float64},
    prob :: MLEProblem
) 
    # evaluate ∂K∂l
    ∂K∂l_eval = ∂K∂l(L_inv_buffer, M, prob);
    score = -0.5*tr(K\∂K∂l_eval)+0.5*Kinvu_buffer'*∂K∂l_eval*Kinvu_buffer;
    return score;
end
function score∂nu(
    L_inv_buffer :: Matrix{Float64},
    K :: Symmetric{Float64, Matrix{Float64}},
    Kinvu_buffer :: Vector{Float64},
    prob :: MLEProblem
) 
    # evaluate ∂K∂nu
    ∂K∂nu_eval = ∂K∂nu(L_inv_buffer, prob);
    score = -0.5*tr(K\∂K∂nu_eval)+0.5*Kinvu_buffer'*∂K∂nu_eval*Kinvu_buffer;
    return score;
end
function score∂kappa(
    ∂Linv∂kappa :: Matrix{Float64},
    MLinv_T_buffer :: Matrix{Float64},
    LinvM_buffer :: Matrix{Float64},
    K :: Symmetric{Float64, Matrix{Float64}},
    Kinvu_buffer :: Vector{Float64},
    prob :: MLEProblem
) 
    # evaluate ∂K∂kappa
    ∂K∂kappa_eval = ∂K∂kappa(∂Linv∂kappa, MLinv_T_buffer, LinvM_buffer, prob);
    score = -0.5*tr(K\∂K∂kappa_eval)+0.5*Kinvu_buffer'*∂K∂kappa_eval*Kinvu_buffer;
    return score;
end
function score∂v(
    ∂Linv∂v :: Dict{String, Vector{Matrix{Float64}}},
    MLinv_T_buffer :: Matrix{Float64},
    LinvM_buffer :: Matrix{Float64},
    K :: Symmetric{Float64, Matrix{Float64}},
    Kinvu_buffer :: Vector{Float64},
    prob :: MLEProblem
)
    # preallocate dictionary to store scores
    v_model_p1 = length(prob.v_model.v1_theta);
    v_model_p2 = length(prob.v_model.v2_theta);
    res = Dict{String, Vector{Float64}}(
        "grad1" => Vector{Float64}(undef, v_model_p1),
        "grad2" => Vector{Float64}(undef, v_model_p2)
    );
    # evaluate ∂K∂v
    ∂K∂v_eval = ∂K∂v(∂Linv∂v, MLinv_T_buffer, LinvM_buffer, prob);
    for i = 1:v_model_p1
        res["grad1"][i] = -0.5*tr(K\∂K∂v_eval["grad1"][i])+0.5*Kinvu_buffer'*∂K∂v_eval["grad1"][i]*Kinvu_buffer;
    end
    for i = 1:v_model_p2
        res["grad2"][i] = -0.5*tr(K\∂K∂v_eval["grad2"][i])+0.5*Kinvu_buffer'*∂K∂v_eval["grad2"][i]*Kinvu_buffer;
    end
    return res;
end

function score∂c(
    ∂Linv∂c :: Matrix{Float64},
    MLinv_T_buffer :: Matrix{Float64},
    LinvM_buffer :: Matrix{Float64},
    K :: Symmetric{Float64, Matrix{Float64}},
    Kinvu_buffer :: Vector{Float64},
    prob :: MLEProblem
) 
    # evaluate ∂K∂kappa
    ∂K∂c_eval = ∂K∂c(∂Linv∂c, MLinv_T_buffer, LinvM_buffer, prob);
    score = -0.5*tr(K\∂K∂c_eval)+0.5*Kinvu_buffer'*∂K∂c_eval*Kinvu_buffer;
    return score;
end

function score(
    M :: Matrix{Float64},
    prob :: MLEProblem
)
    """
        Given an MLE problem in its current parameteric state, compute
        the score (stored as a vector) with respect to all trainable 
        parameters in `update_manual`.

        * WARNING: uses exact matrices, HODLR is recommended for large 
        problems.
    """
    # evaluate operator
    L_eval = L(prob);
    M_eval = M;
    L_inv_eval = L_eval\I(size(L_eval, 1));

    # evaluate observation covariance
    K_eval = K!(M_eval, L_inv_eval, prob);
    K_inv_u_eval = K_eval\prob.u_noisy[:];

    # compute score 
    
    # initialize with full vector and NaN's, then filter
    tmp = mask_parameters(prob);

    # compute new buffers if PDE parameters are being optimized
    if "kappa" in prob.update_manual || "v1_theta" in prob.update_manual || "v2_theta" in prob.update_manual || "c" in prob.update_manual
        M_prod_L_inv_T_eval = M_eval*transpose(L_inv_eval);
        # L is sparse, L\M has better performance than L^-1*M
        L_inv_prod_M_eval = L_eval\M_eval;
    end

    # if not in the update manual, no need to compute gradient
    if "sigma_phi" in prob.update_manual
        tmp[1] = score∂sigma(L_inv_eval, M_eval, K_eval, K_inv_u_eval, prob);
    end

    if "nu" in prob.update_manual
        tmp[2] = score∂nu(L_inv_eval, K_eval, K_inv_u_eval, prob);
    end

    if "l" in prob.update_manual
        tmp[3] = score∂l(L_inv_eval, M_eval, K_eval, K_inv_u_eval, prob);
    end

    if "kappa" in prob.update_manual
        # compute ∂Linv∂kappa
        ∂Linv∂kappa_eval = ∂Linv∂kappa(L_inv_eval, prob);
        tmp[4] = score∂kappa(∂Linv∂kappa_eval, M_prod_L_inv_T_eval, L_inv_prod_M_eval, K_eval, K_inv_u_eval, prob);
    end

    if "v1_theta" in prob.update_manual || "v2_theta" in prob.update_manual
        # number of parameters
        v1_params = length(prob.v_model.v1_theta);
        v2_params = length(prob.v_model.v2_theta);

        # compute ∂Linv∂v
        ∂Linv∂v_eval = ∂Linv∂v(L_inv_eval, prob);
        # compute score∂v
        score∂v_eval = score∂v(∂Linv∂v_eval, M_prod_L_inv_T_eval, L_inv_prod_M_eval, K_eval, K_inv_u_eval, prob);
        if "v1_theta" in prob.update_manual
            tmp[5:5+v1_params-1] .= score∂v_eval["grad1"][:];
        end

        if "v2_theta" in prob.update_manual
            tmp[5+v1_params:5+v1_params+v2_params-1] .= score∂v_eval["grad2"][:];
        end
    end

    if "c" in prob.update_manual
        # compute ∂Linv∂c
        ∂Linv∂c_eval = ∂Linv∂c(L_inv_eval, prob);
        score∂c_eval = score∂c(∂Linv∂c_eval, M_prod_L_inv_T_eval, L_inv_prod_M_eval, K_eval, K_inv_u_eval, prob);
        tmp[end] = score∂c_eval;
    end
    # filter and update G (negative score)
    res = filter(!isnan, tmp);
    return res;
end


######################################################################
# Warm start strategies
######################################################################
function warm_start_esteq!(
    prob :: MLEProblem,
    param_constraints :: Matrix{Float64}
)
    """
        Perform a warm start by solving the estimating equations detailed in 
        ``An Inversion-Free Estimating Equations Approach for Gaussian Process Models``.
        See equation (7), we minimize the negative of (7).

        * WARNING: exact method, not recommended for large-scale optimization, use 
        the HODLR version instead.
    """
    # unpack
    u_obs = prob.u_noisy;
    nx, ny = length(prob.data.xgrid), length(prob.data.ygrid);
    # define optimization problem
    θ_init = dump_trainable_parameters(prob);
    # pre-evaluate M
    M_eval = M(prob);
    function _f!(θ :: Vector{Float64})
        """
            Internal function used for evaluating the estimating optimization problem.
        """
        # update problem parametric state
        update!(prob, θ);
        # only evaluate M if latent parameters are being optimized
        if "sigma_phi" in prob.update_manual || "l" in prob.update_manual || "nu" in prob.update_manual
            M_eval = M(prob);
        end
        # evaluate -h(θ)
        L_eval = L(prob);
        L_inv_eval = L_eval\I(nx*ny);
        K_eval = K!(M_eval, L_inv_eval, prob);
        K_mul_u_eval = K_eval*u_obs;
        # trace is sum of eigenvalues
        trace_K_sq_eval = sum(eigvals(K_eval).^2);
        return 0.5*trace_K_sq_eval - u_obs'*K_mul_u_eval;
    end
    _lower = param_constraints[:, 1];
    _upper = param_constraints[:, 2];
    # no gradient supplied (for a warm start, providing gradient is overkill)
    println("* Computing estimating equations warm start ... ")
    res = optimize(
        _f!,  
        _lower, 
        _upper, 
        θ_init, Fminbox(), 
        Optim.Options(
            outer_iterations = 10,
            iterations=50,
            g_tol = 1e-3,
            store_trace = false,
            show_trace = true,
            time_limit = 300
        )
    );
    θ_started = Optim.minimizer(res);
    update!(prob, θ_started);
    println("* Estimating equations warm start completed. ")
end


function warm_start_nonlinear_ls!(
    prob :: MLEProblem,
    param_constraints :: Matrix{Float64}
)
    """
        Perform a warm start by solving a nonlinear least squares problem.
        This stratgy attempts to minimize:
            || D*[L(θ)]^-1*ϕ - u_obs ||_2 over θ
        where ϕ is a random sample from the Matern random field.

        This warm strategy only applies when latent parameters are fixed. 

        * WARNING: this warm start routine uses `Cholesky()` to sample from
        Matern field, which may be expensive for large-scale problem. Consider
        supplying a pre-computed Matern sample. 

    """
    if "sigma_phi" in prob.update_manual || "l" in prob.update_manual || "nu" in prob.update_manual
        @warn "Cannot perform nonlinear least squares with trainable latent parameters. Exiting ... "
        return;
    end
    # use non-noisy observations
    u_obs = prob.data.u_observed;
    # sample from Matern kernel
    grf = GaussianRandomField(prob.covfunc, GaussianRandomFields.Cholesky(), prob.data.xgrid, prob.data.ygrid);
    phi_sample = GaussianRandomFields.sample(grf)[:];
    # define optimization problem
    θ_init = dump_trainable_parameters(prob);
    function _f!(θ :: Vector{Float64})
        """
            Internal function used for evaluating L^2 distance of observations and 
            parameterized PDE operator.
        """
        # update problem parametric state
        update!(prob, θ);
        # evaluate L^2 distance
        return norm((L(prob)\phi_sample)[prob.data.obs_local_inds, :]-u_obs);
    end
    _lower = param_constraints[:, 1];
    _upper = param_constraints[:, 2];
    # no gradient supplied (for a warm start, providing gradient is overkill)
    println("* Computing nonlinear least squares warm start ... ")
    res = optimize(
        _f!,  
        _lower, 
        _upper, 
        θ_init, Fminbox(), 
        Optim.Options(
            outer_iterations = 10,
            iterations=50,
            g_tol = 1e-3,
            store_trace = false,
            show_trace = true,
            time_limit = 300
        )
    );
    θ_started = Optim.minimizer(res);
    update!(prob, θ_started);
    println("* Nonlinear least squares warm start completed. ")
end


######################################################################
# Wrapper functions for `Optim.optimize`
######################################################################
function update!(prob :: MLEProblem, theta :: Vector)
    """
        A mutating function that updates the MLE problem parameters
        by checking the update manual. The (fixed) ordering of parameters in
        `theta` is specified in the comment at the start of the module.

        `theta` must be of the same length as the number of trainable parameters.
    """ 
    # find parameter indices  
    mask = mask_parameters(prob);
    non_mask_inds = findall(!isnan, mask);
    # update only appropriate parameters
    mask[non_mask_inds] .= theta;
    
    # sizes of velocity model
    v_model_p1 = length(prob.v_model.v1_theta);
    v_model_p2 = length(prob.v_model.v2_theta);
    
    # unpack the vector and update the problem
    sigma_phi, nu, l = mask[1], mask[2], mask[3];
    kappa = mask[4];
    v1_theta, v2_theta = mask[5:5+v_model_p1-1], mask[(5+v_model_p1):(5+v_model_p1+v_model_p2-1)];
    c = mask[end];

    if length(prob.update_manual) > 0
        # update components of the problem based on update manual
        # if update manual is not empty. 
        if "sigma_phi" in prob.update_manual
            prob.sigma_phi = sigma_phi;
        end

        if "l" in prob.update_manual
            prob.l = l;
        end

        if "nu" in prob.update_manual
            prob.nu = nu;
        end

        if "kappa" in prob.update_manual
            prob.kappa = kappa;
        end

        if "v1_theta" in prob.update_manual
            prob.v_model.v1_theta[:] .= v1_theta;
        end

        if "v2_theta" in prob.update_manual
            prob.v_model.v2_theta[:] .= v2_theta;
        end

        if "c" in prob.update_manual
            prob.c = c;
        end
    end
end

function solve!(
    prob :: MLEProblem,
    param_constraints :: Matrix{Float64},
    warm_start :: Union{String, Nothing}=nothing
)
    """
        Main optimization routine that optimizes a (subset) of the 
        PDE/latent parameters stored in the SST partition Gaussian process
        problem. This method is mutating and updates the problem parametric
        state repeatedly until convergence of the gradient.

        initialize the parameters as a vector, overwrites `prob` 
        parameters (contained in `update_manual`) created outside of this function.
        static variables should be initialized outside of the optimization.
        this function only initializes modifiable parameters by different possible warm \
        start strategies, default uses the values initialized outside.

        Inputs:
            prob                        an MLE instance with p total parameters. A subset
                                        of p parameters will be optimized.

            param_constraints           likely regions of optimal parameters, specified as
                                        a (p x 2) matrix.

            warm_start                  A string selecting from a set of pre-defined choices
                                        of warm start strategies. Defaults to :nothing, available
                                        options include:
                                            * :nothing      uses parameters supported when the 
                                                            problem was created.

                                            * random:       uniformly randomly sample from the 
                                                            allowed parameter ranges.
                                            * nlls:         nonlinear least squares

                                            * esteq:        solves estimating eqautions, detailed in:
                                            ``An Inversion-Free Estimating Equations 
                                            Approach for Gaussian Process Models``
    """
    # number of trainable parameters
    num_trainable = size(param_constraints, 1);
    # compute warm start
    if warm_start !== nothing
        if warm_start == "nlls"
            # solves nonlinear least squares problem
            warm_start_nonlinear_ls!(prob, param_constraints);
        elseif warm_start == "random"
            tmp = zeros(num_trainable);
            for i = eachindex(tmp)
                # sample randomly
                tmp_bounds = param_constraints[i, :];
                tmp[i] = rand(Uniform(tmp_bounds[1], tmp_bounds[2]));
            end
            update!(prob, tmp);
        elseif warm_start == "esteq"
            # solves the set of estimating equations
            warm_start_esteq!(prob, param_constraints);
        end
    end

    # initial parameters
    _params_init = dump_trainable_parameters(prob);

    # precompute M if we do not intend on changing latent parameters (only need computed once)
    M_flag = "sigma_phi" in prob.update_manual || "l" in prob.update_manual || "nu" in prob.update_manual;
    if !M_flag
        M_eval = M(prob);
    else
        M_eval = nothing;
    end

    # function that computes both objective and gradient at once, reusing calculations
    function _fg!(F, G, theta)
        # first update problem state
        update!(prob, theta);
        # ----------------------------------------
        # Common computations
        # ----------------------------------------
        # evaluate operator
        L_eval = L(prob);
        L_inv_eval = L_eval\I(size(L_eval, 1));
        # if not optimizing latent parameters, no need to recompute M
        if M_flag
            M_eval = M(prob);
        end
        # evaluate observation covariance
        K_eval = K!(M_eval, L_inv_eval, prob);
        K_inv_u_eval = K_eval\prob.u_noisy[:];

        # compute score first
        # ----------------------------------------
        # Score (only those in `update_manual`)
        # ----------------------------------------
        if G !== nothing
            # initialize with full vector and NaN's, then filter
            tmp = mask_parameters(prob);

            # compute new buffers if PDE parameters are being optimized
            if "kappa" in prob.update_manual || "v1_theta" in prob.update_manual || "v2_theta" in prob.update_manual || "c" in prob.update_manual
                M_prod_L_inv_T_eval = M_eval*transpose(L_inv_eval);
                # L is sparse, L\M has better performance than L^-1*M
                L_inv_prod_M_eval = L_eval\M_eval;
            end

            # if not in the update manual, no need to compute gradient
            if "sigma_phi" in prob.update_manual
                tmp[1] = score∂sigma(L_inv_eval, M_eval, K_eval, K_inv_u_eval, prob);
            end

            if "nu" in prob.update_manual
                tmp[2] = score∂nu(L_inv_eval, K_eval, K_inv_u_eval, prob);
            end

            if "l" in prob.update_manual
                tmp[3] = score∂l(L_inv_eval, M_eval, K_eval, K_inv_u_eval, prob);
            end

            if "kappa" in prob.update_manual
                # compute ∂Linv∂kappa
                ∂Linv∂kappa_eval = ∂Linv∂kappa(L_inv_eval, prob);
                tmp[4] = score∂kappa(∂Linv∂kappa_eval, M_prod_L_inv_T_eval, L_inv_prod_M_eval, K_eval, K_inv_u_eval, prob);
            end

            if "v1_theta" in prob.update_manual || "v2_theta" in prob.update_manual
                # number of parameters
                v1_params = length(prob.v_model.v1_theta);
                v2_params = length(prob.v_model.v2_theta);

                # compute ∂Linv∂v
                ∂Linv∂v_eval = ∂Linv∂v(L_inv_eval, prob);
                # compute score∂v
                score∂v_eval = score∂v(∂Linv∂v_eval, M_prod_L_inv_T_eval, L_inv_prod_M_eval, K_eval, K_inv_u_eval, prob);
                if "v1_theta" in prob.update_manual
                    tmp[5:5+v1_params-1] .= score∂v_eval["grad1"][:];
                end

                if "v2_theta" in prob.update_manual
                    tmp[5+v1_params:5+v1_params+v2_params-1] .= score∂v_eval["grad2"][:];
                end
            end

            if "c" in prob.update_manual
                # compute ∂Linv∂c
                ∂Linv∂c_eval = ∂Linv∂c(L_inv_eval, prob);
                score∂c_eval = score∂c(∂Linv∂c_eval, M_prod_L_inv_T_eval, L_inv_prod_M_eval, K_eval, K_inv_u_eval, prob);
                tmp[end] = score∂c_eval;
            end
            # filter and update G (negative score)
            G[:] .= -filter(!isnan, tmp);
        end

        # ----------------------------------------
        # Likelihood
        # ----------------------------------------
        if F !== nothing
            return -log_likelihood!(K_eval, K_inv_u_eval, prob);
        end
    end

    # optimization
    _lower = param_constraints[:, 1];
    _upper = param_constraints[:, 2];
    # initialize optimization algorithm
    _optimization_alg = LBFGS();

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

# helper functions for formatting
function mask_parameters(prob :: MLEProblem)
    """
        Puts all trainable parameters contained in the MLE problem in a vector,
        masking the non-trainable positions with `NaN`.
    """
    # number of velocity params
    v1_params = length(prob.v_model.v1_theta);
    v2_params = length(prob.v_model.v2_theta);
    # compute length of vector
    n_params = 1+1+1+1+v1_params+v2_params+1;
    # create initial NaN vector
    params = fill(NaN, n_params);
    # replace with trainable values 
    if "sigma_phi" in prob.update_manual
        params[1] = prob.sigma_phi;
    end
    if "l" in prob.update_manual
        params[2] = prob.l;
    end
    if "nu" in prob.update_manual
        params[3] = prob.nu;
    end
    if "kappa" in prob.update_manual
        params[4] = prob.kappa;
    end
    if "v1_theta" in prob.update_manual
        params[5:5+v1_params-1] .= prob.v_model.v1_theta;
    end
    if "v2_theta" in prob.update_manual
        params[5+v1_params:5+v1_params+v2_params-1] .= prob.v_model.v2_theta;
    end
    if "c" in prob.update_manual
        params[end] = prob.c;
    end
    return params;
end

function dump_all_parameters(prob :: MLEProblem)
    """
        Puts all parameters in teh MLE problem in a vector, both trained and 
        untrained. 
    """
    v1_params, v2_params = length(prob.v_model.v1_theta), length(prob.v_model.v2_theta);
    params = zeros(Float64, 1+1+1+1+v1_params+v2_params+1);
    params[1] = prob.sigma_phi;
    params[2] = prob.nu;
    params[3] = prob.l;
    params[4] = prob.kappa;
    params[5:5+v1_params-1] .= prob.v_model.v1_theta;
    params[5+v1_params:5+v1_params+v2_params-1] .= prob.v_model.v2_theta;
    params[end] = prob.c;
    return params;
end

function dump_trainable_parameters(prob :: MLEProblem)
    """
        Puts all trainable parameters contained in the MLE problem in a vector,
        conforming to the default ordering (see beginning of this file).
    """
    params = mask_parameters(prob);
    # only return non-NaN values
    result = filter(x -> !isnan(x), params);
    return result;
end

########################################################################
# Global optimization pre-processing, warm starts
########################################################################
function boundary_velocity(prob1 :: MLEProblem, prob2 :: MLEProblem)
    """
        Given two MLE problems in their respective parameteric states, 
        determine (if any) the partition boundary (by intersecting spatial
        locations) and respectively evaluate velocity field as a function of space.

        The partition boundary must not be masked by cloud.
    """

end

######################################################################
# Computing Hessian / Fisher information matrix (exact)
######################################################################
function fisher(prob :: MLEProblem)
    """
        Given an MLE problem in its current parameteric state,
        computes the full Fisher information matrix of all trained
        parameters.

        Currently only supports PDE parameters.
    """
    # evaluate all matrix derivatives in the update manual
    update_manual = prob.update_manual;
    num_trained = length(update_manual);
    all_∂K = Any[];
    # precompute some matrices
    M_eval = M(prob);
    L_eval = L(prob);
    LinvM_eval = L_eval\M_eval;
    MLinv_t_eval = (L_eval\M_eval)';
    for i = eachindex(update_manual)
        error()
    end
end





######################################################################
# Post processing, imputations
######################################################################
function velocity_field(prob :: MLEProblem)
    """
        Given an MLE problem in its current parameter state, evaluates 
        the velocity function on all points in its effective spatial domain.
    """
    xgrid, ygrid = prob.data.xgrid, prob.data.ygrid;
    nx, ny = length(xgrid), length(ygrid);
    v1 = zeros(Float64, nx, ny);
    v2 = zeros(Float64, nx, ny);
    for i = 1:nx
        for j = 1:ny
            x = xgrid[i]; y = ygrid[j];
            point = [x, y];
            # evaluate 
            field_eval = prob.v_model(point);
            v1[i, j] = field_eval[1];
            v2[i, j] = field_eval[2];
        end
    end
    return v1, v2;
end


function imputation_statistics(prob :: MLEProblem)
    """
        Given a MLE problem in its parameteric state, Computes posterior
        mean and covariance for generating imputed values.

        * WARNING: use exact matrices. Consider using HODLR format for
        global problem.
    """
    n_obs = length(prob.u_noisy);
    
    n_full = length(prob.data.u_full);
    # number of hidden values
    n_hidden = n_full - n_obs;
    
    # observation noise
    sigma_u = prob.sigma_u;
    # observations 
    u_obs = prob.u_noisy;
    # precompute matrices
    L_eval = L(prob);
    M_eval = M(prob);

    # form full covariance matrix (not formed in `K!` evaluation since during optimization
    # it needs to be evaluated many repeatedly, but full covariance matrix is fine here).

    # evaluate L^-1*M*L^-T with two back solves
    K_full_eval = L_eval\M_eval;
    K_full_eval = (L_eval\K_full_eval')';

    # if there are no observations, by default return prior distribution
    if iszero(n_obs)
        @assert n_hidden == n_full
        u_hid_mean = zeros(n_hidden);
        u_hid_cov = Symmetric(K_full_eval);
        @warn "... All observations are masked, returning prior statistics. "
        return u_hid_mean, u_hid_cov;
    end

    # projections
    obs_local_idx = prob.data.obs_local_inds;
    hid_local_idx = prob.data.mask_local_inds;
    K_hidhid = K_full_eval[hid_local_idx, hid_local_idx];
    K_hidobs = K_full_eval[hid_local_idx, obs_local_idx]; 
    K_obsobs = K_full_eval[obs_local_idx, obs_local_idx];

    # add noise to observation covariance
    tmp = K_obsobs[diagind(K_obsobs)];
    K_obsobs[diagind(K_obsobs)] .= tmp .+ (sigma_u).^2;
    # find posterior mean and covariance
    u_hid_mean = K_hidobs*(K_obsobs\u_obs);
    u_hid_cov = Symmetric(
        K_hidhid - K_hidobs*(K_obsobs\K_hidobs')
    );
    return u_hid_mean, u_hid_cov;
end

function sample_imputations(
    u_hid_mean :: Vector{Float64}, 
    u_hid_cov :: Symmetric{Float64},
    n :: Int64
)
    """
        Given mean and covariance, generates N samples 
        from the predictive distribution.
    """
    # create density
    distrib = MvNormal(vec(u_hid_mean), u_hid_cov);
    res = rand(distrib, n);
    return res;
end

function impute!(prob :: MLEProblem)
    """
        Given an MLE problem in its current parametric state, generates
        a random imputation from the posterior distribution and store
        it in the problem instance.
    """
    # compute posterior statistics
    u_hid_mean, u_hid_cov = imputation_statistics(prob);
    # generate one sample from imputation distribution
    u_hid_sample = sample_imputations(u_hid_mean, u_hid_cov, 1);
    # put back to data
    prob.data.u_imputations[:] .= u_hid_sample;
end


"""
    (03/18/2023)
    
    2d finite difference (FDM) differential operators. The spatial grid
    is assumed to be uniform in each axis and the PDE parameters are 
    parameterized. The ordering of points is by default column-wise
    flattening. 

"""
######################################################################
# PDE operators 
######################################################################
function reaction_diffusion_homogeneous_neumann(
    xgrid,
    ygrid,
    kappa :: Float64,
    c :: Float64
)
    """
        Diffusion-reaction operator with homogeneous 
        Neumann boundary condition using custom parameters 
        specifying diffusion and reaction terms.

        General form:
            Lu = -∇⋅(κ∇u) + cu
    """
    # step sizes 
    hx = xgrid[2]-xgrid[1];
    hy = ygrid[2]-ygrid[1];

    nx = length(xgrid);
    ny = length(ygrid);

    # parameters for operator
    # U_i,j
    p_a = (2kappa/hx^2)+(2kappa/hy^2)+c;
    # U_i+1,j
    p_b = -kappa/hx^2;
    # U_i-1,j
    p_c = -kappa/hx^2; 
    # U_i,j+1
    p_d = -kappa/hy^2;
    # U_i,j-1
    p_e = -kappa/hy^2;

    # linearized index query
    linidx(i, j) = sub2ind((nx,ny),i,j);
    # only store three vectors
    # - row indices
    # - col indices
    # - entry at (row,col)
    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();
    for j = 1:ny
        for i = 1:nx
            # evaluate velocity and derivatives with respect to each parameters
            # row index is always column major in-order 
            idx = linidx(i, j);
            if i == 1 && j == 1
                # ********************
                # U_i,j dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);
                # ********************
                # U_i+1,j dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                push!(entry, p_b + p_c);
                # ********************
                # U_i,j+1 dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                push!(entry, p_d + p_e);
            elseif i == 1 && j == ny
                # ********************
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                push!(entry, p_d + p_e);
                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);
                # ********************
                # L[idx, linidx(i+1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                push!(entry, p_b + p_c);
            elseif i == nx && j == 1
                # ********************
                # L[idx, linidx(i-1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                push!(entry, p_b + p_c);
                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);
                # ********************
                # L[idx, linidx(i,j+1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                push!(entry, p_d + p_e);
            elseif i == nx && j == ny
                # ********************
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                push!(entry, p_d + p_e);
                # ********************
                # L[idx, linidx(i-1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                push!(entry, p_b + p_c);
                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);
            else
                # ----------------------------------------
                # boundary, non-corner
                # ----------------------------------------
                if i == 1 && 2 <= j <= ny-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_e);
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b + p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b + p_c);
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d);
                elseif i == nx && 2 <= j <= ny-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_e);
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_b + p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_b + p_c);
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d);
                elseif j == 1 && 2 <= i <= nx-1
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_c);
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b);
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d + p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d + p_e);
                elseif j == ny && 2 <= i <= nx-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_d + p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_d + p_e);
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_c);
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b);
                else
                    # ----------------------------------------
                    # Within boundary
                    # ----------------------------------------

                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_e);
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_c);
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b);
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d);
                end
            end
        end
    end
    # create sparse operator matrix 
    L = sparse(row_ind, col_ind, entry);
    return L;
end

function advection_diffusion_reaction_homogeneous_neumann(
    prob :: MLEProblem
)
    """
        Given an MLE problem instance and its current parameteric
        states, creates a 2d ADR differential operator assuming
        homogeneous Neumann boundary conditions.
    """
    # step sizes 
    xgrid = prob.data.xgrid;
    ygrid = prob.data.ygrid;
    hx = xgrid[2]-xgrid[1];
    hy = ygrid[2]-ygrid[1];

    nx = length(xgrid);
    ny = length(ygrid);

    # unpack mutable PDE parameters
    kappa = prob.kappa;
    v_model = prob.v_model;
    c = prob.c;
    # parameters for operator
    # U_i,j
    p_a = (2kappa/hx^2)+(2kappa/hy^2)+c;
    # linearized index query
    linidx(i, j) = sub2ind((nx,ny),i,j);
    # only store three vectors
    # - row indices
    # - col indices
    # - entry at (row,col)

    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();
    for j = 1:ny
        for i = 1:nx
            # physical domain location
            x1 = xgrid[i];
            x2 = ygrid[j];
            # evaluate velocity and derivatives with respect to each parameters
            v_eval = v_model([x1, x2]);
            # U_i+1,j
            p_b = (v_eval[1]/2hx-kappa/hx^2);
            # U_i-1,j
            p_c = (-v_eval[1]/2hx-kappa/hx^2); 
            # U_i,j+1
            p_d = (v_eval[2]/2hy-kappa/hy^2);
            # U_i,j-1
            p_e = (-v_eval[2]/2hy-kappa/hy^2);
            
            # row index is always column major in-order 
            idx = linidx(i, j);
            if i == 1 && j == 1
                # ********************
                # U_i,j dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);
                # ********************
                # U_i+1,j dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                push!(entry, p_b + p_c);
                # ********************
                # U_i,j+1 dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                push!(entry, p_d + p_e);
            elseif i == 1 && j == ny
                # ********************
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                push!(entry, p_d + p_e);
                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);
                # ********************
                # L[idx, linidx(i+1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                push!(entry, p_b + p_c);
            elseif i == nx && j == 1
                # ********************
                # L[idx, linidx(i-1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                push!(entry, p_b + p_c);
                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);
                # ********************
                # L[idx, linidx(i,j+1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                push!(entry, p_d + p_e);
            elseif i == nx && j == ny
                # ********************
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                push!(entry, p_d + p_e);
                # ********************
                # L[idx, linidx(i-1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                push!(entry, p_b + p_c);
                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);
            else
                # ----------------------------------------
                # boundary, non-corner
                # ----------------------------------------
                if i == 1 && 2 <= j <= ny-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_e);
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b + p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b + p_c);
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d);
                elseif i == nx && 2 <= j <= ny-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_e);
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_b + p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_b + p_c);
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d);
                elseif j == 1 && 2 <= i <= nx-1
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_c);
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b);
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d + p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d + p_e);
                elseif j == ny && 2 <= i <= nx-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_d + p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_d + p_e);
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_c);
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b);
                else
                    # ----------------------------------------
                    # Within boundary
                    # ----------------------------------------

                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_e);
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_c);
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b);
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d);
                end
            end
        end
    end
    # create sparse operator matrix 
    L = sparse(row_ind, col_ind, entry);
    return L;
end

# ******************************************************************
function advection_diffusion_reaction_homogeneous_neumann_∂v(
    prob :: MLEProblem
)
    """
        Given an MLE problem instance and its current parameteric
        states, creates the matrix derivatives of a 2d ADR differential 
        operator assuming homogeneous Neumann boundary conditions, with 
        respect to a parameterization of velocity.
    """
    # step sizes 
    xgrid = prob.data.xgrid;
    ygrid = prob.data.ygrid;
    hx = xgrid[2]-xgrid[1];
    hy = ygrid[2]-ygrid[1];

    nx = length(xgrid);
    ny = length(ygrid);

    # unpack mutable PDE parameters
    v_model = prob.v_model;
    # number of parameters in velocity model
    v_model_p1 = length(v_model.v1_theta);
    v_model_p2 = length(v_model.v2_theta);

    # no dependence on any velocity components
    p_a_dv = 0.0;

    # linearized index query
    linidx(i, j) = sub2ind((nx,ny),i,j);
    # only store three vectors
    # - row indices
    # - col indices
    # - entry at (row,col)

    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    # velocity derivatives
    entry_dv = Dict{String, Vector{Vector{Float64}}}(
        "grad1" => [Vector{Float64}() for i in 1:v_model_p1],
        "grad2" => [Vector{Float64}() for i in 1:v_model_p2]
    );

    # precompute certain values and initialize a buffer for gradient storage during assembly

    # U_i+1,j
    p_b_dv1_buffer = zeros(Float64, v_model_p1);
    p_b_dv2_buffer = zeros(Float64, v_model_p2);
    
    # U_i-1,j
    p_c_dv1_buffer = zeros(Float64, v_model_p1);
    p_c_dv2_buffer = zeros(Float64, v_model_p2);
    
    # U_i,j+1
    p_d_dv1_buffer = zeros(Float64, v_model_p1);
    p_d_dv2_buffer = zeros(Float64, v_model_p2);
    
    # U_i,j-1
    p_e_dv1_buffer = zeros(Float64, v_model_p1);
    p_e_dv2_buffer = zeros(Float64, v_model_p2);


    for j = 1:ny
        for i = 1:nx
            # physical domain location
            x1 = xgrid[i];
            x2 = ygrid[j];
            # evaluate velocity and derivatives with respect to each parameters
            v_eval_grad = ∂v∂θ(v_model, [x1, x2]);

            # ∂U_i+1,j/∂θ_1
            p_b_dv1_buffer[:] .= v_eval_grad["grad1"] ./ 2hx;
            # ∂U_i+1,j/∂θ_2
            p_b_dv2_buffer[:] .= 0.0;

            # ∂U_i-1,j/∂θ_1
            p_c_dv1_buffer[:] .= -p_b_dv1_buffer[:];
            # ∂U_i-1,j/∂θ_2
            p_c_dv2_buffer[:] .= 0.0;
 
            # ∂U_i,j+1/∂θ_1
            p_d_dv1_buffer[:] .= 0.0;
            # ∂U_i,j+1/∂θ_2
            p_d_dv2_buffer[:] .= v_eval_grad["grad2"] ./ 2hy;

            # ∂U_i,j-1/dθ_1
            p_e_dv1_buffer[:] .= 0.0;
            # ∂U_i,j-1/dθ_2
            p_e_dv2_buffer[:] .= -p_d_dv2_buffer[:];

            # row index is always column major in-order 
            idx = linidx(i, j);

            if i == 1 && j == 1
                # ********************
                # U_i,j dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                # --------------------
                # Derivatives
                # --------------------
                # velocity components, v1
                for idx = 1:v_model_p1
                    push!(entry_dv["grad1"][idx], p_a_dv);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    push!(entry_dv["grad2"][idx], p_a_dv);
                end


                # ********************
                # U_i+1,j dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                # --------------------
                # Derivatives
                # --------------------
                # velocity components, v1
                for idx = 1:v_model_p1
                    # p_b_dv1 + p_c_dv1 = 0
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    # p_b_dv2 + p_c_dv2 = 0
                    push!(entry_dv["grad2"][idx], 0.0);
                end

                # ********************
                # U_i,j+1 dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                # --------------------
                # Derivatives
                # --------------------
                # velocity components, v1
                for idx = 1:v_model_p1
                    # p_d_dv1 + p_e_dv1 = 0
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    # p_d_dv2 + p_e_dv2 = 0
                    push!(entry_dv["grad2"][idx], 0.0);
                end

            elseif i == 1 && j == ny
                # ********************
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                # --------------------
                # Derivatives
                # --------------------
                # velocity components, v1
                for idx = 1:v_model_p1
                    # p_d_dv1 + p_e_dv1 = 0
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    # p_d_dv2 + p_e_dv2 = 0
                    push!(entry_dv["grad2"][idx], 0.0);
                end

                
                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                # --------------------
                # Derivatives
                # --------------------
                # velocity components, v1
                for idx = 1:v_model_p1
                    push!(entry_dv["grad1"][idx], p_a_dv);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    # p_d_dv2 + p_e_dv2 = 0
                    push!(entry_dv["grad2"][idx], p_a_dv);
                end

                # ********************
                # L[idx, linidx(i+1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                # --------------------
                # Derivatives
                # --------------------
                # velocity components, v1
                for idx = 1:v_model_p1
                    # p_b_dv1 + p_c_dv1 = 0
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    # p_b_dv2 + p_c_dv2 = 0
                    push!(entry_dv["grad2"][idx], 0.0);
                end


            elseif i == nx && j == 1
                # ********************
                # L[idx, linidx(i-1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                # --------------------
                # Derivatives
                # --------------------
                # velocity components, v1
                for idx = 1:v_model_p1
                    # p_b_dv1 + p_c_dv1 = 0
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    # p_b_dv2 + p_c_dv2 = 0
                    push!(entry_dv["grad2"][idx], 0.0);
                end

                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                # --------------------
                # Derivatives
                # --------------------
                # velocity components, v1
                for idx = 1:v_model_p1
                    push!(entry_dv["grad1"][idx], p_a_dv);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    push!(entry_dv["grad2"][idx], p_a_dv);
                end


                # ********************
                # L[idx, linidx(i,j+1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                # --------------------
                # Derivatives
                # --------------------
                # velocity components, v1
                for idx = 1:v_model_p1
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    push!(entry_dv["grad2"][idx], 0.0);
                end
            elseif i == nx && j == ny
                # ********************
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                # --------------------
                # Derivatives
                # --------------------
                # velocity components, v1
                for idx = 1:v_model_p1
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    push!(entry_dv["grad2"][idx], 0.0);
                end
                # ********************
                # L[idx, linidx(i-1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                # --------------------
                # Derivatives
                # --------------------
                # velocity components, v1
                for idx = 1:v_model_p1
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    push!(entry_dv["grad2"][idx], 0.0);
                end
                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                # --------------------
                # Derivatives
                # --------------------
                # velocity components, v1
                for idx = 1:v_model_p1
                    push!(entry_dv["grad1"][idx], p_a_dv);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    push!(entry_dv["grad2"][idx], p_a_dv);
                end
            else
                # ----------------------------------------
                # boundary, non-corner
                # ----------------------------------------
                if i == 1 && 2 <= j <= ny-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_e_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_e_dv2_buffer[idx]);
                    end
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_a_dv);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_a_dv);
                    end
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b + p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], 0.0);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], 0.0);
                    end

                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_d_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_d_dv2_buffer[idx]);
                    end
                elseif i == nx && 2 <= j <= ny-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_e_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_e_dv2_buffer[idx]);
                    end
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_b + p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], 0.0);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], 0.0);
                    end
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_a_dv);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_a_dv);
                    end
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_d_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_d_dv2_buffer[idx]);
                    end
                elseif j == 1 && 2 <= i <= nx-1
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_c_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_c_dv2_buffer[idx]);
                    end
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_a_dv);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_a_dv);
                    end
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_b_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_b_dv2_buffer[idx]);
                    end
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d + p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], 0.0);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], 0.0);
                    end
                elseif j == ny && 2 <= i <= nx-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_d + p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], 0.0);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], 0.0);
                    end
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_c_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_c_dv2_buffer[idx]);
                    end
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_a_dv);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_a_dv);
                    end
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_b_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_b_dv2_buffer[idx]);
                    end
                else
                    # ----------------------------------------
                    # Within boundary
                    # ----------------------------------------

                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_e_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_e_dv2_buffer[idx]);
                    end
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_c_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_c_dv2_buffer[idx]);
                    end
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_a_dv);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_a_dv);
                    end
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_b_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_b_dv2_buffer[idx]);
                    end
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_d_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_d_dv2_buffer[idx]);
                    end
                end
            end
        end
    end
    # create sparse operator matrix 
    ∂L∂v = Dict{String, Vector{SparseMatrixCSC}}(
        "grad1" => Vector{SparseMatrixCSC}(undef, v_model_p1),
        "grad2" => Vector{SparseMatrixCSC}(undef, v_model_p2)
    );
    for i = 1:v_model_p1
        ∂L∂v["grad1"][i] = sparse(row_ind, col_ind, entry_dv["grad1"][i]);
    end
    for i = 1:v_model_p2
        ∂L∂v["grad2"][i] = sparse(row_ind, col_ind, entry_dv["grad2"][i]);
    end
    return ∂L∂v;
end

# ******************************************************************
function advection_diffusion_reaction_homogeneous_neumann_∂kappa(
    prob :: MLEProblem
)
    """
        See `advection_diffusion_reaction_homogeneous_neumann`, 
        this function returns matrix derivative with respect to
        kappa, the diffusion coeffcient. 
    """
    # step sizes 
    xgrid = prob.data.xgrid;
    ygrid = prob.data.ygrid;
    hx = xgrid[2]-xgrid[1];
    hy = ygrid[2]-ygrid[1];

    nx = length(xgrid);
    ny = length(ygrid);

    # analytic derivatives with respect to each PDE parameters
    p_a_dkappa = (2/hx^2)+(2/hy^2);

    # linearized index query
    linidx(i, j) = sub2ind((nx,ny),i,j);
    # only store three vectors
    # - row indices
    # - col indices
    # - entry at (row,col)

    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    # velocity derivatives
    entry_dkappa = Vector{Float64}();

    # precompute certain values and initialize a buffer for gradient storage during assembly

    # U_i+1,j
    p_b_dkappa = -1.0/hx^2;
    # U_i-1,j
    p_c_dkappa = -1.0/hx^2;
    # U_i,j+1
    p_d_dkappa = -1.0/hy^2;
    # U_i,j-1
    p_e_dkappa = -1.0/hy^2;
    for j = 1:ny
        for i = 1:nx
            # physical domain location
            x1 = xgrid[i];
            x2 = ygrid[j];
            # row index is always column major in-order 
            idx = linidx(i, j);

            if i == 1 && j == 1
                # ********************
                # U_i,j dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_a_dkappa);


                # ********************
                # U_i+1,j dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_b_dkappa + p_c_dkappa);

                # ********************
                # U_i,j+1 dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_d_dkappa + p_e_dkappa);
                
            elseif i == 1 && j == ny
                # ********************
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_d_dkappa + p_e_dkappa);
                
                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_a_dkappa);
            
                # ********************
                # L[idx, linidx(i+1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_b_dkappa + p_c_dkappa);
            elseif i == nx && j == 1
                # ********************
                # L[idx, linidx(i-1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_b_dkappa + p_c_dkappa);
            
                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_a_dkappa);

                # ********************
                # L[idx, linidx(i,j+1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_d_dkappa + p_e_dkappa);
                
            elseif i == nx && j == ny
                # ********************
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_d_dkappa + p_e_dkappa);
                
                # ********************
                # L[idx, linidx(i-1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_b_dkappa + p_c_dkappa);
                
                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_a_dkappa);
                
            else
                # ----------------------------------------
                # boundary, non-corner
                # ----------------------------------------
                if i == 1 && 2 <= j <= ny-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_e_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_a_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b + p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_b_dkappa + p_c_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_d_dkappa);
                    
                elseif i == nx && 2 <= j <= ny-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_e_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_b + p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_b_dkappa + p_c_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_a_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_d_dkappa);
                    
                elseif j == 1 && 2 <= i <= nx-1
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_c_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_a_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_b_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d + p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_d_dkappa + p_e_dkappa);
                    
                elseif j == ny && 2 <= i <= nx-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_d + p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_d_dkappa + p_e_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_c_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_a_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_b_dkappa);
                    
                else
                    # ----------------------------------------
                    # Within boundary
                    # ----------------------------------------

                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_e_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_c_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_a_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_b_dkappa);
                    
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_d_dkappa);
                end
            end
        end
    end
    # create sparse operator matrix 
    ∂L∂kappa = sparse(row_ind, col_ind, entry_dkappa);
    return ∂L∂kappa;
end
# **********************************************************************
function advection_diffusion_reaction_homogeneous_neumann_∂c(
    prob :: MLEProblem
)
    nx = length(prob.data.xgrid);
    ny = length(prob.data.ygrid);
    return sparse(I(nx*ny));
end
# **********************************************************************



############################################################
# Deprecated
############################################################
function advection_diffusion_reaction_homogeneous_neumann_deprecated(
    prob :: MLEProblem
)
    """
        Given an MLE problem instance and its current parameteric
        states, creates a 2d ADR differential operator assuming
        homogeneous Neumann boundary conditions along with the 
        matrix derivatives with respect to each PDE parameter.

        !!! WARNING !!!
        (1) Matrix derivative: Can be made more efficient by directly 
        noting: 
            -κΔu + v⋅∇u + cu = 0

            ∂/∂κ = -Δu
            ∂/∂c = u
            ∂/∂θ₁ = ∂/∂v₁⋅∂v₁/∂θ₁
            ∂/∂θ₂ = ∂/∂v₂⋅∂v₂/∂θ₂
        instead of looping and computing each value.

        Deprecated: (03/22/2023), related functions:
        advection_diffusion_reaction_homogeneous_neumann
        advection_diffusion_reaction_homogeneous_neumann_∂kappa
        advection_diffusion_reaction_homogeneous_neumann_∂v
        advection_diffusion_reaction_homogeneous_neumann_∂c
    """
    # step sizes 
    xgrid = prob.data.xgrid;
    ygrid = prob.data.ygrid;

    hx = xgrid[2]-xgrid[1];
    hy = ygrid[2]-ygrid[1];

    nx = length(xgrid);
    ny = length(ygrid);

    # unpack mutable PDE parameters
    kappa = prob.kappa;
    v_model = prob.v_model;
    # number of parameters in velocity model
    v_model_p1 = length(v_model.v1_theta);
    v_model_p2 = length(v_model.v2_theta);

    c = prob.c;

    # parameters for operator
    # U_i,j
    p_a = (2kappa/hx^2)+(2kappa/hy^2)+c;

    # analytic derivatives with respect to each PDE parameters
    p_a_dkappa = (2/hx^2)+(2/hy^2);
    # no dependence on any velocity components
    p_a_dv = 0.0;
    p_a_dc = 1.0;

    # linearized index query
    linidx(i, j) = sub2ind((nx,ny),i,j);

    # only store three vectors
    # - row indices
    # - col indices
    # - entry at (row,col)

    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();
    # initialize all matrix derivatives with respect to each parameters
    entry_dkappa = Vector{Float64}();
    
    # velocity derivatives
    entry_dv = Dict{String, Vector{Vector{Float64}}}(
        "grad1" => [Vector{Float64}() for i in 1:v_model_p1],
        "grad2" => [Vector{Float64}() for i in 1:v_model_p2]
    );
    # reaction derivative
    entry_dc = Vector{Float64}();

    # precompute certain values and initialize a buffer for gradient storage during assembly

    # U_i+1,j
    p_b_dkappa = -1.0/hx^2;
    p_b_dv1_buffer = zeros(Float64, v_model_p1);
    p_b_dv2_buffer = zeros(Float64, v_model_p2);
    p_b_dc = 0.0;

    # U_i-1,j
    p_c_dkappa = -1.0/hx^2;
    p_c_dv1_buffer = zeros(Float64, v_model_p1);
    p_c_dv2_buffer = zeros(Float64, v_model_p2);
    p_c_dc = 0.0;

    # U_i,j+1
    p_d_dkappa = -1.0/hy^2;
    p_d_dv1_buffer = zeros(Float64, v_model_p1);
    p_d_dv2_buffer = zeros(Float64, v_model_p2);
    p_d_dc = 0.0;

    # U_i,j-1
    p_e_dkappa = -1.0/hy^2;
    p_e_dv1_buffer = zeros(Float64, v_model_p1);
    p_e_dv2_buffer = zeros(Float64, v_model_p2);
    p_e_dc = 0.0;

    for j = 1:ny
        for i = 1:nx
            # physical domain location
            x1 = xgrid[i];
            x2 = ygrid[j];
            # evaluate velocity and derivatives with respect to each parameters
            pnt = [x1, x2];
            v_eval = v_model(pnt);
            v_eval_grad = ∂v∂θ(v_model, pnt);
            # U_i+1,j
            p_b = (v_eval[1]/2hx-kappa/hx^2);
            # ∂U_i+1,j/∂θ_1
            p_b_dv1_buffer[:] .= v_eval_grad["grad1"] ./ 2hx;
            # ∂U_i+1,j/∂θ_2
            p_b_dv2_buffer[:] .= 0.0;

            # U_i-1,j
            p_c = (-v_eval[1]/2hx-kappa/hx^2); 
            # ∂U_i-1,j/∂θ_1
            p_c_dv1_buffer[:] .= -p_b_dv1_buffer[:];
            # ∂U_i-1,j/∂θ_1
            p_c_dv2_buffer[:] .= 0.0;

            # U_i,j+1
            p_d = (v_eval[2]/2hy-kappa/hy^2);
            # ∂U_i,j+1/∂θ_1
            p_d_dv1_buffer[:] .= 0.0;
            # ∂U_i,j+1/∂θ_2
            p_d_dv2_buffer[:] .= v_eval_grad["grad2"] ./ 2hy;

            # U_i,j-1
            p_e = (-v_eval[2]/2hy-kappa/hy^2);
            # ∂U_i,j-1/dθ_1
            p_e_dv1_buffer[:] .= 0.0;
            # ∂U_i,j-1/dθ_2
            p_e_dv2_buffer[:] .= -p_d_dv2_buffer[:];

            # row index is always column major in-order 
            idx = linidx(i, j);

            # ----------------------------------------
            # Assemble operator matrix + derivatives
            # ----------------------------------------

            # ----------------------------------------
            # corner points
            # ----------------------------------------
            if i == 1 && j == 1
                # ********************
                # U_i,j dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_a_dkappa);
                # velocity components, v1
                for idx = 1:v_model_p1
                    push!(entry_dv["grad1"][idx], p_a_dv);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    push!(entry_dv["grad2"][idx], p_a_dv);
                end
                # reaction 
                push!(entry_dc, p_a_dc);

                # ********************
                # U_i+1,j dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                push!(entry, p_b + p_c);
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_b_dkappa + p_c_dkappa);
                # velocity components, v1
                for idx = 1:v_model_p1
                    # p_b_dv1 + p_c_dv1 = 0
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    # p_b_dv2 + p_c_dv2 = 0
                    push!(entry_dv["grad2"][idx], 0.0);
                end
                # reaction 
                push!(entry_dc, 0.0);

                # ********************
                # U_i,j+1 dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                push!(entry, p_d + p_e);
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_d_dkappa + p_e_dkappa);
                # velocity components, v1
                for idx = 1:v_model_p1
                    # p_d_dv1 + p_e_dv1 = 0
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    # p_d_dv2 + p_e_dv2 = 0
                    push!(entry_dv["grad2"][idx], 0.0);
                end
                # reaction 
                push!(entry_dc, 0.0);
                

            elseif i == 1 && j == ny
                # ********************
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                push!(entry, p_d + p_e);
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_d_dkappa + p_e_dkappa);
                # velocity components, v1
                for idx = 1:v_model_p1
                    # p_d_dv1 + p_e_dv1 = 0
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    # p_d_dv2 + p_e_dv2 = 0
                    push!(entry_dv["grad2"][idx], 0.0);
                end
                # reaction 
                push!(entry_dc, 0.0);

                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_a_dkappa);
                # velocity components, v1
                for idx = 1:v_model_p1
                    push!(entry_dv["grad1"][idx], p_a_dv);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    # p_d_dv2 + p_e_dv2 = 0
                    push!(entry_dv["grad2"][idx], p_a_dv);
                end
                # reaction 
                push!(entry_dc, p_a_dc);

                # ********************
                # L[idx, linidx(i+1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                push!(entry, p_b + p_c);
                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_b_dkappa + p_c_dkappa);
                # velocity components, v1
                for idx = 1:v_model_p1
                    # p_b_dv1 + p_c_dv1 = 0
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    # p_b_dv2 + p_c_dv2 = 0
                    push!(entry_dv["grad2"][idx], 0.0);
                end
                # reaction 
                push!(entry_dc, 0.0);

            elseif i == nx && j == 1
                # ********************
                # L[idx, linidx(i-1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                push!(entry, p_b + p_c);

                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_b_dkappa + p_c_dkappa);
                # velocity components, v1
                for idx = 1:v_model_p1
                    # p_b_dv1 + p_c_dv1 = 0
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    # p_b_dv2 + p_c_dv2 = 0
                    push!(entry_dv["grad2"][idx], 0.0);
                end
                # reaction 
                push!(entry_dc, 0.0);

                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);

                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_a_dkappa);
                # velocity components, v1
                for idx = 1:v_model_p1
                    push!(entry_dv["grad1"][idx], p_a_dv);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    push!(entry_dv["grad2"][idx], p_a_dv);
                end
                # reaction 
                push!(entry_dc, p_a_dc);

                # ********************
                # L[idx, linidx(i,j+1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                push!(entry, p_d + p_e);

                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_d_dkappa + p_e_dkappa);
                # velocity components, v1
                for idx = 1:v_model_p1
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    push!(entry_dv["grad2"][idx], 0.0);
                end
                # reaction 
                push!(entry_dc, 0.0);

            elseif i == nx && j == ny
                # ********************
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                push!(entry, p_d + p_e);

                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_d_dkappa + p_e_dkappa);
                # velocity components, v1
                for idx = 1:v_model_p1
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    push!(entry_dv["grad2"][idx], 0.0);
                end
                # reaction 
                push!(entry_dc, 0.0);

                # ********************
                # L[idx, linidx(i-1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                push!(entry, p_b + p_c);

                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_b_dkappa + p_c_dkappa);
                # velocity components, v1
                for idx = 1:v_model_p1
                    push!(entry_dv["grad1"][idx], 0.0);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    push!(entry_dv["grad2"][idx], 0.0);
                end
                # reaction 
                push!(entry_dc, 0.0);

                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);

                # --------------------
                # Derivatives
                # --------------------
                # kappa
                push!(entry_dkappa, p_a_dkappa);
                # velocity components, v1
                for idx = 1:v_model_p1
                    push!(entry_dv["grad1"][idx], p_a_dv);
                end
                # velocity components, v2
                for idx = 1:v_model_p2
                    push!(entry_dv["grad2"][idx], p_a_dv);
                end
                # reaction 
                push!(entry_dc, p_a_dc);
            else
                # ----------------------------------------
                # boundary, non-corner
                # ----------------------------------------
                if i == 1 && 2 <= j <= ny-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_e);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_e_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_e_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_e_dv2_buffer[idx]);
                    end
                    # reaction 
                    push!(entry_dc, p_e_dc);

                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_a_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_a_dv);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_a_dv);
                    end
                    # reaction 
                    push!(entry_dc, p_a_dc);

                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b + p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b + p_c);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_b_dkappa + p_c_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], 0.0);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], 0.0);
                    end
                    # reaction 
                    push!(entry_dc, 0.0);

                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_d_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_d_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_d_dv2_buffer[idx]);
                    end
                    # reaction 
                    push!(entry_dc, p_d_dc);

                elseif i == nx && 2 <= j <= ny-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_e);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_e_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_e_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_e_dv2_buffer[idx]);
                    end
                    # reaction 
                    push!(entry_dc, p_e_dc);

                    # ********************
                    # L[idx, linidx(i-1,j)] = p_b + p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_b + p_c);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_b_dkappa + p_c_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], 0.0);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], 0.0);
                    end
                    # reaction 
                    push!(entry_dc, 0.0);


                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_a_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_a_dv);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_a_dv);
                    end
                    # reaction 
                    push!(entry_dc, p_a_dc);


                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_d_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_d_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_d_dv2_buffer[idx]);
                    end
                    # reaction 
                    push!(entry_dc, p_d_dc);

                elseif j == 1 && 2 <= i <= nx-1
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_c);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_c_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_c_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_c_dv2_buffer[idx]);
                    end
                    # reaction 
                    push!(entry_dc, p_c_dc);

                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_a_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_a_dv);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_a_dv);
                    end
                    # reaction 
                    push!(entry_dc, p_a_dc);

                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_b_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_b_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_b_dv2_buffer[idx]);
                    end
                    # reaction 
                    push!(entry_dc, p_b_dc);

                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d + p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d + p_e);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_d_dkappa + p_e_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], 0.0);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], 0.0);
                    end
                    # reaction 
                    push!(entry_dc, 0.0);


                elseif j == ny && 2 <= i <= nx-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_d + p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_d + p_e);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_d_dkappa + p_e_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], 0.0);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], 0.0);
                    end
                    # reaction 
                    push!(entry_dc, 0.0);


                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_c);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_c_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_c_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_c_dv2_buffer[idx]);
                    end
                    # reaction 
                    push!(entry_dc, p_c_dc);

                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_a_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_a_dv);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_a_dv);
                    end
                    # reaction 
                    push!(entry_dc, p_a_dc);

                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_b_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_b_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_b_dv2_buffer[idx]);
                    end
                    # reaction 
                    push!(entry_dc, p_b_dc);
                    
                else
                    # ----------------------------------------
                    # Within boundary
                    # ----------------------------------------

                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_e);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_e_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_e_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_e_dv2_buffer[idx]);
                    end
                    # reaction 
                    push!(entry_dc, p_e_dc);

                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_c);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_c_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_c_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_c_dv2_buffer[idx]);
                    end
                    # reaction 
                    push!(entry_dc, p_c_dc);

                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_a_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_a_dv);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_a_dv);
                    end
                    # reaction 
                    push!(entry_dc, p_a_dc);

                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_b_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_b_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_b_dv2_buffer[idx]);
                    end
                    # reaction 
                    push!(entry_dc, p_b_dc);


                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d);

                    # --------------------
                    # Derivatives
                    # --------------------
                    # kappa
                    push!(entry_dkappa, p_d_dkappa);
                    # velocity components, v1
                    for idx = 1:v_model_p1
                        push!(entry_dv["grad1"][idx], p_d_dv1_buffer[idx]);
                    end
                    # velocity components, v2
                    for idx = 1:v_model_p2
                        push!(entry_dv["grad2"][idx], p_d_dv2_buffer[idx]);
                    end
                    # reaction 
                    push!(entry_dc, p_d_dc);
                end
            end
        end
    end
    # create sparse operator matrix along with derivatives
    L = sparse(row_ind, col_ind, entry);
    ∂L∂kappa = sparse(row_ind, col_ind, entry_dkappa);
    ∂L∂v = Dict{String, Vector{SparseMatrixCSC}}(
        "grad1" => Vector{SparseMatrixCSC}(undef, v_model_p1),
        "grad2" => Vector{SparseMatrixCSC}(undef, v_model_p2)
    );
    for i = 1:v_model_p1
        ∂L∂v["grad1"][i] = sparse(row_ind, col_ind, entry_dv["grad1"][i]);
    end
    for i = 1:v_model_p2
        ∂L∂v["grad2"][i] = sparse(row_ind, col_ind, entry_dv["grad2"][i]);
    end
    ∂L∂c = sparse(row_ind, col_ind, entry_dc);
    return L, ∂L∂kappa, ∂L∂v, ∂L∂c

end

function advection_diffusion_reaction_homogeneous_neumann_∂c_deprecated(
    prob :: MLEProblem
)
    """
        See `advection_diffusion_reaction_homogeneous_neumann`, 
        this function returns matrix derivative with respect to
        c, the reaction coeffcient. 

        Deprecated: (03/22/2023), it is the same as identity, see comment
        in WARNINGS.
    """
    # step sizes 
    xgrid = prob.data.xgrid;
    ygrid = prob.data.ygrid;
    hx = xgrid[2]-xgrid[1];
    hy = ygrid[2]-ygrid[1];

    nx = length(xgrid);
    ny = length(ygrid);

    # unpack mutable PDE parameters
    kappa = prob.kappa;
    v_model = prob.v_model;
    # number of parameters in velocity model
    v_model_p1 = length(v_model.v1_theta);
    v_model_p2 = length(v_model.v2_theta);

    c = prob.c;

    # derivative of components with respect to c
    p_a_dc = 1.0;

    # linearized index query
    linidx(i, j) = sub2ind((nx,ny),i,j);
    # only store three vectors
    # - row indices
    # - col indices
    # - entry at (row,col)

    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    # reaction derivative
    entry_dc = Vector{Float64}();

    # precompute certain values and initialize a buffer for gradient storage during assembly

    # U_i+1,j
    p_b_dc = 0.0;
    # U_i-1,j
    p_c_dc = 0.0;
    # U_i,j+1
    p_d_dc = 0.0;
    # U_i,j-1
    p_e_dc = 0.0;
    for j = 1:ny
        for i = 1:nx
            # physical domain location
            x1 = xgrid[i];
            x2 = ygrid[j];
            # row index is always column major in-order 
            idx = linidx(i, j);

            if i == 1 && j == 1
                # ********************
                # U_i,j dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                # --------------------
                # Derivatives
                # --------------------
                # reaction 
                push!(entry_dc, p_a_dc);

                # ********************
                # U_i+1,j dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                # --------------------
                # Derivatives
                # --------------------
                # reaction 
                push!(entry_dc, 0.0);

                # ********************
                # U_i,j+1 dependence
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                # --------------------
                # Derivatives
                # --------------------
                # reaction
                push!(entry_dc, 0.0);
                
            elseif i == 1 && j == ny
                # ********************
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                # --------------------
                # Derivatives
                # --------------------
                # reaction
                push!(entry_dc, 0.0);
                
                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                # --------------------
                # Derivatives
                # --------------------
                # reaction
                push!(entry_dc, p_a_dc);
            
                # ********************
                # L[idx, linidx(i+1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                # --------------------
                # Derivatives
                # --------------------
                # reaction
                push!(entry_dc, 0.0);

            elseif i == nx && j == 1
                # ********************
                # L[idx, linidx(i-1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                # --------------------
                # Derivatives
                # --------------------
                # reaction 
                push!(entry_dc, 0.0);

                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                # --------------------
                # Derivatives
                # --------------------
                # reaction 
                push!(entry_dc, p_a_dc);

                # ********************
                # L[idx, linidx(i,j+1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                # --------------------
                # Derivatives
                # --------------------
                # reaction 
                push!(entry_dc, 0.0);

            elseif i == nx && j == ny
                # ********************
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                # --------------------
                # Derivatives
                # --------------------
                # reaction 
                push!(entry_dc, 0.0);

                # ********************
                # L[idx, linidx(i-1,j)] = p_b + p_c;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                # --------------------
                # Derivatives
                # --------------------
                # reaction 
                push!(entry_dc, 0.0);

                # ********************
                # L[idx, linidx(i,j)] = p_a;
                # ********************
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                # --------------------
                # Derivatives
                # --------------------
                # reaction 
                push!(entry_dc, p_a_dc);
                
            else
                # ----------------------------------------
                # boundary, non-corner
                # ----------------------------------------
                if i == 1 && 2 <= j <= ny-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_e_dc);
                    
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_a_dc);
                    
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b + p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, 0.0);

                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_d_dc);
                    
                elseif i == nx && 2 <= j <= ny-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_e_dc);
                    
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_b + p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, 0.0);

                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_a_dc);

                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_d_dc);
                    
                elseif j == 1 && 2 <= i <= nx-1
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_c_dc);

                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_a_dc);

                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_b_dc);
                    
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d + p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, 0.0);

                elseif j == ny && 2 <= i <= nx-1
                    # ********************
                    # L[idx, linidx(i,j-1)] = p_d + p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, 0.0);

                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_c_dc);

                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_a_dc);
                    
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_b_dc);

                else
                    # ----------------------------------------
                    # Within boundary
                    # ----------------------------------------

                    # ********************
                    # L[idx, linidx(i,j-1)] = p_e;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_e_dc);
                    
                    # ********************
                    # L[idx, linidx(i-1,j)] = p_c;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_c_dc);
                    
                    # ********************
                    # L[idx, linidx(i,j)] = p_a;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_a_dc);
                    
                    # ********************
                    # L[idx, linidx(i+1,j)] = p_b;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_b_dc);
                    
                    # ********************
                    # L[idx, linidx(i,j+1)] = p_d;
                    # ********************
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    # --------------------
                    # Derivatives
                    # --------------------
                    # reaction 
                    push!(entry_dc, p_d_dc);
                end
            end
        end
    end
    # create sparse operator matrix 
    ∂L∂c = sparse(row_ind, col_ind, entry_dc);
    return ∂L∂c;
end