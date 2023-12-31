# Main routines for constructing HODLR factorizations of sparse
# covariance matrices, including HODLR algebra.

# Some good practices
# - dealing with unwanted adjoints: https://discourse.julialang.org/t/best-practices-regarding-adjoint-when-should-it-be-used-instead-of-array-best-way-to-style-it-out/43525
using Statistics
using LinearAlgebra
using Random
using BlockDiagonals

import LinearAlgebra: rank

include("./dyadic_idx.jl");
include("./rsvd.jl");


# !!!!!! may want to stop using mutable struct, weird inaccuracies 
# may occur when fields get mutated.
############################################################################################
# HODLR struct
############################################################################################
mutable struct hodlr
    """
        This struct holds a sparse square symmetric 
        matrix in HODLR format. Not all matrix entries
        need to be stored but instead the:
            1. (dense) block diagonals
            2. off-diagonal low rank factors
            3. indices at which the HODLR entries are nontrivial
    """
    # level of HODLR approximation
    max_level::Int64
    # all block diagonal matrices
    leaves::Vector{Matrix{Float64}} 
    # all off-diagonal factors
    U::Vector{Vector{Matrix{Float64}}}
    V::Vector{Vector{Matrix{Float64}}}
    Ul::Vector{Vector{Matrix{Float64}}}
    Vl::Vector{Vector{Matrix{Float64}}}
    # index information to reconstruct full matrix
    idx_tree::Vector{Any}
    row_idx_tree::Vector{Any}
    col_idx_tree::Vector{Any}
    # internal constructor for validation
    function hodlr(max_level, leaves, U, V, idx_tree)
        @assert(length(leaves) == 2^max_level);
        @assert(length(U) == max_level);
        @assert(length(V) == max_level);
        @assert(length(idx_tree) == max_level)
        new(max_level, leaves, U, V, V, U, idx_tree, idx_tree, idx_tree)
    end
end

############################################################################################
# HODLR one-way factorization
############################################################################################
struct hodlr_fact
    """
        An extension of the base HODLR format. Factorizes
        A :: hodlr 
        A = blkdiag(A) * (I + U1V1) * ... * (I + UkVk)
        where k is maximum number of levels.

        The U, V blocks are different from the off-diagonal 
        low-rank factorization of the base format. See 
        Appendix B of the paper for a reference.
    """
    # level of HODLR approximation
    max_level
    # all block diagonal matrices
    leaves
    # all off-diagonal factors
    U
    V
    # index information to reconstruct full matrix
    idx_tree::Vector{Any}
    # internal constructor for validation
    function hodlr_fact(max_level, leaves, U, V, idx_tree)
        @assert(length(U) == max_level);
        @assert(length(V) == max_level);
        new(max_level, leaves, U, V, idx_tree)
    end
end

############################################################################################
# Nonsymmetric HODLR 
############################################################################################
mutable struct hodlr_nonsymmetric
    """
        This struct holds a sparse square nonsymmetric 
        matrix in HODLR format. Not all matrix entries
        need to be stored but instead the:
            1. (dense) block diagonals
            2. off-diagonal low rank factors
            3. indices at which the HODLR entries are nontrivial
    """
    # level of HODLR approximation
    max_level
    # all block diagonal matrices
    leaves
    # all off-diagonal factors
    U
    V
    Ul
    Vl
    # index information to reconstruct full matrix
    row_idx_tree::Vector{Any}
    col_idx_tree::Vector{Any}
    idx_tree::Vector{Any}
    # internal constructor for validation
    function hodlr_nonsymmetric(max_level, leaves, U, V, Ul, Vl, row_idx_tree, col_idx_tree=nothing)
        @assert(length(leaves) == 2^max_level);
        @assert(length(U) == max_level);
        @assert(length(V) == max_level);
        @assert(length(Ul) == max_level);
        @assert(length(Vl) == max_level);
        if isnothing(col_idx_tree)
            new(max_level, leaves, U, V, Ul, Vl, row_idx_tree, row_idx_tree, row_idx_tree)
        else
            new(max_level, leaves, U, V, Ul, Vl, row_idx_tree, col_idx_tree, row_idx_tree)
        end
    end
end

############################################################################################
# Nonsymmetric HODLR with low rank updates
############################################################################################
mutable struct hodlr_nonsymmetric_with_updates
    max_level
    leaves
    U
    V
    Ul
    Vl
    U_additive
    V_additive
    idx_tree
    function hodlr_nonsymmetric_with_updates(max_level, leaves, U, V, Ul, Vl, U_additive, V_additive, idx_tree)
        new(max_level, leaves, U,V, Ul, Vl, U_additive, V_additive, idx_tree)
    end
end

############################################################################################
# Reconstruction of full matrix given HODLR struct
############################################################################################
function hodlr_to_full(A_hodlr :: hodlr)
    """
        Given an HODLR struct, reconstructs the full matrix.

        The full matrix is assumed to be square and symmetric.
    """
    # recover original matrix size
    max_level = A_hodlr.max_level;
    leave_size = size(A_hodlr.leaves[1], 1);
    n = Int(leave_size * 2^max_level);
    # create zero matrix
    A = zeros(n, n);
    leave_idx = A_hodlr.idx_tree[end];
    # put back leaves
    for i = eachindex(leave_idx)
        idx = leave_idx[i];
        A[idx, idx] .= A_hodlr.leaves[i];
    end
    # start with leave level (reconstruct off-diagonal level by level)
    peel = leave_idx;
    for l = max_level:-1:1
        # number of off-diagonal matrices at level l
        U_l = A_hodlr.U[l];
        V_l = A_hodlr.V[l];
        for i = 1:2^(l-1)
            idx1 = peel[2*(i-1)+1]; idx2 = peel[2*i];
            # off diagonal factor matrices
            Ui = U_l[i]; Vi = V_l[i];
            B = Ui*Vi';
            # put back
            A[idx1, idx2] .= B;
            A[idx2, idx1] .= B';
        end
        # merge to reconstruct the level above
        peel = dyadic_merge(peel, 1);
    end
    return A
end
############################################################################################
# Reconstruction of full matrix given nonsymmetric struct with additive updates
############################################################################################
function hodlr_to_full(A :: hodlr_nonsymmetric_with_updates)
    """ 
        Restore original matrix from nonsymmetric HODLR with
        additive updates.
    """
    # unpack variables
    max_level = A.max_level;
    idx_tree = A.idx_tree;
    leaves = A.leaves;
    U = A.U;
    V = A.V;
    Ul = A.Ul;
    Vl = A.Vl;
    leave_size = size(leaves[1],1);
    # full matrix size
    n = Int(leave_size * 2^max_level)
    res = zeros(n, n);
    # fill off diagonals
    for i = 1:max_level
        idx = idx_tree[i];
        for j = 1:Int(floor(length(idx)/2))
            idx1 = idx[Int((j-1)*2+1)];
            idx2 = idx[Int(j*2)];
            res[idx1, idx2] .= U[i][j]*(V[i][j]');
            res[idx2, idx1] .= Ul[i][j]*(Vl[i][j]');
        end
    end
    # fill leaf levels
    idx = idx_tree[end];
    for i = eachindex(idx)
        res[idx[i], idx[i]] .= leaves[i];
    end
    # additive low rank updates 
    for i = 1:max_level
        tp = Array{Matrix}(undef, Int(2^(i-1)));
        for j = 1:Int(2^(i-1))
            tp[j] = A.U_additive[i][j]*transpose(A.V_additive[i][j]);
        end
        # update 
        tp2 = zeros(n, n);
        for j = 1:Int(2^(i-1))
            sub_n = size(tp[j], 1);
            tp2[sub_n*(j-1)+1:sub_n*(j), sub_n*(j-1)+1:sub_n*(j)] .= tp[j];
        end
        res = res + tp2;
    end
    return res
end

############################################################################################
# Basic HODLR algorithm with full matrix access
############################################################################################
function hodlr(K::Matrix{Float64}, max_level::Int64, local_rank::Int64)
    """
        Given a square matrix K, constructs a max_level
        HODLR factorization with off-diagonal matrices
        having local_rank. This generic routine assumes
        square, sparse symmetric matrix with dimensions 
        being powers of 2 (for divisibility).

        Inputs:
            K :: Matrix{Float64}    -> (n x n) matrix for which
                                    an HODLR factorization is built.
            max_level :: Int        -> maximum number of levels of 
                                    approximation. At each level l,
                                    2^l off-diagonal matrices need
                                    to be approximated.
            local_rank :: Int       -> a fixed rank for truncating
                                    each off-diagonal matrix.
    """
    n = size(K, 1);
    @assert(size(K, 2) == n, "Input matrix must be square. ");
    @assert(mod(n, 2) == 0, "Matrix size must be powers of 2. ");
    # generate all index sets up to leave level
    leave_idx = dyadic_idx(n, max_level);
    # store all levels of index sets 
    idx_tree = Array{Any}(undef, max_level);
    # get block diagonals of leave level 
    num_components = length(leave_idx);
    leaves = Array{Any}(undef, num_components);
    for i = 1:num_components
        idx = leave_idx[i];
        leaves[i] = view(K, idx, idx);
    end
    # approximate off-diagonal blocks (due to symmetry, enough to only take upper diagonal)
    U = Array{Any}(undef, max_level);
    V = Array{Any}(undef, max_level);
    # start with leave level (peel level by level)
    peel = leave_idx;
    for l = max_level:-1:1
        # number of off-diagonal matrices at level l
        U_l = Array{Any}(undef, 2^(l-1));
        V_l = Array{Any}(undef, 2^(l-1));
        for i = 1:2^(l-1)
            idx1 = peel[2*(i-1)+1]; idx2 = peel[2*i];
            # off diagonal matrix
            B = view(K, idx1, idx2);
            Ui, Vi, local_err = interp_decomp(B, local_rank);
            U_l[i] = Ui; V_l[i] = Vi;
        end
        # store index sets at level l 
        idx_tree[l] = peel;
        # merge to peel the level above
        peel = dyadic_merge(peel, 1);
        # store factor matrices
        U[l] = U_l; V[l] = V_l;
    end
    return hodlr(max_level, leaves, U, V, idx_tree);
end

function hodlr(K::Matrix{Float64}, max_level::Int64, eps::Float64=1e-12, ord::Union{Int, String}=2)
    """Given symmetric square matrix K, construct an HODLR apporoximation with max_level levels and all
       off-diagonal blocks are approximated to relative accuracy of rank_tol.

    Args:
        K : dense matrix to compress 
        max_level : max level for the HODLR approximation 
        eps : relative tolarence for the off-diagonal ranks 
        ord : norm of the matrix used

    Returns:
        HODLR approximation of the dense matrix K.
    """
    # Assertion 
    @assert(eps > 0, "Threshold parameter eps should be positive.")

    # Compression 
    n = size(K, 1);
    idx_tree = [];
    for lvl = 1:max_level 
        push!(idx_tree, dyadic_idx(n, lvl));
    end
    # Construct leaves 
    leaves = Vector{Matrix{Float64}}(undef, 0);
    for i = eachindex(idx_tree[end])
        push!(leaves, K[idx_tree[end][i], idx_tree[end][i]]);
    end
    # Construct the off-diagonal factors 
    #!!! This is not the most efficient implementation, consider direct thresholing with SVD
    U = Vector{Vector{Matrix{Float64}}}(undef, 0);
    V = Vector{Vector{Matrix{Float64}}}(undef, 0);
    for lvl = 1:max_level 
        tmpU, tmpV = Vector{Matrix{Float64}}(undef, 0), Vector{Matrix{Float64}}(undef, 0);
        for id = 1:2^(lvl-1)
            tmp = svd(K[idx_tree[lvl][2 * (id - 1) + 1], idx_tree[lvl][2 * (id - 1) + 2]]);
            l, r = compress_factors(tmp.U * Diagonal(tmp.S), copy(tmp.V), eps, ord, false);
            
            push!(tmpU, l); push!(tmpV, r);
        end
        push!(U, tmpU); push!(V, tmpV);
    end

    return hodlr(max_level, leaves, U, V, idx_tree);
end

function hodlr_nonsymmetric(K::Matrix{Float64}, max_level::Int64, eps::Float64=1e-12, ord=Union{Int, String}=2)
    """Given nonsymmetric square matrix K, construct an HODLR apporoximation with max_level levels and 
       all off-diagonal blocks are approximated to relative accuracy of rank_tol.

    Args:
        K : dense matrix to compress 
        max_level : max level for the HODLR approximation 
        eps : relative tolarence for the off-diagonal ranks 
        ord : norm of the matrix used

    Returns:
        Nonsymmetric HODLR approximation of the dense matrix K.
    """
    # Assertion 
    @assert(eps > 0, "Threshold parameter eps should be positive.")

    # Compression 
    rown, coln = size(K);
    row_idx_tree = []; col_idx_tree = [];
    for lvl = 1:max_level 
        push!(row_idx_tree, dyadic_idx(rown, lvl));
        push!(col_idx_tree, dyadic_idx(coln, lvl));
    end
    # Construct leaves 
    leaves = Vector{Matrix{Float64}}(undef, 0);
    for i = eachindex(row_idx_tree[end])
        push!(leaves, K[row_idx_tree[end][i], col_idx_tree[end][i]]);
    end
    # Construct the off-diagonal factors 
    #!!! This is not the most efficient implementation, consider direct thresholing with SVD
    U = Vector{Vector{Matrix{Float64}}}(undef, 0);
    V = Vector{Vector{Matrix{Float64}}}(undef, 0);
    Ul = Vector{Vector{Matrix{Float64}}}(undef, 0);
    Vl = Vector{Vector{Matrix{Float64}}}(undef, 0);
    for lvl = 1:max_level 
        tmpU, tmpV = Vector{Matrix{Float64}}(undef, 0), Vector{Matrix{Float64}}(undef, 0);
        for id = 1:2^(lvl-1)
            tmp = svd(K[row_idx_tree[lvl][2 * (id - 1) + 1], col_idx_tree[lvl][2 * (id - 1) + 2]]);
            l, r = compress_factors(tmp.U * Diagonal(tmp.S), tmp.V, eps, ord, false);
            
            push!(tmpU, l); push!(tmpV, r);
        end
        push!(U, tmpU); push!(V, tmpV);

        tmpUl, tmpVl = Vector{Matrix{Float64}}(undef, 0), Vector{Matrix{Float64}}(undef, 0);
        for id = 1:2^(lvl-1)
            tmp = svd(K[row_idx_tree[lvl][2 * (id - 1) + 2], col_idx_tree[lvl][2 * (id - 1) + 1]]);
            l, r = compress_factors(tmp.U * Diagonal(tmp.S), tmp.V, eps, ord, false);
            
            push!(tmpUl, l); push!(tmpVl, r);
        end
        push!(Ul, tmpUl); push!(Vl, tmpVl);
    end

    return hodlr_nonsymmetric(max_level, leaves, U, V, Ul, Vl, row_idx_tree, col_idx_tree);
end

############################################################################################
# HODLR algorithm with black-box matvec access (using randomized range finder)
############################################################################################
function hodlr(mv_query::Function, n::Int64, max_level::Int64, local_rank::Int64, c::Int64)
    """
        A variant of the HODLR algorithm without assuming
        full matrix access. Instead, the matrix is able to
        be queried via an external routine `mv_query` which 
        has an underlying matrix `K`, and returns for any 
        input vector `x` the product `K*x`.

        The underlying matrix `K` is assumed to be square,
        symmetric. HODLR factorization with off-diagonal 
        matrices having `local_rank`. The rank truncation
        algorithm is randomized SVD, where `c` serves as 
        an upsampling parameter. 

        Inputs:
            mv_query :: Function    -> function handle. `mv` is 
                                    short for "matrix-vector multiply". 

            n :: Int                -> size of the target matrix `K`. 

            max_level :: Int        -> maximum number of levels of 
                                    approximation. At each level l,
                                    2^l off-diagonal matrices need
                                    to be approximated.
            local_rank :: Int       -> a fixed rank for truncating
                                    each off-diagonal matrix.

            c :: Int                -> upsampling parameter for randomized
                                    SVD (RSVD).
    """
    # generate all levels of indices
    idx_tree = Array{Any}(undef, max_level);
    tmp = dyadic_idx(n, max_level);
    for ii = max_level:-1:1
        idx_tree[ii] = tmp;
        tmp = dyadic_merge(tmp, 1);
    end

    # approximate off-diagonal blocks
    U = Array{Any}(undef, max_level);
    V = Array{Any}(undef, max_level);
    for ii = 1:max_level
        tmp_idx = idx_tree[ii];
        num_idx = length(tmp_idx);
        
        # right sampling matrix
        R = zeros(n, local_rank+c);
        for jj = 1:Int(num_idx/2)
            R[tmp_idx[2*jj], :] = randn(length(tmp_idx[2*jj]), local_rank+c);
        end
        # fast matvec query: see eqn (3.9)
        K_mult_R = mv_query(R);
        # off-diagonal product: see eqn (3.9)
        B_mult_R = offdiag_prod(idx_tree, U, V, ii, R);

        # subtract: see eqn (3.9)
        tmp_prod = K_mult_R .- B_mult_R;

        # HODLR low rank factors
        Q_blocks = Array{Any}(undef, Int(num_idx/2));
        for kk = 1:Int(num_idx/2)
            Q_blocks[kk] = qr(tmp_prod[tmp_idx[2*kk-1], :]).Q[:, 1:local_rank+c];
        end

        # left sampling matrix (by computing Q^T*K)
        R = zeros(n, local_rank + c);
        for kk = 1:Int(num_idx/2)
            R[tmp_idx[2*kk-1], :] .= Q_blocks[kk];
        end

        # multiply with K using fast query
        K_mult_R = mv_query(R);
        # off-diagonal product: see eqn (3.9)
        B_mult_R = offdiag_prod(idx_tree, U, V, ii, R);
        # peel HODLR
        tmp_prod = K_mult_R - B_mult_R;
        # R factors / Q^T*B factors
        R_blocks = Array{Any}(undef, Int(num_idx/2));
        for kk = 1:Int(num_idx/2)
            R_blocks[kk] = tmp_prod[tmp_idx[2*kk], :];
        end

        U[ii] = Q_blocks;
        V[ii] = R_blocks;
    end

    # construct leave matrices
    leaves = Array{Any}(undef, 2^max_level);
    # create sampling matrix (block identities)
    R = repeat(I(Int(n/2^max_level)), 2^max_level);

    # fast matvec query: see eqn (3.9)
    K_mult_R = mv_query(R);
    # off-diagonal product: see eqn (3.9)
    B_mult_R = offdiag_prod(idx_tree, U, V, max_level+1, R);

    # subtract: see eqn (3.9)
    tmp_prod = K_mult_R - B_mult_R;

    # store leaves (!!! may want to symmetrize first)
    for ii = 1:2^max_level
        leaves[ii] = tmp_prod[idx_tree[end][ii], :];
    end
    # return struct
    return hodlr(max_level, leaves, U, V, idx_tree);
end


############################################################################################
# Basic HODLR gradient algorithm with full matrix access
############################################################################################
function hodlr_grad(param_matvec_grad::Function, n::Int64, max_level::Int64, local_rank::Int64, c::Int64, θ::Vector)
    """
        Given fast matrix-vector query access through `param_matvec_grad`,
        computes matrix compression and derivative of parameterized
        matrix `K(θ)` (available via query) both in HODLR format. 
        HODLR is fixed rank prescribed by `local_rank`, plus any
        upsampling `c`. 

        `θ` should be numerical values and respect the same input
        order as parameterized matrix `K(θ)`. `ForwardDiff` is used
        for forward-mode autodifferentiation with respect to each
        θ parameters.

        Returnss `K(θ)` in HODLR format, and ∂θ₁K, ..., ∂θₚK, all in HODLR format.
    """
    # create index tree
    idx_tree = Array{Any}(undef, max_level);
    tmp = dyadic_idx(n, max_level);
    for ii = max_level:-1:1
        idx_tree[ii] = tmp;
        tmp = dyadic_merge(tmp, 1);
    end
    # number of parameters
    p = length(θ);                  

    # preallocate (K)
    U = Array{Any}(undef, max_level);
    V = Array{Any}(undef, max_level);

    # preallocate (∂K)
    Udθ = Array{Array{Any}}(undef, p);
    Vdθ = Array{Array{Any}}(undef, p);
    for k = 1:p
        Udθ[k] = Array{Any}(undef, max_level);
        Vdθ[k] = Array{Any}(undef, max_level);
    end
    
    # construct off-diagonals
    for ii = 1:max_level
        tmp_idx = idx_tree[ii];
        num_idx = length(tmp_idx);
        # right sampling matrix
        tmp_sample1 = zeros(n, local_rank+c);
        for jj = 1:Int(num_idx/2)
            # temporarily changed to ones for debugging, change back !!!!! Bob: 11/15/2022
            tmp_sample1[tmp_idx[2*jj], :] .= randn(length(tmp_idx[2*jj]), local_rank+c);
            #tmp_sample1[tmp_idx[2*jj], :] .= ones(length(tmp_idx[2*jj]), local_rank+c);
        end
        # fast matvec + gradient query
        tmp_prod, tmp_prod_grad = param_matvec_grad(θ, n, tmp_sample1);
        # off-diagonal product
        tprod = offdiag_prod(idx_tree, U, V, ii, tmp_sample1);

        # subtract: (K - B)*Ω
        tmp_prod = tmp_prod .- tprod;
        # subtract: (∂K - ∂B)*Ω
        for k = 1:p
            tprod = offdiag_prod(idx_tree, Udθ[k], Vdθ[k], ii, tmp_sample1);
            tmp_prod_grad[k] = tmp_prod_grad[k] .- tprod;
        end
        # # truncate the product to account for singularity
        # for k = 1:Int(length(tmp_idx)/2)
        #     tmp_mat = tmp_prod[tmp_idx[2*k-1], :];
        #     # do pivoted QR
        #     qr_tmp_mat = qr(tmp_mat, ColumnNorm());
        #     tmp_R = qr_tmp_mat.R; 
        #     perm = qr_tmp_mat.p;
        #     # sum of |r_ii|
        #     total_R = sum(abs.(diag(tmp_R)));
        #     num_comp = findfirst(cumsum(abs.(diag(tmp_R))).>total_R*0.99999);
        #     # truncate
        #     tmp_prod = tmp_prod[:, perm[1:num_comp]];
        #     for jj = 1:p
        #         # truncate the derivative as well
        #         tmp_grad = tmp_prod_grad[jj][:,perm[1:num_comp]];
        #         tmp_prod_grad[jj] = tmp_grad;
        #     end
        # end

        # HODLR low rank factors
        block_Q = Array{Any}(undef, Int(num_idx/2));
        # HODLR factor derivatives
        block_Q_grad = Array{Array{Any}}(undef, p);
        # for each parameter, initialize array for Qdθ
        for k = 1:p
            block_Q_grad[k] = Array{Any}(undef, Int(num_idx/2));
        end

        for jj = 1:Int(num_idx/2)
            tmp_mat = tmp_prod[tmp_idx[2*jj-1], :];
            # compute QR factorization
            tmp_matqr = qr(tmp_mat);
            tmp_Q = tmp_matqr.Q[:, 1:local_rank+c];
            # save low rank factor
            block_Q[jj] = tmp_Q;
            # save R for gradient computation
            tmp_R = tmp_matqr.R[1:local_rank+c, 1:local_rank+c];
            # dQdθ: Algorithm 3.2
            for k = 1:p
                # tmp_Phi = (tmp_Q')*(((tmp_R')\(tmp_prod_grad[k][tmp_idx[2*jj-1], :]'))');
                # phi1 = tril(tmp_Phi,-1);
                # dOmega = phi1 .- phi1';
                # tprod = (tmp_R'\(tmp_prod_grad[k][tmp_idx[2*jj-1], :]'))';
                # dQ2 = tmp_Q*dOmega+tprod-tmp_Q*(tmp_Q'*tprod);

                dB = tmp_prod_grad[k][tmp_idx[2*jj-1], :];
                dQ, _ = grad_qr(dB, tmp_Q, tmp_R);
                #println(norm(dQ-dQ2))

                # store dQdθ
                block_Q_grad[k][jj] = dQ;
            end
        end

        # draw left range by computing Q'*K
        tmp_sample1 = zeros(n, local_rank+c);
        tmp_sample_grad = Array{Matrix}(undef, p);
        for k = 1:p
            tmp_sample_grad[k] = zeros(n, local_rank+c);
        end

        for jj = 1:Int(num_idx/2)
            tmp_sample1[tmp_idx[2*jj-1], :] .= block_Q[jj];
            # construct sampling matrix using dQ
            for k = 1:p
                tmp = tmp_sample_grad[k];
                tmp[tmp_idx[2*jj-1], :] .= block_Q_grad[k][jj];
                tmp_sample_grad[k] = tmp;
            end
        end

        # multiply with K using fast query
        tmp_prod, tmp_prod_grad = param_matvec_grad(θ, n, tmp_sample1);
        # off-diagonal product
        tprod = offdiag_prod(idx_tree, U, V, ii, tmp_sample1);
        # peel HODLR
        tmp_prod = tmp_prod - tprod;
        # peel HODLR grad
        for jj = 1:p
            tprod = offdiag_prod(idx_tree, Udθ[jj], Vdθ[jj], ii, tmp_sample1);
            tmp_prod_grad[jj] = tmp_prod_grad[jj] - tprod;
        end
        tmp_prod_dQ = Array{Matrix}(undef, p);
        for k = 1:p
            tmp1, _ = param_matvec_grad(θ, n, tmp_sample_grad[k]);
            tprod = offdiag_prod(idx_tree, U, V, ii, tmp_sample_grad[k]);
            tmp_prod_dQ[k] = tmp1 - tprod;
        end
        # compute P / Q^T*B factors
        block_P = Array{Any}(undef, Int(num_idx/2));
        block_P_grad = Array{Array{Any}}(undef, p);
        # for each parameter, initialize array for Pdθ
        for k = 1:p
            block_P_grad[k] = Array{Any}(undef, Int(num_idx/2));
        end

        for jj = 1:Int(num_idx/2)
            # store P blocks
            block_P[jj] = tmp_prod[tmp_idx[2*jj], :];
            # for each parameter θᵢ, store Pdθᵢ
            for k = 1:p
                block_P_grad[k][jj] = tmp_prod_grad[k][tmp_idx[2*jj], :] + tmp_prod_dQ[k][tmp_idx[2*jj], :];
            end
        end

        # dB = dQ*P' + Q*dP' = [dQ, Q]*[P, dP]'
        for jj = 1:Int(num_idx/2)
            for k = 1:p
                block_Q_grad[k][jj] = [block_Q_grad[k][jj] block_Q[jj]];
                block_P_grad[k][jj] = [block_P[jj] block_P_grad[k][jj]];
            end
        end
        # update local low rank blocks
        U[ii] = block_Q; 
        V[ii] = block_P;
        for k = 1:p
            Udθ[k][ii] = block_Q_grad[k];
            Vdθ[k][ii] = block_P_grad[k];
        end
    end

    # construct leaves along with their derivatives
    leaves = Array{Any}(undef, 2^max_level);
    leaves_grad = Array{Array{Any}}(undef, p);
    for k = 1:p
        leaves_grad[k] = Array{Any}(undef, 2^max_level);
    end
    tmp_idx = idx_tree[max_level];
    # sampling matrix (block identities)
    R = repeat(I(Int(n/2^max_level)), 2^max_level);
    # fast matvec: diagonals and their derivatives
    tmp_prod, tmp_prod_grad = param_matvec_grad(θ, n, R);
    # off-diagonal product
    tprod = offdiag_prod(idx_tree, U, V, max_level+1, R);
    # peel
    tmp_prod = tmp_prod - tprod;
    for k = 1:p
        # for each parameter, peel similarly
        tprod = offdiag_prod(idx_tree, Udθ[k], Vdθ[k], max_level+1, R);
        tmp_prod_grad[k] = tmp_prod_grad[k] - tprod;
    end

    # store leaves and their derivatives
    for ii = 1:2^max_level
        idx = tmp_idx[ii];
        # !!! may want to symmetrize the leaves
        leaves[ii] = tmp_prod[idx, 1:length(idx)];
        for k = 1:p
            # # !!! may want to symmetrize the leaves
            leaves_grad[k][ii] = tmp_prod_grad[k][idx, 1:length(idx)];
        end
    end
    # create HODLR objects and return
    K_hodlr = hodlr(max_level, leaves, U, V, idx_tree);
    Kdθ_hodlr = Array{hodlr}(undef, p);
    for k = 1:p
        Kdθ_hodlr[k] = hodlr(max_level, leaves_grad[k], Udθ[k], Vdθ[k], idx_tree);
    end
    return K_hodlr, Kdθ_hodlr;
end

############################################################################################
# HODLR matvec product (only off-diagonal part)
############################################################################################
function offdiag_prod(idx_tree, U, V, current_level, X)
    """
        (Used for peeling in HODLR matvec construction) 
        E.g. for level-2 peeling
        Computes:
        [ 0     B₁
        
          B₁ᵀ   0 ] * X 
        where B₁ is available as U*Vᵀ
    """
    sol = zeros(size(X));
    for l = 1:current_level-1
        # number of blocks at level l
        num_blocks = length(U[l]);
        @assert length(V[l]) == num_blocks
        for k = 1:num_blocks
            Uk = U[l][k];
            Vk = V[l][k];
            idx1 = idx_tree[l][(k-1)*2+1];
            idx2 = idx_tree[l][(k-1)*2+1+1];
            # upper right block
            sol[idx1, :] .= sol[idx1, :] .+ Uk * (Vk' * X[idx2, :]);

            # lower left block
            sol[idx2, :] .= sol[idx2, :] .+ Vk * (Uk' * X[idx1, :]);
        end
    end
    return sol
end

############################################################################################
# one-way factorization of HODLR
############################################################################################
function hodlr_factorize(A :: hodlr)
    """ 
        Computes the one-way factorization of 
        A in HODLR form.
    """
    # unpack parameters
    idx_tree = copy(A.idx_tree);
    # add one more level to idx_tree
    idx_tree = prepend!(idx_tree, [dyadic_merge(idx_tree[1], 1)]);
    max_level = A.max_level;
    leaves = A.leaves;
    U = A.U;
    V = A.V;

    # initialize parameters for the new struct
    U_new = Array{Any}(undef, max_level);
    V_new = Array{Any}(undef, max_level);

    # preallocate for component matrices

    # upper right block
    U_ur = Array{Array{Matrix}}(undef, max_level);
    V_ur = Array{Array{Matrix}}(undef, max_level);
    # lower left block
    U_ll = Array{Array{Matrix}}(undef, max_level);
    V_ll = Array{Array{Matrix}}(undef, max_level);

    # factorize
    for i = max_level:-1:1
        # original off-diagonal blocks at level i
        tmp_U = U[i];
        tmp_V = V[i];
        num_blocks = length(tmp_U);
        @assert(length(tmp_V) == num_blocks)
        # number of factor matrices
        fact = Int(floor(2^max_level/num_blocks));
        # initialize arrays for component matrices at this level
        tmp_U_ur = Array{Matrix}(undef, num_blocks);
        tmp_V_ur = Array{Matrix}(undef, num_blocks);
        tmp_U_ll = Array{Matrix}(undef, num_blocks);
        tmp_V_ll = Array{Matrix}(undef, num_blocks);
        # off-diagonal updates: see (B.3)
        for j = 1:num_blocks
            tmp_mat1 = BlockDiagonal(leaves[Int((j-1)*fact+1):Int((j-1)*fact+floor(0.5*fact))]);
            tmp_mat2 = BlockDiagonal(leaves[Int((j-1)*fact+floor(0.5*fact)+1):Int(j*fact)])
            tmp_U_ur[j] = tmp_mat1\tmp_U[j];
            tmp_V_ur[j] = tmp_V[j];
            tmp_U_ll[j] = tmp_mat2\tmp_V[j];
            tmp_V_ll[j] = tmp_U[j];
        end
        U_ur[i] = tmp_U_ur;
        V_ur[i] = tmp_V_ur;
        U_ll[i] = tmp_U_ll;
        V_ll[i] = tmp_V_ll;
    end

    # factorize off-diagonals bottom-up
    for i = max_level-1:-1:0
        num_blocks = length(idx_tree[i+1]);
        tmp_U = Array{Matrix{Float64}}(undef, num_blocks);
        tmp_V = Array{Matrix{Float64}}(undef, num_blocks);
        # Get rank 2k update: see (B.8)
        for j = 1:num_blocks
            tmp_fact1 = BlockDiagonal([U_ur[i+1][j], U_ll[i+1][j]]);
            tmp_fact2 = [
                zeros(size(V_ll[i+1][j],1), size(V_ur[i+1][j],2)) V_ll[i+1][j];
                V_ur[i+1][j] zeros(size(V_ur[i+1][j],1), size(V_ll[i+1][j],2))
            ];
            tmp_U[j] = tmp_fact1;
            tmp_V[j] = tmp_fact2;
        end

        # get one-way factors
        for k = i:-1:1
            tp_U_ur = U_ur[k]; tp_V_ur = V_ur[k];
            tp_U_ll = U_ll[k]; tp_V_ll = V_ll[k];

            tmp_U_ur = Array{Matrix}(undef, size(tp_U_ur, 1));
            tmp_V_ur = Array{Matrix}(undef, size(tp_U_ur, 1));
            tmp_U_ll = Array{Matrix}(undef, size(tp_U_ur, 1));
            tmp_V_ll = Array{Matrix}(undef, size(tp_U_ur, 1));
            fact     = floor(size(tmp_U,1)/size(tp_U_ur,1));
            for j = eachindex(tp_U_ur)
                # (11/17/2022) there was a strange error of BlockDiagonal having trouble taking
                # in a list of length 1
                if length(tmp_U[Int((j-1)*fact+1):Int((j-1)*fact+floor(0.5*fact))]) == 1
                    blkdiag1 = BlockDiagonal([tmp_U[Int((j-1)*fact+1):Int((j-1)*fact+floor(0.5*fact))][1]]);
                else
                    blkdiag1 = BlockDiagonal(tmp_U[Int((j-1)*fact+1):Int((j-1)*fact+floor(0.5*fact))]);
                end
                if length(tmp_V[Int((j-1)*fact+1):Int((j-1)*fact+floor(0.5*fact))]) == 1
                    blkdiag2 = transpose(BlockDiagonal([tmp_V[Int((j-1)*fact+1):Int((j-1)*fact+floor(0.5*fact))][1]]));
                else
                    blkdiag2 = transpose(BlockDiagonal(tmp_V[Int((j-1)*fact+1):Int((j-1)*fact+floor(0.5*fact))]));
                end
                tmp_U_ur[j] = woodbury_inv(blkdiag1, blkdiag2, tp_U_ur[j]); 
                tmp_V_ur[j] = tp_V_ur[j];
                if length(tmp_U[Int((j-1)*fact+floor(0.5*fact)+1):Int((j)*fact)]) == 1
                    blkdiag1 = BlockDiagonal([tmp_U[Int((j-1)*fact+floor(0.5*fact)+1):Int((j)*fact)][1]]);
                else
                    blkdiag1 = BlockDiagonal(tmp_U[Int((j-1)*fact+floor(0.5*fact)+1):Int((j)*fact)]);
                end
                if length(tmp_V[Int((j-1)*fact+floor(0.5*fact)+1):Int((j)*fact)]) == 1
                    blkdiag2 = transpose(BlockDiagonal([tmp_V[Int((j-1)*fact+floor(0.5*fact)+1):Int((j)*fact)][1]]));
                else
                    blkdiag2 = transpose(BlockDiagonal(tmp_V[Int((j-1)*fact+floor(0.5*fact)+1):Int((j)*fact)]));
                end
                tmp_U_ll[j] = woodbury_inv(blkdiag1, blkdiag2, tp_U_ll[j]);
                tmp_V_ll[j] = tp_V_ll[j];
            end
            U_ur[k] = tmp_U_ur;
            V_ur[k] = tmp_V_ur;
            U_ll[k] = tmp_U_ll;
            V_ll[k] = tmp_V_ll;
        end
        # create one way factoriztaion
        U_new[i+1] = tmp_U;
        V_new[i+1] = tmp_V;
    end
    return hodlr_fact(max_level, leaves, U_new, V_new, idx_tree)
end



############################################################################################
# factorized HODLR inverse
############################################################################################
function hodlr_solve(A :: hodlr_fact, b :: Union{Matrix, Vector})
    """ 
        Solves linear system where A is in HODLR one-way factorized format. 
        Returnss A^-1*b.

        A = blkdiag(A) * (I + UkVk') * ... * (I + U1V1')
        then:
        A_inv = (I + U1V1')^-1 * ... * (I + UkVk')^-1 * blkdiag(A)^-1
    """
    max_level = A.max_level;
    idx_tree = A.idx_tree;
    # apply matrix inverses from bottom up
    res = copy(b);
    # apply leaf level first
    idx = idx_tree[max_level+1];
    for ii = 1:2^max_level
        res[idx[ii], :] .= A.leaves[ii] \ res[idx[ii], :];
    end
    # nonleaf levels (bottom up)
    for l = max_level:-1:1
        Ul = A.U[l];
        Vl = A.V[l];
        num_factors = length(Ul);
        idx = idx_tree[l];
        for ii = 1:num_factors
            res[idx[ii], :] .= woodbury_inv(Ul[ii], Vl[ii]', res[idx[ii], :]);
        end
    end
    return res;
end

function hodlr_tr(A :: hodlr)
    """ 
        Compute trace of an HODLR matrix.
    """
    # accumulate traces of all leaves
    res = 0;
    for i = 1:2^A.max_level
        res += LinearAlgebra.tr(A.leaves[i]);
    end
    return res
end

function hodlr_tr(A :: hodlr_nonsymmetric_with_updates)
    """
        Computes trace of HODLR matrix with additive updates.
    """
    res = 0;
    # accumulate traces of all leaves
    for i = 1:2^A.max_level
        res += LinearAlgebra.tr(A.leaves[i]);
    end
    # accumulate additive low rank contributions
    @assert length(A.V_additive) == length(A.U_additive);
    @assert length(A.U_additive) == A.max_level
    for k = 1:A.max_level
        Uk_additive = A.U_additive[k];
        @assert length(Uk_additive) == length(A.V_additive[k])
        for i = eachindex(Uk_additive)
            res += LinearAlgebra.tr(A.V_additive[k][i]'*A.U_additive[k][i]);
        end
    end
    return res;
end

function hodlr_logdet(A :: hodlr_fact)
    """
        Computes log determinant of a factorized HODLR.

        Currently uses `logabsdet`, may want to change later.
    """
    # accumulate output
    res = 0;
    # leaf level contributions
    for i = eachindex(A.leaves)
        tmp, sign = LinearAlgebra.logabsdet(A.leaves[i]);
        if sign == -1
            @warn "Negative determinant encountered in leave $(i), forcing abs value ..."
        end
        res = res + tmp;
    end
    # rank-k updates
    for l = A.max_level:-1:1
        tmp_U = A.U[l];
        tmp_V = A.V[l];
        for j = eachindex(tmp_U)
            n = size(tmp_U[j], 2);
            tmp, sign = LinearAlgebra.logabsdet(I(n)+transpose(tmp_V[j])*tmp_U[j]);
            if sign == -1
                @warn "Negative determinant encountered, forcing absolute value ... "
            end
            res = res + tmp;
        end
    end
    return res;
end

function hodlr_fact_transpose(A :: hodlr_fact)
    """ 
        Stores the transpose of a one-way HODLR factorization.
        See (A.1), (A.2) for two ways of factorizing.

        Note: A must be symmetric for this to be valid.
    """
    # transpose the low rank factors
    U_new = copy(A.V);
    V_new = copy(A.U);
    leaves_new = Array{Any}(undef, length(A.leaves));

    # transpose the leaves
    for i = eachindex(A.leaves)
        leaves_new[i] = copy(transpose(A.leaves[i]));
    end
    idx_tree_new = copy(A.idx_tree);
    return hodlr_fact(A.max_level, leaves_new, U_new, V_new, idx_tree_new)
end

############################################################################################
# HODLR matvec product (full HODLR)
############################################################################################
function hodlr_prod(hodlr, x)
    """
        Given an HODLR struct K and a vector x, 
        computes matrix-vector product K*x.
    """
    max_level = hodlr.max_level;
    U = hodlr.U; V = hodlr.V;
    num_leaves = length(hodlr.leaves);
    # get matrix size
    leave_size = size(hodlr.leaves[1], 1);
    n = Int(leave_size * 2^max_level);
    @assert(size(x, 1) == n);
    # compute as K*x = (D + M)*x = D*x + M*x where D is block diagonal
    sol = zeros(size(x));
    # diagonal
    leave_idx = hodlr.idx_tree[end];
    for i = eachindex(leave_idx)
        idx = leave_idx[i];
        # compute block-wise mat-vec
        sol[idx, :] .= sol[idx, :] .+ hodlr.leaves[i] * x[idx, :];
    end
    # off-diagonal
    for l = max_level:-1:1
        Ul = hodlr.U[l]; Vl = hodlr.V[l];
        for ii = eachindex(Ul)
            # take local factors
            Uli = Ul[ii];
            Vli = Vl[ii];
            idx1 = hodlr.idx_tree[l][2*(ii-1)+1]
            idx2 = hodlr.idx_tree[l][2*ii]
            # upper diagonal --> sol1 = U*V'*b2
            sol[idx1, :] .= sol[idx1, :] .+ Uli * (Vli' * x[idx2, :]);

            # lower diagonal --> sol2 = V*U'*b1
            sol[idx2, :] .= sol[idx2, :] .+ Vli * (Uli' * x[idx1, :]);
        end
    end
    return sol
end

function hodlr_invmult(A :: hodlr_fact, B :: hodlr)
    """ 
        Given A (factorized HODLR) and B (base HODLR),
        computes A^-1*B in HODLR form directly.

        B is assumed to be symmetric.
    """
    max_level = A.max_level;
    # transpose the factorization to have form:
    # A = (I+V1U1)*...*(I+VkUk)*blkdiag(A)
    A = hodlr_fact_transpose(A);
    # then A^-1 = blkdiag(A)^-1 * (I + VkUk)^-1 * ... * (I + V1U1)^-1

    # unpack variables
    max_level = A.max_level; 
    # assumption: A and B have the same HODLR levels
    @assert B.max_level == max_level
    # preallocate to store additive low rank updates
    U_additive = Array{Any}(undef, max_level);
    V_additive = Array{Any}(undef, max_level);

    # lower blocks of the resulting HODLR matrix
    U_new_l = copy(B.V);
    V_new_l = copy(B.U)
    idx_tree_new = copy(B.idx_tree);
    leaves_new = copy(B.leaves);
    # create new output and modify 
    output_B = hodlr_nonsymmetric(max_level, leaves_new, copy(B.U), copy(B.V), U_new_l, V_new_l, idx_tree_new);
    # level 1: step 1, (A.3)
    U_additive[1] = [-A.U[1][1]];
    # should be B^T*V, but B is symmetric
    tmp = [hodlr_prod(B, A.V[1][1])];
    #@show norm(hodlr_prod(B, A.V[1][1]))
    tmp[1] = ((I(size(A.V[1][1],2))+A.V[1][1]'*A.U[1][1])\(tmp[1]'))';
    #@show norm(tmp[1])
    V_additive[1] = tmp;
    # level 2 onwards
    for l = 2:max_level
        # update additive low rank terms for all previous levels
        for k = 1:l-1
            # for each of the coarser levels, update the additive factors
            # Uk <- (I - Ul(I + Vl^TUl)\Vl^T)*Uk
            U_additive[k] = block_inv_update(U_additive[k], A.U[l], A.V[l]);
            #@show l, k 
            #@show norm(U_additive[k][1])
        end
        # update off-diagonal blocks of B :: hodlr
        for k = 1:l-1
            # update the low rank factors of B at each of the coarser levels
            # see (A.7) for off diagonal blocks update rules

            # interleave upper and lower diagonal blocks
            tmp = copy([output_B.U[k] output_B.Ul[k]]')[:];
            for tmp_idx = eachindex(tmp)
                tmp[tmp_idx] = copy(transpose(tmp[tmp_idx]));
            end
            #@show norm(tmp[1]) norm(tmp[2])
            #@show norm(output_B.V[1][1]) norm(output_B.V[2][1]) norm(output_B.V[2][2])
            #@show output_B.V[1][1][1:3,1:3]
            tmp = block_inv_update(tmp, A.U[l], A.V[l]);
            
            #@show output_B.V[1][1][1:3,1:3]
            #@show norm(output_B.V[1][1]) norm(output_B.V[2][1]) norm(output_B.V[2][2])
            
            #@show norm(tmp[1]) norm(tmp[2])
            # unpack 
            output_B.U[k] = tmp[1:2:end];
            output_B.Ul[k] = tmp[2:2:end];
            #@show norm(output_B.U[k][1]) norm(output_B.Ul[k][1])
        end
        
        # get additive low rank update for the newer level
        num_factors = length(A.U[l]);
        tmp_U = Array{Any}(undef, num_factors);
        tmp_V = Array{Any}(undef, num_factors);
        for j = 1:num_factors
            tmp_U[j] = -A.U[l][j];
            #@show j, norm(tmp_U[j])
            tp = hodlr_truncate(output_B, l-1, j);
            tp = hodlr_prod(tp, A.V[l][j]);
            #@show j, norm(tp)
            tp = (I(size(A.V[l][j],2)) + transpose(A.V[l][j])*A.U[l][j]) \ (tp');
            #@show j, norm(tp)
            tmp_V[j] = tp';
        end
        # update
        U_additive[l] = tmp_U;
        V_additive[l] = tmp_V;
    end
    #@show norm(output_B.V[1][1]) norm(output_B.V[2][1]) norm(output_B.V[2][2])
    # modify leaf level
    for j = eachindex(B.leaves)
        output_B.leaves[j] = A.leaves[j]\output_B.leaves[j]; 
        #@show j, norm(output_B.leaves[j])
    end
    # apply off-diagonal blocks
    for k = max_level:-1:1
        # update the low rank factors of B at each of the coarser levels
        # see (A.7) for off diagonal blocks update rules

        # interleave upper and lower diagonal blocks
        tmp = copy([output_B.U[k] output_B.Ul[k]]')[:];
        for tmp_idx = eachindex(tmp)
            tmp[tmp_idx] = copy(transpose(tmp[tmp_idx]));
        end
        #@show k, [norm(tmp[i]) for i = 1:length(tmp)]
        tmp = block_inv_update2(tmp, A.leaves);
        #@show k, [norm(tmp[i]) for i = 1:length(tmp)]
        # unpack 
        output_B.U[k] = tmp[1:2:end];
        output_B.Ul[k] = tmp[2:2:end];
    end
    # update all low-rank factors
    for k = max_level:-1:1
        tp = U_additive[k];
        #@show k, [norm(tp[i]) for i = 1:length(tp)]
        tp = block_inv_update2(tp, A.leaves);
        #@show k, [norm(tp[i]) for i = 1:length(tp)]
        U_additive[k] = tp; 
    end
    # output struct
    return hodlr_nonsymmetric_with_updates(output_B.max_level, output_B.leaves, output_B.U, output_B.V, output_B.Ul, output_B.Vl, U_additive, V_additive, output_B.idx_tree);
end

function hodlr_hadamard(A :: hodlr, B :: hodlr)
    """ 
        Given two matrices in HODLR format, computes
        a new HODLR matrix as their Hadamard product.
        See Fig. 9 of https://arxiv.org/pdf/1909.07909.pdf
        for more details. 

        A, B are assumed to have the same levels, and 
        symmetric.
    """
    idx_tree = A.idx_tree;
    num_levels = A.max_level;
    @assert(B.max_level == num_levels);
    # original matrix size
    n = Int(Int(2^num_levels)*size(A.leaves[1], 1));
    full_idx = 1:n;
    # new leaves
    new_leaves = Array{Matrix}(undef, Int(2^num_levels));
    for i = eachindex(A.leaves)
        new_leaves[i] = A.leaves[i].*B.leaves[i];
    end
    # off-diagonal factors
    U_new = Array{Any}(undef, length(A.U));
    V_new = Array{Any}(undef, length(A.V));
    for l = num_levels:-1:1
        Ul_new = Array{Matrix}(undef, length(A.U[l]));
        Vl_new = Array{Matrix}(undef, length(A.V[l]));
        for k = eachindex(Ul_new)
            # compute new U factor
            Ul_new[k] = tkrp_fast(A.U[l][k], B.U[l][k]);
            # compute new V factor
            Vl_new[k] = tkrp_fast(A.V[l][k], B.V[l][k]);
        end
        U_new[l] = Ul_new;
        V_new[l] = Vl_new;
    end
    return hodlr(num_levels, new_leaves, U_new, V_new, idx_tree);
end

function hodlr_hadamard(A :: hodlr_nonsymmetric, B :: hodlr_nonsymmetric)
    """ 
        Given two matrices in HODLR format, computes
        a new HODLR matrix as their Hadamard product.
        See Fig. 9 of https://arxiv.org/pdf/1909.07909.pdf
        for more details. 

        A, B are assumed to have the same levels, and 
        not necessarily symmetric.
    """
    row_idx_tree = A.row_idx_tree;
    col_idx_tree = A.col_idx_tree;
    num_levels = A.max_level;
    @assert(B.max_level == num_levels);
    # original matrix size
    n = Int(Int(2^num_levels)*size(A.leaves[1], 1));
    full_idx = 1:n;
    # new leaves
    new_leaves = Array{Matrix}(undef, Int(2^num_levels));
    for i = eachindex(A.leaves)
        new_leaves[i] = A.leaves[i].*B.leaves[i];
    end
    # off-diagonal factors
    U_new = Array{Any}(undef, length(A.U));
    V_new = Array{Any}(undef, length(A.V));
    Ul_new = Array{Any}(undef, length(A.Ul));
    Vl_new = Array{Any}(undef, length(A.Vl));
    for l = num_levels:-1:1
        Ul_new = Array{Matrix}(undef, length(A.U[l]));
        Vl_new = Array{Matrix}(undef, length(A.V[l]));
        Ul_lower_new = Array{Matrix}(undef, length(A.Ul[l]));
        Vl_lower_new = Array{Matrix}(undef, length(A.Vl[l]));
        for k = eachindex(Ul_new)
            # compute new U factor
            Ul_new[k] = tkrp_fast(A.U[l][k], B.U[l][k]);
            # compute new V factor
            Vl_new[k] = tkrp_fast(A.V[l][k], B.V[l][k]);
            # compute new Ul factor
            Ul_lower_new[k] = tkrp_fast(A.Ul[l][k], B.Ul[l][k]);
            # compute new Vl factor
            Vl_lower_new[k] = tkrp_fast(A.Vl[l][k], B.Vl[l][k]);
        end
        U_new[l] = Ul_new;
        V_new[l] = Vl_new;
        Ul_new[l] = Ul_lower_new;
        Vl_new[l] = Vl_lower_new;
    end
    return hodlr_nonsymmetric(num_levels, new_leaves, U_new, V_new, Ul_new, Vl_new, row_idx_tree, col_idx_tree);
end
############################################################################################
# HODLR recompression routines to control the off-diagonal ranks
############################################################################################
function compress_factors(U::Matrix{Float64}, V::Matrix{Float64}, eps::Float64=1e-12, ord::Union{Int, String}=2, 
    inplace::Bool=true)::Union{Nothing, Tuple{Matrix{Float64}, Matrix{Float64}}}
    """Compress old factorization (U * V') using more efficient factorizations.
       The tolerance is controlled by eps (relative error tolerance) and ord (norm of the matrix).

    Args:
        U : left factor
        V : right factor
        eps : relative error tolerance
        ord : norm of the matrix used
        inplace : whether modifying the input arrays inplace

    Returns:
        U : recompressed left factor 
        V : recompressed right factor
    """
    # Assertion
    @assert(eps > 0, "Threshold parameter eps should be positive.")

    # Get column and row space 
    qrU = qr(U); qrV = qr(V);
    usv = svd(qrU.R * qrV.R');

    # Get numerical rank of the recompressed factors 
    if ord == 2
        topsv = maximum(usv.S);
        rk = sum(usv.S .> topsv * eps);
    elseif ord == "fro"
        tmp = sqrt(cumsum(usv.S[end:-1:1].^2));
        rk = sum(tmp .> norm(usv.S) * eps);
    else
        throw(ArgumentError("Matrix norm not supported."))
    end

    # Recompression 
    if inplace 
        U = qrU.Q * usv.U[:, 1:rk] * Diagonal(usv.S[1:rk]);
        V = qrV.Q * usv.V[:, 1:rk];
    else 
        Unew = qrU.Q * usv.U[:, 1:rk] * Diagonal(usv.S[1:rk]);
        Vnew = qrV.Q * usv.V[:, 1:rk];
        return Unew, Vnew
    end
end


function recompression(A::hodlr, eps::Float64=1e-12, ord::Union{Int, String}=2, inplace::Bool=true)::Union{Nothing, hodlr}
    """Recompression routine for all the off-diagonal blocks to control the rank.

    Args:
        A : HOLDR matrix as hodlr object
        eps : targeting relative error 
        ord : norm of the matrix used
        inplace : whether modifying the input HODLR matrix inplace

    Returns:
        A : HODLR matrix with all the off-diagonal blocks recompressed with relative error eps.
    """
    # Assertion 
    @assert(eps > 0, "Threshold parameter eps should be positive.")

    # Recompression 
    output = inplace ? A : deepcopy(A);

    for lvl = 1:output.max_level 
        for idx = 1:length(output.U[lvl])
            compress_factors(output.U[lvl][idx], output.V[lvl][idx], eps, ord);
        end
    end

    if !inplace 
        return output 
    end
end

############################################################################################
# HODLR operations by overloading base operators
# TODO: Extend the operations to nonsymmetric HODLR matrices 
############################################################################################
function is_compatible(A::Union{hodlr, hodlr_nonsymmetric}, B::Union{hodlr, hodlr_nonsymmetric})::Bool 
    """Check if two HODLR matrices are compatible in partitioning 

    Args:
        A : HODLR matrix 1 
        B : HODLR matrix 2

    Returns:
        A boolean indicating if the two HODLR matrices have the same partitioning tree.
    """
    
    if A.max_level != B.max_level
        return false
    end

    # Check partitioning tree 
    for i = eachindex(A.row_idx_tree)
        if length(A.row_idx_tree[i]) != length(B.row_idx_tree[i]) || 
            length(A.col_idx_tree[i]) != length(B.col_idx_tree[i])
            return false
        end

        for j = eachindex(A.row_idx_tree[i])
            if A.row_idx_tree[i][j] != B.row_idx_tree[i][j] ||
                A.col_idx_tree[i][j] != B.col_idx_tree[i][j]
                return false
            end
        end
    end
    
    return true 
end

function Base.:+(A::Union{hodlr, hodlr_nonsymmetric}, B::Union{hodlr, hodlr_nonsymmetric}
    )::Union{hodlr, hodlr_nonsymmetric} 
    """HODLR matrix-matrix additions. This operation can inflate off-diagonal ranks.

    Args:
        A : HODLR matrix 1 
        B : HODLR matrix 2

    Returns:
        C : An HODLR matrix C = A + B. 
    """
    # Assertion 
    @assert(is_compatible(A, B), "HODLR additions now only support matrices with the same partitioning.")

    # Off-diagonal additions
    C = typeof(A) == hodlr ? deepcopy(B) : deepcopy(A);
    for lvl = 1:C.max_level 
        for i = 1:length(C.U[lvl])
            C.U[lvl][i] = cat(A.U[lvl][i], B.U[lvl][i], dims=2);
            C.V[lvl][i] = cat(A.V[lvl][i], B.V[lvl][i], dims=2);
            C.Ul[lvl][i] = cat(A.Ul[lvl][i], B.Ul[lvl][i], dims=2);
            C.Vl[lvl][i] = cat(A.Vl[lvl][i], B.Vl[lvl][i], dims=2);
        end
    end

    # Leaf additions 
    for i = 1:length(C.leaves)
        C.leaves[i] = A.leaves[i] + B.leaves[i];
    end

    return C 
end

function Base.:+(A::hodlr, iden::UniformScaling{Float64})::hodlr
    """Compute HODLR matrix + identity.

    Args:
        A : HODLR matrix 
        iden : identity matrix or constant * identity matrix 

    Returns:
        A + iden. 
    """

    ret = deepcopy(A);
    # Update the leaves 
    for i = eachindex(ret.leaves)
        ret.leaves[i] += iden;
    end

    return ret;
end

function Base.:+(iden::UniformScaling{Float64}, A::hodlr)::hodlr
    """Compute HODLR matrix + identity.

    Args:
        iden : identity matrix or constant * identity matrix 
        A : HODLR matrix 

    Returns:
        iden + A. 
    """

    return A + iden;
end


function Base.:*(A::hodlr, x::Union{Matrix{Float64}, Adjoint{Float64, Matrix{Float64}}})::Matrix{Float64}
    """Compute HODLR * dense matrix.

    Args:
        A : HODLR matrix 
        x : dense matrix

    Returns:
        A matrix = A * x.
    """

    # TODO: Extend to hodlr_nonsymmetric
    max_level = A.max_level;
    U = A.U; V = A.V; Ul = A.Ul; Vl = A.Vl;
    num_leaves = length(A.leaves);
    # get matrix size
    leave_size = size(A.leaves[1], 1);
    n = Int(leave_size * 2^max_level);
    @assert(size(x, 1) == n);
    # compute as K*x = (D + M)*x = D*x + M*x where D is block diagonal
    sol = zeros(size(x));
    # diagonal
    leave_idx = A.idx_tree[end];
    for i = eachindex(leave_idx)
        idx = leave_idx[i];
        # compute block-wise mat-vec
        sol[idx, :] .= sol[idx, :] .+ A.leaves[i] * x[idx, :];
    end
    # off-diagonal
    for l = max_level:-1:1
        Ul = A.U[l]; Vl = A.V[l];
        for ii = eachindex(Ul)
            # take local factors
            Uli = Ul[ii];
            Vli = Vl[ii];
            idx1 = A.idx_tree[l][2*(ii-1)+1];
            idx2 = A.idx_tree[l][2*ii];
            # upper diagonal --> sol1 = U*V'*b2
            sol[idx1, :] .= sol[idx1, :] .+ Uli * (Vli' * x[idx2, :]);
            
            # lower diagonal --> sol2 = V*U'*b1
            sol[idx2, :] .= sol[idx2, :] .+ Vli * (Uli' * x[idx1, :]);
        end
    end
    return sol
end

function Base.:*(x::Union{Matrix{Float64}, Adjoint{Float64, Matrix{Float64}}}, A::hodlr)::Matrix{Float64}
    """Compute dense matrix. * HODLR.

    Args:
        x : dense matrix
        A : HODLR matrix 

    Returns:
        A matrix = x * A.

    """
    sol = A * x';
    return sol'
end

function Base.:*(D::Diagonal{Float64, Vector{Float64}}, A::Union{hodlr, hodlr_nonsymmetric})::Union{hodlr, hodlr_nonsymmetric}
    """Compute Diagonal * HODLR matrix.

    Args:
        D : Diagonal matrix 
        A : HODLR matrix

    Returns:
        A matrix = D * A.

    """
    ret = hodlr_transpose(deepcopy(A));
    ret = hodlr_transpose(ret * D);

    return ret
end

function Base.:*(A::Union{hodlr, hodlr_nonsymmetric}, B::Union{hodlr, hodlr_nonsymmetric}
    )::Union{hodlr, hodlr_nonsymmetric} 
    """HODLR matrix-matrix products. This operation can inflate off-diagonal ranks.

    Args:
        A : HODLR matrix 1 
        B : HODLR matrix 2

    Returns:
        C : An HODLR matrix C = A * B. 
    """
    # Assertion 
    @assert(is_compatible(A, B), "HODLR products now only support matrices with the same partitioning.")

    # Recursive computation 
    # Obtain sub-blocks 
    A1 = hodlr_truncate(A, 1, 1); A2 = hodlr_truncate(A, 1, 2);
    B1 = hodlr_truncate(B, 1, 1); B2 = hodlr_truncate(B, 1, 2);
    AU = A.U[1][1]; AV = A.V[1][1]; BU = B.U[1][1]; BV = B.V[1][1];
    AUl = A.Ul[1][1]; AVl = A.Vl[1][1]; BUl = B.Ul[1][1]; BVl = B.Vl[1][1];

    # Computation 
    tmp1 = A1 * B1; tmp2 = AU * (AV' * BUl);
    C1 = hodlr_plus_lowrank(tmp1, tmp2, BVl);

    tmp1 = A2 * B2; tmp2 = AUl * (AVl' * BU);
    C2 = hodlr_plus_lowrank(tmp1, tmp2, BV);

    tmp1 = A1 * BU; tmp2 = AV' * B2;
    U = hcat(tmp1, AU); V = hcat(BV, tmp2');

    tmp1 = AVl' * B1; tmp2 = A2 * BUl;
    Ul = hcat(AUl, tmp2); Vl = hcat(tmp1', BVl);

    # Reconstruction of the output matrix 
    return block_hodlr_construction(C1, C2, U, V, Ul, Vl);
end

function Base.:\(
    A::hodlr, 
    B::Union{Vector{Float64}, Matrix{Float64}}
)::Union{Vector{Float64}, Matrix{Float64}}
    """Matrix linear system solve with HODLR LHS and general matrix RHS.

    Args:
        A : full-rank HODLR matrix. Need to be cautious about the condition number of A 
        B : dense matrix as RHS 

    Returns:
        A^{-1} * B.
    """
    # Assertions 
    @assert(A.col_idx_tree[end][end][end] == size(B, 1), "Dimension mismatches.")

    # Factorization + solve 
    A_fact = hodlr_factorize(A);
    return hodlr_solve(A_fact, B);
end

function Base.:\(A_fact::hodlr_fact, B::Union{hodlr, hodlr_nonsymmetric})#::Union{hodlr, hodlr_nonsymmetric}
    """Compute A / B as another nonsymmetric HODLR matrix.

    Args:
        A: factorization of HODLR matrix
        B: symmetric or nonsymmetric HODLR matrices

    Returns:
        A / B as another nonsymmetric HODLR matrix.
    """

    max_level = A_fact.max_level;
    # transpose the factorization to have form:
    # A = (I+V1U1)*...*(I+VkUk)*blkdiag(A)
    A_fact = hodlr_fact_transpose(A_fact);
    # then A^-1 = blkdiag(A)^-1 * (I + VkUk)^-1 * ... * (I + V1U1)^-1

    # Sequential applications 
    ret = deepcopy(B);
    for lvl = 1:max_level
        if lvl == 1
            tmp_U = A_fact.U[lvl][1];
            tmp_V = A_fact.V[lvl][1];
            new_V = ((I + tmp_V' * tmp_U) \ (tmp_V' * ret))';
            new_V = Matrix{Float64}(new_V);

            ret = hodlr_plus_lowrank(ret, -tmp_U, new_V)

            continue
        end

        # Other levels 
        # Generate the modified diagonal blocks for the finest level 
        tmp_diag = [];
        for i = 1:2^(lvl - 1)
            # Update the diagonal blocks 
            diag_block = hodlr_truncate(ret, lvl - 1, i);
            tmp_U = A_fact.U[lvl][i];
            tmp_V = A_fact.V[lvl][i];
            new_V = ((I + tmp_V' * tmp_U) \ (tmp_V' * diag_block))';
            new_V = Matrix{Float64}(new_V);
            diag_block = hodlr_plus_lowrank(diag_block, -tmp_U, new_V)

            push!(tmp_diag, diag_block);
        end

        # Sequentially merge to reconstruct the HODLR matrix 
        for i = lvl - 1:-1:1
            tmp_off_diag_l = [];
            tmp_off_diag_r = [];
            for j = 1:2^(i - 1)
                push!(tmp_off_diag_l, ret.U[i][j]);
                push!(tmp_off_diag_l, ret.Ul[i][j]);
                push!(tmp_off_diag_r, ret.V[i][j]);
                push!(tmp_off_diag_r, ret.Vl[i][j]);
            end

            # Update the off-diagonal low-rank factors 
            # @show size(A_fact.U[i + 1][1]), A_fact.U[i + 1][1][1:10,1:10];
            tmp_off_diag_l = block_inv_update(tmp_off_diag_l, A_fact.U[lvl], A_fact.V[lvl]);
            tmp_off_diag_r = tmp_off_diag_r;

            # Merge with the diagonal blocks 
            new_tmp_diag = [];
            for j = 1:2^(i - 1)
                diag1 = tmp_diag[(j - 1) * 2 + 1];
                diag2 = tmp_diag[(j - 1) * 2 + 2];

                tmp_U = tmp_off_diag_l[(j - 1) * 2 + 1];
                tmp_Ul = tmp_off_diag_l[(j - 1) * 2 + 2];
                tmp_V = tmp_off_diag_r[(j - 1) * 2 + 1];
                tmp_Vl = tmp_off_diag_r[(j - 1) * 2 + 2];

                tmp = block_hodlr_construction(diag1, diag2, tmp_U, tmp_V, tmp_Ul, tmp_Vl);
                push!(new_tmp_diag, tmp);
            end

            # Update tmp_diag
            tmp_diag = new_tmp_diag;
        end

        # Update ret 
        ret = deepcopy(tmp_diag[1]);

        @assert(length(tmp_diag) == 1, "Something is wrong.")
    end

    # Applications of the leaf level diagonal blocks 
    for i = 1:length(ret.leaves)
        ret.leaves[i] = A_fact.leaves[i] \ ret.leaves[i];
    end
    for i = 1:max_level
        tmp_off_diag_l = [];
        tmp_off_diag_r = [];
        for j = 1:2^(i - 1)
            push!(tmp_off_diag_l, ret.U[i][j]);
            push!(tmp_off_diag_l, ret.Ul[i][j]);
            push!(tmp_off_diag_r, ret.V[i][j]);
            push!(tmp_off_diag_r, ret.Vl[i][j]);
        end

        # Update the off-diagonal low-rank factors 
        tmp_off_diag_l = block_inv_update2(tmp_off_diag_l, A_fact.leaves);
        tmp_off_diag_r = tmp_off_diag_r;

        # Write them back 
        for j = 1:2^(i - 1)
            ret.U[i][j] = tmp_off_diag_l[(j - 1) * 2 + 1];
            ret.Ul[i][j] = tmp_off_diag_l[(j - 1) * 2 + 2];
            ret.V[i][j] = tmp_off_diag_r[(j - 1) * 2 + 1];
            ret.Vl[i][j] = tmp_off_diag_r[(j - 1) * 2 + 2];
        end
    end

    return ret 
end

function Base.:\(A::hodlr, B::Union{hodlr, hodlr_nonsymmetric})#::Union{hodlr, hodlr_nonsymmetric}
    """Compute A / B as another nonsymmetric HODLR matrix.

    Args:
        A: HODLR matrix
        B: symmetric or nonsymmetric HODLR matrices

    Returns:
        A / B as another nonsymmetric HODLR matrix.
    """

    # transpose the factorization to have form:
    # A = (I+V1U1)*...*(I+VkUk)*blkdiag(A)
    A_fact = hodlr_factorize(A);
    
    return A_fact \ B;
end

############################################################################################
# HODLR-specific utility functions
############################################################################################
function block_inv_update(Uk_additive :: Array, Ul :: Array, Vl :: Array)
    """ 
        Used extensively in `hodlr_invmult`, a subroutine
        to update the low rank U factors of the previous levels.

        Uk_additive are additive low rank factors for the B matrix,
        whereas Ul and Vl are low rank off-diagonals of the A matrix.
        The subroutine applies (I + Ul*Vl')^-1*Uk_additive, with appropriate
        level slicing.
    """
    num_fine_blocks = length(Ul);
    @assert length(Vl) == num_fine_blocks
    num_coarse_blocks = length(Uk_additive);
    factor = Int(num_fine_blocks / num_coarse_blocks);
    # size of fine blocks
    n_fine = size(Ul[1], 1);
    # update each factor 
    for i = eachindex(Uk_additive)
        tmp_U = Uk_additive[i];
        for j = 1:factor
            Ulj = Ul[Int((i-1)*factor+j)];
            Vlj = Vl[Int((i-1)*factor+j)];
            # update coarse-blockwise
            idx = Int((j-1)*n_fine+1):Int(j*n_fine);
            #@show j idx Int((i-1)*factor+j) size(tmp_U)
            tmp_U[idx, :] = woodbury_inv(Ulj, Vlj', tmp_U[idx, :]);
        end
        Uk_additive[i] = tmp_U;
    end
    return Uk_additive;
end

function block_inv_update2(Uk_additive :: Array, leaves :: Array)
    """ 
        Used extensively in `hodlr_invmult`, a subroutine
        to update the leaves.
    """
    num_fine_blocks = length(leaves);
    num_coarse_blocks = length(Uk_additive);
    factor = Int(num_fine_blocks / num_coarse_blocks);
    # size of fine blocks
    n_fine = size(leaves[1], 1);
    # update each leaf block
    for i = eachindex(Uk_additive)
        tmp_U = Uk_additive[i];
        for j = 1:factor
            leaf_j = leaves[Int((i-1)*factor+j)];
            #@show norm(leaf_j)
            # update coarse-blockwise
            idx = Int((j-1)*n_fine+1):Int(j*n_fine);
            #@show norm(tmp_U[idx, :])
            #@show norm(leaf_j \ tmp_U[idx, :])
            tmp_U[idx, :] = leaf_j \ tmp_U[idx, :];
            
        end
        Uk_additive[i] = tmp_U;
        #@show norm(tmp_U)
    end
    return Uk_additive
end

function hodlr_truncate(A::Union{hodlr, hodlr_nonsymmetric}, truncate_level::Int, j::Int)::Union{
    hodlr, hodlr_nonsymmetric, Matrix{Float64}}
    """ 
        Truncates the HODLR structure, given level and index.
    """
    # new maximum level
    max_level = A.max_level - truncate_level;

    # Corner case: truncate to the leaf level 
    if max_level == 0
        return A.leaves[j];
    end

    # new index tree
    row_start_id = A.row_idx_tree[truncate_level][j][1];
    col_start_id = A.col_idx_tree[truncate_level][j][1];
    new_row_idx_tree = Array{Any}(undef, max_level);
    new_col_idx_tree = Array{Any}(undef, max_level);
    for i = 1:max_level
        new_row_idx_tree[i] = A.row_idx_tree[truncate_level+i][Int((j-1)*2^(i)+1):Int(j*2^(i))];
        new_col_idx_tree[i] = A.col_idx_tree[truncate_level+i][Int((j-1)*2^(i)+1):Int(j*2^(i))];
        # reset indexing
        for k = eachindex(new_row_idx_tree[i])
            new_row_idx_tree[i][k] = new_row_idx_tree[i][k] .- row_start_id .+ 1;
            new_col_idx_tree[i][k] = new_col_idx_tree[i][k] .- col_start_id .+ 1;
        end
    end
    # new leaves
    i = max_level;
    leaves = A.leaves[Int((j-1)*2^(i)+1):Int(j*2^(i))];
    # new U and V
    new_U = Array{Any}(undef, max_level);
    new_V = Array{Any}(undef, max_level);
    # lower blocks
    new_Ul = Array{Any}(undef, max_level);
    new_Vl = Array{Any}(undef, max_level);
    for i = 1:max_level
        start_idx = Int((j-1)*(2^(i-1))+1);
        end_idx = Int((j)*(2^(i-1)))
        new_U[i] = A.U[truncate_level+i][start_idx:end_idx];
        new_V[i] = A.V[truncate_level+i][start_idx:end_idx];
        new_Ul[i] = A.Ul[truncate_level+i][start_idx:end_idx];
        new_Vl[i] = A.Vl[truncate_level+i][start_idx:end_idx];
    end

    # construct new HODLR 
    if typeof(A) == hodlr
        return hodlr(max_level, leaves, new_U, new_V, new_row_idx_tree);
    else
        return hodlr_nonsymmetric(max_level, leaves, new_U, new_V, new_Ul, new_Vl, new_row_idx_tree, 
        new_col_idx_tree);
    end
end

function hodlr_plus_lowrank(A::Union{hodlr, hodlr_nonsymmetric, Matrix{Float64}}, U::Matrix{Float64}, 
    V::Matrix{Float64})
    """Compute HODLR or dense matrix A + global low-rank matrix defined by U * V'. This operation may
       inflate the rank of off-diagonal blocks. If A is a dense matrix, return dense matrix as output.

    Args:
        A : HODLR matrix or dense matrix
        U : Left global low-rank factor 
        V : right global low-rank factor 

    Returns:
        A nonsymmetric HODLR or dense matrix = A + U * V.'.
    """
    # Corner case: A is dense matrix
    if typeof(A) == Matrix{Float64}
        return A + U * V';
    end

    # Assertion
    @assert(size(A.U[1][1], 1) + size(A.V[1][1], 1) == size(U, 1), "Matrix shape needs to be compatible.")
    @assert(size(A.U[1][1], 1) + size(A.V[1][1], 1) == size(V, 1), "Matrix shape needs to be compatible.")

    # Generate output 
    output = deepcopy(A);

    # Off-diagonal blocks 
    for lvl = 1:output.max_level 
        for i = 1:length(output.U[lvl])
            # Upper off-diagonal blocks 
            row_idx = output.row_idx_tree[lvl][2 * (i - 1) + 1];
            col_idx = output.col_idx_tree[lvl][2 * (i - 1) + 2];

            output.U[lvl][i] = cat(output.U[lvl][i], U[row_idx, :], dims=2);
            output.V[lvl][i] = cat(output.V[lvl][i], V[col_idx, :], dims=2);

            # Lower off-diagonal blocks 
            output.Ul[lvl][i] = cat(output.Ul[lvl][i], U[col_idx, :], dims=2);
            output.Vl[lvl][i] = cat(output.Vl[lvl][i], V[row_idx, :], dims=2);
        end
    end

    # Leaf blocks 
    for i = 1:length(output.leaves)
        idx_set = output.row_idx_tree[end][i];
        output.leaves[i] += U[idx_set, :] * V[idx_set, :]';
    end

    return output;
end

function hodlr_plus_diagonal(
    A::hodlr,
    d::Vector{Float64}
)::hodlr
    """Add a diagonal matrix to HODLR matrix.

    Args:
        A: HODLR matrix to work with 
        d: diagonal entries as a vector added to the HODLR matrix 

    Returns:
        hodlr matrix = A + Diagonal(d).
    """
    # Update the leaves 
    leaves = deepcopy(A.leaves);
    for i = eachindex(leaves)
        idx = A.idx_tree[end][i];
        leaves[i] += Diagonal(d[idx]);
    end

    return hodlr(A.max_level, leaves, deepcopy(A.U), deepcopy(A.V), deepcopy(A.idx_tree));
end

function rank(A::Union{hodlr, hodlr_nonsymmetric})::Vector{Vector{Int}}
    """Returns the rank of the nonsymmetric HODLR off-diagonal blocks for each level.
    Args:
        A : nonsymmetric HODLR matrix to work with.
    Returns:
        A vector containing the ranks of each level.
    """
    ret = Vector{Vector{Int}}(undef, 0);
    for lvl = 1:A.max_level
        tmp = Vector{Int}(undef, 0);
        for matid = 1:2^(lvl - 1)
            if typeof(A) == hodlr
                push!(tmp, size(A.U[lvl][matid], 2));
            else
                push!(tmp, size(A.U[lvl][matid], 2));
                push!(tmp, size(A.Ul[lvl][matid], 2))
            end
        end
        push!(ret, tmp);
    end
    
    return ret
end

function Base.size(
    A::Union{hodlr, hodlr_nonsymmetric},
    dims::Union{Nothing, Int64}=nothing
)::Union{Int64, Tuple}
    """Returns the size of HODLR matrix

    Args:
        A : nonsymmetric HODLR or symmetric HODLR matrix to work with
        dims: dimension of the size to return 

    Returns:
        Int64 if typeof(dims) == Int64, Tuple if dims === nothing
    """
    
    row_dim, col_dim = A.row_idx_tree[1][end][end], A.col_idx_tree[1][end][end];
    
    if dims === nothing
        return (row_dim, col_dim)
    elseif typeof(dims) == Int64 
        @assert(dims == 1 || dims == 2, "HODLR matrices only have two dimensions.")

        return dims == 1 ? row_dim : col_dim;
    else
        error("Input type not supported.")
    end
end

function block_hodlr_construction(
    A1::Union{hodlr, hodlr_nonsymmetric, Matrix{Float64}}, 
    A2::Union{hodlr, hodlr_nonsymmetric, Matrix{Float64}}, 
    U::Matrix{Float64}, 
    V::Matrix{Float64}, 
    Ul::Matrix{Float64}=nothing, 
    Vl::Matrix{Float64}=nothing
    )::Union{hodlr, hodlr_nonsymmetric}
    """Construct HODLR matrix with 2x2 blocks.

    Args:
        A1 : HODLR or nonsymmetric HODLR matrix located at the upper left diagonal block
        A2 : HODLR or nonsymmetric HODLR matrix located at the lower right diagonal block
        U : Left low-rank factor matrix located at the upper right off-diagonal block
        V : Right low-rank factor matrix located at the upper right off-diagonal block
        Ul : Left low-rank factor matrix located at the lower left off-diagonal block. Defaults to nothing.
        Vl : Right low-rank factor matrix located at the lower left off-diagonal block. Defaults to nothing.

    Returns:
        A new HODLR or nonsymmetric HODLR matrix of form [A1, U * V'; V * U', A2].
    """
    # For simplicity
    is_symm =  Ul === nothing;
    Ul = Ul === nothing ? V : Ul;
    Vl = Vl === nothing ? U : Vl;

    # Corner case: If diagonal blocks are dense matrices 
    if typeof(A1) == Matrix{Float64} && typeof(A2) == Matrix{Float64} 
        @assert(size(A1, 1) == size(U, 1), "Matrix size not compatible.")
        @assert(size(A1, 2) == size(Vl, 1), "Matrix size not compatible.")
        @assert(size(A2, 1) == size(Ul, 1), "Matrix size not compatible.")
        @assert(size(A2, 2) == size(V, 1), "Matrix size not compatible.")

        # Construct output hodlr matrix 
        max_level = 1;
        leaves = [A1, A2];
        row_idx_tree = [[1:size(A1, 1), size(A1, 1) + 1:size(A1, 1) + size(A2, 1)]];
        col_idx_tree = [[1:size(A1, 2), size(A1, 2) + 1:size(A1, 2) + size(A2, 2)]];
        if isapprox(A1, A1', rtol=1e-12) && isapprox(A2, A2', rtol=1e-12) && is_symm
            return hodlr(max_level, leaves, [[U]], [[V]], row_idx_tree);
        else
            return hodlr_nonsymmetric(max_level, leaves, [[U]], [[V]], [[Ul]], [[Vl]], row_idx_tree, 
            col_idx_tree);
        end
    elseif typeof(A1) == Matrix{Float64} || typeof(A2) == Matrix{Float64}
        error("Diagonal structure not compatible. One of the diagonal blocks is dense.");
    end

    # Assertion 
    A1_rdim = A1.row_idx_tree[1][end][end];
    A1_cdim = A1.col_idx_tree[1][end][end];
    A2_rdim = A2.row_idx_tree[1][end][end];
    A2_cdim = A2.col_idx_tree[1][end][end];
    @assert(A1_rdim == size(U)[1], "Matrix size not compatible.")
    @assert(A1_cdim == size(Vl)[1], "Matrix size not compatible.")
    @assert(A2_rdim == size(Ul)[1], "Matrix size not compatible.")
    @assert(A2_cdim == size(V)[1], "Matrix size not compatible.")
    @assert(size(U, 2) == size(V, 2), "Matrix size not compatible.")
    @assert(size(Ul, 2) == size(Vl, 2), "Matrix size not compatible.")
    @assert(A1.max_level == A2.max_level, "Hierarchcal levels are not compatible.")

    # Construction 
    # Construct the new idx_tree 
    # Lift the second one 
    new_row_idx_tree = [];
    tmp = []; 
    push!(tmp, dyadic_merge(A1.row_idx_tree[1], 1));
    push!(tmp, dyadic_merge(A2.row_idx_tree[1], 1) .+ A1_rdim);
    push!(row_idx_tree, tmp);
    for lvl = 1:A1.max_level
        tmp = [];
        for i = eachindex(A1.row_idx_tree[lvl])
            push!(tmp, A1.row_idx_tree[lvl][i]);
        end
        for i = eachindex(A2.row_idx_tree[lvl])
            push!(tmp, A2.row_idx_tree[lvl][i] .+ A1_rdim);
        end
        push!(new_row_idx_tree, tmp);
    end

    new_col_idx_tree = [];
    tmp = []; 
    push!(tmp, dyadic_merge(A1.col_idx_tree[1], 1));
    push!(tmp, dyadic_merge(A2.col_idx_tree[1], 1) .+ A1_cdim);
    push!(col_idx_tree, tmp);
    for lvl = 1:A1.max_level
        tmp = [];
        for i = eachindex(A1.col_idx_tree[lvl])
            push!(tmp, A1.col_idx_tree[lvl][i]);
        end
        for i = eachindex(A2.col_idx_tree[lvl])
            push!(tmp, A2.col_idx_tree[lvl][i] .+ A1_cdim);
        end
        push!(new_col_idx_tree, tmp);
    end

    # Construct the new off-diagonal blocks
    new_U = Vector{Vector{Matrix{Float64}}}(undef, 0);
    tmp = Vector{Matrix{Float64}}(undef, 0);
    push!(tmp, U);
    push!(new_U, tmp);
    for lvl = 1:A1.max_level 
        push!(new_U, vcat(A1.U[lvl], A2.U[lvl]));
    end

    new_V = Vector{Vector{Matrix{Float64}}}(undef, 0);
    tmp = Vector{Matrix{Float64}}(undef, 0);
    push!(tmp, V);
    push!(new_V, tmp);
    for lvl = 1:A1.max_level 
        push!(new_V, vcat(A1.V[lvl], A2.V[lvl]));
    end

    new_Ul = Vector{Vector{Matrix{Float64}}}(undef, 0);
    tmp = Vector{Matrix{Float64}}(undef, 0);
    push!(tmp, Ul);
    push!(new_Ul, tmp);
    for lvl = 1:A1.max_level 
        push!(new_Ul, vcat(A1.Ul[lvl], A2.Ul[lvl]));
    end

    new_Vl = Vector{Vector{Matrix{Float64}}}(undef, 0);
    tmp = Vector{Matrix{Float64}}(undef, 0);
    push!(tmp, Vl);
    push!(new_Vl, tmp);
    for lvl = 1:A1.max_level 
        push!(new_Vl, vcat(A1.Vl[lvl], A2.Vl[lvl]));
    end

    # Construct the new leaves 
    new_leaves = vcat(A1.leaves, A2.leaves);

    # Construct the output 
    if typeof(A1) == hodlr && typeof(A2) == hodlr && is_symm
        output = hodlr(A1.max_level + 1, new_leaves, new_U, new_V, new_row_idx_tree);
    else
        output = hodlr_nonsymmetric(A1.max_level + 1, new_leaves, new_U, new_V, new_Ul, new_Vl, 
        new_row_idx_tree, new_col_idx_tree);
    end

    return output
end

function hodlr_transpose(A::Union{hodlr, hodlr_nonsymmetric})::Union{hodlr, hodlr_nonsymmetric}
    """Take transpose of the given HODLR or nonsymmetric HODLR matrix.

    Args:
        A: HODLR or nonsymmetric HODLR matrix

    Returns:
        A.'
    """
    if typeof(A) == hodlr_nonsymmetric
        return hodlr_nonsymmetric(
            A.max_level, 
            [Matrix{Float64}(transpose(A.leaves[i])) for i = eachindex(A.leaves)], 
            A.Vl, 
            A.Ul, 
            A.V, 
            A.U, 
            A.col_idx_tree, 
            A.row_idx_tree);
    elseif typeof(A) == hodlr
        return hodlr(
            A.max_level, 
            [Matrix{Float64}(transpose(A.leaves[i])) for i = eachindex(A.leaves)], 
            A.Vl, 
            A.Ul, 
            A.V, 
            A.U, 
            A.col_idx_tree, 
            A.row_idx_tree);
    else
        throw(TypeError("Matrix type not supported."))
    end
end

function hodlr_eye(template::hodlr)::hodlr
    """Create an identity matrix in HODLR format.

    Args:
        template: template HODLR matrix, the identity matrix follows the same idx_tree

    Returns:
        hodlr identity matrix.
    """

    idx_tree = deepcopy(template.idx_tree);
    leaves = deepcopy(template.leaves);
    U, V = deepcopy(template.U), deepcopy(template.V);
    # Update the leaves
    for i = eachindex(leaves)
        leaves[i] = Diagonal(ones(size(leaves[i], 1)));
    end

    # Update the off-diagonal blocks 
    for lvl = 1:template.max_level
        for i = eachindex(U[lvl])
            U[lvl][i] = zeros(size(U[lvl][i], 1), 1);
            V[lvl][i] = zeros(size(V[lvl][i], 1), 1);
        end
    end

    return hodlr(template.max_level, leaves, U, V, idx_tree);
end

function hodlr_diag(A::hodlr)::Vector{Float64}
    """Extract the diagonal of an HODLR matrix.

    Args:
        A: HODLR matrix 

    Returns:
        Vector containing the diagonal of the HODLR matrix.
    """
    tmp = [];
    for i = eachindex(A.leaves)
        push!(tmp, diag(A.leaves[i]))
    end

    return vcat(tmp...);
end

function hodlr_rank_heatmap(A::Union{hodlr, hodlr_nonsymmetric})
    """Plot the rank heatmap of the given HODLR matrix.

    Args:
        A: HODLR matrix to examine

    Returns:
        Plot of the heatmap of the rank.
    """
    tmp_rank = rank(A);
    row_sz, col_sz = A.row_idx_tree[1][2][end], A.col_idx_tree[1][2][end];
    ret = zeros(row_sz, col_sz);
    for lvl = 1:A.max_level
        for i = 1:2^(lvl - 1)
            row_idx, col_idx = A.row_idx_tree[lvl][2 * (i - 1) + 1], A.col_idx_tree[lvl][2 * i];
            ret[row_idx, col_idx] .= ones(size(ret[row_idx, col_idx])) * tmp_rank[lvl][2 * (i - 1) + 1];

            row_idx, col_idx = A.row_idx_tree[lvl][2 * i], A.col_idx_tree[lvl][2 * (i - 1) + 1];
            ret[row_idx, col_idx] .= ones(size(ret[row_idx, col_idx])) * tmp_rank[lvl][2 * i];
        end
    end

    # Leaves 
    for i = eachindex(A.leaves)
        row_idx, col_idx = A.row_idx_tree[end][i], A.col_idx_tree[end][i];
        ret[row_idx, col_idx] .= ones(size(ret[row_idx, col_idx])) * minimum(size(ret[row_idx, col_idx]));
    end

    p = heatmap(ret, yflip=true, c=:darkrainbow);

    return p;
end

