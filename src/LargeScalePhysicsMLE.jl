#     (04/11/2023) 
#     An extension of `PhysicsMLE.jl` to full-scale problems, the methods 
#     defined in this module are designed to handle large matrices (~10^6
#     observations). 

#     This module imports `PhysicsMLE.jl`, refer to the description therein 
#     for method definitions.

#     Help information:
#         Matern is positive definite (can store Cholesky factor only?):
#             https://stats.stackexchange.com/questions/322523/what-is-the-rationale-of-the-mat%C3%A9rn-covariance-function
#             https://groups.google.com/g/stan-users/c/7dFVAStRjj0
#         Runtime for Cholesky decomposition:
#             https://math.stackexchange.com/questions/2840755/what-is-the-computation-time-of-lu-cholesky-and-qr-decomposition
#         Fast matrix-vector multiplication with triangular matrix:
#         https://www.mathworks.com/matlabcentral/answers/812175-symmetric-matrix-vector-multiplication-with-only-lower-triangular-stored/?s_tid=ans_lp_feed_leaf

#     Benchmark information:
#         Fast vector dot-products: http://blog.wouterkoolen.info/Julia-DotProduct/post.html

######################################################################
# Preprocessing
######################################################################

######################################################################
# Full-scale likelihood function evaluations
######################################################################
function matern_query(x1 :: Vector{Float64}, x2 :: Vector{Float64}, prob :: MLEProblem)
    """
        Queries the Matern covariance function given two 2d points
        x1, x2 and latent parameters stored in an MLE problem instance.
    """
    field = prob.covfunc.cov;
    sigma_phi = field.σ;
    z = norm(x1 - x2, 2);
    res = apply(field, z);
    return (sigma_phi.^2).*res;
end

function matern_query(
        i :: Int64, 
        j :: Int64, 
        prob :: MLEProblem
    )
    """
        Queries the full-scale Matern latent Gaussian covariance matrix
        provided (uniform) spatial grids and query indices (in column-major
        linearized ordering).
    """
    nx, ny = length(prob.data.xgrid), length(prob.data.ygrid);
    @assert 1 <= i <= nx*ny && 1 <= j <= nx*ny;
    # convert into Cartesian indices
    ix1, jx1 = ind2sub((nx, ny), i);
    ix2, jx2 = ind2sub((nx, ny), j);
    x1 = [prob.data.xgrid[ix1], prob.data.ygrid[jx1]];
    x2 = [prob.data.xgrid[ix2], prob.data.ygrid[jx2]];
    return matern_query(x1, x2, prob);
end

function matern_query(col_idx :: Int64, prob :: MLEProblem)
    """
        For the 2d Matern covariance matrix, returns a requested column.
        The column index should be specified as a linearized Cartesian index
        in 2d.
    """
    nx, ny = length(prob.data.xgrid), length(prob.data.ygrid);
    @assert 1 <= col_idx <= nx*ny;
    ix, iy = ind2sub((nx, ny), col_idx);
    column_point = [prob.data.xgrid[ix], prob.data.ygrid[iy]];
    res = zeros(nx*ny);
    for i = 1:nx*ny
        row_ix, row_iy = ind2sub((nx, ny), i);
        row_point = [prob.data.xgrid[row_ix], prob.data.ygrid[row_iy]];
        res[i] = matern_query(row_point, column_point, prob);
    end
    return res;
end

function matern_query(
    col_indices :: Union{StepRange{Int64, Int64}, Vector{Int64}, UnitRange{Int64}},
    prob :: MLEProblem
)
    """
        Queries multiple columns at once for the full-scale Matern covariance matrix.
    """
    nx, ny = length(prob.data.xgrid), length(prob.data.ygrid);
    # preallocate
    num_cols = length(col_indices);
    C = zeros(nx*ny, num_cols);
    for i = 1:num_cols
        col_idx = col_indices[i];
        C[:, i] .= matern_query(col_idx, prob);
    end
    return C;
end

# Covariance tapering
function wendland1(x, y, tol :: Float64=0.1)
    """
        Implements covariance tapering using Wendland-1.

        If the distance between two points x, y are 
        over tolerance tol, they are considered uncorrelated.
    """
    h = norm(x - y, 2);
    if h > tol
        return 0.0;
    else
        return ( (1 - h/tol)^4 ) * (1 + 4h/tol);
    end
end

function M_wendland1(prob :: MLEProblem, tol :: Float64=0.6)
    """
        A sparse storage tapered Matern covariance matrix 
        using the Wendland-1 taper function.

        The default tapering range is correlation length of the Matern.
    """
    # get spatial grids
    xgrid, ygrid = prob.data.xgrid, prob.data.ygrid;
    nx, ny = length(xgrid), length(ygrid);

    # only store queried indices
    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();

    for i = 1:nx*ny
        row_ix, row_iy = ind2sub((nx, ny), i);
        row_point = [xgrid[row_ix], ygrid[row_iy]];
        for j = 1:nx*ny
            col_ix, col_iy = ind2sub((nx, ny), j);
            col_point = [xgrid[col_ix], ygrid[col_iy]];
            # compute point distance to decide whether to store 0
            h = norm(row_point - col_point, 2);
            if h <= tol
                tapered_cov_val = wendland1(row_point, col_point, tol) *
                    matern_query(row_point, col_point, prob);
                # store in sparse structure
                push!(row_ind, i);
                push!(col_ind, j);
                push!(entry, tapered_cov_val);
            end
        end
    end
    # return sparse Matern matrix
    M_eval = sparse(row_ind, col_ind, entry);
    return M_eval;
end

function M_wendland2(prob :: MLEProblem, tol :: Float64=0.6, num_diags=500)
    """
        A sparse storage tapered Matern covariance matrix 
        using the Wendland-1 taper function.

        The default tapering range is correlation length of the Matern.

        This routine only checks a constant number of diagonals
        to save computation time. Empirically, 500 diagonals (default)
        is enough for tapering tolerance between 0.6 and 1.0, thus preventing
        the need to compute a spatial distance at every point.
    """
    # get spatial grids
    xgrid, ygrid = prob.data.xgrid, prob.data.ygrid;
    nx, ny = length(xgrid), length(ygrid);

    # only store queried indices
    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();

    for i = 1:nx*ny
        row_ix, row_iy = ind2sub((nx, ny), i);
        row_point = [xgrid[row_ix], ygrid[row_iy]];
        # only query a set number of diagonals (j=i ± num_diags)
        for j = i:(i+num_diags)
            if j <= nx*ny
                col_ix, col_iy = ind2sub((nx, ny), j);
                col_point = [xgrid[col_ix], ygrid[col_iy]];
                # compute point distance to decide whether to store 0
                h = norm(row_point - col_point, 2);
                if h <= tol
                    tapered_cov_val = wendland1(row_point, col_point, tol) *
                        matern_query(row_point, col_point, prob);
                    if j == i
                        # store in sparse structure
                        push!(row_ind, i);
                        push!(col_ind, j);
                        push!(entry, tapered_cov_val);
                    else
                        # store in sparse structure
                        push!(row_ind, i);
                        push!(col_ind, j);
                        push!(entry, tapered_cov_val);
                        # also the lower diagonal
                        push!(row_ind, j);
                        push!(col_ind, i);
                        push!(entry, tapered_cov_val);
                    end
                end
            end
        end
    end
    # return sparse Matern matrix
    M_eval = sparse(row_ind, col_ind, entry);
    return M_eval;
end



## Fast and storage-efficient matrix-vector algorithms

#### CUR Based 
function M_mul_u_query!(
    C_buffer :: Matrix{Float64},
    u :: Union{Matrix, Vector},
    matern_skip :: Int64=40
)
    """
        A reduced runtime procedure to compute matrix-vector products
        with the latent covariance using CUR decomposition.
    """
    n = size(u, 1);
    # queried indices 
    col_indices = 1:matern_skip:n;

    # matrix vector product
    # multiply with C^T             => runtime:O(N^2/matern_skip)
    #                                  storage:O(N/matern_skip)
    #                                  vector dimension: N/matern_skip
    v_tmp = C_buffer'u; 

    # backsolve with U if invertible, else pseudoinverse => worst-case runtime:O((N/matern_skip)^3)
    try
        v_tmp[:, :] .= C_buffer[col_indices, :]\v_tmp[:, :];
    catch e
        @warn "... ... Backsolve U failed due to rank deficiency, computing pseudoinverse ...";
        v_tmp[:, :] .= pinv(C_buffer[col_indices, :])*v_tmp[:, :];
    end

    # multiply with C               => runtime:O(N^2/matern_skip)
    #                                  vector dimension: N
    u[:, :] .= C_buffer*v_tmp[:, :];
end

function K_mul_u_query(
    C_buffer :: Matrix{Float64}, 
    u :: Union{Matrix, Vector},
    prob :: MLEProblem,
    matern_skip :: Int64=40
)
    """
        Fast matrix-vector multiplication of observed covariance (with noise)
        implemented with minimal runtime and allocation.

        The latent Gaussian covariance matrix (Matern) is computed with 
        CUR decomposition with a default rank (r=n/10) where n is the problem
        size.

        Recall the covariance:

        K * v = D * (L __backslash__ M) * (L^T) __backslash__ D^T * v

        M is replaced by standard Nystrom decomposition / CUR for symmetric matrix.

        Benchmarking sparse operations:
            Example:
                >test_v = spzeros(1000);
                >test_v[1:30, :] .= randn(30);
                >test_v2 = Vector(test_v);
                >test_A = sparse(1:1000, 1:1000, randn(1000));
                >test_A2 = Matrix(test_A);
                >@benchmark test_A__backslash__test_v (sparse by sparse)
                >@benchmark test_A2__backslash__test_v (full by sparse)
                >@benchmark test_A__backslash__test_v2 (sparse by full)
                >@benchmark test_A2__backslash__test_v2 (full by full)

                >>>> Results (max runtime):
                69.853 ms
                74.659 ms
                59.028 ms < no need to store full matrix
                2.075 ms
        
        Inputs:
            C_buffer                   Queried columns from Matern covariance, can be 
                                       reused across iterations. 

            u

            prob

            matern_skip                 Reduction factor for storing Matern covariance as 
                                        CUR decomposition, defaults to 20 (rank N is divided by 20).
    """

    # unwrap parameters
    nx, ny = length(prob.data.xgrid), length(prob.data.ygrid);
    obs_idx = prob.data.obs_local_inds;
    # matrix problem size
    n_obs = size(u, 1);
    n_full = length(prob.data.u_full);
    n_samples = size(u, 2);

    n_full = nx*ny;
    # create sparse opeartor matrix => storage:O(N) (banded)
    L_eval = L(prob);
    
    # querying columns of M         => storage:O(N^2/matern_skip)
    col_indices = 1:matern_skip:n_full;
    
    # enlarge vector                => storage:O(N/matern_skip)
    v = zeros(Float64, n_full, n_samples);     # WARNING: best to use sparse, which does not work with backslash, https://github.com/JuliaLang/julia/issues/33637
    v[obs_idx, :] .= u[:, :];
    
    # solve sparse system           => runtime:O(k^2*N) where k is bandwidth
    v[:, :] .= (L_eval')\v[:, :];

    # multiply with C^T             => runtime:O(N^2/matern_skip)
    #                                  storage:O(N/matern_skip)
    #                                  vector dimension: N/matern_skip
    v_tmp = C_buffer'v; 

    # backsolve with U if invertible, else pseudoinverse => worst-case runtime:O((N/matern_skip)^3)
    try
        v_tmp[:, :] .= C_buffer[col_indices, :]\v_tmp[:, :];
    catch e
        @warn "... ... Backsolve U failed due to rank deficiency, computing pseudoinverse ...";
        v_tmp[:, :] .= pinv(C_buffer[col_indices, :])*v_tmp[:, :];
    end

    # multiply with C               => runtime:O(N^2/matern_skip)
    #                                  vector dimension: N
    v[:, :] .= C_buffer*v_tmp;

    # solve another sparse system   => runtime:O(k^2*N)
    v[:, :] .= L_eval\v[:, :];

    # sub-select to obtain D * u       vector dimension: N/matern_skip
    v_tmp = v[obs_idx, :];

    # add perturbation              => runtime:O(N)
    v_tmp[:, :] .= v_tmp[:, :] .+ (prob.sigma_u.^2).*u[:, :];
    return v_tmp;
end

#### Exact, taking advantage of Matern structure
function K_mul_u_query(
    matern_blocks :: Vector{Matrix{Float64}},
    block_permutations :: Matrix{Int64},
    u :: Union{Matrix, Vector},
    prob :: MLEProblem
)
    """
        Computes observed covariance matrix-vector product relying
        on a storage-efficient Matern-vector product querying routine.
    """
    # unwrap parameters
    nx, ny = length(prob.data.xgrid), length(prob.data.ygrid);
    obs_idx = prob.data.obs_local_inds;
    # matrix problem size
    n_obs = size(u, 1);
    n_full = length(prob.data.u_full);
    n_samples = size(u, 2);

    n_full = nx*ny;
    # create sparse opeartor matrix => storage:O(N) (banded)
    L_eval = L(prob);

    # enlarge vector                => storage:O(N/matern_skip)
    v = zeros(Float64, n_full, n_samples);     # WARNING: best to use sparse, which does not work with backslash, https://github.com/JuliaLang/julia/issues/33637
    v[obs_idx, :] .= u[:, :];
    
    # solve sparse system           => runtime:O(k^2*N) where k is bandwidth
    v[:, :] .= (L_eval')\v[:, :];

    # multiply with Matern          => runtime:O(N^2) (!!! may be a problem)
    v[:, :] .= M_mul_u_query(matern_blocks, block_permutations, v[:, :], prob);

    # solve another sparse system   => runtime:O(k^2*N)
    v[:, :] .= L_eval\v[:, :];

    # sub-select to obtain D * u       vector dimension: N/matern_skip
    v_tmp = v[obs_idx, :];

    # add perturbation              => runtime:O(N)
    v_tmp[:, :] .= v_tmp[:, :] .+ (prob.sigma_u.^2).*u[:, :];
    return v_tmp;
end

#### Tapered covariance
function K_mul_u_query(
    M_taper :: SparseMatrixCSC,
    L_eval :: SparseMatrixCSC,
    u :: Union{Matrix, Vector},
    prob :: MLEProblem
)
    """
        Matrix-vector query with tapered Matern latent covariance.
    """
    # unwrap parameters
    nx, ny = length(prob.data.xgrid), length(prob.data.ygrid);
    obs_idx = prob.data.obs_local_inds;
    # matrix problem size
    n_obs = size(u, 1);
    n_full = length(prob.data.u_full);
    n_samples = size(u, 2);

    n_full = nx*ny;

    # enlarge vector                => storage:O(N/matern_skip)
    v = zeros(Float64, n_full, n_samples);     # WARNING: best to use sparse, which does not work with backslash, https://github.com/JuliaLang/julia/issues/33637
    v[obs_idx, :] .= u[:, :];
    
    # solve sparse system           => runtime:O(k^2*N) where k is bandwidth
    v[:, :] .= (L_eval')\v[:, :];

    # multiply with Matern          => runtime:O(b*N) where b is bandwidth in tapered covariance
    v[:, :] .= M_taper'v[:, :];

    # solve another sparse system   => runtime:O(k^2*N)
    v[:, :] .= L_eval\v[:, :];

    # sub-select to obtain D * u       vector dimension: N/matern_skip
    v_tmp = v[obs_idx, :];

    # add perturbation              => runtime:O(N)
    v_tmp[:, :] .= v_tmp[:, :] .+ (prob.sigma_u.^2).*u[:, :];
    return v_tmp;
end

#### Block-exact Matern matrix-vector query
function M_mul_u_query(
    matern_blocks :: Vector, 
    block_permutations :: Matrix{Int64},
    u :: Union{Matrix, Vector},
    prob :: MLEProblem
)
    """
        Exact matrix-vector multiplication with Matern latent covariance.
        Takes O(nx^2 * ny) storage only and O(n^2) runtime where n = nx * ny.

        Inputs:
            matern_blocks                   A vector of block matrices describing
                                            local covariance. There are `ny` of 
                                            the local covariance matrices.

            block_permutations              The permutations matrix of size 
                                            (ny x ny) to compute the correct 
                                            matrix-vector product.

            u                               Returns M*u
            

            prob                            An instance of the MLE problem in its
                                            current parameteric state.
    """
    result_vec = zeros(Float64, size(u));
    local_nx, local_ny = length(prob.data.xgrid), length(prob.data.ygrid);
    @inbounds for i = 1:local_ny
        select_index_block_i = select_index(i, local_nx);
        @inbounds for j = 1:local_ny
            select_index_block_j = select_index(j, local_nx);
            result_vec[select_index_block_i, :] .+= # matern blocks are symmetric, see next line
                    matern_blocks[block_permutations[i, j]]'u[select_index_block_j, :];
        end
    end

    return result_vec;
end 


# helper functions
function select_index(i :: Int64, local_nx :: Int64)
    """
        Selects the range (in the global matrix) of 
        indices of the i-th (nx x nx) local block.
    """
    return (i-1)*local_nx+1:(i)*local_nx;
end


function select_matern_query(i :: Int64, j :: Int64, prob :: MLEProblem)
    """
        Queries the [Block(i), Block(j)] part of the full-scale
        Matern matrix (treated as a block matrix) organized by 
        enumerating y grid points.
    """
    local_xgrid, local_ygrid = prob.data.xgrid, prob.data.ygrid;
    local_nx, local_ny = length(local_xgrid), length(local_ygrid);
    @assert 1 <= i <= local_ny && 1 <= j <= local_ny
    block_row = i; block_col = j;
    ypt1, ypt2 = local_ygrid[block_row], local_ygrid[block_col];
    all_local_pts1 = Vector{Vector{Float64}}(undef, local_nx);
    all_local_pts2 = Vector{Vector{Float64}}(undef, local_nx);
    for kk = eachindex(local_xgrid)
        # create 2d points to compute correlation
        all_local_pts1[kk] = [local_xgrid[kk], ypt1];
        all_local_pts2[kk] = [local_xgrid[kk], ypt2];
    end
    # assemble local matern matrix
    matern_local = zeros(Float64, local_nx, local_nx);
    for ii = eachindex(all_local_pts1)
        for jj = eachindex(all_local_pts2)
            matern_local[ii, jj] = matern_query(all_local_pts1[ii], all_local_pts2[jj], prob);
        end
    end
    return matern_local;
end

function matern_query_blocks(prob :: MLEProblem)
    """
        Queries the necessary blocks to rebuild the entire Matern
        covariance matrix in 2d (essentially the first column) of
        blocks.
    """
    local_xgrid, local_ygrid = prob.data.xgrid, prob.data.ygrid;
    local_nx, local_ny = length(local_xgrid), length(local_ygrid);
    block_queries = Vector{Matrix{Float64}}(undef, local_ny);
    # fix block column to 1, only need the first column
    block_col = 1;
    for i = eachindex(block_queries)
        block_row = i;
        block_queries[i] = select_matern_query(block_row, block_col, prob);
    end
    return block_queries;
end

function generate_permutation(local_ny :: Int64)
    """
        Generates the block-wise permutation matrix for computing
        correct matrix-vector product.
    """
    # create permutation matrix 
    permutation = [];
    for i = 1:local_ny
        if isone(i)
            # diagonal
            push!(permutation, Pair(0, ones(Int64, local_ny)));
        else
            # upper diagonal and lower diagonal
            push!(permutation, Pair(i-1, i*ones(Int64, local_ny-i+1)));
            push!(permutation, Pair(-(i-1), i*ones(Int64, local_ny-i+1)))
        end
    end
    permutation_matrix = diagm(permutation...);
    return permutation_matrix;
end

# deprecated 
function tril_matvec(
    L :: Union{SparseMatrixCSC{Float64, Int64}, Matrix{Float64}}, 
    x :: Union{SparseVector{Float64}, Matrix{Float64}, Vector{Float64}, SparseMatrixCSC{Float64, Int64}}
)
    """
        Computes matrix-vector product A*x, where A is symmetric,
        and we are only provided with L, where L is the lower 
        triangular part of A. 

        Modifies right hand side vector/matrix in-place.

        More expensive compared to directly multiplying A and x.
    """
    n_col = size(x, 2);
    # preallocate
    y = zeros(size(x));
    for i = 1:n_col
        x_col = x[:, i];
        y[:, i] .= (L*x_col) .+ (x_col'L)' .- diag(L).*x_col;
    end
    return y;
end

function tril_matvec!(
    L :: Union{SparseMatrixCSC{Float64, Int64}, Matrix{Float64}}, 
    x :: Union{SparseVector{Float64}, Matrix{Float64}, Vector{Float64}, SparseMatrixCSC{Float64, Int64}}
)
    """
        Computes matrix-vector product A*x, where A is symmetric,
        and we are only provided with L, where L is the lower 
        triangular part of A. 

        Modifies right hand side vector/matrix in-place.

        More expensive compared to directly multiplying A and x.
    """
    n_col = size(x, 2);
    for i = 1:n_col
        x_col = x[:, i];
        x[:, i] .= (L*x_col) .+ (x_col'L)' .- diag(L).*x_col;
    end
end


## HODLR methods
### CUR version of building HODLR: O(N^2/skip) runtime and storage
function K_hodlr(
    C_buffer :: Matrix{Float64},
    prob :: MLEProblem,
    n :: Int64,
    max_level :: Int64=2, 
    local_rank :: Int64=round(Int64, n/8),
    c :: Int64=10,
    matern_skip :: Int64=40
)
    """
        HODLR version of the observed covariance matrix, computed using 
        fast matrix-vector product.

        Uses decomposition method for the latent covariance matrix.

        Inputs:
            C_buffer                Queried columns from Matern covariance. 

            prob                    MLE problem instance.

            n                       Problem size, typically number of total 
                                    observations.
            max_level               Maximum number of levels for compression.

            local_rank              Rank of off-diagonal blocks, by default 
                                    n / 8
            c                       Oversampling parameter.

            matern_skip             Reduction factor for CUR of latent covariance. 
                                    Defaults to reduction by 10.
            
    """
    # ensure divisible by required number of levels
    @assert iszero(mod(n, Int(2^max_level)));
    K_hodlr = hodlr(
        x -> K_mul_u_query(C_buffer, x, prob, matern_skip), 
        n, 
        max_level, 
        local_rank, 
        c
    );
    return K_hodlr;
end

### Blocked-exact version of HODLR O(N^2) runtime, O(ny*nx^2) storage
function K_hodlr(
    matern_blocks :: Vector{Matrix{Float64}}, 
    block_permutations :: Matrix{Int64},
    prob :: MLEProblem,
    n :: Int64,
    max_level :: Int64=2, 
    local_rank :: Int64=round(Int64, n/8),
    c :: Int64=10
)
    """
        HODLR matrix construction relying on an O(N^2) matrix-vector
        evaluation of Matern covariance, which is exact and has minimal
        storage requirements.
    """
    # ensure divisible by required number of levels
    @assert iszero(mod(n, Int(2^max_level)));
    K_hodlr = hodlr(
        x -> K_mul_u_query(matern_blocks, block_permutations, x, prob),
        n, 
        max_level, 
        local_rank, 
        c
    );
    return K_hodlr;
end

### Covariance tapering approach
function K_hodlr(
    M_tapered :: SparseMatrixCSC, 
    L_eval :: SparseMatrixCSC,
    prob :: MLEProblem,
    n :: Int64,
    max_level :: Int64=2, 
    local_rank :: Int64=round(Int64, n/8),
    c :: Int64=10
)
    """
        HODLR matrix construction relying on an O(N^2) matrix-vector
        evaluation of Matern covariance, which is exact and has minimal
        storage requirements.
    """
    # ensure divisible by required number of levels
    @assert iszero(mod(n, Int(2^max_level)));
    K_hodlr = hodlr(
        x -> K_mul_u_query(M_tapered, L_eval, x, prob),
        n, 
        max_level, 
        local_rank, 
        c
    );
    return K_hodlr;
end

######################################################################
# Full-scale score function evaluations
######################################################################
function ∂K∂kappa_mul_u_query(
    M_taper_buffer :: SparseMatrixCSC,
    L_eval :: SparseMatrixCSC,
    prob :: MLEProblem,
    u :: Union{Matrix, Vector}
)
    """
        Wrapper for computing u -> ∂K∂kappa * u
    """
    # evaluate sparse ∂L∂kappa 
    ∂L∂kappa_eval = ∂L∂kappa(prob);
    return ∂K∂PDE_param_mul_u_query(M_taper_buffer, L_eval, ∂L∂kappa_eval, prob, u);
end

function ∂K∂v_mul_u_query(
    M_taper_buffer :: SparseMatrixCSC,
    L_eval :: SparseMatrixCSC,
    prob :: MLEProblem,
    u :: Union{Matrix, Vector}
)
    """
        Wrapper for computing u -> ∂K∂kappa * u
    """
    v_model_p1 = length(prob.v_model.v1_theta);
    v_model_p2 = length(prob.v_model.v2_theta);
    res = Dict{String, Vector{Union{Matrix, Vector}}}(
        "grad1" => Vector{Union{Matrix, Vector}}(undef, v_model_p1),
        "grad2" => Vector{Union{Matrix, Vector}}(undef, v_model_p2)
    );
    # query all ∂L∂v
    ∂L∂v_eval = ∂L∂v(prob);

    # evaluate sparse ∂L∂v for all components
    for i = 1:v_model_p1
        res["grad1"][i] = ∂K∂PDE_param_mul_u_query(
            M_taper_buffer, L_eval, ∂L∂v_eval["grad1"][i], prob, u
        );
    end

    for i = 1:v_model_p2
        res["grad2"][i] = ∂K∂PDE_param_mul_u_query(
            M_taper_buffer, L_eval, ∂L∂v_eval["grad2"][i], prob, u
        );
    end

    return res;
end

function ∂K∂c_mul_u_query(
    M_taper_buffer :: SparseMatrixCSC,
    L_eval :: SparseMatrixCSC,
    prob :: MLEProblem,
    u :: Union{Matrix, Vector}
)
    """
        Wrapper for computing u -> ∂K∂c * u
    """
    # evaluate sparse ∂L∂c
    ∂L∂c_eval = ∂L∂c(prob);
    return ∂K∂PDE_param_mul_u_query(M_taper_buffer, L_eval, ∂L∂c_eval, prob, u);
end


function ∂K∂PDE_param_mul_u_query(
    M_taper_buffer :: SparseMatrixCSC,
    L_eval :: SparseMatrixCSC,
    ∂L∂theta_eval :: SparseMatrixCSC,
    prob :: MLEProblem,
    u :: Union{Matrix, Vector}
)
    """
        Efficient O(N) matvec query algorithm for computing 
            u -> ∂K∂θ * u
        where θ is a PDE parameter.

        ∂K∂kappa = 
            D * 
            [
                ( -L_inv * ∂L∂kappa * L_inv * M * L_inv )
                    +
                (-L_inv * M * L_inv^T * ∂L∂kappa^T * L_inv^T )
            ] 
            * D^T
    """
    # unpack parameters
    nx, ny = length(prob.data.xgrid), length(prob.data.ygrid);
    obs_idx = prob.data.obs_local_inds;
    # matrix problem size
    n_full = length(prob.data.u_full);
    n_samples = size(u, 2);
    n_full = nx*ny;


    v = zeros(Float64, n_full, n_samples);     
    v[obs_idx, :] .= u[:, :];

    # First term:

    # solve sparse system 
    w1 = L_eval \ v[:, :];

    # multiply with sparse latent covariance
    w1[:, :] .= M_taper_buffer'w1[:, :];

    # another sparse solve
    w1[:, :] .= L_eval \ w1[:, :];

    # multiply with sparse derivative
    w1[:, :] .= ∂L∂theta_eval * w1[:, :];

    # another sparse solve and negate
    w1[:, :] .= -L_eval \ w1[:, :];

    # subselect
    w1 = w1[obs_idx, :];

    # --------------------------------------------------

    # Second term:

    # solve sparse system 
    w2 = (L_eval')\v[:, :];

    # multiply with sparse derivative (transpose)
    w2[:, :] .= ∂L∂theta_eval'w2[:, :];

    # sparse solve (transpose)
    w2[:, :] .= (L_eval') \ w2[:, :];

    # multiply sparse latent covariance
    w2[:, :] .= M_taper_buffer'w2[:, :];

    # another sparse solve and negate
    w2[:, :] .= -L_eval \ w2[:, :];

    # subselect
    w2 = w2[obs_idx, :];

    # --------------------------------------------------
    # Final result

    res = w1 + w2;
    return res;
end

#### HODLR derivative constructions
function ∂K∂kappa_hodlr(
    M_tapered :: Any, 
    L_eval :: SparseMatrixCSC,
    prob :: MLEProblem,
    n :: Int64,
    max_level :: Int64=2, 
    local_rank :: Int64=round(Int64, n/8),
    c :: Int64=10
)  
    @assert iszero(mod(n, Int(2^max_level)));
    hodlr_eval = hodlr(
        x -> ∂K∂kappa_mul_u_query(M_tapered, L_eval, prob, x),
        n, 
        max_level, 
        local_rank, 
        c
    );
    return hodlr_eval;
end

function ∂K∂v_hodlr(
    M_tapered :: Any, 
    L_eval :: SparseMatrixCSC,
    prob :: MLEProblem,
    n :: Int64,
    max_level :: Int64=2, 
    local_rank :: Int64=round(Int64, n/8),
    c :: Int64=10
)  
    v_model_p1 = length(prob.v_model.v1_theta);
    v_model_p2 = length(prob.v_model.v2_theta);
    @assert iszero(mod(n, Int(2^max_level)));
    # query all ∂L∂v (sparse)
    ∂L∂v_eval = ∂L∂v(prob);

    res = Dict{String, Vector{hodlr}}(
        "grad1" => Vector{hodlr}(undef, v_model_p1),
        "grad2" => Vector{hodlr}(undef, v_model_p2)
    );

    # compute HODLR for each parameter
    for i = 1:v_model_p1
        matvec = x -> ∂K∂PDE_param_mul_u_query(
            M_tapered, L_eval, ∂L∂v_eval["grad1"][i], prob, x
        );
        res["grad1"][i] = hodlr(
            matvec,
            n, 
            max_level, 
            local_rank, 
            c
        );
    end

    for i = 1:v_model_p2
        matvec = x -> ∂K∂PDE_param_mul_u_query(
            M_tapered, L_eval, ∂L∂v_eval["grad2"][i], prob, x
        );
        res["grad2"][i] = hodlr(
            matvec,
            n, 
            max_level, 
            local_rank, 
            c
        );
    end
    return res;
end

function ∂K∂c_hodlr(
    M_tapered :: Any, 
    L_eval :: SparseMatrixCSC,
    prob :: MLEProblem,
    n :: Int64,
    max_level :: Int64=2, 
    local_rank :: Int64=round(Int64, n/8),
    c :: Int64=10
)  
    @assert iszero(mod(n, Int(2^max_level)));
    hodlr_eval = hodlr(
        x -> ∂K∂c_mul_u_query(M_tapered, L_eval, prob, x),
        n, 
        max_level, 
        local_rank, 
        c
    );
    return hodlr_eval;
end

#### HODLR-based score evaluations
function score∂PDE_param(
    K_inv_u_eval :: Vector{Float64},
    K_hodlr_fact_eval :: hodlr_fact,
    ∂K∂PDE_param_eval :: hodlr,
    prob :: MLEProblem
) 
    """
        General routine for evaluating HODLR-based score function
        of PDE parameters provided HODLR structs.

        Assumes mean 0.
    """
    u = prob.u_noisy;
    # compute K_inv_K_j
    K_inv_K_j_eval = hodlr_invmult(K_hodlr_fact_eval, ∂K∂PDE_param_eval);
    # first term, compute trace
    trace_K_inv_K_j_eval = hodlr_tr(K_inv_K_j_eval);
    # second term
    K_inv_K_j_K_inv_u_eval = hodlr_prod(K_inv_K_j_eval, K_inv_u_eval);
    u_T_K_inv_K_j_K_inv_u_eval = u'K_inv_K_j_K_inv_u_eval;

    # aggregate results
    score = -0.5 * trace_K_inv_K_j_eval + 0.5 * u_T_K_inv_K_j_K_inv_u_eval;
    return score;
end

function score∂v(
    K_inv_u_eval :: Vector{Float64},
    K_hodlr_fact_eval :: hodlr_fact,
    ∂K∂v_eval :: Dict{String, Vector{hodlr}},
    prob :: MLEProblem
)
    """
        Wrapper for evaluating the score function with respect to 
        all velocity parameters, in HODLR format.
    """
    # preallocate dictionary to store scores
    v_model_p1 = length(prob.v_model.v1_theta);
    v_model_p2 = length(prob.v_model.v2_theta);
    res = Dict{String, Vector{Float64}}(
        "grad1" => Vector{Float64}(undef, v_model_p1),
        "grad2" => Vector{Float64}(undef, v_model_p2)
    );
    for i = 1:v_model_p1
        res["grad1"][i] = score∂PDE_param(K_inv_u_eval, K_hodlr_fact_eval, ∂K∂v_eval["grad1"][i], prob);
    end
    for i = 1:v_model_p2
        res["grad2"][i] = score∂PDE_param(K_inv_u_eval, K_hodlr_fact_eval, ∂K∂v_eval["grad2"][i], prob);
    end
    return res;
end






######################################################################
# Full-scale likelihood function evaluations
######################################################################
## Likelihood computations

function log_likelihood(
    K_hodlr :: hodlr,
    prob :: MLEProblem
)
    """
        HODLR version of log-likelihood evaluation, suitable for full-scale MLE
        problems.
    """
    # compute factorization
    K_hodlr_fact = hodlr_factorize(K_hodlr);
    # unpack parameters
    u = prob.u_noisy;
    # hodlr evaluation of log-determinant
    logabsdet_K_hodlr = hodlr_logdet(K_hodlr_fact);
    # K^-1u
    tmp = hodlr_solve(K_hodlr_fact, prob.u_noisy);
    res = -0.5*logabsdet_K_hodlr-0.5*u'*tmp;
    return res;
end

function log_likelihood(
    K_inv_u_eval :: Vector{Float64},
    K_hodlr :: hodlr,
    prob :: MLEProblem
)
    """
        HODLR version of log-likelihood evaluation, suitable for full-scale MLE
        problems. K^-1*u is precomputed.
    """
    # compute factorization
    K_hodlr_fact = hodlr_factorize(K_hodlr);
    # unpack parameters
    u = prob.u_noisy;
    # hodlr evaluation of log-determinant
    logabsdet_K_hodlr = hodlr_logdet(K_hodlr_fact);

    res = -0.5*logabsdet_K_hodlr-0.5*u'*K_inv_u_eval;
    return res;
end

function solve_large!(
    prob :: MLEProblem,
    param_constraints :: Matrix{Float64},
    warm_start :: Union{String, Vector, Nothing},
    M_query_mode :: String,
    optimizer :: Any,
    max_level :: Int64,
    local_rank :: Int64,
    c :: Int64=20
)
    """
        Main function for solving full-scale MLE. Refer to `solve!` for more details 
        on functionality. Due to large-scale, only "nlls" warm start strategy is provided
        for comparison.

        This code does not allow identification of latent parameters (i.e. Matern 
        matrix is considered fixed).

    """
    # number of trainable parameters
    num_trainable = size(param_constraints, 1);
    # compute warm start
    if warm_start !== nothing
        if warm_start == "nlls"
            # solves nonlinear least squares problem
            warm_start_nonlinear_ls!(prob, param_constraints);
        elseif isa(warm_start, Vector)
            # if we inputted global warm starts, simply update
            update!(prob, warm_start);
        else
            # use random starting point
            tmp = zeros(num_trainable);
            for i = eachindex(tmp)
                # sample randomly
                tmp_bounds = param_constraints[i, :];
                tmp[i] = rand(Uniform(tmp_bounds[1], tmp_bounds[2]));
            end
        end
    end

    # initial parameters
    _params_init = dump_trainable_parameters(prob);
    nx = length(prob.data.xgrid); 
    ny = length(prob.data.ygrid);
    n_obs = length(prob.u_noisy);

    # precompute M if we do not intend on changing latent parameters (only need computed once)
    M_flag = "sigma_phi" in prob.update_manual || "l" in prob.update_manual || "nu" in prob.update_manual;
    if M_flag
        error("... Optimizing latent parameters not currently supported! ");
    else
        # query (blocks of) M only once
        println("[ Full Scale Optimization ]=>Querying Matern latent via Wendland. ");
        M_query = M_wendland2(prob);
        println("[ Full Scale Optimization ]=>Matern query finished. ");
    end

    # internal function that evaluates (negative) log-likelihood as loss function
    function _f(theta)
        @info theta
        # update problem state
        update!(prob, theta);
        # ----------------------------------------
        # Precompute important quantities
        # ----------------------------------------
        # evaluate observation covariance in HODLR form
        L_eval = L(prob);
        println("[ Full Scale Optimization ]=>Computing HODLR. ")
        K_eval = K_hodlr(M_query, L_eval, prob, n_obs, max_level, local_rank, c);
        println("[ Full Scale Optimization ]=>HODLR Finished. ")
        println("[ Full Scale Optimization ]=>Computing HODLR factorization. ");
        K_fact_eval = hodlr_factorize(K_eval);
        println("[ Full Scale Optimization ]=>Computing HODLR inverse vector. ")
        K_inv_u_eval = hodlr_solve(K_fact_eval, prob.u_noisy);
        println("[ Full Scale Optimization ]=>HODLR inverse vector finished. ")
        # compute likelihood
        return -log_likelihood(K_inv_u_eval, K_eval, prob);
    end

    # optimization
    _lower = param_constraints[:, 1];
    _upper = param_constraints[:, 2];
    # initialize optimization algorithm (gradient free)
    _optimization_alg = optimizer;

    # define function to be optimized (uses forward autodiff)
    _optimization_function = _f;
        #OnceDifferentiable(_f, _params_init; autodiff=:forward);

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

    # return optimizer information
    return _optimizer_result;
end

function solve_large!(
    prob :: MLEProblem,
    param_constraints :: Matrix{Float64},
    warm_start :: Union{String, Vector, Nothing},
    optimizer :: Any,
    max_level :: Int64,
    local_rank :: Int64,
    c :: Int64=20
)
    """
        Main function for solving full-scale MLE. Refer to `solve!` for more details 
        on functionality. Due to large-scale, only "nlls" warm start strategy is provided
        for comparison.

        This code assumes access to score equations in HODLR format and
        the use of a gradient-based method is recommended.

        This code does not allow identification of latent parameters (i.e. Matern 
        matrix is considered fixed).

        TODO: solve score equations directly to prevent non-PSD issue.

    """
    # number of trainable parameters
    num_trainable = size(param_constraints, 1);
    # compute warm start
    if warm_start !== nothing
        if warm_start == "nlls"
            # solves nonlinear least squares problem
            warm_start_nonlinear_ls!(prob, param_constraints);
        elseif isa(warm_start, Vector)
            # if we inputted global warm starts, simply update
            update!(prob, warm_start);
        else
            # use random starting point
            tmp = zeros(num_trainable);
            for i = eachindex(tmp)
                # sample randomly
                tmp_bounds = param_constraints[i, :];
                tmp[i] = rand(Uniform(tmp_bounds[1], tmp_bounds[2]));
            end
        end
    end

    # initial parameters
    _params_init = dump_trainable_parameters(prob);
    nx = length(prob.data.xgrid); 
    ny = length(prob.data.ygrid);
    n_obs = length(prob.u_noisy);

    # precompute M if we do not intend on changing latent parameters (only need computed once)
    M_flag = "sigma_phi" in prob.update_manual || "l" in prob.update_manual || "nu" in prob.update_manual;
    if M_flag
        error("... Optimizing latent parameters not currently supported! ");
    else
        # query (blocks of) M only once
        println("[ Full Scale Optimization ]=>Querying Matern latent via Wendland. ");
        M_query = M_wendland2(prob);
        println("[ Full Scale Optimization ]=>Matern query finished. ");
    end

    # function that computes both objective and gradient at once, reusing calculations
    function _fg!(F, G, theta)
        @info theta
        # first update problem state
        update!(prob, theta);
        # ----------------------------------------
        # Common computations
        # ----------------------------------------
        # number of observations
        n = length(prob.u_noisy);
        # evaluate operator
        L_eval = L(prob);
        # evaluate observation covariance in HODLR
        K_eval = K_hodlr(M_query, L_eval, prob, n, max_level, local_rank, c);
        # evaluate factorized HODLR observation covariance
        K_fact_eval = hodlr_factorize(K_eval);
        # pre-compute K_inv_u
        K_inv_u_eval = hodlr_solve(K_fact_eval, prob.u_noisy);

        # compute score first
        # ----------------------------------------
        # Score (only those in `update_manual`)
        # ----------------------------------------
        if G !== nothing
            # initialize with full vector and NaN's, then filter
            tmp = mask_parameters(prob);

            if "kappa" in prob.update_manual
                # evaluate HODLR form of ∂K∂kappa
                ∂K∂kappa_eval = ∂K∂kappa_hodlr(M_query, L_eval, prob, n, max_level, local_rank, c);
                tmp[4] = score∂PDE_param(K_inv_u_eval, K_fact_eval, ∂K∂kappa_eval, prob);
            end

            if "v1_theta" in prob.update_manual || "v2_theta" in prob.update_manual
                # number of parameters
                v1_params = length(prob.v_model.v1_theta);
                v2_params = length(prob.v_model.v2_theta);
                # compute ∂K∂v in HODLR
                ∂K∂v_eval = ∂K∂v_hodlr(M_query, L_eval, prob, n, max_level, local_rank, c);
                # compute score∂v
                score∂v_eval = score∂v(K_inv_u_eval, K_fact_eval, ∂K∂v_eval, prob);
                if "v1_theta" in prob.update_manual
                    tmp[5:5+v1_params-1] .= score∂v_eval["grad1"][:];
                end

                if "v2_theta" in prob.update_manual
                    tmp[5+v1_params:5+v1_params+v2_params-1] .= score∂v_eval["grad2"][:];
                end
            end

            if "c" in prob.update_manual
                # evaluate HODLR form of ∂K∂c
                ∂K∂c_eval = ∂K∂c_hodlr(M_query, L_eval, prob, n, max_level, local_rank, c);
                score∂c_eval = score∂PDE_param(K_inv_u_eval, K_fact_eval, ∂K∂c_eval, prob);
                tmp[end] = score∂c_eval;
            end
            # filter and update G (negative score)
            G[:] .= -filter(!isnan, tmp);
        end

        # ----------------------------------------
        # Likelihood
        # ----------------------------------------
        if F !== nothing
            return -log_likelihood(K_inv_u_eval, K_eval, prob);
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

    # return optimizer information
    return _params_final;
end
######################################################################
# Computing second order information (HODLR)
######################################################################
function fisher_hodlr(
    prob :: MLEProblem,
    max_level :: Int64,
    local_rank :: Int64,
    c :: Int64=20
)
    """
        Given an MLE problem in its current parameteric state,
        computes the full Fisher information matrix of all trained
        parameters. In HODLR format with respecified hyperparameters.
    """

end

######################################################################
# Postprocessing and sampling imputations
######################################################################
#### Predictive sampling, CUR version
function predictive_mean(
    prob :: MLEProblem,
    K_hodlr_fact :: hodlr_fact,
    skip :: Int64=40
)
    """
        Computes predictive mean for hidden observations.

        This version uses CUR decomposition and is not exact.
    """
    # build HODLR factorization using CUR
    u = prob.u_noisy;

    n_full = length(prob.data.u_full);
    
    # query CUR factor
    col_indices = 1:skip:n_full;
    C_buffer = matern_query(col_indices, prob);
    v = hodlr_solve(K_hodlr_fact, u);
    # pad with 0's
    v_tilde = zeros(Float64, n_full);
    v_tilde[prob.data.obs_local_inds] .= v;
    # sparse solve with PDE operator
    L_eval = L(prob);
    w = (L_eval')\v_tilde;
    # multiply with M using CUR
    w[:] .= M_mul_u_query!(C_buffer, w[:], skip);
    # another sparse solve
    w[:] .= L_eval\w[:];
    # subselect on hidden indices
    return w[prob.data.mask_local_inds];
end

#### predictive mean (block exact version)
function predictive_mean(
    prob :: MLEProblem,
    K_hodlr_fact :: hodlr_fact,
    matern_blocks :: Vector{Matrix{Float64}},
    block_permutations :: Matrix{Int64}
)
    """
        Uses exact block matrix-vector procedure to compute predictive mean.
    """
    # build HODLR factorization using CUR
    u = prob.u_noisy;

    n_full = length(prob.data.u_full);

    v = hodlr_solve(K_hodlr_fact, u);

    # pad with 0's
    v_tilde = zeros(Float64, n_full);

    v_tilde[prob.data.obs_local_inds] .= v;

    # sparse solve with PDE operator
    L_eval = L(prob);

    w = (L_eval')\v_tilde;
    # multiply with M using block matrix-vector query
    w[:] .= M_mul_u_query(matern_blocks, block_permutations, w[:], prob);

    # another sparse solve
    w[:] .= L_eval\w[:];

    # subselect on hidden indices
    return w[prob.data.mask_local_inds];
end

#### predictive covariance matvec query (CUR)
function predictive_covariance_mul_u_query(
    prob :: MLEProblem,
    C_buffer :: Matrix{Float64},
    K_hodlr_fact :: hodlr_fact,
    v :: Union{Matrix, Vector},
    skip :: Int64=40
)
    """
        A routine to query matrix-vector product with predictive
        covariance matrix in quadratic runtime (reduced by a factor).
        
        This routine uses CUR decomposition and is not exact.
    """
    n_hid = length(prob.data.mask_local_inds);
    n_obs = length(prob.data.u_observed);
    n_full = length(prob.data.u_full);

    # pad with 0's
    v_tilde = zeros(Float64, n_full);
    v_tilde[prob.data.mask_local_inds] .= v;

    # sparse solve
    L_eval = L(prob);
    w = (L_eval')\v_tilde;

    # multiply with M using CUR

    # query CUR factor (can be reused)
    w[:] .= M_mul_u_query!(C_buffer, w[:], skip);

    # another sparse solve
    w[:] .= L_eval\w[:];

    # subselect observation locations and hidden locations
    z1, z2 = w[prob.data.mask_local_inds], w[prob.data.obs_local_inds];

    # evaluate HODLR
    z2[:] .= hodlr_solve(K_hodlr_fact, z2);

    # pad with 0's
    z2_new = zeros(Float64, n_full);
    z2_new[prob.data.obs_local_inds] .= z2;
    
    # solve with PDE operator
    z2_new[:] .= (L_eval')\z2_new[:];

    # multiply with M using CUR
    z2_new[:] .= M_mul_u_query!(C_buffer, z2_new[:], skip);

    # another sparse solve
    z2_new[:] .= L_eval\z2_new[:];

    # subselect hidden index
    q2 = z2_new[prob.data.mask_local_inds];

    # result vector
    return z1 - q2;
end

## predictive covariance matvec query (exact block)
function predictive_covariance_mul_u_query(
    prob :: MLEProblem,
    matern_blocks :: Vector{Matrix{Float64}},
    block_permutations :: Matrix{Int64},
    K_hodlr_fact :: hodlr_fact,
    v :: Union{Matrix, Vector}
)
    """
        Computes matrix-vector product with the predictive covariance
        matrix, using block-wise multiplication to achieve exact Matern
        matrix-vector product.
    """
    n_full = length(prob.data.u_full);

    # pad with 0's
    v_tilde = zeros(Float64, n_full);
    v_tilde[prob.data.mask_local_inds] .= v;

    # sparse solve
    L_eval = L(prob);
    w = (L_eval')\v_tilde;

    # query exact block matvec procedure
    w[:] .= M_mul_u_query(matern_blocks, block_permutations, w, prob);

    # another sparse solve
    w[:] .= L_eval\w[:];

    # subselect observation locations and hidden locations
    z1, z2 = w[prob.data.mask_local_inds], w[prob.data.obs_local_inds];

    # evaluate HODLR
    z2[:] .= hodlr_solve(K_hodlr_fact, z2);

    # pad with 0's
    z2_new = zeros(Float64, n_full);
    z2_new[prob.data.obs_local_inds] .= z2;
    
    # solve with PDE operator
    z2_new[:] .= (L_eval')\z2_new[:];

    # query exact block matvec procedure
    z2_new[:] .= M_mul_u_query(matern_blocks, block_permutations, z2_new, prob);

    # another sparse solve
    z2_new[:] .= L_eval\z2_new[:];

    # subselect hidden index
    q2 = z2_new[prob.data.mask_local_inds];

    # result vector
    return z1 - q2;
end

## Imputations using CUR decomposition
function impute_large!(
    prob :: MLEProblem,
    x0 :: Vector{Float64},
    b :: Vector{Float64},
    tol :: Float64=1e-3,
    num_iter :: Int64=100,
    skip :: Int64=80,
    hodlr_num_levels :: Int64=2,
    hodlr_rank :: Int64=256
)
    """
        Generates a sample from the Gaussian predictive distribution 
        using a conjugate-gradient based iterative method. The routine
        only requires procedures to evaluate the matrix-vector product
        with the covariance matrix.

        x0, b                       Initial solution vector and right hand
                                    side for starting the iterations.

        tol                         Tolerance on the residual norm to stop the 
                                    iteration.
                                    
        num_iter                    number of iterations for the CG, 
                                    recommended to be larger than 60.

        This procedure uses CUR decomposition for multiplications 
        with Matern covariance.

        See:
            Sampling Gaussian Distributions in Krylov Spaces with 
            Conjugate Gradients.
    """
    n_obs = length(prob.data.u_observed);
    n_full = length(prob.data.u_full);

    # form CUR matrix by querying
    col_indices = 1:skip:n_full;
    println("[ Full Scale ]=>Querying CUR buffer ... ");
    C_buffer = matern_query(col_indices, prob);
    println("... ... Query finished ... ")

    # form K matrix in HODLR form
    println("[ Full Scale ]=>Constructing HODLR ... ");
    K_hodlr_eval = K_hodlr(C_buffer, prob, n_obs, hodlr_num_levels, hodlr_rank, 10, skip);
    println("... ... HODLR finished ... ")

    println("[ Full Scale ]=>Factorizing HODLR ... ");
    K_hodlr_fact = hodlr_factorize(K_hodlr_eval);
    println("... ... HODLR factorization finished ... ")

    # begin Krylov iteration to sample from N(0, K_hid^-1)
    x = x0;
    if iszero(norm(x))
        r = b;
    else
        println("[ Full Scale ]=>Computing large matrix-vector query ... ");
        r = b - predictive_covariance_mul_u_query(prob, C_buffer, K_hodlr_fact, x, skip);
        println("... ... Finished ... ")
    end
    p = r;
    println("[ Full Scale ]=>Computing large matrix-vector query ... ");
    tmp = predictive_covariance_mul_u_query(prob, C_buffer, K_hodlr_fact, p, skip);
    println("... ... Finished ... ")
    d = p'tmp;
    y = x;
    iter = 1;
    while iter <= num_iter
        r_norm = r'r;
        println("... CG sampling step=$(iter) with r_norm=$(r_norm). ")
        if r_norm <= tol
            println("... CG sampler terminated early with iter=$(iter), residual norm = $(r_norm). ")
            break;
        end
        # step 1
        # iterative CG step
        gamma = r_norm/d;
        # step 2
        x[:] .= x[:] .+ gamma .* p[:];
        # step 3
        z = randn(size(y));
        y[:] .= y[:] .+ (z./sqrt(d)).*p[:];
        # step 4: recompute residual
        println("[ Full Scale ]=>Computing large matrix-vector query ... ");
        r[:] .= r[:] .- gamma .* predictive_covariance_mul_u_query(prob, C_buffer, K_hodlr_fact, p, skip);
        println("... ... Finished ... ")

        # step 5: 
        beta = r'r / r_norm;  # the paper says -r'r/r_norm, however that diverges
        # step 6:
        p[:] .= r[:] .- beta .* p[:];
        # step 7:
        println("[ Full Scale ]=>Computing large matrix-vector query ... ");
        tmp = predictive_covariance_mul_u_query(prob, C_buffer, K_hodlr_fact, p, skip);
        println("... ... Finished ... ")
        d = p'tmp;
        iter = iter + 1;
    end
    # one more matrix-vector to generate sample from N(0, K_hid)
    println("[ Full Scale ]=>Computing large matrix-vector query ... ");
    c = predictive_covariance_mul_u_query(prob, C_buffer, K_hodlr_fact, y, skip);
    println("... ... Finished ... ")
    # add predictive mean to generate N(m_hid, K_hid)
    m_predictive = predictive_mean(prob, K_hodlr_fact, skip);
    c[:] .= c[:] .+ m_predictive[:];

    # store imputations in problem data
    prob.data.u_imputations[:] .= c[:];
    return c;
end

## Imputations with exact Matern covariance block matvec query
function impute_large!(
    prob :: MLEProblem,
    x0 :: Vector{Float64},
    b :: Vector{Float64},
    tol :: Float64=1e-3,
    num_iter :: Int64=100,
    hodlr_num_levels :: Int64=2,
    hodlr_rank :: Int64=256
)
    """
        A large-scale imputation routine that uses exact block matrix-vector
        query to compute Matern covariance.
    """
    n_obs = length(prob.data.u_observed);
    n_full = length(prob.data.u_full);

    # physical domain spatial grid sizes 
    local_nx, local_ny = length(prob.data.xgrid), length(prob.data.ygrid);

    println("[ Full Scale ]=>Querying Matern local blocks ... ");
    # query necessary blocks of Matern matrix
    matern_blocks = matern_query_blocks(prob);
    block_permutations = generate_permutation(local_ny);
    println("... ... Query finished ... ")

    # form K matrix in HODLR form
    println("[ Full Scale ]=>Constructing HODLR ... ");
    K_hodlr_eval = K_hodlr(matern_blocks, block_permutations, prob, n_obs, hodlr_num_levels, hodlr_rank, 10);
    println("... ... HODLR finished ... ")

    println("[ Full Scale ]=>Factorizing HODLR ... ");
    K_hodlr_fact = hodlr_factorize(K_hodlr_eval);
    println("... ... HODLR factorization finished ... ")

    # begin Krylov iteration to sample from N(0, K_hid^-1)
    x = x0;
    if iszero(norm(x))
        r = b;
    else
        println("[ Full Scale ]=>Computing large matrix-vector query ... ");
        r = b - predictive_covariance_mul_u_query(prob, matern_blocks, block_permutations, K_hodlr_fact, x);
        println("... ... Finished ... ")
    end
    p = r;
    println("[ Full Scale ]=>Computing large matrix-vector query ... ");
    tmp = predictive_covariance_mul_u_query(prob, matern_blocks, block_permutations, K_hodlr_fact, p);
    println("... ... Finished ... ")
    d = p'tmp;
    y = x;
    iter = 1;
    while iter <= num_iter
        r_norm = r'r;
        println("... CG sampling step=$(iter) with r_norm=$(r_norm). ")
        if r_norm <= tol
            println("... CG sampler terminated early with iter=$(iter), residual norm = $(r_norm). ")
            break;
        end
        # step 1
        # iterative CG step
        gamma = r_norm/d;
        # step 2
        x[:] .= x[:] .+ gamma .* p[:];
        # step 3
        z = randn(size(y));
        y[:] .= y[:] .+ (z./sqrt(d)).*p[:];
        # step 4: recompute residual
        println("[ Full Scale ]=>Computing large matrix-vector query ... ");
        r[:] .= r[:] .- gamma .* predictive_covariance_mul_u_query(prob, matern_blocks, block_permutations, K_hodlr_fact, p);
        println("... ... Finished ... ")

        # step 5: 
        beta = r'r / r_norm;  # the paper says -r'r/r_norm, however that diverges
        # step 6:
        p[:] .= r[:] .- beta .* p[:];
        # step 7:
        println("[ Full Scale ]=>Computing large matrix-vector query ... ");
        tmp = predictive_covariance_mul_u_query(prob, matern_blocks, block_permutations, K_hodlr_fact, p);
        println("... ... Finished ... ")
        d = p'tmp;
        iter = iter + 1;
    end
    # one more matrix-vector to generate sample from N(0, K_hid)
    println("[ Full Scale ]=>Computing large matrix-vector query ... ");
    c = predictive_covariance_mul_u_query(prob, matern_blocks, block_permutations, K_hodlr_fact, y);
    println("... ... Finished ... ")
    # add predictive mean to generate N(m_hid, K_hid)
    m_predictive = predictive_mean(prob, K_hodlr_fact, matern_blocks, block_permutations);
    c[:] .= c[:] .+ m_predictive[:];

    # store imputations in problem data
    prob.data.u_imputations[:] .= c[:];
    return c;
end

######################################################################
# Additional helper functions
######################################################################
function matern_spde_operator(
    prob :: MLEProblem
)
    """
        The stochastic partial differential equations (SPDE) approach
        for generating Matern latent covariance. This routine generates
        the operator:

            (1/l^2 - Δ) (γ * ⋅) 

        Note: this is a special case of the fractional operator:

            (1/l^2 - Δ)^((ν + d/2)/2) (γ * ⋅) 
        where ν=1, d=2.

        !!! Not working, does not give correct covariance.
    """
    @assert prob.nu == 1
    d = 2;
    nu = 1;
    # compute SPDE parameters
    l = prob.l;
    sigma_phi = prob.sigma_phi;
    # ν = 1, Γ(1) = Γ(2) = 1
    # compute gamma, related to Matern marginal variance
    gamma = l / sigma_phi / 2 / sqrt(π);

    xgrid = prob.data.xgrid;
    ygrid = prob.data.ygrid;

    # SPDE parameters
    kappa = gamma;
    c = gamma / l / l;
    return reaction_diffusion_homogeneous_neumann(xgrid, ygrid, kappa, c);
end