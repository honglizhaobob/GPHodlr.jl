# Helper functions for creating matrices in the `basic hodlr accuracy`
# test suite.

using GPHodlr: dyadic_idx, dyadic_merge, randn_symmetric
######################################################################
function dummy_matvec_query(K, x)
    """
        A dummy query procedure used mainly for
        development. It is a black-box that returns
        K * x. This black-box computational procedure
        may be arbitrarily complex; hence a black-box.
    """
    return K * x;
end

function dummy_param_mat(θ, n)
    """
        A dummy parameterized, symmetric matrix of size n.
        θ has 3 parameters.
    """
    θ1 = θ[1];
    θ2 = θ[2];
    θ3 = θ[3];
    A = Tridiagonal(repeat([θ1^2], n-1), repeat([θ2^3], n), repeat([θ3^2], n-1));
    A = A'*A;
    return A
end

function param_matvec(θ, n, X)
    """
        Returns A(θ)*X.
    """
    return dummy_param_mat(θ, n)*X;
end

function param_matvec_grad(θ, n, X)
    """
        Returns ∂ᵢA(θ)*X for parameter 1 ≤ i ≤ p.
    """
    p = length(θ);
    # forward evaluation
    matvec = param_matvec(θ, n, X);
    # function handle for ForwardDiff
    jac = ForwardDiff.jacobian(θ->param_matvec(θ, n, X), θ);
    grad = Array{Matrix}(undef, p);
    for ii = 1:p
        grad[ii] .= reshape(jac[:, ii], size(matvec, 1), size(matvec, 2));
    end
    return matvec, grad
end

# Creating simple HODLR matrices to test recovery
function create_low_rank_two_level_matrix(n)
    """ 
        Creates a matrix of size n such that it can be
        recovered exactly with a 2-level HODLR. The matrix
        is symmetric and each off-diagonal matrix has low
        rank (rank=n/2^3);
    """
    @assert(ispow2(n))
    # exact rank
    k = Int(n/2^3);
    exact_level = 2;
    idx = dyadic_idx(n, exact_level);
    # fill the diagonals (all symmetric)
    A = zeros(n, n);
    A[idx[1], idx[1]] = randn_symmetric(Int(n / 4));
    A[idx[2], idx[2]] = randn_symmetric(Int(n / 4));
    A[idx[3], idx[3]] = randn_symmetric(Int(n / 4));
    A[idx[4], idx[4]] = randn_symmetric(Int(n / 4));
    # fill the off-diagonals
    tmp = randn_symmetric(Int(n / 4));
    A[idx[2], idx[1]] = tmp;
    A[idx[1], idx[2]] = tmp';
    tmp = randn_symmetric(Int(n / 4));
    A[idx[4], idx[3]] = tmp;
    A[idx[3], idx[4]] = tmp';

    # fill the off-diagonals of the prior level
    idx = dyadic_merge(idx, 1);
    tmp = randn_symmetric(Int(n / 2));
    A[idx[2], idx[1]] = tmp;
    A[idx[1], idx[2]] = tmp';
    @assert(issymmetric(A))
    # modify blocks such that off-diagonals are low rank
    idx = dyadic_idx(n, exact_level);
    tmp = A[idx[2], idx[1]];
    tmp[:, Int(n/8)+1:end] = tmp[:, 1:Int(n/8)];
    A[idx[2], idx[1]] .= tmp;
    A[idx[1], idx[2]] .= tmp';
    tmp = A[idx[4], idx[3]];
    tmp[:, Int(n/8)+1:end] = tmp[:, 1:Int(n/8)];
    A[idx[4], idx[3]] .= tmp;
    A[idx[3], idx[4]] .= tmp';
    # modify blocks from prior level
    idx = dyadic_merge(idx, 1);
    tmp = A[idx[2], idx[1]];
    tmp[:, Int(n/8)+1:end] = [tmp[:, 1:Int(n/8)] tmp[:, 1:Int(n/8)] tmp[:, 1:Int(n/8)]];
    A[idx[2], idx[1]] .= tmp;
    A[idx[1], idx[2]] .= tmp';
    @assert(issymmetric(A))
    return A
end

function create_low_rank_two_level_matrix_two_params(θ :: AbstractVector{T}, n :: Int) where T
    """ 
        For testing HODLR grad, creates a parametrized 
        (with 2 parameters) matrix that can be recovered 
        exactly with a 2-level HODLR. Each off-diagonal 
        factor has low-rank (rank=n/2^3).

        The off-diagonal blocks are simply:
        randn_symmetric(n/2^i, n/2^i) * θi^2 
        i.e. first level off-diag blocks depend on θ1,
        second level off-diag blocks depend on θ2

        Warning: Input θ must be of type Float64, otherwise 
        will get error conflicting with type T.
        https://discourse.julialang.org/t/error-with-forwarddiff-no-method-matching-float64/41905
    """
    @assert(length(θ) == 2)
    k = Int(n / 2^3);               # exact rank
    exact_level = 2;
    idx = dyadic_idx(n, exact_level);
    # fill the diagonals (all symmetric)
    A = zeros(T, n, n);
    θ1, θ2 = θ;
    A[idx[1], idx[1]] = randn_symmetric(Int(n / 4));
    A[idx[2], idx[2]] = randn_symmetric(Int(n / 4));
    A[idx[3], idx[3]] = randn_symmetric(Int(n / 4));
    A[idx[4], idx[4]] = randn_symmetric(Int(n / 4));
    # fill the off-diagonals
    tmp = randn_symmetric(Int(n / 4));
    A[idx[2], idx[1]] = tmp;
    A[idx[1], idx[2]] = tmp';
    tmp = randn_symmetric(Int(n / 4));
    A[idx[4], idx[3]] = tmp;
    A[idx[3], idx[4]] = tmp';

    # fill the off-diagonals of the prior level
    idx = dyadic_merge(idx, 1);
    tmp = randn_symmetric(Int(n / 2));
    A[idx[2], idx[1]] = tmp;
    A[idx[1], idx[2]] = tmp';
    @assert(issymmetric(A))
    # modify blocks such that off-diagonals are low rank
    idx = dyadic_idx(n, exact_level);
    tmp = A[idx[2], idx[1]];
    tmp[:, Int(n/8)+1:end] = tmp[:, 1:Int(n/8)];
    # add parametric dependence
    tmp = tmp .* (θ2^2);

    A[idx[2], idx[1]] .= tmp;
    A[idx[1], idx[2]] .= tmp';
    tmp = A[idx[4], idx[3]];
    tmp[:, Int(n/8)+1:end] = tmp[:, 1:Int(n/8)];
    # add parametric dependence
    tmp = tmp .* (θ2^2);

    A[idx[4], idx[3]] .= tmp;
    A[idx[3], idx[4]] .= tmp';
    # modify blocks from prior level
    idx = dyadic_merge(idx, 1);
    tmp = A[idx[2], idx[1]];
    tmp[:, Int(n/8)+1:end] = [tmp[:, 1:Int(n/8)] tmp[:, 1:Int(n/8)] tmp[:, 1:Int(n/8)]];
    # add parametric dependence
    tmp = tmp .* (θ1^2);

    A[idx[2], idx[1]] .= tmp;
    A[idx[1], idx[2]] .= tmp';
    @assert(issymmetric(A))
    return A
end

function dummy_param_mat(A, θ :: AbstractVector{T}) where T
    """ 
        Helper for testing HODLR grad. Takes a fixed 
        matrix and adds parametric dependence on θ1, θ2,
        for ease of taking derivative.

        A should be created from `create_low_rank_two_level_matrix()`, 
        see TEST_CASE = 6.
    """
    n = size(A, 1);
    exact_level = 2;
    idx = dyadic_idx(n, exact_level);
    A = convert(Matrix{T}, A);
    tmp = A[idx[2], idx[1]] .* (θ[2]^2);
    # add parametric dependence
    A[idx[2], idx[1]] .= tmp;
    A[idx[1], idx[2]] .= tmp';
    tmp = A[idx[4], idx[3]] .* (θ[2]^2);
    A[idx[4], idx[3]] .= tmp;
    A[idx[3], idx[4]] .= tmp';
    idx = dyadic_merge(idx, 1);
    tmp = A[idx[2], idx[1]] .* (θ[1]^2);
    A[idx[2], idx[1]] .= tmp;
    A[idx[1], idx[2]] .= tmp';
    return A;
end

function two_level_matvec_grad(θ, n, v)
    """ 
        Mimics the fast mat-vec functionality, computes 
        A(θ)*v where A is given by 
        `create_low_rank_two_level_matrix_two_params`,
        along with its jacobian matrices with respect
        to each parameter θ1, θ2.

        Size of matrix n is fixed at 1024.

        v is allowed to be multiple vectors.
    """
    n = 1024;
    @assert(size(v, 1) == n)
    k = size(v, 2);
    # function wrapper for jacobian
    wrapper(θ) = create_low_rank_two_level_matrix_two_params(θ, n) * v;
    # product
    product = wrapper(θ);
    # take jacobian
    jac = ForwardDiff.jacobian(wrapper, θ);
    jac = [reshape(jac[:, 1], n, k), reshape(jac[:, 2], n, k)];
    return product, jac
end