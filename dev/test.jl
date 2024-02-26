using GPHodlr
using LinearAlgebra

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

n = 1024;
A = create_low_rank_two_level_matrix(n);
#Ah = GPHodlr.hodlr(A, 3, Int(n / 8));