############################################################################################
# Randomized range finder via reduced QR 
############################################################################################
function randcols(B, k, c)
    """
        Finds a k-column space from B 
        using Gaussian random matrices. 
        Alg. 3.1 in Yian's paper.

        Inputs:
            B :: Matrix{Float64}    -> (p x q) matrix from which we would 
                                       like to find a reduced column space
                                       of dimension k.
            k :: Int                -> Target rank, k < q

        Outputs:
            Bhat :: Matrix{Float64} -> (p x 1) matrix of approximate column
                                       space of B (of rank k).
    """
    p, q = size(B);
    @assert(k <= q);
    # Gaussian random matrix
    Omega = randn(q, k+c);
    Y = B*Omega;
    # QR with column pivoting
    Q, R, perm = qr(Y, ColumnNorm());
    # reduced Q (take the first k pivoted columns)
    column_idx = perm[1:k+c];
    Q_reduced = Q[:, column_idx];
    Bhat = Q_reduced*(Q_reduced'*B);
    return Bhat
end


############################################################################################
# Deterministic interpolative decomposition via truncated SVD
############################################################################################
function interp_decomp(B, k)
    """
        Best rank-k approximation to B.

        Inputs:
            B :: Matrix{Float64}    -> (p x q) matrix from which we would 
                                       like to find a reduced column space
                                       of dimension k.
            k :: Int                -> Target rank, k < q

        Outputs:
            A1 ::   Matrix{Float64} -> (p x k) matrix of approximate column
                                       space of B.
            A2 ::   Matrix{Float64} -> (p x k) matrix of approximate row
                                       space of B. Such that:
                                       B = A1*A2'
            rel_err :: Float64      -> relative error in two norm, or sigma_k+1/sigma_max
    """
    p, q = size(B);
    @assert(k <= q);
    # SVD
    U, S, V = svd(B);
    # truncate
    A1 = U[:,1:k]*diagm(sqrt.(S)[1:k]);
    A2 = V[:,1:k]*diagm(sqrt.(S)[1:k]);

    # max singular value
    if k == q
        rel_err = norm(A1*A2'-B)/S[1];
    else
        rel_err = S[k+1]/S[1];
    end
    return A1, A2, rel_err
end
############################################################################################
# Algorithm 3.2: differentiate QR decompositions
############################################################################################
function grad_qr(dB, Q, R)
    """
        Let B = QR, computes dQ, dR assuming dB is known.
        dB is typically computed with forward mode AD.

        B has size (p x q), Q (p x q), R (q x q)
    """
    # step 1: Y' = R'\(dB'*Q)
    Y = (R'\(dB'*Q))';
    # step 2: unique determine Y = dΩ + dR*R^-1
    dΩ, dR_times_R_inv = ssut_decompose(Y);
    # step 3: solve for dQ, dR
    dR = dR_times_R_inv*R;
    dB_times_R_inv = (R'\(dB'))';
    dQ = Q*dΩ + dB_times_R_inv - Q*(Q'*dB_times_R_inv);
    return dQ, dR
end
############################################################################################
# Decomposition of a matrix into skew symmetric + upper triangular
############################################################################################
function ssut_decompose(A)
    """
        Given a matrix A, writes A = Omega + R
        where Omega is skew-symmetric (ss), and 
        R is upper triangular (ut).

        See: https://j-towns.github.io/papers/qr-derivative.pdf
    """
    # take strict lower triangular part to form Omega
    lt = tril(A, -1);
    Omega = lt - lt';
    R = A - Omega;
    return Omega, R
end

############################################################################################
# Sherman-Morrison Woodbury formula
############################################################################################
function woodbury_inv(U, V, b)
    """
        Computes (I + U*V)^-1 * b without explicitly inverting.
    """
    n = size(V, 1);
    VU = V*U;
    Vb = V*b;
    return b - U*( (I(n) + VU)\Vb );
end

############################################################################################
# Symmetrize a matrix
############################################################################################
function symmetrize!(A)
    """ `symmetrize()` in-place """
    @assert(size(A, 1) == size(A, 2))
    A = 0.5 * (A + A');
end

function randn_symmetric(n)
    """ creates a `randn(n)` matrix and symmetrize it. """
    A = randn(n, n);
    A = symmetrize!(A);
end

############################################################################################
# transpose Khatri-Rao product
############################################################################################
function tkrp(C :: Matrix, D :: Matrix)
    """
        For more details see https://arxiv.org/pdf/1909.07909.pdf
        Equation (4)
    """
    # unpack matrix dimensions
    n, q = size(C);
    @assert(size(D, 1) == n);
    _, m = size(D);
    res = zeros(n, Int(q*m));
    for i = 1:n
        res[i, :] .= kron(C[i, :], D[i, :]);
    end
    return res;
end

function tkrp_fast(C :: Matrix, D :: Matrix)
    """
        Faster and cleaner implementation.
        For more details see https://arxiv.org/pdf/1909.07909.pdf
        Equation (4)
    """
    # unpack matrix dimensions
    n, q = size(C);
    @assert(size(D, 1) == n);
    _, m = size(D);
    res = kron.(eachrow(C), eachrow(D));
    return mapreduce(permutedims, vcat, res);
end