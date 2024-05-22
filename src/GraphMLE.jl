function laplacian(A, normalized)
    """ 
        Computes (normalized) graph Laplacian given 
        an adjacency matrix. The adjacency can be 
        weighted. 
    """
    # degree matrix 
    degs = sum(A, dims=2);
    if normalized
        rootD = diagm(sqrt.(vec(degs)));
        # adjacency matrix is symmetric, Lnorm = D^(-1/2)*(D - A)*D^(-1/2)
        L = -rootD\(rootD\A)';
        # add identity
        L[diagind(L)] .= diag(L) .+ 1.0;
    else
        L = -A;
        # add diagonal 
        L[diagind(L)] .= diag(L) .+ vec(degs);
    end
    return L;
end


# An implementation of the stochastic block model to generate random 
# adjacency matrices with clusters. 
#
#
# Author: Hongli Zhao
# Date: 03/05/2024
#
# References:
#
#   1. [C] Compressive graph clustering from random sketches
#   2. [HLL] Stochastic blockmodels: First steps
###########
using SparseArrays, LinearAlgebra
###########
function sbm(n :: Int, r :: Int, p :: Real, q :: Real)
    """ 
        A simple implementation of the stochastic block generative 
        model for generating adjacency matrices. 

        The clusters are assumed to have the same size, with uniform 
        random assignment.

        n    total number of nodes in the graph

        r    number of clusters 

        p    probability of connection for any two nodes in the same 
             cluster. 
        
        q    probability of connection for any two nodes not in the 
             same cluster.

        p and q do not necessarily sum to 1.
        Returns:
        
        A    adjacency matrix (sparse)

        idx  shuffled index
    """
    @assert mod(n, r) == 0;
    idx = shuffle(collect(1:n));
    # size of partition 
    k = Int(n / r);
    # create dictionary assigning nodes to blocks 
    idx = shuffle(1:n);
    mapper = Dict();
    for i = eachindex(idx)
        id = idx[i];
        block_number = floor(id / k)+1;
        mapper[id] = block_number;
    end
    # create sparse adjacency matrix 
    A = spzeros(n, n);
    for i = 1:n
        for j = 1:i-1
            # shuffled index 
            ii, jj = idx[i], idx[j];
            # check membership and create connectivity
            cluster_i = mapper[ii];
            cluster_j = mapper[jj];
            # uniform random criteria
            b = rand();
            if cluster_i == cluster_j 
                # two nodes are in the same cluster 
                if b < p 
                    A[ii, jj] = A[jj, ii] = 1;
                    #A[ii, jj] = A[jj, ii] = exp(-(ii-jj)^2);
                end
            else 
                # two nodes are not in the same cluster 
                if b < q 
                    A[ii, jj] = A[jj, ii] = 1;
                    #A[ii, jj] = A[jj, ii] = exp(-(ii-jj)^2);
                end
            end
        end
    end
    return A, idx;
end