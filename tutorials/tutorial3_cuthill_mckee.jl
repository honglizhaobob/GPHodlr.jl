# This tutorial tests the Cuthill Mckee algorithm for re-ordering 
# a graph adjacency such that it is almost banded.
# We will use this code: https://github.com/rleegates/CuthillMcKee.jl

# Remark:
#   The algorithm is best for very sparse matrix, and it does minimize 
#   fill-in and gives us a banded matrix. However, it is not suitable 
#   for random clustered matrices. In clustered matrices case with 
#   few interconnection among clusters. It is best to keep them as is
#   and not apply the Cuthill algorithm.

using SparseArrays, CuthillMcKee, UnicodePlots, LinearAlgebra;
using Plots, Colors;
# Example 1: random sparse matrix
N = 1024;
A = sprand(N, N, 1/N)
A = A+A'+30I
b = rand(N)
@time p = symrcm(A); # indices that permutes A to fill-in minimized matrix
ip = symrcm(A, true, true) # inverse indices
AP = rcmpermute(A)
@assert norm( (AP*b[p])[ip]-A*b ) < 1e-12;
display(spy(A));
display(spy(AP));

println(norm(A[p,p]-AP));
println(norm(AP[ip,ip]-A));

error();
# Example 2: generated graph 
include("utils.jl");
# create a graph matrix with clusters 
n = 1024;
r = 16;
p, q = 0.4, 0.1;
# returns graph adjacency 
A, idx = sbm(
    n, r, p, q
); 
# plot(Gray.(A))
AP = rcmpermute(A);