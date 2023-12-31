# Gradient descent testing with custom functions
using Optim, NLSolversBase, Random, LinearAlgebra


@testset begin
#----------
# test 1: minimize a nonnegative quadratic form, solution is clearly 0.0
#----------
theta = theta = [0.1, -0.5];
n = 10;
v0 = randn(n) / 10;
A = randn(n,n);
# make matrix s.p.d 
A = A'*A;
cholesky(A);
function f(x, A)
    n = length(x);
    # create dummy matrix
    return x'*A*x;
end
# define solution box
lower = repeat([-1], n);
upper = repeat([1], n);
minimizer = optimize(x -> f(x, A), lower, upper, v0, Fminbox(GradientDescent()), Optim.Options(iterations = 10^4))
@test norm(minimizer.minimizer) <= 1e-6;
#----------
end