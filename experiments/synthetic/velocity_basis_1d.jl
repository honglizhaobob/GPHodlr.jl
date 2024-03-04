using GPHodlr
# fix random seed
using Random
using GaussianRandomFields
using Plots
Random.seed!(10);
######################################################################
# (03/04/2024) Define a 1D Gaussian process using a basis expansion
# velocity and verify that it is recoverable.
######################################################################

# generate data without masking 
n = 2^10;
@assert ispow2(n);
# left and right boundaries
xmin, xmax = -1., 1.;
pts = range(xmin, stop=xmax, length=n);
dx = pts[2]-pts[1];
# sample Gaussian random field
m(x) = 20. .* exp.(-(x.^2) ./2 ./0.2 ./ 0.2);
# latent parameters
l, nu = 0.05, 1.0;
sigma_phi = 1.0;
# nugget level 
nugget = 0.05;

# physical parameters
kappa = 1e-2; 
c = 0.5;
# true PDE velocity model
p = 3; # up to quadratic
vparams = ones(p);
vparams[1] = -2.0;
vtrue = GPHodlr.PhysicsMLE1d.ChebyshevVelocity1d(xmin, xmax, vparams);


# generate sample data
nmc = 10;
u, M_u, sigma_u = GPHodlr.PhysicsMLE1d.generate_samples(n, xmin, xmax, m,
        [sigma_phi, nu, l], [kappa, c], vtrue, [],
        nmc, "CirculantEmbedding", nugget, true);
#p = plot(u', label="");

# define problem, assume no masking
u_observed = copy(u[1, :]);
obs_local_inds = collect(1:n);
# define model velocity 
vmodel = GPHodlr.PhysicsMLE1d.ChebyshevVelocity1d(xmin, xmax, zeros(3))

prob = GPHodlr.PhysicsMLE1d.MLEProblem1d(collect(pts), u_observed, obs_local_inds, nugget, 
        0.37,
        vmodel,
        c+0.1,
        sigma_phi,
        nu, 
        l
    );

# optimization box constraints 
constraints = ones(1+p+1, 2);
constraints[:, 1] .= -10.0 .* constraints[:, 1];
constraints[:, 2] .= 10.0 .* constraints[:, 2];
res = GPHodlr.PhysicsMLE1d.solve!(prob, constraints);