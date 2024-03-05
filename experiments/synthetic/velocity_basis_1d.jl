using GPHodlr
# fix random seed
using Random
using GaussianRandomFields
using Plots
Random.seed!(5);
######################################################################
# (03/04/2024) Define a 1D Gaussian process using a basis expansion
# velocity and verify that it is recoverable.
######################################################################

# generate data without masking 
n = 2^8;
@assert ispow2(n);
# left and right boundaries
xmin, xmax = -1., 1.;
pts = range(xmin, stop=xmax, length=n);
dx = pts[2]-pts[1];
# sample Gaussian random field (mean zero)
m(x) = zeros(length(x));
# latent parameters
l, nu = 0.05, 1.0;
sigma_phi = 1.0;
# nugget level 
nugget = 0.1;

# physical parameters
kappa = 1e-2; 
c = 0.1;
# true PDE velocity model (taken to be cosine)
_vtrue_function = x -> 2.0.*cos.(2.0.*x);
vtrue = GPHodlr.PhysicsMLE1d.CustomVelocity1d(xmin, xmax, _vtrue_function);


# generate sample data
nmc = 2^8;
u, M_u, sigma_u = GPHodlr.PhysicsMLE1d.generate_samples(n, xmin, xmax, m,
        [sigma_phi, nu, l], [kappa, c], vtrue, [],
        nmc, "CirculantEmbedding", nugget, true);
#p = plot(u', label="");

# define problem, assume no masking
u_observed = copy(u[1, :]);
obs_local_inds = collect(1:n);
# number of parameterization for velocity
p = 5;
# define model velocity (initialize at all one's)
vmodel = GPHodlr.PhysicsMLE1d.ChebyshevVelocity1d(xmin, xmax, ones(p))

prob = GPHodlr.PhysicsMLE1d.MLEProblem1d(collect(pts), 
        u_observed, obs_local_inds, nugget, 
        kappa,
        vmodel,
        c,
        sigma_phi,
        nu, 
        l
    );

# optimization box constraints 
res = GPHodlr.PhysicsMLE1d.solve!(prob);