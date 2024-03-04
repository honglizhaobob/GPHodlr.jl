# Helper script to build finite-difference differential operators
#
# References:
# -- Finite difference: LeVeque, Finite Difference Methods for Ordinary and Partial Differential Equations
# -- https://utminers.utep.edu/oktweneboah/files/talk.pdf
# -- https://github.com/luraess/parallel-gpu-workshop-JuliaCon21/tree/main/scripts

# Ferrite: https://github.com/Ferrite-FEM/Ferrite.jl

using LinearAlgebra
using BlockArrays
using Plots
using SparseArrays
using ForwardDiff

############################################################################################
# 1d Advection-diffusion-reaction
############################################################################################
function adv_mat1d(gridpts, c, v, θ, kappa)
    """ 
        Returns finite difference matrix for 1d. 
        `v` is assumed to have form v(x, θ) where
        θ are parameterization. 

        Example:
            v(x, θ) = θ[1]*x^3 + sin(θ[2]*x)
    """
    n = length(gridpts);
    h = gridpts[2] - gridpts[1];
    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();
    for i = 1:n
        # grid location
        x = gridpts[i];
        # advection velocity
        v_val = v(x, θ);
        # compute coefficients
        # form: a1U_i-1 + a2U_i + a3U_i+1 = f_i
        a1 = (-v_val/2h-kappa/h^2);
        a2 = (c+2kappa/h^2);
        a3 = (v_val/2h-kappa/h^2);
        # store coefficients
        if i == 1
            # U_i 
            push!(row_ind, i);
            push!(col_ind, i);
            push!(entry, a2);
            # U_i+1
            push!(row_ind, i);
            push!(col_ind, i+1);
            push!(entry, a1+a3);
        elseif i == n
            # U_i-1
            push!(row_ind, i);
            push!(col_ind, i-1);
            push!(entry, a1+a3);
            # U_i
            push!(row_ind, i);
            push!(col_ind, i);
            push!(entry, a2);
        else
            # U_i-1
            push!(row_ind, i);
            push!(col_ind, i-1);
            push!(entry, a1);
            # U_i 
            push!(row_ind, i);
            push!(col_ind, i);
            push!(entry, a2);
            # U_i+1
            push!(row_ind, i);
            push!(col_ind, i+1);
            push!(entry, a3);
        end
    end
    # create differential operator as a matrix
    L = sparse(row_ind, col_ind, entry);
    return L;
end

function adv_mat1d_grad(gridpts, c, v, θ, kappa)
    """ 
        Returns Jacobian of the finite difference 
        matrix for 1d, in all parameters θ through
        automatic differentiation. The result is 
        returned as an array of matrices.

        `v` is assumed to have form v(x, θ) where
        θ are parameterization. 

        Example:
            v(x, θ) = θ[1]*x^3 + sin(θ[2]*x)
    """
    n = length(gridpts);
    h = gridpts[2] - gridpts[1];
    p = length(θ);
    Ldθ = Array{Matrix}(undef, p);
    for pp = 1:p
        #println("Parameter $pp")
        row_ind = Vector{Int64}();
        col_ind = Vector{Int64}();
        entry = Vector{Float64}();
        for i = 1:n
            # grid location
            x = gridpts[i];
            # advection velocity
            v_val(θ) = v(x, θ);
            # compute coefficients
            # form: a1U_i-1 + a2U_i + a3U_i+1 = f_i
            a1_var(θ) = (-v_val(θ)/2h-kappa/h^2);
            a2_var(θ) = (c+2kappa/h^2);
            a3_var(θ) = (v_val(θ)/2h-kappa/h^2);

            # take derivative with respect to θ using autodiff
            a1 = ForwardDiff.gradient(a1_var, θ)[pp];
            a2 = ForwardDiff.gradient(a2_var, θ)[pp];
            a3 = ForwardDiff.gradient(a3_var, θ)[pp];
            
            # store coefficients
            if i == 1
                # U_i 
                push!(row_ind, i);
                push!(col_ind, i);
                push!(entry, a2);
                # U_i+1
                push!(row_ind, i);
                push!(col_ind, i+1);
                push!(entry, a1+a3);
            elseif i == n
                # U_i-1
                push!(row_ind, i);
                push!(col_ind, i-1);
                push!(entry, a1+a3);
                # U_i
                push!(row_ind, i);
                push!(col_ind, i);
                push!(entry, a2);
            else
                # U_i-1
                push!(row_ind, i);
                push!(col_ind, i-1);
                push!(entry, a1);
                # U_i 
                push!(row_ind, i);
                push!(col_ind, i);
                push!(entry, a2);
                # U_i+1
                push!(row_ind, i);
                push!(col_ind, i+1);
                push!(entry, a3);
            end
        end
        Ldθ[pp] = sparse(row_ind, col_ind, entry);
    end
    return Ldθ;
end


############################################################################################
# 2d Advection-diffusion-reaction
############################################################################################

######################################################################
# Autodifferentation with matrix parameterization
######################################################################
function adv_mat(gridpts :: Union{Vector, StepRangeLen}, c :: Real, v :: Function, theta :: Vector, kappa :: Real)
    """
        Re-written matrix creation of `advection_diffusion_reaction`
        by creating sparse matrix from vectors. Here `v` is allowed 
        to be any arbitrary function of grid points (x, y).

        Advection velocity is assumed to have input form: v(x, θ)

        Example:
            v(x, θ) = [x[1]+θ[1], x[2]+θ[2]];

        `gridpts` define the uniform grid used for both x and y directions.
    """
    h = gridpts[2]-gridpts[1];
    n = length(gridpts);
    # parameters for operator
    # U_i,j
    p_a = (4kappa/h^2+c);

    # linearized index query
    linidx(i, j) = sub2ind((n,n),i,j);
    # only store three vectors
    # - row indices
    # - col indices
    # - entry at (row,col)
    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();
    for j = 1:n
        for i = 1:n
            # physical domain location
            x1 = gridpts[i];
            x2 = gridpts[j];
            # evaluate velocity
            v_eval = v([x1,x2], theta);
            # U_i+1,j
            p_b = (v_eval[1]/2h-kappa/h^2);
            # U_i-1,j
            p_c = (-v_eval[1]/2h-kappa/h^2); 
            # U_i,j+1
            p_d = (v_eval[2]/2h-kappa/h^2);
            # U_i,j-1
            p_e = (-v_eval[2]/2h-kappa/h^2);
            # row index is always column major in-order 
            idx = linidx(i, j);
            # ----------------------------------------
            # corner points
            # ----------------------------------------
            if i == 1 && j == 1
                # U_i,j dependence
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);

                # U_i+1,j dependence
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                push!(entry, p_b + p_c);

                # U_i,j+1 dependence
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                push!(entry, p_d + p_e);

            elseif i == 1 && j == n
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                push!(entry, p_d + p_e);

                # L[idx, linidx(i,j)] = p_a;
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);

                # L[idx, linidx(i+1,j)] = p_b + p_c;
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                push!(entry, p_b + p_c);

            elseif i == n && j == 1
                # L[idx, linidx(i-1,j)] = p_b + p_c;
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                push!(entry, p_b + p_c);

                # L[idx, linidx(i,j)] = p_a;
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);

                # L[idx, linidx(i,j+1)] = p_d + p_e;
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                push!(entry, p_d + p_e);

            elseif i == n && j == n
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                push!(entry, p_d + p_e);

                # L[idx, linidx(i-1,j)] = p_b + p_c;
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                push!(entry, p_b + p_c);

                # L[idx, linidx(i,j)] = p_a;
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);
            else
                # ----------------------------------------
                # boundary, non-corner
                # ----------------------------------------
                if i == 1 && 2 <= j <= n-1
                    # L[idx, linidx(i,j-1)] = p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_e);
                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # L[idx, linidx(i+1,j)] = p_b + p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b + p_c);
                    # L[idx, linidx(i,j+1)] = p_d;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d);
                elseif i == n && 2 <= j <= n-1
                    # L[idx, linidx(i,j-1)] = p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_e);
                    # L[idx, linidx(i-1,j)] = p_b + p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_b + p_c);
                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # L[idx, linidx(i,j+1)] = p_d;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d);
                elseif j == 1 && 2 <= i <= n-1
                    # L[idx, linidx(i-1,j)] = p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_c);
                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # L[idx, linidx(i+1,j)] = p_b;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b);
                    # L[idx, linidx(i,j+1)] = p_d + p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d + p_e);
                elseif j == n && 2 <= i <= n-1
                    # L[idx, linidx(i,j-1)] = p_d + p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_d + p_e);
                    # L[idx, linidx(i-1,j)] = p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_c);
                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # L[idx, linidx(i+1,j)] = p_b;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b);
                else
                
                    # ----------------------------------------
                    # Within boundary
                    # ----------------------------------------
                    # L[idx, linidx(i,j-1)] = p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_e);
                    # L[idx, linidx(i-1,j)] = p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_c);
                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # L[idx, linidx(i+1,j)] = p_b;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b);
                    # L[idx, linidx(i,j+1)] = p_d;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d);
            
                end
            end
        end
    end
    # create sparse operator matrix
    L = sparse(row_ind, col_ind, entry);
    return L
end



function adv_mat_grad(gridpts :: Union{Vector, StepRangeLen}, c :: Real, v :: Function, theta :: Vector, kappa :: Real)
    """
        Matrix gradient creation via ForwardDiff, allowing 
        arbitrary parameterized advection velocities. 

        The velocity is assumed to have format:
            v(x, theta) where x is a vector of grid positions (x, y),
        theta is a vector of other parameters.

        The function returns an array of matrices of length p, where
        p is the number of parameters in `theta`, each of the 
        same size as the original `adv_mat`:

            [∂θ₁L, ∂θ₂L, ..., ∂θₚL]

        Example (linear in each spatial axis):
            v(x, theta) = [theta[1]*x[1]+theta[2], theta[3]*x[2]+theta[4]];

        `gridpts` define the uniform grid used for both x and y directions.
    """
    # parameters
    p = length(theta);
    h = gridpts[2]-gridpts[1];
    n = length(gridpts);

    # linearized index query
    linidx(i, j) = sub2ind((n,n),i,j);
    # create array of matrices
    Ldθ = Array{Matrix}(undef, p);
    for pp = 1:p
        #println("Parameter $pp")
        # only store three vectors
        # - row indices
        # - col indices
        # - entry at (row,col)
        row_ind = Vector{Int64}();
        col_ind = Vector{Int64}();
        entry = Vector{Float64}();
        for j = 1:n
            for i = 1:n
                # physical domain location
                x1 = gridpts[i];
                x2 = gridpts[j];
                # evaluate velocity at spatial points (create function handle)
                v_eval(θ) = v([x1, x2], θ);
                # parameters for operator
                # U_i,j
                p_a_var(θ) = (4kappa/h^2+c);
                # U_i+1,j
                p_b_var(θ) = (v_eval(θ)[1]/2h-kappa/h^2);
                # U_i-1,j
                p_c_var(θ) = (-v_eval(θ)[1]/2h-kappa/h^2); 
                # U_i,j+1
                p_d_var(θ) = (v_eval(θ)[2]/2h-kappa/h^2);
                # U_i,j-1
                p_e_var(θ) = (-v_eval(θ)[2]/2h-kappa/h^2);

                # take gradient of each parameter using autodiff, and take the `idx`-th partial derivative
                p_a = ForwardDiff.gradient(p_a_var, theta)[pp];
                p_b = ForwardDiff.gradient(p_b_var, theta)[pp];
                p_c = ForwardDiff.gradient(p_c_var, theta)[pp];
                p_d = ForwardDiff.gradient(p_d_var, theta)[pp];
                p_e = ForwardDiff.gradient(p_e_var, theta)[pp];

                # row index is always column major in-order 
                idx = linidx(i, j);
                # ----------------------------------------
                # corner points
                # ----------------------------------------
                if i == 1 && j == 1
                    # U_i,j dependence
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);

                    # U_i+1,j dependence
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b + p_c);

                    # U_i,j+1 dependence
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d + p_e);

                elseif i == 1 && j == n
                    # L[idx, linidx(i,j-1)] = p_d + p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_d + p_e);

                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);

                    # L[idx, linidx(i+1,j)] = p_b + p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b + p_c);

                elseif i == n && j == 1
                    # L[idx, linidx(i-1,j)] = p_b + p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_b + p_c);

                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);

                    # L[idx, linidx(i,j+1)] = p_d + p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d + p_e);

                elseif i == n && j == n
                    # L[idx, linidx(i,j-1)] = p_d + p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_d + p_e);

                    # L[idx, linidx(i-1,j)] = p_b + p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_b + p_c);

                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                else
                    # ----------------------------------------
                    # boundary, non-corner
                    # ----------------------------------------
                    if i == 1 && 2 <= j <= n-1
                        # L[idx, linidx(i,j-1)] = p_e;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j-1));
                        push!(entry, p_e);
                        # L[idx, linidx(i,j)] = p_a;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j));
                        push!(entry, p_a);
                        # L[idx, linidx(i+1,j)] = p_b + p_c;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i+1,j));
                        push!(entry, p_b + p_c);
                        # L[idx, linidx(i,j+1)] = p_d;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j+1));
                        push!(entry, p_d);
                    elseif i == n && 2 <= j <= n-1
                        # L[idx, linidx(i,j-1)] = p_e;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j-1));
                        push!(entry, p_e);
                        # L[idx, linidx(i-1,j)] = p_b + p_c;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i-1,j));
                        push!(entry, p_b + p_c);
                        # L[idx, linidx(i,j)] = p_a;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j));
                        push!(entry, p_a);
                        # L[idx, linidx(i,j+1)] = p_d;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j+1));
                        push!(entry, p_d);
                    elseif j == 1 && 2 <= i <= n-1
                        # L[idx, linidx(i-1,j)] = p_c;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i-1,j));
                        push!(entry, p_c);
                        # L[idx, linidx(i,j)] = p_a;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j));
                        push!(entry, p_a);
                        # L[idx, linidx(i+1,j)] = p_b;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i+1,j));
                        push!(entry, p_b);
                        # L[idx, linidx(i,j+1)] = p_d + p_e;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j+1));
                        push!(entry, p_d + p_e);
                    elseif j == n && 2 <= i <= n-1
                        # L[idx, linidx(i,j-1)] = p_d + p_e;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j-1));
                        push!(entry, p_d + p_e);
                        # L[idx, linidx(i-1,j)] = p_c;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i-1,j));
                        push!(entry, p_c);
                        # L[idx, linidx(i,j)] = p_a;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j));
                        push!(entry, p_a);
                        # L[idx, linidx(i+1,j)] = p_b;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i+1,j));
                        push!(entry, p_b);
                    else
                    
                        # ----------------------------------------
                        # Within boundary
                        # ----------------------------------------
                        # L[idx, linidx(i,j-1)] = p_e;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j-1));
                        push!(entry, p_e);
                        # L[idx, linidx(i-1,j)] = p_c;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i-1,j));
                        push!(entry, p_c);
                        # L[idx, linidx(i,j)] = p_a;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j));
                        push!(entry, p_a);
                        # L[idx, linidx(i+1,j)] = p_b;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i+1,j));
                        push!(entry, p_b);
                        # L[idx, linidx(i,j+1)] = p_d;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j+1));
                        push!(entry, p_d);
                
                    end
                end
            end
        end
        Ldθ[pp] = sparse(row_ind, col_ind, entry);
    end
    return Ldθ
end


############################################################################
# ADR 2d, non-uniform grid, parameterized velocity and arbitrary const kappa.
############################################################################
function adv_mat(
    xgrid :: Union{Vector, StepRangeLen}, 
    ygrid :: Union{Vector, StepRangeLen},
    c :: Real, 
    v :: Function, 
    theta :: Vector
)
    """
        Discretized operator matrix creation of `advection_diffusion_reaction`
        by creating sparse matrix from vectors. Here `v` is allowed 
        to be any arbitrary function of grid points (x, y).

        Advection velocity is assumed to have input form: v(x, θ)

        The first argument of `theta` is assumed to contain the 
        estimated (const) diffusivity parameter. 

        Example:
            kappa = θ[1];
            v(x, θ) = [x[1]+θ[2], x[2]+θ[3]];

        `xgrid, ygrid` define the grids used for both x and y directions, which
        can have different discretization sizes.
    """
    hx = xgrid[2]-xgrid[1];
    hy = ygrid[2]-ygrid[1];

    nx = length(xgrid);
    ny = length(ygrid);

    kappa = theta[1];

    # the velocity part of `theta`
    theta_v = theta[2:end];


    # parameters for operator
    # U_i,j
    p_a = (2kappa/hx^2)+(2kappa/hy^2)+c;

    # linearized index query
    linidx(i, j) = sub2ind((nx,ny),i,j);
    # only store three vectors
    # - row indices
    # - col indices
    # - entry at (row,col)
    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();
    for j = 1:ny
        for i = 1:nx
            # physical domain location
            x1 = xgrid[i];
            x2 = ygrid[j];
            # evaluate velocity
            v_eval = v([x1,x2], theta_v);
            # U_i+1,j
            p_b = (v_eval[1]/2hx-kappa/hx^2);
            # U_i-1,j
            p_c = (-v_eval[1]/2hx-kappa/hx^2); 
            # U_i,j+1
            p_d = (v_eval[2]/2hy-kappa/hy^2);
            # U_i,j-1
            p_e = (-v_eval[2]/2hy-kappa/hy^2);
            # row index is always column major in-order 
            idx = linidx(i, j);
            # ----------------------------------------
            # corner points
            # ----------------------------------------
            if i == 1 && j == 1
                # U_i,j dependence
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);

                # U_i+1,j dependence
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                push!(entry, p_b + p_c);

                # U_i,j+1 dependence
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                push!(entry, p_d + p_e);

            elseif i == 1 && j == ny
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                push!(entry, p_d + p_e);

                # L[idx, linidx(i,j)] = p_a;
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);

                # L[idx, linidx(i+1,j)] = p_b + p_c;
                push!(row_ind, idx);
                push!(col_ind, linidx(i+1,j));
                push!(entry, p_b + p_c);

            elseif i == nx && j == 1
                # L[idx, linidx(i-1,j)] = p_b + p_c;
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                push!(entry, p_b + p_c);

                # L[idx, linidx(i,j)] = p_a;
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);

                # L[idx, linidx(i,j+1)] = p_d + p_e;
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j+1));
                push!(entry, p_d + p_e);

            elseif i == nx && j == ny
                # L[idx, linidx(i,j-1)] = p_d + p_e;
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j-1));
                push!(entry, p_d + p_e);

                # L[idx, linidx(i-1,j)] = p_b + p_c;
                push!(row_ind, idx);
                push!(col_ind, linidx(i-1,j));
                push!(entry, p_b + p_c);

                # L[idx, linidx(i,j)] = p_a;
                push!(row_ind, idx);
                push!(col_ind, linidx(i,j));
                push!(entry, p_a);
            else
                # ----------------------------------------
                # boundary, non-corner
                # ----------------------------------------
                if i == 1 && 2 <= j <= ny-1
                    # L[idx, linidx(i,j-1)] = p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_e);
                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # L[idx, linidx(i+1,j)] = p_b + p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b + p_c);
                    # L[idx, linidx(i,j+1)] = p_d;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d);
                elseif i == nx && 2 <= j <= ny-1
                    # L[idx, linidx(i,j-1)] = p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_e);
                    # L[idx, linidx(i-1,j)] = p_b + p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_b + p_c);
                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # L[idx, linidx(i,j+1)] = p_d;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d);
                elseif j == 1 && 2 <= i <= nx-1
                    # L[idx, linidx(i-1,j)] = p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_c);
                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # L[idx, linidx(i+1,j)] = p_b;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b);
                    # L[idx, linidx(i,j+1)] = p_d + p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d + p_e);
                elseif j == ny && 2 <= i <= nx-1
                    # L[idx, linidx(i,j-1)] = p_d + p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_d + p_e);
                    # L[idx, linidx(i-1,j)] = p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_c);
                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # L[idx, linidx(i+1,j)] = p_b;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b);
                else
                
                    # ----------------------------------------
                    # Within boundary
                    # ----------------------------------------
                    # L[idx, linidx(i,j-1)] = p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_e);
                    # L[idx, linidx(i-1,j)] = p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_c);
                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                    # L[idx, linidx(i+1,j)] = p_b;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b);
                    # L[idx, linidx(i,j+1)] = p_d;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d);
            
                end
            end
        end
    end
    # create sparse operator matrix
    L = sparse(row_ind, col_ind, entry);
    return L
end

function adv_mat_grad(
    xgrid :: Union{Vector, StepRangeLen}, 
    ygrid :: Union{Vector, StepRangeLen}, 
    c :: Real, 
    v :: Function, 
    theta :: Vector
)
    """ 
        Overloaded, with parameterization:
        kappa = theta[1];
        theta_v = theta[2:end];
    """
    # parameters
    p = length(theta);
    hx = xgrid[2]-xgrid[1];
    hy = ygrid[2]-ygrid[1];
    nx = length(xgrid);
    ny = length(ygrid);

    # linearized index query
    linidx(i, j) = sub2ind((nx,ny),i,j);

    # create array of matrices
    Ldθ = Array{Matrix}(undef, p);
    for pp = 1:p
        row_ind = Vector{Int64}();
        col_ind = Vector{Int64}();
        entry = Vector{Float64}();
        for j = 1:ny
            for i = 1:nx
                # physical domain location
                x1 = xgrid[i];
                x2 = ygrid[j];
                # evaluate velocity at spatial points (create function handle)
                v_eval(θ) = v([x1, x2], θ);
                # parameters for operator
                # U_i,j
                p_a_var(θ) = (2θ[1]/hx^2)+(2θ[1]/hy^2)+c;
                # U_i+1,j
                p_b_var(θ) = (v_eval(θ[2:end])[1]/2hx-θ[1]/hx^2);
                # U_i-1,j
                p_c_var(θ) = (-v_eval(θ[2:end])[1]/2hx-θ[1]/hx^2); 
                # U_i,j+1
                p_d_var(θ) = (v_eval(θ[2:end])[2]/2hy-θ[1]/hy^2);
                # U_i,j-1
                p_e_var(θ) = (-v_eval(θ[2:end])[2]/2hy-θ[1]/hy^2);

                # take gradient of each parameter using autodiff, and take the `idx`-th partial derivative
                p_a = ForwardDiff.gradient(p_a_var, theta)[pp];
                p_b = ForwardDiff.gradient(p_b_var, theta)[pp];
                p_c = ForwardDiff.gradient(p_c_var, theta)[pp];
                p_d = ForwardDiff.gradient(p_d_var, theta)[pp];
                p_e = ForwardDiff.gradient(p_e_var, theta)[pp];

                # row index is always column major in-order 
                idx = linidx(i, j);
                # ----------------------------------------
                # corner points
                # ----------------------------------------
                if i == 1 && j == 1
                    # U_i,j dependence
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);

                    # U_i+1,j dependence
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b + p_c);

                    # U_i,j+1 dependence
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d + p_e);

                elseif i == 1 && j == ny
                    # L[idx, linidx(i,j-1)] = p_d + p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_d + p_e);

                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);

                    # L[idx, linidx(i+1,j)] = p_b + p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i+1,j));
                    push!(entry, p_b + p_c);

                elseif i == nx && j == 1
                    # L[idx, linidx(i-1,j)] = p_b + p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_b + p_c);

                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);

                    # L[idx, linidx(i,j+1)] = p_d + p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j+1));
                    push!(entry, p_d + p_e);

                elseif i == nx && j == ny
                    # L[idx, linidx(i,j-1)] = p_d + p_e;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j-1));
                    push!(entry, p_d + p_e);

                    # L[idx, linidx(i-1,j)] = p_b + p_c;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i-1,j));
                    push!(entry, p_b + p_c);

                    # L[idx, linidx(i,j)] = p_a;
                    push!(row_ind, idx);
                    push!(col_ind, linidx(i,j));
                    push!(entry, p_a);
                else
                    # ----------------------------------------
                    # boundary, non-corner
                    # ----------------------------------------
                    if i == 1 && 2 <= j <= ny-1
                        # L[idx, linidx(i,j-1)] = p_e;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j-1));
                        push!(entry, p_e);
                        # L[idx, linidx(i,j)] = p_a;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j));
                        push!(entry, p_a);
                        # L[idx, linidx(i+1,j)] = p_b + p_c;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i+1,j));
                        push!(entry, p_b + p_c);
                        # L[idx, linidx(i,j+1)] = p_d;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j+1));
                        push!(entry, p_d);
                    elseif i == nx && 2 <= j <= ny-1
                        # L[idx, linidx(i,j-1)] = p_e;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j-1));
                        push!(entry, p_e);
                        # L[idx, linidx(i-1,j)] = p_b + p_c;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i-1,j));
                        push!(entry, p_b + p_c);
                        # L[idx, linidx(i,j)] = p_a;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j));
                        push!(entry, p_a);
                        # L[idx, linidx(i,j+1)] = p_d;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j+1));
                        push!(entry, p_d);
                    elseif j == 1 && 2 <= i <= nx-1
                        # L[idx, linidx(i-1,j)] = p_c;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i-1,j));
                        push!(entry, p_c);
                        # L[idx, linidx(i,j)] = p_a;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j));
                        push!(entry, p_a);
                        # L[idx, linidx(i+1,j)] = p_b;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i+1,j));
                        push!(entry, p_b);
                        # L[idx, linidx(i,j+1)] = p_d + p_e;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j+1));
                        push!(entry, p_d + p_e);
                    elseif j == ny && 2 <= i <= nx-1
                        # L[idx, linidx(i,j-1)] = p_d + p_e;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j-1));
                        push!(entry, p_d + p_e);
                        # L[idx, linidx(i-1,j)] = p_c;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i-1,j));
                        push!(entry, p_c);
                        # L[idx, linidx(i,j)] = p_a;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j));
                        push!(entry, p_a);
                        # L[idx, linidx(i+1,j)] = p_b;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i+1,j));
                        push!(entry, p_b);
                    else
                    
                        # ----------------------------------------
                        # Within boundary
                        # ----------------------------------------
                        # L[idx, linidx(i,j-1)] = p_e;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j-1));
                        push!(entry, p_e);
                        # L[idx, linidx(i-1,j)] = p_c;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i-1,j));
                        push!(entry, p_c);
                        # L[idx, linidx(i,j)] = p_a;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j));
                        push!(entry, p_a);
                        # L[idx, linidx(i+1,j)] = p_b;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i+1,j));
                        push!(entry, p_b);
                        # L[idx, linidx(i,j+1)] = p_d;
                        push!(row_ind, idx);
                        push!(col_ind, linidx(i,j+1));
                        push!(entry, p_d);
                
                    end
                end
            end
        end
        Ldθ[pp] = sparse(row_ind, col_ind, entry);
    end
    return Ldθ
end

############################################################################
# Advection-diffusion-reaction (ADR) 1d, time dependent, periodic boundary
############################################################################
function adv_periodic_mat_1d(
        gridpts :: Union{Matrix, Vector}, 
        c :: Real, 
        v :: Function, 
        θ :: Union{Matrix, Vector}, 
        kappa :: Real,
        dt :: Real
    )
    # do not need the last grid point due to periodic boundary
    gridpts_effective = gridpts[1:end-1];
    n = length(gridpts_effective);
    dx = gridpts_effective[2] - gridpts_effective[1];
    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();
    for i = 1:n
        # grid location
        x = gridpts_effective[i];
        # advection velocity
        v_val = v(x, θ);
        # compute coefficients
        # form: a1U_i-1 + a2U_i + a3U_i+1
        a1 = (kappa*dt/(dx^2)) + ((v_val*dt)/(2dx));
        a2 = (1 - ((2kappa*dt)/(dx^2)) - c*dt);
        a3 = (kappa*dt/(dx^2)) - ((v_val*dt)/(2dx));
        # store coefficients
        if i == 1
            # U_1
            push!(row_ind, i);
            push!(col_ind, i);
            push!(entry, a2);
            # U_2
            push!(row_ind, i);
            push!(col_ind, i+1);
            push!(entry, a3);
            # boundary U_0 = U_n
            push!(row_ind, i);
            push!(col_ind, n);
            push!(entry, a1);
        elseif i == n
            # U_n-1
            push!(row_ind, i);
            push!(col_ind, i-1);
            push!(entry, a1);
            # U_n
            push!(row_ind, i);
            push!(col_ind, i);
            push!(entry, a2);
            # boundary: U_n+1 = U_1
            push!(row_ind, i);
            push!(col_ind, 1);
            push!(entry, a3);
        else
            # U_i-1
            push!(row_ind, i);
            push!(col_ind, i-1);
            push!(entry, a1);
            # U_i 
            push!(row_ind, i);
            push!(col_ind, i);
            push!(entry, a2);
            # U_i+1
            push!(row_ind, i);
            push!(col_ind, i+1);
            push!(entry, a3);
        end
    end
    # create differential operator as a matrix (has 1 step smaller than original grid)
    L = sparse(row_ind, col_ind, entry);
    return L;
end

function adv_periodic_mat_1d_grad(
        gridpts :: Union{Matrix, Vector}, 
        c :: Real, 
        v :: Function, 
        θ :: Union{Matrix, Vector}, 
        kappa :: Real,
        dt :: Real
    )
    """ 
        Returns Jacobian of the finite difference 
        matrix for 1d, in all parameters θ through
        automatic differentiation. The result is 
        returned as an array of matrices.

        `v` is assumed to have form v(x, θ) where
        θ are parameterization. 

        Example:
            v(x, θ) = θ[1]*x^3 + sin(θ[2]*x)
    """
    n = length(gridpts);
    h = gridpts[2] - gridpts[1];
    p = length(θ);
    Ldθ = Array{Matrix}(undef, p);
    for pp = 1:p
        #println("Parameter $pp")
        row_ind = Vector{Int64}();
        col_ind = Vector{Int64}();
        entry = Vector{Float64}();
        for i = 1:n
            # grid location
            x = gridpts[i];
            # advection velocity
            v_val(θ) = v(x, θ);
            # compute coefficients
            # form: a1U_i-1 + a2U_i + a3U_i+1 = f_i
            a1_var(θ) = (kappa*dt/(dx^2)) + ((v_val(θ)*dt)/(2dx));
            a2_var(θ) = (1 - ((2kappa*dt)/(dx^2)) - c*dt);
            a3_var(θ) = (kappa*dt/(dx^2)) - ((v_val(θ)*dt)/(2dx));

            # take derivative with respect to θ using autodiff
            a1 = ForwardDiff.gradient(a1_var, θ)[pp];
            a2 = ForwardDiff.gradient(a2_var, θ)[pp];
            a3 = ForwardDiff.gradient(a3_var, θ)[pp];
            
            # store coefficients
            # store coefficients
            if i == 1
                # U_1
                push!(row_ind, i);
                push!(col_ind, i);
                push!(entry, a2);
                # U_2
                push!(row_ind, i);
                push!(col_ind, i+1);
                push!(entry, a3);
                # boundary U_0 = U_n
                push!(row_ind, i);
                push!(col_ind, n);
                push!(entry, a1);
            elseif i == n
                # U_n-1
                push!(row_ind, i);
                push!(col_ind, i-1);
                push!(entry, a1);
                # U_n
                push!(row_ind, i);
                push!(col_ind, i);
                push!(entry, a2);
                # boundary: U_n+1 = U_1
                push!(row_ind, i);
                push!(col_ind, 1);
                push!(entry, a3);
            else
                # U_i-1
                push!(row_ind, i);
                push!(col_ind, i-1);
                push!(entry, a1);
                # U_i 
                push!(row_ind, i);
                push!(col_ind, i);
                push!(entry, a2);
                # U_i+1
                push!(row_ind, i);
                push!(col_ind, i+1);
                push!(entry, a3);
            end
        end
        Ldθ[pp] = sparse(row_ind, col_ind, entry);
    end
    return Ldθ;
end

######################################################################
# Helper functions
######################################################################
function sub2ind(arr_size, i, j)
    """
        Implementation of `sub2ind` functionality
        from MATLAB.

        Inputs:
            arr_size :: Tuple{Int, Int} -> size of the array being
                                        linearized (column-wise).
            i, j     :: Int             -> Cartesian index to be
                                        converted to linear index.
        Output:
            idx      :: Int             -> linearized Cartesian index.
    """
    return LinearIndices(arr_size)[CartesianIndex.(i, j)]
end