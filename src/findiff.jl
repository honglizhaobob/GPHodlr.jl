# Helper script for building differential operators by finite difference
#
# References:
# -- Finite difference: LeVeque, Finite Difference Methods for Ordinary and Partial Differential Equations
# -- https://utminers.utep.edu/oktweneboah/files/talk.pdf
# -- https://github.com/luraess/parallel-gpu-workshop-JuliaCon21/tree/main/scripts

# Ferrite: https://github.com/Ferrite-FEM/Ferrite.jl

############################################################################################
# 1d Advection-diffusion-reaction
############################################################################################
function advection_diffusion_reaction1d(gridpts, phi, c, v, kappa)
    """
        Creates the discretized stationary ADR opeartor in 1d. 
        Assuming Neumann condition.

        Advection velocity v is constant.

        Eqn:
            -κ(Uxx) + vUx + cU = phi
            Ux = 0 at boundary
    """
    L = adv_mat1d(gridpts, c, v, kappa);
    # solve
    sol = L\phi;
    return sol
end

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

######################################################################
# Constant velocity and its jacobian, for testing
######################################################################

function adv_mat1d_cstvel(gridpts, c, v, kappa)
    """ Returns finite difference matrix for 1d. """
    n = length(gridpts);
    h = gridpts[2] - gridpts[1];
    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();
    for i = 1:n
        # grid location
        x = gridpts[i];
        # advection velocity
        v_val = v;
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

function adv_mat1d_cstvel_dv(gridpts, c, v, kappa)
    """ Returns finite difference matrix for 1d. """
    n = length(gridpts);
    h = gridpts[2] - gridpts[1];
    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();
    for i = 1:n
        # grid location
        x = gridpts[i];
        # advection velocity
        dv = 1;
        # compute coefficients
        # form: a1U_i-1 + a2U_i + a3U_i+1 = f_i
        a1 = (-dv/2h);
        a2 = 0; # no dependence on v
        a3 = (dv/2h);
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

######################################################################
# Linear velocity v(x) = a*x + b and its Jacobian, for testing
######################################################################

function adv_mat1d_linvel(gridpts, c, θ, kappa)
    """ Returns finite difference matrix for 1d. """
    n = length(gridpts);
    h = gridpts[2] - gridpts[1];
    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();
    for i = 1:n
        # grid location
        x = gridpts[i];
        # advection velocity
        v_val = θ[1]*x + θ[2];
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

function adv_mat1d_linvel_dv1(gridpts, c, θ, kappa)
    """ 
        Returns first Jabobian (there are two) of
        finite difference matrix for 1d. 
    """
    n = length(gridpts);
    h = gridpts[2] - gridpts[1];
    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();
    for i = 1:n
        # grid location
        x = gridpts[i];
        # compute coefficients
        # form: a1U_i-1 + a2U_i + a3U_i+1 = f_i
        a1 = (-x/2h);
        a2 = 0; # no dependence on v
        a3 = (x/2h);
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

function adv_mat1d_linvel_dv2(gridpts, c, θ, kappa)
    """ 
        Returns second Jabobian (there are two) of
        finite difference matrix for 1d. 
    """
    n = length(gridpts);
    h = gridpts[2] - gridpts[1];
    row_ind = Vector{Int64}();
    col_ind = Vector{Int64}();
    entry = Vector{Float64}();
    for i = 1:n
        # grid location
        x = gridpts[i];
        # compute coefficients
        # form: a1U_i-1 + a2U_i + a3U_i+1 = f_i
        a1 = (-1/2h);
        a2 = 0; # no dependence on v
        a3 = (1/2h);
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

function adv_mat1d_linvel_dv_full(gridpts, c, θ, kappa)
    """ 
        Linear velocity Jacobian, stored as an array.
    """
    res = Array{Any}(undef, 2);
    res[1] = adv_mat1d_linvel_dv1(gridpts, c, θ, kappa);
    res[2] = adv_mat1d_linvel_dv2(gridpts, c, θ, kappa);
    return res;
end


############################################################################################
# 2d Advection-diffusion-reaction
############################################################################################


##############################################################################
# General velocity v(x) and its Jacobian, !!! need to integrate with Autodiff
##############################################################################

function advection_diffusion_reaction(gridpts :: Union{StepRangeLen, Vector}, phi :: Matrix, c :: Real, v :: Function, kappa :: Real)
    """
        Creates the discretized stationary ADR operator in 
        2d. Advection velocity depends on location.
    """
    # form: aU_ij+bU_(i+1)j+cU_(i-1)j+dU_i(j+1)+eU_i(j-1) = f_ij
    n = length(gridpts);
    # sparse operator matrix
    L = adv_mat(gridpts, c, v, kappa);
    # column-major vectorize force term
    f = phi[:];
    # reshape back
    sol = L\f;
    U = reshape(sol,(n,n));
    return U;
end


function advection_diffusion_reaction(phi :: Matrix, n :: Int, c :: Real, v :: Vector, h :: Real, kappa :: Real)
    """
        Creates the discretized stationary ADR operator in 
        2d. Advection velocity is constant.

        The first order derivatives are discretized using
        centered difference. The operator is represented
        as a matrix assuming uniform grid.

        Inputs:
            phi :: Matrix             -> an (n x n) matrix containing
                                      source term at each grid point. 

            n   :: Int                -> size of grid

            c   :: Float              -> (parameter) reaction speed

            v   :: Vector{Any}        -> (parameter) a 2d vector containing
                                      advection velocities.
            h   :: Float              -> grid size (assumed dx = dy)

            kappa :: Float          -> (parameter) diffusivity
    """
    # form: aU_ij+bU_(i+1)j+cU_(i-1)j+dU_i(j+1)+eU_i(j-1) = f_ij

    # sparse operator matrix
    L = adv_mat(n, c, v, h, kappa);
    # column-major vectorize force term
    f = phi[:];
    # reshape back
    sol = L\f;
    U = reshape(sol,(n,n));
    return U;
end

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


function adv_mat(n :: Int, c :: Real, v :: Vector, h :: Real, kappa :: Real)
    """
        Re-written matrix creation of `advection_diffusion_reaction`
        by creating sparse matrix from vectors (in lieu of initializing
        zero matrix first).
    """
    # parameters for operator

    # U_i,j
    p_a = (4kappa/h^2+c);
    # U_i+1,j
    p_b = (v[1]/2h-kappa/h^2);
    # U_i-1,j
    p_c = (-v[1]/2h-kappa/h^2); 
    # U_i,j+1
    p_d = (v[2]/2h-kappa/h^2);
    # U_i,j-1
    p_e = (-v[2]/2h-kappa/h^2);

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
        println("Parameter $pp")
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

############################################################################################
# Archived
############################################################################################

############################################################################################
# 2d Laplace diffusion operator
############################################################################################
function laplacian(n)
    """
        Creates the Laplace 5-point stencil as a matrix,
        assuming natural row-wise ordering 
        see LeVeque finite difference Sec.3.3, (3.11)).

        Note: natural row-wise ordering in LeVeque is transpose
        of MATLAB matrix ordering.

        Inputs:
            n :: Int                -> size of numerical grid. Grid 
                                    is assumed to be square.
    """
    A = BlockArray(fill(0, n^2, n^2), repeat([n],n), repeat([n],n));
    # tridiag matrix for each row
    T = Tridiagonal(repeat([1],n-1), repeat([-4],n), repeat([1],n-1));
    # identity matrix
    id = I(n);
    # fill in blocks
    for ii = 1:n
        A[Block(ii,ii)] .= T; 
    end
    for ii = 1:n-1
        A[Block(ii,ii+1)] .= id;
        A[Block(ii+1,ii)] .= id;
    end
    return A
end