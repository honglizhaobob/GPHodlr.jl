""" 
    (03/17/2023) 
    2-dimensional SST advection velocity field models that are
    parameterized basis expansions and trained using MLE.

    The velocity models are evaluable at each spatial point, 
    and also returns the derivatives at that point with respect
    to the input parameters (used for MLE gradient descent). The
    format is assumed to be a dictionary with two fields, each 
    for a velocity component.

    Resources:
        * Inheritance in Julia:
            https://discourse.julialang.org/t/composition-and-inheritance-the-julian-way/11231

"""


######################################################################
abstract type VelocityModel end
######################################################################
mutable struct ConstantVelocity{T <: AbstractFloat} <: VelocityModel
    """
        A basic velocity model that parameterizes the field
        as constants.
    """
    # mutable parameterizations
    v1_theta :: T
    v2_theta :: T
    # the derivative is exactly known for constant model
    v_grad :: Dict{String, Vector{T}}
    function ConstantVelocity(v1_theta :: T, v2_theta :: T) where T <: AbstractFloat
        res_grad = Dict{String, Vector{T}}(
            "grad1" => ones(1),
            "grad2" => ones(1)
        );
        return new{T}(v1_theta, v2_theta, res_grad);
    end
end

function (v :: ConstantVelocity)(x :: Vector{Float64})
    """ 
        Evaluable method for `ConstantVelocity` model. 
        Also returns derivatives with respect to parameters.


        Input:
            x           2 dimensional input.
        Output:
            res         2 dimensional velocity evaluations.
    """
    res = [v.v1_theta, v.v2_theta];
    return res
end

function ∂v∂θ(v_model :: ConstantVelocity, x :: Vector{Float64})
    """
        Evaluate gradient of velocity with respect to parameters,
        as a dictionary for each component.
    """
    return v_model.res_grad;
end



######################################################################
mutable struct ChebyshevVelocity{T <: AbstractFloat} <: VelocityModel
    """
        Velocity field is parameterized as basis 
        expansion using (shifted) Chebyshev polynomials
        of the first kind. The Chebyshev polynomials
        are normalized with respect to the weight function
        corresponding to a general domain [a, b].

        See:
            https://mathworld.wolfram.com/ChebyshevPolynomialoftheFirstKind.html
        for normalizing constant in L^2.

        Each velocity component `v1`, `v2` only depend on 
        `x`, `y`. Also assumes the bases are the same for
        both `v1` and `v2`.

        The order of `v1_theta` and `v2_theta` reflects 
        that of:
            https://docs.juliahub.com/SpecialPolynomials/LrhA0/0.1.0/

        
        See also:
            * Chebyshev polynomials change much more quickly outside [-1, 1]:
                https://math.stackexchange.com/questions/2047666/chebyshev-polynomials-increase-more-quickly-than-any-other-polynomial-outside?rq=1
                http://www.cameronmusco.com/personal_site/pdfs/retreatTalk.pdf
            * Chebyshev polynomials are not used for stats:
                https://stats.stackexchange.com/questions/434936/polynomial-chebyshev-regression-versus-multi-linear-regression#:~:text=The%20Chebyshev%20polynomial%20is%20defined,(x)%20on%20the%20interval.
    """
    # domain 
    xmin :: T
    xmax :: T
    ymin :: T
    ymax :: T
    # mutable parameterizations
    v1_theta :: Vector{T}
    v2_theta :: Vector{T}

end

function (v :: ChebyshevVelocity)(x :: Vector{Float64})
    """ 
        Evaluable method for `ChebyshevVelocity` model. 


        Input:
            x           2 dimensional input.
        Output:
            res         2 dimensional velocity evaluations.

    """
    res = zeros(Float64, 2);
    p = length(v.v1_theta);
    @assert p == length(v.v2_theta)
    # rescale coefficients such that bases are orthonormal
    tmp = ones(p, 2);
    # adjust for built-in normalization constant
    tmp[1, :] .= tmp[1, :] ./ sqrt(pi);
    tmp[2:end, :] .= tmp[2:end, :] ./ sqrt(pi ./ 2);
    # adjust for domain shifting 
    tmp[:, 1] .= tmp[:, 1] .* sqrt(2 ./ (v.xmax .- v.xmin));
    tmp[:, 2] .= tmp[:, 2] .* sqrt(2 ./ (v.ymax .- v.ymin));

    # rescale input coefficients
    tmp[:, 1] .= tmp[:, 1] .* v.v1_theta;
    tmp[:, 2] .= tmp[:, 2] .* v.v2_theta;

    # shifted Chebyshev polynomials
    y1 = (2*x[1]-(v.xmax+v.xmin))/(v.xmax-v.xmin);
    y2 = (2*x[2]-(v.ymax+v.ymin))/(v.ymax-v.ymin);
    res[1] = Chebyshev(tmp[:, 1])(y1);
    res[2] = Chebyshev(tmp[:, 2])(y2);

    return res
end

function ∂v∂θ(v_model :: ChebyshevVelocity, x :: Vector{Float64})
    p = length(v_model.v1_theta);
    @assert p == length(v_model.v2_theta)
    # evalute gradient with respect to each parameters

    # ∂θᵢ∑θᵢTᵢ = Tᵢ
    res_grad = Dict{String, Vector{Float64}}(
            "grad1" => zeros(p),
            "grad2" => zeros(p)
    );
    y1 = (2*x[1]-(v_model.xmax+v_model.xmin))/(v_model.xmax-v_model.xmin);
    y2 = (2*x[2]-(v_model.ymax+v_model.ymin))/(v_model.ymax-v_model.ymin);
    for i = 1:p
        if i == 1
            res_grad["grad1"][i] = sqrt(2 ./ (v_model.xmax .- v_model.xmin)) .* basis(Chebyshev, i-1)(y1) ./ sqrt(pi);
            res_grad["grad2"][i] = sqrt(2 ./ (v_model.ymax .- v_model.ymin)) .* basis(Chebyshev, i-1)(y2) ./ sqrt(pi);
        else
            res_grad["grad1"][i] = sqrt(2 ./ (v_model.xmax .- v_model.xmin)) .* basis(Chebyshev, i-1)(y1) ./ sqrt(pi ./ 2);
            res_grad["grad2"][i] = sqrt(2 ./ (v_model.ymax .- v_model.ymin)) .* basis(Chebyshev, i-1)(y2) ./ sqrt(pi ./ 2);
        end
    end
    return res_grad;
end

######################################################################
mutable struct LegendreVelocityModel{T <: AbstractFloat} <: VelocityModel
    """ 
        A Legendre polynomial model for each components of the velocity
        field, i.e. `v1`, `v2` are expanded in shifted Legendre bases in
        general domain [a, b].

        See also:
        
        * Normalization constant: 
            https://en.wikipedia.org/wiki/Legendre_polynomials
    """
    # domain
    xmin :: T
    xmax :: T
    ymin :: T
    ymax :: T
    # mutable parameterizations
    v1_theta :: Vector{T}
    v2_theta :: Vector{T}
end

function (v :: LegendreVelocityModel)(x :: Vector{Float64})
    res = zeros(Float64, 2);
    p = length(v.v1_theta);
    @assert p == length(v.v2_theta)
    # rescale coefficients such that bases are orthonormal
    tmp = ones(p, 2);
    # adjust for built-in normalization constant
    tmp[:, 1] .= ( sqrt(2) .* tmp[:, 1] ) ./ sqrt.(2 .* (0:(p-1)) .+ 1);
    tmp[:, 2] .= ( sqrt(2) .* tmp[:, 2] ) ./ sqrt.(2 .* (0:(p-1)) .+ 1);

    # adjust for domain shifting 
    tmp[:, 1] .= tmp[:, 1] .* sqrt(2 ./ (v.xmax .- v.xmin));
    tmp[:, 2] .= tmp[:, 2] .* sqrt(2 ./ (v.ymax .- v.ymin));

    # rescale input coefficients
    tmp[:, 1] .= tmp[:, 1] .* v.v1_theta;
    tmp[:, 2] .= tmp[:, 2] .* v.v2_theta;

    # shifted Chebyshev polynomials
    y1 = (2*x[1]-(v.xmax+v.xmin))/(v.xmax-v.xmin);
    y2 = (2*x[2]-(v.ymax+v.ymin))/(v.ymax-v.ymin);
    res[1] = Legendre(tmp[:, 1])(y1);
    res[2] = Legendre(tmp[:, 2])(y2);
    return res
end

function ∂v∂θ(v_model :: LegendreVelocityModel, x :: Vector{Float64})
    p = length(v_model.v1_theta);
    @assert p == length(v_model.v2_theta)
    # compute gradient with respect to parameters
    # ∂θᵢ∑θᵢPᵢ = Pᵢ
    res_grad = Dict{String, Vector{Float64}}(
        "grad1" => zeros(p),
        "grad2" => zeros(p)
    );
    # shifted Chebyshev polynomials
    y1 = (2*x[1]-(v_model.xmax+v_model.xmin))/(v_model.xmax-v_model.xmin);
    y2 = (2*x[2]-(v_model.ymax+v_model.ymin))/(v_model.ymax-v_model.ymin);
    for i = 1:p
        res_grad["grad1"][i] = sqrt(2 ./ (v_model.xmax .- v_model.xmin)) .* basis(Legendre, i-1)(y1) .* ( sqrt(2) ./ sqrt(2 .* (i-1) .+ 1));
        res_grad["grad2"][i] = sqrt(2 ./ (v_model.ymax .- v_model.ymin)) .* basis(Legendre, i-1)(y2) .* ( sqrt(2) ./ sqrt(2 .* (i-1) .+ 1));
    end
    return res_grad
end

#############################################################################
# Velocity models with cross terms in each component
#############################################################################
mutable struct ChebyshevCrossVelocity{T <: AbstractFloat} <: VelocityModel
    """
        Velocity field is parameterized as 2d basis 
        expansion using (shifted) Chebyshev polynomials
        of the first kind. The Chebyshev polynomials
        are normalized with respect to the weight function
        corresponding to a general domain [a, b]. The basis 
        terms are enumerated according to column-major order 
        and from low to high degrees.

        The number of coefficients in each dimension will be 
        O(p^2) for p being the highest degree polynomial.

    """
    # domain 
    xmin :: T
    xmax :: T
    ymin :: T
    ymax :: T
    # mutable parameterizations
    v1_theta :: Vector{T}
    v2_theta :: Vector{T}

    function ChebyshevCrossVelocity(
        xmin :: T,
        xmax :: T,
        ymin :: T,
        ymax :: T,
        v1_theta :: Vector{T},
        v2_theta :: Vector{T}
    ) where T <: AbstractFloat
        """
            Constructor with parameter validation.
        """
        # check that number of parameters in each velocity component is perfect square
        p1 = length(v1_theta);
        p2 = length(v2_theta);
        tmp1 = round(Int64, sqrt(p1));
        tmp2 = round(Int64, sqrt(p2));
        @assert tmp1^2 == p1;
        @assert tmp2^2 == p2;
        return new{T}(
            xmin, xmax,
            ymin, ymax,
            v1_theta, v2_theta
        );
    end
end

function (v :: ChebyshevCrossVelocity)(x :: Vector{Float64})
    """ 
        Evaluable method for `ChebyshevCrossVelocity` model. 

        Currently only supports the same model for each velocity component.

        Input:
            x           2 dimensional input.
        Output:
            res         2 dimensional velocity evaluations.

    """
    p1 = length(v.v1_theta);
    p2 = length(v.v2_theta);
    @assert p1 == p2;
    # highest degree
    q = round(Int64, sqrt(p1));
    
    # create matrix of cross evaluations
    x1 = x[1]; x2 = x[2];
    _coef_matrix1 = reshape(v.v1_theta, q, q);
    _coef_matrix2 = reshape(v.v2_theta, q, q);

    # rescale coefficients such that bases are orthonormal in each dimension
    tmp = ones(q, 2);

    # adjust for built-in normalization constant
    tmp[1, :] .= tmp[1, :] ./ sqrt(pi);
    tmp[2:end, :] .= tmp[2:end, :] ./ sqrt(pi ./ 2);
    # adjust for domain shifting 
    tmp[:, 1] .= tmp[:, 1] .* sqrt(2 ./ (v.xmax .- v.xmin));
    tmp[:, 2] .= tmp[:, 2] .* sqrt(2 ./ (v.ymax .- v.ymin));

    _renormalize_matrix = tmp[:, 1] .* tmp[:, 2]';

    # shifted Chebyshev polynomials
    y1 = (2*x1-(v.xmax+v.xmin))/(v.xmax-v.xmin);
    y2 = (2*x2-(v.ymax+v.ymin))/(v.ymax-v.ymin);

    # create matrix of cross-basis evaluations
    basis_i = Float64[basis(Chebyshev, i)(y1) for i in 0:q-1];
    basis_j = Float64[basis(Chebyshev, i)(y2) for i in 0:q-1];
    _cross_basis_matrix_ij = basis_i .* basis_j';

    # evaluate each velocity component
    res = zeros(Float64, 2);
    res[1] = sum(_coef_matrix1 .* _renormalize_matrix .* _cross_basis_matrix_ij);
    res[2] = sum(_coef_matrix2 .* _renormalize_matrix .* _cross_basis_matrix_ij);
    return res
end

function ∂v∂θ(v_model :: ChebyshevCrossVelocity, x :: Vector{Float64})
    p = length(v_model.v1_theta);
    @assert p == length(v_model.v2_theta)

    # highest degree in each dimension
    q = round(Int64, sqrt(p));

    # evalute gradient with respect to each parameters

    # ∂θᵢⱼ∑∑θᵢⱼ*TᵢTⱼ = TᵢTⱼ
    res_grad = Dict{String, Vector{Float64}}(
            "grad1" => zeros(p),
            "grad2" => zeros(p)
    );

    # rescale coefficients such that bases are orthonormal in each dimension
    tmp = ones(q, 2);

    # adjust for built-in normalization constant
    tmp[1, :] .= tmp[1, :] ./ sqrt(pi);
    tmp[2:end, :] .= tmp[2:end, :] ./ sqrt(pi ./ 2);
    # adjust for domain shifting 
    tmp[:, 1] .= tmp[:, 1] .* sqrt(2 ./ (v_model.xmax .- v_model.xmin));
    tmp[:, 2] .= tmp[:, 2] .* sqrt(2 ./ (v_model.ymax .- v_model.ymin));
    _renormalize_matrix = tmp[:, 1] .* tmp[:, 2]';

    # shifted input 
    y1 = (2*x[1]-(v_model.xmax+v_model.xmin))/(v_model.xmax-v_model.xmin);
    y2 = (2*x[2]-(v_model.ymax+v_model.ymin))/(v_model.ymax-v_model.ymin);
    # create matrix of cross-basis evaluations
    basis_i = Float64[basis(Chebyshev, i)(y1) for i in 0:q-1];
    basis_j = Float64[basis(Chebyshev, i)(y2) for i in 0:q-1];
    _cross_basis_matrix_ij = basis_i .* basis_j';

    res = _renormalize_matrix .* _cross_basis_matrix_ij;
    res_grad["grad1"][:] .= copy(res[:]);
    res_grad["grad2"][:] .= copy(res[:]);
    return res_grad;
end