module GeneralPositiveKappa

export kappa_true, PRODUCTION_NX, PRODUCTION_NY

function kappa_true(x::Real, y::Real)
    x = Float64(x)
    y = Float64(y)

    log_kappa =
        log(0.8) +
        0.28 * cos(2.0 * pi * x) +
        0.22 * sin(2.0 * pi * y) +
        0.16 * cos(2.0 * pi * (x + y)) +
        0.10 * cos(2.0 * pi * (5.0 * x - 3.0 * y)) 
        +0.075 * sin(2.0 * pi * (9.0 * x + 4.0 * y)) 
        +0.050 * cos(2.0 * pi * (14.0 * x - 11.0 * y))

    return exp(log_kappa)
end

const PRODUCTION_NX = 32
const PRODUCTION_NY = 32

end
