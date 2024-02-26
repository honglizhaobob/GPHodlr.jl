module GPHodlr

    # Global imports for the module

    # for random sampling
    using Random, Statistics

    # linear algebra
    using LinearAlgebra

    # block wise linear algebra
    using BlockDiagonals, BlockArrays

    # autodifferentiation
    using ForwardDiff

    # user-defined helpers

    include("dyadic_idx.jl")  # creating indices that are powers of 2 (for dividing domain into halves)
    include("rsvd.jl")        # randomized matrix decompositions
    include("hodlr.jl")       # main routines defining HODLR operations 
    include("findiff.jl")     # main routines defining finite difference matrices
    # todo                    # main routines defining finite elements


    # export a few helper functions (so can be used in unit tests)
    export dyadic_idx, dyadic_merge
    export randn_symmetric

end