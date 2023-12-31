using SafeTestsets 


######################################################################
# HODLR matrix implementation tests
######################################################################
# @safetestset "basic hodlr construction accuracy" begin
#     include("basic_hodlr_accuracy_test.jl");
# end

######################################################################
# HODLR operation: A*B for B a regular matrix
######################################################################
# @safetestset "hodlr matmat" begin
#     include("hodlr_matmat_test.jl");
# end

######################################################################
# HODLR operation: A\b
######################################################################
# @safetestset "hodlr solve" begin 
#     include("hodlr_solve_test.jl");
# end

######################################################################
# HODLR operation: A⊙B (Hadamard product) for B another HODLR matrix
######################################################################
# @safetestset "hodlr hadamard" begin
#     include("hodlr_hadamard_test.jl");
# end

######################################################################
# HODLR operation: A\B for B another HODLR matrix
######################################################################
# @safetestset "hodlr inverse mat multiply" begin 
#     include("hodlr_invmult_test.jl");
# end


######################################################################
# HODLR operation: matrix gradients with respect to parameters
######################################################################
@safetestset "hodlr gradients" begin
    include("hodlr_grad_test.jl");
end