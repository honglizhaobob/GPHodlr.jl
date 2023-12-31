using SafeTestsets 

######################################################################
# Utility function test suites
######################################################################


######################################################################
# HODLR matrix implementation tests
######################################################################
@safetestset "basic hodlr construction" begin
    include("basic_hodlr_accuracy_test.jl");
end