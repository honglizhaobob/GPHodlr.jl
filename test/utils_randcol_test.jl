using GPHodlr
using Statistics
using LinearAlgebra
using Random

@testset begin
#----------
# test across 20 random seeds
for seednum = 1:20
    Random.seed!(seednum);
    # randomly generate a large rect. matrix (note #rows must >> #cols)
    B = 10*randn(1000, 100);
    # augment B by creating useless columns
    B_aug = [B B[:, 1:100]];
    B_aug_hat = GPHodlr.randcols(B_aug, 100, 10);
    @test (norm(B_aug-B_aug_hat)) <= 1e-6
end
#----------
end

