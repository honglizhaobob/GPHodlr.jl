# Test scalability of HODLR operations and output plots, these run 
# for a long time and are not suitable for unit tests.

### COME BACK LATER

##########
# Test 1: HODLR construction
##########

all_sizes = 2 .^ (8:12);
all_runtimes = zeros(length(all_sizes));
for i = eachindex(all_sizes)
    println(i)
    # take size
    n = all_sizes[i]
    # create HODLR matrix
    A = create_low_rank_two_level_matrix(n);
    dt = @elapsed begin
        A_hodlr = holdr(v->dummy_matvec_query(A, v), n, 2, 32, 10);
    end
    # store time
    all_runtimes[i] = dt;
end
# compare with O(nlogn)
nlogn = all_sizes .* ((8:12).^2);
p = plot(9:12, log.(nlogn)[2:end], markershape = :auto, linewidth=2., label=L"$N\log^2 N$")
plot!(p, 9:12, log.(all_runtimes)[2:end], markershape = :auto, linewidth=2., label="Runtime", legend=:topleft, dpi=200)
savefig(p, "./fig/hodlr/A_hodlr_construct_scale.png")



##########
# Test 2: scaling of A*v
##########
all_sizes = 2 .^ (8:15);
all_runtimes = zeros(length(all_sizes));
for i = eachindex(all_sizes)
    println(i)
    # take size
    n = all_sizes[i]
    # create HODLR matrix
    A = create_low_rank_two_level_matrix(n);
    A_hodlr = hodlr(A, 2, 32);
    # create random vector
    v = randn(n, 1);
    dt = @elapsed begin
        # execute HODLR matvec
        res = hodlr_prod(A_hodlr, v);
    end
    # store time
    all_runtimes[i] = dt;
end
# compare with O(nlogn)
nlogn = all_sizes .* (9:15);
p = plot(9:15, log.(nlogn[2:end]), markershape = :auto, linewidth=2., label=L"$N\log N$")
plot!(p, 9:15, log.(all_runtimes)[2:end], markershape = :auto, linewidth=2., label="Runtime", legend=:topleft, dpi=200)
savefig(p, "./fig/hodlr/Av_hodlr_scale.png")


