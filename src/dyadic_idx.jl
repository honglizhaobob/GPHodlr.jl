# Helper routine to bisect indices

function dyadic_idx(n::Int64, level::Int64)
    """
        Assuming 1-based indexing, returns bi-partitioned 
        segments of sequential indices. 

        E.g. n = 128, level = 2 will divide the 
        index array 1:128 twice, yielding 4 components:
            1:32
            33:64
            65:96
            97:128
    """
    @assert(mod(n, 2) == 0, "Indices must be multiples of 2. ");
    @assert(0 <= level <= log2(n), "Too many / few levels. ")
    # initialize array
    idx_array = Array{Any}(undef, 2^level);
    # size of each component
    size = Int(n / 2^level);
    for i in 1:2^level
        idx_array[i] = (1+size*(i-1)):(1+size*(i-1)+size-1);
    end
    return idx_array
end

function dyadic_merge(index_set::Vector{Any}, back_level::Int64)
    """
        Given a set of index components generated using
        `dyadic_idx(n, level)`, restore index components 
        of previous level, i.e. `dyadic_idx(n, level-back_level)`.

        This computation is cheaper than computing index 
        components from scratch when we have access to
        finer levels of indices.

        e.g. If `a` is a level=5 partition for n=256, 
        then `dyadic_merge(a, back_level=5)` should
        recover 1:256
    """
    # number of components
    n = length(index_set);
    # new number of components = n / 2^back_level
    new_n = Int(n / 2^back_level);
    # initialize array
    idx_array = Array{Any}(undef, new_n);
    # skipped old components
    skipsize = 2^back_level;
    for i in 1:new_n
        left_comp = index_set[1+(i-1)*skipsize];
        right_comp = index_set[1+(i-1)*skipsize+skipsize-1];
        # merge
        idx_array[i] = left_comp[1]:right_comp[end];
    end
    return idx_array
end