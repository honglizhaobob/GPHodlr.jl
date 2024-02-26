""" 
    (03/15/2023) 
    Helper functions to preprocess Sea Surface Temperature (SST) data.

    The raw data is stored in `.jld` files and are assumed to be 
    dictionaries containing the following fields:

    "lat"=> an approximately uniform grid containing latitude locations,
    which is interpreted as the y-axis.

    "lon"=> longitudal recordings, interpreted as the x-axis.

    "cloud_mask"=> 2d matrix of size (ny x nx), containing 0's and 
    1's, where 1's indicate the SST reading is masked by cloud. 

    "ssta"=> 2d matrix of size (ny x nx), containing zero-mean SST
    anomaly readings. The `NaN` values represent land locations.

"""
struct SSTPartition
    """ 
        An immutable struct holding a block of the 
        observed SST anomaly data. Partitions share 
        boundaries with adjacent other partitions.

        u                -> a subset of the observed anomaly data.
        cloud_mask       -> a subset of the cloud mask, should have the 
                            same size as `u`.
        xgrid, ygrid     -> corresponding portions of the spatial grids.
        global_idx_range -> A vector of tuples storing the index ranges of 
                            data covered:
                            [
                                (idx_xmin_partition, idx_xmax_partition),
                                (idx_ymin_partition, idx_ymax_partition),
                            ]
    """
    u :: Matrix{Float64}
    cloud_mask :: Matrix{Float64}
    xgrid :: Vector{Float64}
    ygrid :: Vector{Float64}
    global_idx_range :: Vector{Tuple{Int64, Int64}}

end

struct SSTData
    """
        Preprocessed SST numerical data ready for maximum likelihood 
        computations.

        u_full                  -> full data from the partition including
                                possible NaN values, used as reference.

        u_observed              -> observations fed into Gaussian process;
                                   non-land, and not cloud masked.

        xgrid, ygrid            -> spatial grids for this SST partition.

        obs_local_inds          -> local (linearized) indices of observed 
                                   locations. 
        
        mask_local_inds         -> local (lineared) indices of masked locations.

        obs_global_inds         -> global (Cartesian) indices of observed locations.

        mask_global_inds        -> global (Cartiesian) indices of masked locations.

    """
    u_full :: Vector{Float64}
    u_observed :: Vector{Float64}
    xgrid :: Vector{Float64}
    ygrid :: Vector{Float64}
    obs_local_inds :: Vector{Int64}
    mask_local_inds :: Vector{Int64}
    obs_global_inds :: Vector{Tuple{Int64, Int64}}
    mask_global_inds :: Vector{Tuple{Int64, Int64}}
    # store imputed values, initialized as a vector of 0's
    u_imputations :: Vector{Float64}

    function SSTData(
        u_full :: Vector{Float64},
        u_observed :: Vector{Float64},
        xgrid :: Vector{Float64},
        ygrid :: Vector{Float64},
        obs_local_inds :: Vector{Int64},
        mask_local_inds :: Vector{Int64},
        obs_global_inds :: Vector{Tuple{Int64, Int64}},
        mask_global_inds :: Vector{Tuple{Int64, Int64}}
    )
        # compute number of masked values
        n_masked = length(mask_local_inds);
        # create buffer for imputations
        imputations = zeros(Float64, n_masked);
        return new(
            u_full,
            u_observed,
            xgrid,
            ygrid,
            obs_local_inds,
            mask_local_inds,
            obs_global_inds,
            mask_global_inds,
            imputations
        );
    end
end

# ======================================================================

function load_sst(path :: String)
    """
        Loads the data and adjusts dimensions as needed.

        The data is transposed when building partitions.
    """
    raw_data = load(path);
    # unpack data and convert to the same data type (adjusted ordering)
    u = convert(Matrix{Float64}, transpose(raw_data["ssta"][end:-1:1, :]));
    cloud_mask = convert(Matrix{Float64}, transpose(raw_data["cloud_mask"][end:-1:1, :]));
    @assert all(size(u) .== size(cloud_mask));

    # treat NaN values as masked
    replace!(cloud_mask, NaN=>1.0);
    @assert iszero(sum(isnan.(cloud_mask)));

    xgrid = convert(Vector{Float64}, raw_data["lon"]);
    ygrid = convert(Vector{Float64}, raw_data["lat"][end:-1:1]);
    # make sure grids are in sorted order
    xgrid = sort(xgrid);
    ygrid = sort(ygrid);
    # find coarest step size
    xmin, xmax = minimum(xgrid), maximum(xgrid);
    ymin, ymax = minimum(ygrid), maximum(ygrid);

    dx = minimum(unique(xgrid[2:end].-xgrid[1:end-1]));
    dy = minimum(unique(ygrid[2:end].-ygrid[1:end-1]));
    # rediscretize
    xgrid = convert(Vector{Float64}, xmin:dx:xmax);
    ygrid = convert(Vector{Float64}, ymin:dy:ymax);
    
    # if size of data and size of rediscretized grid do not match, force matching
    # (essentially ignoring the end few points, which should have approximately no effect
    # since step size is small). By construction, grid will only be more than the size of
    # `u`, if they happen to not match (since we are taking the minimum when rediscretizing).
    nx, ny = length(xgrid), length(ygrid);
    if !all((nx, ny).==size(u))
        # for reporting
        tmpnx, tmpny = nx, ny;
        nx, ny = size(u);
        xgrid = xgrid[1:nx];
        ygrid = ygrid[1:ny];
        @warn "Grid size reassigned, previous sizes ($(tmpnx), $(tmpny)), with data size $(size(u))";
    end

    # assert the dimensions are matching
    @assert all((nx, ny).==size(u));
    @assert all((nx, ny).==size(cloud_mask));
    return u, cloud_mask, xgrid, ygrid
end


function generate_partitions(u, cloud_mask, xgrid, ygrid, num_parts_x, num_parts_y)
    """
        Divide preprocessed grid data into partitions, stored as structs.

        The resulting partitions will be a (NUM_PATRS x NUM_PARTS) grid
        of blocks. By default, the size of partitions is determined by 
        [floor(nx/num_parts), floor(nx/num_parts)], which may have left-over
        blocks at the ends of the domain; they can be either pruned, or 
        incorporated normally.
    """
    nx, ny = length(xgrid), length(ygrid);
    part_nx = floor(Int, nx/num_parts_x);
    part_ny = floor(Int, ny/num_parts_y);
    # partition the indices
    partitions = Array{Any}(undef, num_parts_x+1, num_parts_y+1);
    for i = 0:num_parts_x
        for j = 0:num_parts_y
            if i != num_parts_x
                tmp_x = (i*(part_nx-1)+1):((i+1)*(part_nx-1)+1);
            else
                # last block may be non-uniform (left over block)
                tmp_x = (i*(part_nx-1)+1):nx;
            end

            if j != num_parts_y
                tmp_y = (j*(part_ny-1)+1):((j+1)*(part_ny-1)+1);
            else
                tmp_y = (j*(part_ny-1)+1):ny;
            end
            # index into the grid and save data subset
            tmp_u = u[tmp_x, tmp_y];
            tmp_mask = cloud_mask[tmp_x, tmp_y];
            tmp_xgrid = xgrid[tmp_x];
            tmp_ygrid = ygrid[tmp_y];
            tmp_global_idx_range = [
                (tmp_x[1], tmp_x[end]), 
                (tmp_y[1], tmp_y[end])
            ];
            partitions[i+1, j+1] = SSTPartition(tmp_u, tmp_mask, tmp_xgrid, tmp_ygrid, tmp_global_idx_range);
        end
    end
    return partitions
end

function preprocess_partition(partition :: SSTPartition)
    """
        Given a partition of the SST grid data, extracts the
        observations after applying cloud mask.

        Inputs:

            partition     -> a subset of the SST observations

        Outputs:

            u_observed :: Vector{Float64}
                          -> extracted non-land, un-masked
                          observations used for MLE.
    """
    nx = length(partition.xgrid);
    ny = length(partition.ygrid);
    # get global ranges
    xinds = partition.global_idx_range[1][1]:partition.global_idx_range[1][2];
    yinds = partition.global_idx_range[2][1]:partition.global_idx_range[2][2];
    # store values and observed global (Cartesian) indices
    u_full = partition.u[:];
    u_observed = Float64[];
    observed_global_inds = Tuple{Int64, Int64}[];
    masked_global_inds = Tuple{Int64, Int64}[];
    # save local (linearized) indices
    observed_local_inds = Int64[];
    masked_local_inds = Int64[];
    for i = 1:nx
        for j = 1:ny
            tmp = partition.u[i, j];
            # global indices on grid
            tmp_x_ind = xinds[i];
            tmp_y_ind = yinds[j];
            # check that the location is not land
            if !isnan(tmp)
                # check that the location is not masked
                if iszero(partition.cloud_mask[i, j])
                    # save this location to observed values
                    push!(u_observed, tmp);
                    push!(observed_global_inds, (tmp_x_ind, tmp_y_ind));
                    push!(observed_local_inds, sub2ind((nx, ny), i, j));
                else
                    # save this location to unobserved values
                    push!(masked_global_inds, (tmp_x_ind, tmp_y_ind));
                    push!(masked_local_inds, sub2ind((nx, ny), i, j));
                end
            end
        end
    end
    res = SSTData(
        u_full,
        u_observed,
        partition.xgrid,
        partition.ygrid,
        observed_local_inds, 
        masked_local_inds,
        observed_global_inds,
        masked_global_inds
    );
    return res
end

function subsampling(nx :: Int64, ny :: Int64, observed_idx :: Vector{Int64})
    """
        Given 2d grid size and a vector of linearized indices, constructs 
        a matrix of 0's and 1's such that:
            u_obs = D * u
        where `u_obs` contains the observed values recorded in the 
        vector of linearized indices, `u` contains the full set of values.

        nx, ny           -> grid dimensions
        observed_idx     -> linearized (Julia default ordering) indices of 
                         a subset of values.

    """
    return I(nx*ny)[observed_idx, :]
end



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

function ind2sub(arr_size, lin_idx)
    """
        Converts a linearized index into Cartesian indices.
    """
    res = CartesianIndices(arr_size)[lin_idx];
    return res[1], res[2];
end