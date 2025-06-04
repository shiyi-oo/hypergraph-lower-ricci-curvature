using Pkg
Pkg.activate("/work/users/s/h/shiyi/hypergraph_with_curvature/orchid")
using Orchid, SparseArrays, DelimitedFiles, Base.Threads, JSON

# Function to parse the edge list
parse_edgelist(fp) = [parse.(Int, split(r, '\t')) for r in readlines(fp) if r != ""]

# Function to generate incidence matrix
function generate_incidence_matrix(H_list::Vector{Vector{Int64}})
    all_nodes = Set{Int}()
    for edge in H_list
        for node in edge
            push!(all_nodes, node)
        end
    end
    nodes = sort(collect(all_nodes))
    num_edges = length(H_list)
    num_nodes = length(nodes)
    incidence_matrix = zeros(Int, num_edges, num_nodes)
    for (i, edge) in enumerate(H_list)
        for node in edge
            col_index = findfirst(x -> x == node, nodes)
            if col_index !== nothing
                incidence_matrix[i, col_index] = 1
            end
        end
    end
    return sparse(incidence_matrix)
end

function process_site(site_id::Int)
    println("Start site $site_id")
    file_path = "derived_data/stex/stex_$(site_id).tsv"
    H_list = parse_edgelist(file_path)
    H_Inc = generate_incidence_matrix(H_list)
    horc = hypergraph_curvatures(
        Orchid.DisperseWeightedClique,
        Orchid.AggregateMean,
        H_Inc,
        0.0,
        Orchid.CostOndemand
    )

    return site_id => horc
end

function process_all_sites_and_save(start_id::Int, end_id::Int, output_json::String)
    n = end_id - start_id + 1
    nthreads = Base.Threads.nthreads()

    # Each thread gets its own Dict
    thread_results = [Dict{Int, Any}() for _ in 1:nthreads]
    total_start = time()
    Base.Threads.@threads for i in 1:n
        site_id = start_id + i - 1
        result = process_site(site_id)

        # Use threadid() to avoid race conditions
        thread_dict = thread_results[Base.Threads.threadid()]
        thread_dict[result[1]] = result[2]
    end
    total_elapsed = time() - total_start
    println("Elapsed: ", total_elapsed, "s")
    # Merge all thread-local Dicts
    all_data = merge(thread_results...)

    # Save to JSON
    open(output_json, "w") do io
        JSON.print(io, all_data)
    end
end

# Set the range of site_ids you want to process
start_id = parse(Int, ARGS[1])
end_id = parse(Int, ARGS[2])

output_json = "derived_data/horc_$(start_id)_$(end_id).json"

process_all_sites_and_save(start_id, end_id, output_json)
