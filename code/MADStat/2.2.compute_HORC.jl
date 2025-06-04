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

function compute_horc(file_path)
    println("Start $file_path")
    H_list = parse_edgelist(file_path)
    H_Inc = generate_incidence_matrix(H_list)
    horc = hypergraph_curvatures(
        Orchid.DisperseWeightedClique,
        Orchid.AggregateMean,
        H_Inc,
        0.0,
        Orchid.CostOndemand
    )
    return horc
end


file_path = "./derived_data/hyperedges.tsv"
output_json = "./derived_data/horc.json"


timing = @timed compute_horc(file_path);
println("Elapsed: ", timing.time, " s, alloc: ", timing.bytes, " bytes")

open(output_json, "w") do io
    JSON.print(io, timing.value)
end
