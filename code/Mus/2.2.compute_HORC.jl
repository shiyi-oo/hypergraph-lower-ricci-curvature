#!/usr/bin/env julia

using Pkg
project_path = normpath(joinpath(@__DIR__, "..", "..", "orchid"))
Pkg.activate(project_path)
Pkg.instantiate()

using Orchid, SparseArrays, DelimitedFiles, Base.Threads
using CSV
using DataFrames

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

function extract_edge_curvature(result)
    hasproperty(result, :aggregations) || return nothing
    aggregations = result.aggregations
    for entry in aggregations
        if hasproperty(entry, :edge_curvature)
            return entry.edge_curvature
        elseif entry isa AbstractDict
            haskey(entry, :edge_curvature) && return entry[:edge_curvature]
        end
    end
    return nothing
end

function process_site(site_id::Int)
    println("Start site $site_id")
    file_path = "derived_data/mus/mus_$(site_id).tsv"
    H_list = parse_edgelist(file_path)
    H_Inc = generate_incidence_matrix(H_list)
    result = hypergraph_curvatures(
        Orchid.DisperseWeightedClique,
        Orchid.AggregateMean,
        H_Inc,
        0.1,
        Orchid.CostOndemand
    )

    edge_curvature = extract_edge_curvature(result)
    hyperedge_labels = [join(string.(edge), ",") for edge in H_list]
    values = Vector{Union{Missing, Float64}}(undef, length(hyperedge_labels))

    if edge_curvature === nothing
        fill!(values, missing)
    elseif length(edge_curvature) != length(values)
        @warn "Edge curvature count mismatch" site_id length_edges=length(values) length_curvature=length(edge_curvature)
        fill!(values, missing)
    else
        for (idx, value) in enumerate(edge_curvature)
            if value === nothing || value isa Missing
                values[idx] = missing
            else
                values[idx] = Float64(value)
            end
        end
    end

    return DataFrame(
        hg_idx = fill(site_id, length(hyperedge_labels)),
        hyperedge = hyperedge_labels,
        horc = values,
    )
end

function process_all_sites_and_save(start_id::Int, end_id::Int, output_path::String)
    n = end_id - start_id + 1
    nthreads = Base.Threads.nthreads()
    mkpath(dirname(output_path))
    rows_per_thread = [DataFrame(hg_idx=Int[], hyperedge=String[], horc=Union{Missing,Float64}[]) for _ in 1:nthreads]
    total_start = time()
    Base.Threads.@threads for i in 0:(n-1)
        site_id = start_id + i
        df = process_site(site_id)
        append!(rows_per_thread[Base.Threads.threadid()], df)
    end
    total_elapsed = time() - total_start
    println("Elapsed: ", total_elapsed, "s")
    combined = reduce(vcat, rows_per_thread)
    CSV.write(output_path, combined; delim = '\t')
end

# Set the range of site_ids you want to process
start_id = parse(Int, ARGS[1])
end_id = parse(Int, ARGS[2])

output_path = "derived_data/horc/mus_$(start_id)_$(end_id)_horc.tsv"

process_all_sites_and_save(start_id, end_id, output_path)
