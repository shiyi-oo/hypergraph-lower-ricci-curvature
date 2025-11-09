#!/usr/bin/env julia

using Pkg

project_path = normpath(joinpath(@__DIR__, "..", "..", "orchid"))
Pkg.activate(project_path)
Pkg.instantiate()

using CSV
using JSON
using Orchid
using Printf
using SparseArrays

function parse_edge_nodes(nodes_str::AbstractString)
    clean = strip(nodes_str)
    isempty(clean) && return String[]
    json_ready = replace(clean, '\'' => '"')
    isempty(strip(json_ready, ['[', ']', '"'])) && return String[]
    return String.(JSON.parse(json_ready))
end

get_with_fallback(d, key::Symbol) = _get_with_fallback(d, key)
get_with_fallback(d, key::String) = _get_with_fallback(d, Symbol(key))

function _get_with_fallback(d, key::Symbol)
    if d isa AbstractDict
        if haskey(d, key)
            return d[key]
        end
        str_key = String(key)
        if haskey(d, str_key)
            return d[str_key]
        end
    elseif hasproperty(d, key)
        return getproperty(d, key)
    end
    return nothing
end

function load_hypergraph_edges(rows)
    hypergraphs = Dict{Int, Vector{Vector{String}}}()
    row_map = Dict{Int, Vector{Int}}()
    for (row_idx, row) in enumerate(rows)
        hg_raw = get_with_fallback(row, :hg_idx)
        members_raw = get_with_fallback(row, :members)
        if hg_raw === nothing || members_raw === nothing
            continue
        end
        if ismissing(hg_raw) || ismissing(members_raw)
            continue
        end
        hg = hg_raw isa Integer ? Int(hg_raw) : parse(Int, String(hg_raw))
        members = parse_edge_nodes(String(members_raw))
        isempty(members) && continue
        push!(get!(hypergraphs, hg, Vector{Vector{String}}()), members)
        push!(get!(row_map, hg, Int[]), row_idx)
    end
    return hypergraphs, row_map
end

function generate_incidence_matrix(edges::Vector{Vector{String}})
    all_nodes = Set{String}()
    for edge in edges
        for node in edge
            push!(all_nodes, node)
        end
    end
    nodes = sort!(collect(all_nodes))
    node_index = Dict(node => col for (col, node) in enumerate(nodes))
    incidence = zeros(Int, length(edges), length(nodes))
    for (edge_idx, edge) in enumerate(edges)
        for node in edge
            incidence[edge_idx, node_index[node]] = 1
        end
    end
    return sparse(incidence), nodes
end


function extract_edge_curvature(result)
    aggregations = get_with_fallback(result, :aggregations)
    aggregations === nothing && return nothing
    for entry in aggregations
        candidate = get_with_fallback(entry, :edge_curvature)
        candidate === nothing && continue
        return candidate
    end
    return nothing
end


function build_horc_specs()
    dispersion_opts = [
        # (label = "uw_clique", type = Orchid.DisperseUnweightedClique),
        # (label = "w_clique", type = Orchid.DisperseWeightedClique),
        (label = "uw_star", type = Orchid.DisperseUnweightedStar),
    ]
    aggregation_opts = [
        # (label = "mean", type = Orchid.AggregateMean),
        (label = "max", type = Orchid.AggregateMax),
    ]
    # alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    alphas = [0.1]
    specs = NamedTuple[]
    for d in dispersion_opts, a in aggregation_opts, alpha in alphas
        name = if d.type === Orchid.DisperseUnweightedClique &&
                   a.type === Orchid.AggregateMean &&
                   isapprox(alpha, 0.0; atol = 1e-9)
            :horc
        else
            alpha_label = replace(@sprintf("%.1f", alpha), "." => "p")
            Symbol("horc_$(d.label)_$(a.label)_a$(alpha_label)")
        end
        push!(specs, (; name, dispersion = d.type, aggregation = a.type, alpha))
    end
    return specs
end


function compute_horc_for_hypergraphs(
    hypergraphs::Dict{Int, Vector{Vector{String}}},
    row_map::Dict{Int, Vector{Int}},
    row_count::Int,
    specs,
)
    results = Dict{Symbol, Vector{Union{Missing, Float64}}}()
    for spec in specs
        values = Vector{Union{Missing, Float64}}(undef, row_count)
        fill!(values, missing)
        results[spec.name] = values
    end
    for idx in sort(collect(keys(hypergraphs)))
        edges = hypergraphs[idx]
        incidence, _ = generate_incidence_matrix(edges)
        row_indices = get(row_map, idx, Int[])
        isempty(row_indices) && continue
        for spec in specs
            horc_full = try
                hypergraph_curvatures(
                    spec.dispersion,
                    spec.aggregation,
                    incidence,
                    spec.alpha,
                    Orchid.CostOndemand,
                )
            catch err
                @warn "Failed HORC computation" hypergraph=idx spec=spec.name alpha=spec.alpha dispersion=spec.dispersion aggregation=spec.aggregation error=err
                continue
            end
            edge_curvature = extract_edge_curvature(horc_full)
            edge_curvature === nothing && continue
            if length(row_indices) != length(edge_curvature)
                println(
                    "Warning: edge curvature count mismatch for hypergraph $(idx) (spec=$(spec.name))",
                )
                continue
            end
            values = results[spec.name]
            for (row_pos, value) in zip(row_indices, edge_curvature)
                if value === nothing || value isa Missing
                    values[row_pos] = missing
                else
                    values[row_pos] = Float64(value)
                end
            end
        end
    end
    return results
end


function resolve_paths(args::Vector{String})
    data_dir = normpath(joinpath(@__DIR__, "derived_data"))
    default_input = joinpath(data_dir, "nu_hsbm_dataset.csv")
    default_output = joinpath(data_dir, "nu_hsbm_curv.csv")
    input_path = length(args) >= 1 ? args[1] : default_input
    output_path = length(args) >= 2 ? args[2] : default_output
    return input_path, output_path
end


function main()
    input_path, output_path = resolve_paths(ARGS)
    println("Loading hypergraphs from $(input_path)")
    rows = collect(CSV.File(input_path))
    hypergraphs, row_map = load_hypergraph_edges(rows)
    println("Found $(length(hypergraphs)) hypergraphs")
    specs = build_horc_specs()
    spec_names = [spec.name for spec in specs]
    spec_name_set = Set(spec_names)
    horc_values = compute_horc_for_hypergraphs(hypergraphs, row_map, length(rows), specs)

    output_rows = Vector{NamedTuple}(undef, length(rows))
    for (idx, row) in enumerate(rows)
        row_nt = NamedTuple(row)
        pairs_list = collect(pairs(row_nt))
        if !isempty(spec_name_set)
            pairs_list = [p for p in pairs_list if !(first(p) in spec_name_set)]
        end
        insert_pos = findfirst(p -> first(p) == :hlrc, pairs_list)
        insert_at = insert_pos === nothing ? length(pairs_list) + 1 : insert_pos + 1
        for spec in specs
            value = horc_values[spec.name][idx]
            insert!(pairs_list, insert_at, spec.name => value)
            insert_at += 1
        end
        output_rows[idx] = (; pairs_list...)
    end

    mkpath(dirname(output_path))
    CSV.write(output_path, output_rows)
    println("Saved HORC results to $(output_path)")
end


main()

# Custom input and output:
# julia code/hsbm/compute_horc.jl derived_data/u_hsbm_toy_dataset.csv derived_data/u_hsbm_toy_curv.csv