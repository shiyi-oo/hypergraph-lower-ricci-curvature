"""Shared Python-to-Julia interface for ORCHID HORC computation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHID_PROJECT = PROJECT_ROOT / "orchid"

DISPERSIONS = ("uw_clique", "w_clique", "uw_star")
AGGREGATIONS = ("mean", "max")
ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

__all__ = [
    "AGGREGATIONS",
    "ALPHAS",
    "DISPERSIONS",
    "HORC_COLUMNS",
    "compute_horc",
    "horc_column_name",
]


def horc_column_name(dispersion: str, aggregation: str, alpha: float) -> str:
    """Return the legacy CSV column name for one HORC configuration."""
    if dispersion == "uw_clique" and aggregation == "mean" and alpha == 0.0:
        return "horc"
    alpha_label = f"{alpha:.1f}".replace(".", "p")
    return f"horc_{dispersion}_{aggregation}_a{alpha_label}"


HORC_COLUMNS = tuple(
    horc_column_name(dispersion, aggregation, alpha)
    for dispersion in DISPERSIONS
    for aggregation in AGGREGATIONS
    for alpha in ALPHAS
)


# This worker has no community-count filter: every hg_idx is computed. Mean
# and max aggregations share each optimal-transport run.
JULIA_WORKER = r'''
using CSV
using DataFrames
using JSON
using Orchid
using Printf
using SparseArrays

const DISPERSIONS = [
    (label="uw_clique", type=Orchid.DisperseUnweightedClique),
    (label="w_clique", type=Orchid.DisperseWeightedClique),
    (label="uw_star", type=Orchid.DisperseUnweightedStar),
]
const AGGREGATIONS = [
    (label="mean", type=Orchid.AggregateMean),
    (label="max", type=Orchid.AggregateMax),
]
const ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

function horc_name(dispersion, aggregation, alpha)
    if dispersion == "uw_clique" && aggregation == "mean" && iszero(alpha)
        return :horc
    end
    alpha_label = replace(@sprintf("%.1f", alpha), "." => "p")
    Symbol("horc_$(dispersion)_$(aggregation)_a$(alpha_label)")
end

function build_specs()
    specs = NamedTuple[]
    for d in DISPERSIONS, a in AGGREGATIONS, alpha in ALPHAS
        push!(
            specs,
            (
                name=horc_name(d.label, a.label, alpha),
                dispersion=d,
                aggregation=a,
                alpha=alpha,
            ),
        )
    end
    specs
end

function parse_edge_nodes(value)
    ismissing(value) && return String[]
    clean = strip(String(value))
    isempty(clean) && return String[]
    json_ready = replace(clean, '\'' => '"')
    parsed = JSON.parse(json_ready)
    String.(parsed)
end

function load_hypergraphs(data::DataFrame)
    hypergraphs = Dict{Int, Vector{Vector{String}}}()
    row_map = Dict{Int, Vector{Int}}()

    for row_idx in 1:nrow(data)
        hg_raw = data[row_idx, :hg_idx]
        members_raw = data[row_idx, :members]
        (ismissing(hg_raw) || ismissing(members_raw)) && continue

        hg_idx = hg_raw isa Integer ? Int(hg_raw) : parse(Int, String(hg_raw))
        members = parse_edge_nodes(members_raw)
        isempty(members) && continue

        push!(get!(hypergraphs, hg_idx, Vector{Vector{String}}()), members)
        push!(get!(row_map, hg_idx, Int[]), row_idx)
    end

    hypergraphs, row_map
end

function incidence_matrix(edges::Vector{Vector{String}})
    nodes = sort!(collect(Set(node for edge in edges for node in edge)))
    node_index = Dict(node => index for (index, node) in enumerate(nodes))

    rows = Int[]
    columns = Int[]
    for (edge_idx, edge) in enumerate(edges), node in edge
        push!(rows, edge_idx)
        push!(columns, node_index[node])
    end

    sparse(rows, columns, ones(Int, length(rows)), length(edges), length(nodes))
end

function empty_result(row_count::Int)
    values = Vector{Union{Missing, Float64}}(undef, row_count)
    fill!(values, missing)
    values
end

function compute_all_horc(data::DataFrame)
    hypergraphs, row_map = load_hypergraphs(data)
    specs = build_specs()
    results = Dict(spec.name => empty_result(nrow(data)) for spec in specs)
    failures = String[]
    hypergraph_ids = sort!(collect(keys(hypergraphs)))

    println("Found $(length(hypergraph_ids)) hypergraphs and $(nrow(data)) edges")
    println("Computing $(length(specs)) HORC configurations")

    for (position, hg_idx) in enumerate(hypergraph_ids)
        edges = hypergraphs[hg_idx]
        row_indices = row_map[hg_idx]
        incidence = incidence_matrix(edges)

        for dispersion in DISPERSIONS, alpha in ALPHAS
            result = try
                Orchid.hypergraph_curvatures(
                    dispersion.type,
                    [aggregation.type for aggregation in AGGREGATIONS],
                    incidence,
                    alpha,
                    Orchid.CostOndemand,
                )
            catch error
                message = sprint(showerror, error)
                push!(failures, "hg_idx=$(hg_idx), dispersion=$(dispersion.label), alpha=$(alpha): $(message)")
                continue
            end

            for (aggregation_idx, aggregation) in enumerate(AGGREGATIONS)
                name = horc_name(dispersion.label, aggregation.label, alpha)
                edge_curvature = result.aggregations[aggregation_idx].edge_curvature
                if length(edge_curvature) != length(row_indices)
                    push!(
                        failures,
                        "hg_idx=$(hg_idx), spec=$(name): expected $(length(row_indices)) values, got $(length(edge_curvature))",
                    )
                    continue
                end
                for (row_idx, value) in zip(row_indices, edge_curvature)
                    results[name][row_idx] = Float64(value)
                end
            end
        end

        if position == 1 || position % 10 == 0 || position == length(hypergraph_ids)
            println("Completed hypergraph $(position)/$(length(hypergraph_ids)) (hg_idx=$(hg_idx))")
            flush(stdout)
        end
    end

    for spec in specs
        missing_count = count(ismissing, results[spec.name])
        if missing_count > 0
            push!(failures, "spec=$(spec.name): $(missing_count) rows are missing")
        end
    end

    if !isempty(failures)
        preview = join(first(failures, min(length(failures), 20)), "\n")
        error("HORC failed for $(length(failures)) computations:\n$(preview)")
    end

    specs, results
end

function remove_existing_horc_columns!(data::DataFrame, specs)
    existing = Set(propertynames(data))
    for spec in specs
        if spec.name in existing
            select!(data, [name for name in propertynames(data) if name != spec.name])
            delete!(existing, spec.name)
        end
    end
end

function insert_horc_columns!(data::DataFrame, specs, results)
    column_names = propertynames(data)
    hlrc_position = findfirst(==(:hlrc), column_names)
    insert_position = isnothing(hlrc_position) ? length(column_names) + 1 : hlrc_position + 1

    for spec in specs
        insertcols!(data, insert_position, spec.name => results[spec.name])
        insert_position += 1
    end
end

function main()
    length(ARGS) == 2 || error("expected input and output paths")
    input_path, output_path = ARGS

    println("Loading $(input_path)")
    data = CSV.read(input_path, DataFrame)
    specs, results = compute_all_horc(data)
    remove_existing_horc_columns!(data, specs)
    insert_horc_columns!(data, specs, results)

    mkpath(dirname(output_path))
    CSV.write(output_path, data)
    println("Saved HORC results to $(output_path)")
end

main()
'''


def inspect_source(path: Path) -> tuple[int, int, int]:
    """Validate a source CSV and return rows, hypergraphs, and config count."""
    if not path.is_file():
        raise FileNotFoundError(f"source dataset does not exist: {path}")

    header = pd.read_csv(path, nrows=0)
    required_columns = {"hg_idx", "members"}
    missing_columns = required_columns - set(header.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required column(s): {missing}")

    if "config_idx" in header.columns:
        config_column = "config_idx"
    elif "config_name" in header.columns:
        config_column = "config_name"
    else:
        config_column = None

    usecols = ["hg_idx"] + ([config_column] if config_column else [])
    summary = pd.read_csv(path, usecols=usecols)
    config_count = summary[config_column].nunique() if config_column else 1
    return len(summary), summary["hg_idx"].nunique(), config_count


def validate_output(path: Path, expected_rows: int) -> None:
    """Ensure Julia produced every HORC column and preserved row count."""
    header = pd.read_csv(path, nrows=0)
    missing_columns = set(HORC_COLUMNS) - set(header.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RuntimeError(f"HORC output is missing column(s): {missing}")

    with path.open("rb") as handle:
        row_count = sum(1 for _ in handle) - 1
    if row_count != expected_rows:
        raise RuntimeError(
            f"HORC output has {row_count} rows; expected {expected_rows}"
        )


def compute_horc(
    source: Path,
    output: Path,
    *,
    julia: str,
    threads: str = "auto",
    overwrite: bool = False,
) -> Path:
    """Compute all 36 HORC configurations for a hypergraph CSV."""
    row_count, hypergraph_count, config_count = inspect_source(source)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output}\nUse overwrite=True to replace it."
        )
    if not ORCHID_PROJECT.is_dir():
        raise FileNotFoundError(f"Orchid project does not exist: {ORCHID_PROJECT}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.unlink(missing_ok=True)

    print(
        f"{source.name}: {row_count} edges, {hypergraph_count} hypergraphs, "
        f"{config_count} configurations"
    )
    command = [
        julia,
        f"--project={ORCHID_PROJECT}",
        f"--threads={threads}",
        "-",
        str(source.resolve()),
        str(temporary_output.resolve()),
    ]

    try:
        subprocess.run(command, input=JULIA_WORKER, text=True, check=True)
        validate_output(temporary_output, row_count)
        os.replace(temporary_output, output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    return output
