#!/bin/bash

# Define parameters
START_ID=1
END_ID=355
FILES_PER_JOB=100
JULIA_SCRIPT="/work/users/s/h/shiyi/hypergraph_with_curvature/code/Stex/2.2.compute_HORC.jl"
PROJECT_DIR="/work/users/s/h/shiyi/hypergraph_with_curvature/orchid"

# Loop through ranges
for ((i=START_ID; i<=END_ID; i+=FILES_PER_JOB)); do
    current_start=$i
    current_end=$((i + FILES_PER_JOB - 1))
    
    # Ensure we don't exceed END_ID
    if (( current_end > END_ID )); then
        current_end=$END_ID
    fi

    # Submit sbatch job
    sbatch -n 1 --cpus-per-task=4 --mem=60g -t 01:45:00 \
        --wrap="julia --threads 4 --project=$PROJECT_DIR $JULIA_SCRIPT $current_start $current_end"
    
    echo "Submitted job for IDs: $current_start to $current_end"
done