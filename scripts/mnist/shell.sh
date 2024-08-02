#!/bin/bash

# Define the input file
input_file="input_args.txt"

# Define the output file
output_file="output_results.txt"

# Run test.py for each set of arguments in the input file
while read -r method n_compression ; do
    python3 eth_mnist_merge.py --method "$method" --n_compression "$n_compression" >> "$output_file"
done < "$input_file"
