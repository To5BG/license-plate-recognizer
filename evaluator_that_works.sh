#!/bin/bash

T=$(find . -type f -name "*test*" \( -not -path "*git*" \) -and \( -not -path "*venv*" \) -and \( -not -path "*cache*" \) -and \( -not -path "*test_cross_validation*" \))

python3 main.py --file_path $T --output_path ./Output.csv

F=$(find . -type f -name "Output*")

G=$(find . -type f -name "*TruthTest*")

python3 evaluation.py --file_path $F  --ground_truth_path $G
