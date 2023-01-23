#!/bin/bash
#this is only for running inside of VSCode and PyCharm
T=$(find . -type f -name "*test*" \( -not -path "*git*" \) -and \( -not -path "*venv*" \))

python3 main.py --file_path $T --output_path ./Output.csv

F=$(find . -type f -name "Output*")

G=$(find . -type f -name "*TruthTest*")

python3 evaluation.py --file_path $F  --ground_truth_path $G
