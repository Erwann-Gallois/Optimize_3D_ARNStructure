#!/bin/bash

# Interactive script to launch RNA optimizations (Full atom only)

echo "===================================================="
echo "   3D RNA Optimization Launch Interface (Full Atom)"
echo "===================================================="

# 1. Input type selection (Sequence or File)
echo "How would you like to provide the sequence?"
echo "1) Direct sequence (ex: GCAUAGC...)"
echo "2) FASTA file (ex: example.fasta)"
read -p "Your choice (1/2): " input_type

if [ "$input_type" == "1" ]; then
    read -p "Enter RNA sequence: " seq_val
    INPUT_ARG="-s $seq_val"
elif [ "$input_type" == "2" ]; then
    read -p "Enter path to FASTA file: " fasta_path
    INPUT_ARG="-f $fasta_path"
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# 2. Scoring function selection
echo ""
echo "Choose scoring function:"
echo "1) RASP (Full atoms)"
echo "2) DFIRE (Full atoms)"
read -p "Your choice (1/2): " score_choice

case $score_choice in
    1) SCORE="rasp" ;;
    2) SCORE="dfire" ;;
    *) echo "Invalid choice. Using dfire by default." ; SCORE="dfire" ;;
esac

# 3. Optional Parameters
echo ""
read -p "Number of cycles (default 20): " CYCLES
CYCLES=${CYCLES:-20}

read -p "Epochs per cycle (default 50): " EPOCHS
EPOCHS=${EPOCHS:-50}

read -p "Enable verbose mode? (y/n): " VERBOSE_YN
if [ "$VERBOSE_YN" == "y" ]; then
    VERBOSE="-v"
else
    VERBOSE=""
fi

# 4. Output Configuration
echo ""
read -p "Enter output filename/path (press Enter for default): " OUTPUT_NAME
if [ -n "$OUTPUT_NAME" ]; then
    OUTPUT_ARG="-o $OUTPUT_NAME"
else
    OUTPUT_ARG=""
fi

# 5. Execution
echo ""
echo ">>> Running command:"
echo "python cli.py $INPUT_ARG --score $SCORE --cycles $CYCLES --epochs $EPOCHS $VERBOSE $OUTPUT_ARG"
echo ""

# Use conda environment if detected, otherwise just python
if command -v conda &> /dev/null && [ -n "$CONDA_DEFAULT_ENV" ]; then
    python cli.py $INPUT_ARG --score $SCORE --cycles $CYCLES --epochs $EPOCHS $VERBOSE $OUTPUT_ARG
else
    python3 cli.py $INPUT_ARG --score $SCORE --cycles $CYCLES --epochs $EPOCHS $VERBOSE $OUTPUT_ARG
fi
