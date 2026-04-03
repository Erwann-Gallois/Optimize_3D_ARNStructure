#!/bin/bash

# Interactive script to launch RNA optimizations (Full atom only)

echo "===================================================="
echo "   3D RNA Optimization Launch Interface"
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

echo ""
echo "Choose optimization method:"
echo "1) Bead-springs"
echo "2) Full atoms"
read -p "Your choice (1/2): " method_choice

case $method_choice in
    1) METHOD="bead-springs" ;;
    2) METHOD="full-atoms" ;;
    *) echo "Invalid choice. Using bead-springs by default." ; METHOD="bead-springs" ;;
esac

# 2. Scoring function selection
echo ""
echo "Choose scoring function:"
echo "1) RASP"
echo "2) DFIRE"
echo "3) rsRNASP"
read -p "Your choice (1/2/3): " score_choice

case $score_choice in
    1) SCORE="rasp" ;;
    2) SCORE="dfire" ;;
    3) SCORE="rsRNASP" ;;
    *) echo "Invalid choice. Using rasp by default." ; SCORE="rasp" ;;
esac

if [ "$METHOD" == "bead-springs" ]; then
    ./launch_bead_springs.sh "$SCORE" "$INPUT_ARG"
elif [ "$METHOD" == "full-atoms" ]; then
    ./launch_full_atom.sh "$SCORE" "$INPUT_ARG"
fi