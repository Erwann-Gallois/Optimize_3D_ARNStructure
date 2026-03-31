#!/bin/bash

SCORE=$1
INPUT_ARG=$2

# 3. Optional Parameters
echo ""
read -p "Number of cycles (default 20): " CYCLES
CYCLES=${CYCLES:-20}

read -p "Epochs per cycle (default 50): " EPOCHS
EPOCHS=${EPOCHS:-50}

read -p "Enter spring constant (default 40.0): " K
K=${K:-40.0}

read -p "Enter equilibrium length (default 5.5): " L0
L0=${L0:-5.5}

read -p "Enter bead atom (default C3'): " BEAD_ATOM
BEAD_ATOM=${BEAD_ATOM:-"C3'"}

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
echo "python main_bead_springs.py $INPUT_ARG --score $SCORE --cycles $CYCLES --epochs $EPOCHS --k $K --l0 $L0 --bead-atom $BEAD_ATOM $VERBOSE $OUTPUT_ARG"
echo ""

if command -v conda &> /dev/null && [ -n "$CONDA_DEFAULT_ENV" ]; then
    python main_bead_springs.py $INPUT_ARG --score $SCORE --cycles $CYCLES --epochs $EPOCHS --k $K --l0 $L0 --bead-atom $BEAD_ATOM $VERBOSE $OUTPUT_ARG
else
    python3 main_bead_springs.py $INPUT_ARG --score $SCORE --cycles $CYCLES --epochs $EPOCHS --k $K --l0 $L0 --bead-atom $BEAD_ATOM $VERBOSE $OUTPUT_ARG
fi