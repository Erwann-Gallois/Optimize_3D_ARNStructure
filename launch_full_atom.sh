#!/bin/bash

SCORE=$1
INPUT_ARG=$2

# 3. Optional Parameters
echo ""
read -p "Number of cycles (default 20): " CYCLES
CYCLES=${CYCLES:-20}

read -p "Epochs per cycle (default 50): " EPOCHS
EPOCHS=${EPOCHS:-50}



read -p "Enter backbone weight (default 100): " BACKBONE_WEIGHT
BACKBONE_WEIGHT=${BACKBONE_WEIGHT:-100}

read -p "Enter noise coordinate (default 1.5): " NOISE_COORDINATE
NOISE_COORDINATE=${NOISE_COORDINATE:-1.5}

read -p "Enter noise angles (default 0.2): " NOISE_ANGLES
NOISE_ANGLES=${NOISE_ANGLES:-0.2}

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
echo "python main_full_atom.py $INPUT_ARG --score $SCORE --cycles $CYCLES --epochs $EPOCHS --backbone-weight $BACKBONE_WEIGHT --noise-coords $NOISE_COORDINATE --noise-angles $NOISE_ANGLES $VERBOSE $OUTPUT_ARG"
echo ""

if command -v conda &> /dev/null && [ -n "$CONDA_DEFAULT_ENV" ]; then
    python main_full_atom.py $INPUT_ARG --score $SCORE --cycles $CYCLES --epochs $EPOCHS --backbone-weight $BACKBONE_WEIGHT --noise-coords $NOISE_COORDINATE --noise-angles $NOISE_ANGLES $VERBOSE $OUTPUT_ARG
else
    python3 main_full_atom.py $INPUT_ARG --score $SCORE --cycles $CYCLES --epochs $EPOCHS --backbone-weight $BACKBONE_WEIGHT --noise-coords $NOISE_COORDINATE --noise-angles $NOISE_ANGLES $VERBOSE $OUTPUT_ARG
fi