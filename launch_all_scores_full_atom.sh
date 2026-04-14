#!/bin/bash

INPUT_ARG=$1
CIF_ARG=$2

# 3. Optional Parameters
echo ""
read -p "Local patience (iterations without improvement, default 100): " PATIENCE_LOCALE
PATIENCE_LOCALE=${PATIENCE_LOCALE:-100}

read -p "Min delta (min energy gain, default 1e-4): " MIN_DELTA
MIN_DELTA=${MIN_DELTA:-1e-4}

read -p "Global patience (shakes without record, default 5): " PATIENCE_GLOBALE
PATIENCE_GLOBALE=${PATIENCE_GLOBALE:-5}

read -p "Cooling rate (default 0.85): " TAUX_REFROIDISSEMENT
TAUX_REFROIDISSEMENT=${TAUX_REFROIDISSEMENT:-0.85}

read -p "Minimum noise (default 0.01): " BRUIT_MIN
BRUIT_MIN=${BRUIT_MIN:-0.01}

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

SCORES=("rasp" "dfire" "rsRNASP")

for SCORE in "${SCORES[@]}"; do
    CMD_ARGS="$INPUT_ARG --score $SCORE --patience-locale $PATIENCE_LOCALE --min-delta $MIN_DELTA --patience-globale $PATIENCE_GLOBALE --taux-refroidissement $TAUX_REFROIDISSEMENT --bruit-min $BRUIT_MIN --backbone-weight $BACKBONE_WEIGHT --noise-coords $NOISE_COORDINATE --noise-angles $NOISE_ANGLES $VERBOSE $CIF_ARG $OUTPUT_ARG --score_weight $SCORE_WEIGHT"
    echo ""
    echo ">>> Running command:"
    echo "python main_full_atom.py $CMD_ARGS"
    echo ""

    if command -v conda &> /dev/null && [ -n "$CONDA_DEFAULT_ENV" ]; then
        python main_full_atom.py $CMD_ARGS
    else
        python3 main_full_atom.py $CMD_ARGS
    fi
done