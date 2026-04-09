#!/bin/bash

SCORE=$1
INPUT_ARG=$2

# 3. Optional Parameters
echo ""
read -p "Number of epochs before local optimization (default 100): " PATIENCE_LOCALE 
PATIENCE_LOCALE=${PATIENCE_LOCALE:-100}

read -p "Number of epochs before global optimization (default 5): " PATIENCE_GLOBALE
PATIENCE_GLOBALE=${PATIENCE_GLOBALE:-5}

read -p "Minimum delta for local optimization (default 1e-4): " MIN_DELTA
MIN_DELTA=${MIN_DELTA:-1e-4}

read -p "Rate of temperature decrease (default 0.85): " TAUX_REFROIDISSEMENT
TAUX_REFROIDISSEMENT=${TAUX_REFROIDISSEMENT:-0.9}

read -p "Minimum temperature (default 0.01): " BRUIT_MIN
BRUIT_MIN=${BRUIT_MIN:-0.01}

read -p "Noise on coordinates (default 3.0): " NOISE_COORDS
NOISE_COORDS=${NOISE_COORDS:-5.0}

read -p "Enter spring constant (default 45.28): " K
K=${K:-10.0}

read -p "Enter equilibrium length (default 5.726): " L0
L0=${L0:-5.726}

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
echo "python main_bead_springs.py $INPUT_ARG --score $SCORE --patience-locale $PATIENCE_LOCALE --patience-globale $PATIENCE_GLOBALE --min-delta $MIN_DELTA --taux-refroidissement $TAUX_REFROIDISSEMENT --bruit-min $BRUIT_MIN --noise-coords $NOISE_COORDS --k $K --l0 $L0 --bead-atom $BEAD_ATOM $VERBOSE $OUTPUT_ARG"
echo ""

if command -v conda &> /dev/null && [ -n "$CONDA_DEFAULT_ENV" ]; then
    python main_bead_springs.py $INPUT_ARG --score $SCORE --patience-locale $PATIENCE_LOCALE --patience-globale $PATIENCE_GLOBALE --min-delta $MIN_DELTA --taux-refroidissement $TAUX_REFROIDISSEMENT --bruit-min $BRUIT_MIN --noise-coords $NOISE_COORDS --k $K --l0 $L0 --bead-atom $BEAD_ATOM $VERBOSE $OUTPUT_ARG
else
    python3 main_bead_springs.py $INPUT_ARG --score $SCORE --patience-locale $PATIENCE_LOCALE --patience-globale $PATIENCE_GLOBALE --min-delta $MIN_DELTA --taux-refroidissement $TAUX_REFROIDISSEMENT --bruit-min $BRUIT_MIN --noise-coords $NOISE_COORDS --k $K --l0 $L0 --bead-atom $BEAD_ATOM $VERBOSE $OUTPUT_ARG
fi