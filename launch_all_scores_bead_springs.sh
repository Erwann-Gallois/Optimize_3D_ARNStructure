#!/bin/bash

INPUT_ARG=$1
CIF_ARG=$2

# 2. Optional Parameters (Shared across all runs)
echo ""
read -p "Number of epochs before local optimization (default 100): " PATIENCE_LOCALE 
PATIENCE_LOCALE=${PATIENCE_LOCALE:-100}

read -p "Number of epochs before global optimization (default 5): " PATIENCE_GLOBALE
PATIENCE_GLOBALE=${PATIENCE_GLOBALE:-5}

read -p "Minimum delta for local optimization (default 1e-4): " MIN_DELTA
MIN_DELTA=${MIN_DELTA:-1e-4}

read -p "Rate of temperature decrease (default 0.9): " TAUX_REFROIDISSEMENT
TAUX_REFROIDISSEMENT=${TAUX_REFROIDISSEMENT:-0.9}

read -p "Minimum temperature (default 0.01): " BRUIT_MIN
BRUIT_MIN=${BRUIT_MIN:-0.01}

read -p "Noise on coordinates (default 5.0): " NOISE_COORDS
NOISE_COORDS=${NOISE_COORDS:-5.0}

read -p "Enter spring constant (default 10.0): " K
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

# 3. Execution Loop
SCORES=("dfire" "rasp" "rsRNASP")

for SCORE in "${SCORES[@]}"
do
    echo ""
    echo "=========================================================="
    echo ">>> Running Optimization with SCORE: $SCORE"
    echo "=========================================================="
    
    # Check for conda environment or use python3
    if command -v conda &> /dev/null && [ -n "$CONDA_DEFAULT_ENV" ]; then
        PYTHON_CMD="python"
    else
        PYTHON_CMD="python3"
    fi

    $PYTHON_CMD main_bead_springs.py $INPUT_ARG \
        --score $SCORE \
        --patience-locale $PATIENCE_LOCALE \
        --patience-globale $PATIENCE_GLOBALE \
        --min-delta $MIN_DELTA \
        --taux-refroidissement $TAUX_REFROIDISSEMENT \
        --bruit-min $BRUIT_MIN \
        --noise-coords $NOISE_COORDS \
        --k $K \
        --l0 $L0 \
        --bead-atom "$BEAD_ATOM" \
        $VERBOSE $CIF_ARG
done

echo ""
echo "All score optimizations completed."
