#!/bin/bash
#Stop the script if any command fails
set -e

echo "========================================================"
echo " 1. ELKOPY INSTALLATION"
echo "========================================================"
# Moves to the parent directory and installs the latest version of elkopy in editable mode
pip install -e ../..

echo -e "\n========================================================"
echo " 2. CALCULATION WITH LATERAL DISPLACEMENT + DEFAULT PLOT (HEXATRIENE)"
echo "========================================================"
# Calculate the coupling for the hexatriene homodimer along the x-axis and generate a default plot
elkopy hexatriene.xyz -s 1 --basis 3-21g --axis x --range 0.0 8.0 0.5 --offset 0.0 0.0 3.5 --plot -o scan_hexatriene.csv

echo -e "\n========================================================"
echo " 3. ADVANCED POST-PROCESSING WITH ELKOPLOT"
echo "========================================================"
# Take the generated CSV and create a plot with different output name using elkoplot
elkoplot scan_hexatriene.csv -o plot_hexatriene.png

echo -e "\n========================================================"
echo " [OK] Hexatriene example completed successfully!"
echo "========================================================"