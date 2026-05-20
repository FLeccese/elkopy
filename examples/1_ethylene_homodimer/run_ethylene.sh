#!/bin/bash
#Stop the script if any command fails
set -e

echo -e "\n================================================================================"
echo " 1. CALCULATION OF COUPLING + DEFAULT PLOT (ETHYLENE)"
echo "================================================================================"
# Calculate the coupling for the ethylene homodimer along the z-axis and generate a default plot
elkopy ethylene.xyz -s 1 --basis 3-21g --axis z --range 3.5 13.0 0.5 --plot -o scan_ethylene.csv

echo -e "\n================================================================================"
echo " 2. ADVANCED POST-PROCESSING WITH ELKOPLOT (LOG SCALE)"
echo "================================================================================"
# Take the generated CSV and create a log-scale plot using elkoplot
elkoplot scan_ethylene.csv --logy -o plot_ethylene_log.png

echo -e "\n================================================================================"
echo " [OK] Ethylene example completed successfully!"
echo "================================================================================"