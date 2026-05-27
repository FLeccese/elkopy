#!/bin/bash
#Stop the script if any command fails
set -e

echo -e "\n================================================================================"
echo " 1. CALCULATION OF COUPLING + DEFAULT PLOT (ETHYLENE)"
echo "================================================================================"
# Calculate the coupling for the ethylene homodimer along the z-axis and generate a default plot
elkopy ethylene.xyz -s 1 --basis 6-31g* --axis z --range 3.0 6.0 0.2 --fullqm --plot -o scan_ethylene.csv

echo -e "\n================================================================================"
echo " 2. ADVANCED POST-PROCESSING WITH ELKOPLOT (Cm^-1 PLOT)"
echo "================================================================================"
# Take the generated CSV and create a log-scale plot using elkoplot
elkoplot scan_ethylene.csv --cm -o plot_ethylene_cm-1.png

echo -e "\n================================================================================"
echo " [OK] Ethylene example completed successfully!"
echo "================================================================================"