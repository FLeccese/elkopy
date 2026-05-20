#!/bin/bash
#Stop the script if any command fails
set -e

echo -e "\n================================================================================"
echo " 1. CALCULATION WITH NAPHTHALENE HETERODIMER + DEFAULT PLOT"
echo "================================================================================"
# Calculate the coupling for the naphthalene heterodimer along the z-axis and generate a default plot
elkopy Naphthalene_GSgeom.xyz Naphthalene_T1geom.xyz -s 1 --basis 3-21g --axis z --range 4.0 8.0 0.5 --plot -o scan_naphthalene.csv

echo -e "\n================================================================================"
echo " 2. ADVANCED POST-PROCESSING WITH ELKOPLOT"
echo "================================================================================"
# Clean the plot leaving only the Coulomb and Dipole-Dipole coupling for comparison
elkoplot scan_naphthalene.csv -c J_Coul J_DipDip -o comparison_coul_coupling.png

echo -e "\n================================================================================"
echo " [OK] Naphthalene example completed successfully!"
echo "================================================================================"