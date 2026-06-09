# Elkopy

A Python package for computing electronic coupling matrix elements in Electronic Excitation Energy Transfer (EET) processes, starting from the ground and excited states of the monomers involved.

---

## Theoretical Background

The electronic coupling for EET between a donor (D) and an acceptor (A) chromophore is decomposed following the TDFI-TI framework of Fujimoto (2012) and the through-configuration theory of Scholes et al. (1994, 1995):

$$V_{IF} = V_{\text{Coul}} + V_{\text{Exch}} + V_{\text{Ovlp}} + V_{\text{indirect}}$$

### Coulomb term ($V_{\text{Coul}}$)

The Coulomb coupling is computed via the Transition Density Fragment Interaction (TDFI) approach (Fujimoto 2012, Eq. 11):

$$V_{\text{Coul}} = \sum_{\nu\mu \in D} \sum_{\lambda\sigma \in A} P^D_{\nu\mu} P^A_{\lambda\sigma} (\mu\nu|\sigma\lambda)$$

where $P^X_{\nu\mu}$ is the CIS transition density in the atomic orbital representation. 
This is the dominant term for singlet EET at intermediate and long range. At large distances the dipole-dipole interaction is the dominant contribution to the Coulomb term, so it decays as $R^{-3}$.
For triplet-triplet EET this term is zero by spin selection rules.

### Exchange term ($V_{\text{Exch}}$)

The exchange coupling (Fujimoto 2012, Eq. 12) is:

$$V_{\text{Exch}} = -\frac{1}{2} \sum_{\nu\mu \in D} \sum_{\lambda\sigma \in A} P^D_{\nu\mu} P^A_{\lambda\sigma} (\mu\lambda|\sigma\nu)$$

This term represents the simultaneous exchange of two electrons with different energy. According to Dexter, it can be seen as a Coulomb interaction between two orbital overlaps, and it decays exponentially.
As shown by both Scholes et al. and Fujimoto, this term is small compared to the through-configuration interaction at short range (indirect term).

### Overlap/Penetration term ($V_{\text{Pterm}}$)

The penetration (or orbital overlap-dependent direct) term arises from the correction to $T_{14}$ at second order in interchromophore orbital overlap. Following Scholes et al. (1995, Eq. 15b) and applying the Mulliken approximation:

$$V_{\text{Pterm}} = -(S_{a'b'}\beta_{ab} + S_{ab}\beta_{a'b'} - S_{ab}S_{a'b'}(\ldots))$$

where the bond integrals are $\beta_{ab} = h_{ab} - S_{ab}h_{aa}$ and $\beta_{a'b'} = h_{a'b'} - S_{a'b'}h_{a'a'}$, with $h_{pq}$ being the partially screened core Hamiltonian matrix elements. 
For heterodimers, the formula is extended considering the asymmetry of the system.

### Indirect (through-configuration) term ($V_{\text{indirect}}$)

The dominant short-range orbital overlap-dependent term arises from through-configuration interaction via bridging ionic (charge-transfer) configurations $|D^+A^-\rangle$ and $|D^-A^+\rangle$. Following Scholes et al. (1994, Eq. 11) and Fujimoto (2012, Eq. 49):

$$V_{\text{indirect}} = -\frac{2T_{12}T_{13}}{A}$$

which in the Mulliken approximation reduces to (Scholes et al. 1995, Eq. 9a):

$$V_{\text{indirect}} \approx \frac{2\beta_{ab}\beta_{a'b'}}{^1A}$$

where $A$ is the energy gap between the locally excited and ionic configurations:

$$A = (aa|a'a') - (aa|b'b') \mp 2J_0$$

with the upper/lower sign for singlet/triplet states respectively and $J_0 = (a'a|aa')$ the intra-monomer exchange integral. This term is consistently larger than the Dexter exchange interaction, as confirmed numerically by both Scholes et al. and Fujimoto for the ethylene dimer.

### Dipole-dipole approximation ($V_{\text{DipDip}}$)

The point dipole-dipole approximation (Förster limit) is also computed for reference:

$$V_{\text{DipDip}} = \frac{\mu_D \cdot \mu_A - 3(\mu_D \cdot \hat{R})(\mu_A \cdot \hat{R})}{R^3}$$

where $\mu_D$ and $\mu_A$ are the transition dipole moments and $R$ is the interchromophore distance.

---

## Installation

```bash
git clone https://github.com/FLeccese/elkopy.git
cd elkopy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install .
deactivate
```

### Requirements

- Python >= 3.8
- [PySCF](https://pyscf.org/)
- NumPy
- Pandas
- Matplotlib
- psutil

---

## Usage

### Command-line interface

```bash
elkopy donor.xyz [acceptor.xyz] [options]
```

If only one `.xyz` file is provided, a homodimer is assumed.

#### Main options

| Option | Default | Description |
|--------|---------|-------------|
| `-b`, `--basis` | `3-21g` | Basis set |
| `--xc` | `None` (HF/TDA) | DFT functional (e.g. `b3lyp`) |
| `-s`, `--state` | `1` | Excited state index (one value for both, or two: DONOR ACCEPTOR) |
| `--spin` | `singlet` | Spin multiplicity (`singlet` or `triplet`) |
| `--axis` | `z` | Scan axis (`x`, `y`, or `z`) |
| `--range` | `3.0 13.0 0.5` | Scan range: START STOP STEP (Å) |
| `--offset` | `0.0 0.0 0.0` | Base offset vector (Å) |
| `--fullqm` | off | Compute full QM reference via energy splitting (homodimers only) |
| `-o`, `--output` | `coupling_scan_results.csv` | Output CSV filename |
| `--plot` | off | Generate plot after scan |

#### Examples

Homodimer scan of ethylene along z-axis with 6-31G* basis with default plot generated after the scan:

```bash
elkopy ethylene.xyz -b 6-31g* --range 3.0 8.0 0.5 --plot
```

Heterodimer, second excited state of donor and first of acceptor:

```bash
elkopy donor.xyz acceptor.xyz -s 2 1 -b 6-31g* --axis z
```

Triplet EET, scan along x, dimer separated by 4Å:

```bash
elkopy donor.xyz --spin triplet --range 3.0 10.0 0.5 --offset 0 0 4
```

With DFT:

```bash
elkopy donor.xyz --xc cam-b3lyp -b 6-31g*
```

### Plotting results

```bash
elkoplot coupling_scan_results.csv --logy --cm
```

| Option | Description |
|--------|-------------|
| `--logy` | Logarithmic y-axis |
| `--logx` | Logarithmic x-axis |
| `--cm` | Convert coupling from eV to cm⁻¹ |
| `-c COLS` | Select specific columns to plot |
| `-o FILE` | Output PNG filename |

---

## Output

The scan produces a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `Distance` | Interchromophore distance (Å) |
| `J_Coul` | Coulomb coupling (eV) |
| `J_Exch` | Exchange coupling (eV) |
| `J_Pterm` | Penetration/overlap term (eV) |
| `J_indirect` | Through-configuration indirect coupling (eV) |
| `J_DipDip` | Dipole-dipole approximation (eV) |
| `Full_QM` | Energy splitting reference, homodimers only (eV) |
| `J_Total` | Total coupling: Coul + Exch + Pterm + indirect (eV) |

---

## Package Structure

```
elkopy/
├── compchem/
│   └── monomer.py          # Monomer class: SCF, TDA, transition densities
├── physics/
│   ├── el_coupling.py      # ElectronicCoupling class: all coupling terms
│   └── approximations.py   # Dipole-dipole and full QM reference
├── engine/
│   └── scanner.py          # Distance scan engine (generator pattern)
├── analysis/
│   └── plot.py             # CSV plotting utilities
└── utils/
    ├── io.py               # I/O helpers and formatting
    └── system.py           # Performance monitoring

scripts/
├── run_elkopy.py           # CLI entry point for scans
└── run_plot.py             # CLI entry point for plotting
```

---

## Physical Notes

The transition density matrices are computed at the TDA (Tamm-Dancoff Approximation) level. PySCF normalizes the TDA amplitudes such that $\sum_{ia} X_{ia}^2 = 1/2$, so the transition density is constructed with an explicit $\sqrt{2}$ prefactor to recover the standard CIS normalization $\sum_{ia} t_{ia}^2 = 1$.

The energy gap $A$ for the indirect term is estimated from intra- and inter-monomer two-electron integrals following the Koopmans-based approximation of Scholes et al. (1995, Eq. 8) and Fujimoto (2012, Eqs. 46-47), without requiring an explicit calculation of the ionic state energies.

Phase consistency between the transition densities of donor and acceptor is enforced by aligning the transition dipole moments before the scan.

---

## References

1. Z. You, C. Hsu, *Int. J. Quantum Chem.* **114**, 102-115 (2014)
2. K. J. Fujimoto, *J. Chem. Phys.* **137**, 034101 (2012).
3. R. D. Harcourt, G. D. Scholes, K. P. Ghiggino, *J. Chem. Phys.* **101**, 10521 (1994).
4. G. D. Scholes, R. D. Harcourt, K. P. Ghiggino, *J. Chem. Phys.* **102**, 9574 (1995).
5. T. Förster, *Ann. Phys.* **437**, 55 (1948).
6. D. L. Dexter, *J. Chem. Phys.* **21**, 836 (1953).

---

## License

MIT
