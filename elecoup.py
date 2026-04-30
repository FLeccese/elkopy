import os
import time
import argparse
import numpy as np
from monomer import Monomer
from el_coupling import ElectronicCoupling
import utils


def main():
     # 1.Argparse configuration for input
    parser = argparse.ArgumentParser(description="Electronic Coupling calculation.")
    
    # mandatory argument
    parser.add_argument("xyz_file", help="File .xyz with monomer geometry ")
    
    # optional arguments
    parser.add_argument("-s", "--state", type=int, default=0, help="Excited state index (default: 0)")
    parser.add_argument("--spin", choices=['singlet', 'triplet'], default='singlet', help="Spin multiplicity (default: singlet)")
    parser.add_argument("-b", "--basis", type=str, default='sto-6g', help="Basis set (default: sto-6g)")
    
    # scan parameter
    parser.add_argument("--axis", choices=['x', 'y', 'z'], default='z', help="Scanning axis (default: z)")
    parser.add_argument("--range", type=float, nargs=3, metavar=('START', 'STOP', 'STEP'), default=[0.0, 10.0, 0.5], help="Start stop and step (default: 0.0 10.0 0.5)")
    parser.add_argument("--offset", type=float, nargs=3, metavar=('X', 'Y', 'Z'), default=[0.0, 0.0, 0.0], help="Initial offset position for acceptor (default: 0.0 0.0 0.0)")

    args = parser.parse_args()

    # 2. Initialization
    start_time = time.time()
    mem_start = utils.get_memory_usage()
    
    mol_coord = utils.read_xyz(args.xyz_file)
    is_singlet = (args.spin == 'singlet')

    print(f"--- Electronic Coupling Scan ---")
    print(f"File: {args.xyz_file} | Basis: {args.basis} | State: {args.state} ({args.spin})")
    print(f"Scan Axis: {args.axis.upper()} | Range: {args.range[0]} to {args.range[1]} (step {args.range[2]}) | Base Offset: {args.offset}\n")

    # 3. Setup and calculations
    m1 = Monomer(mol_coord, basis=args.basis)
    m2 = Monomer(mol_coord, basis=args.basis)
    
    print("Start HF/CIS calculations on monomers...\n")
    m1.run_calculations(singlet=is_singlet)
    m2.run_calculations(singlet=is_singlet)
    
    rho1 = m1.get_trans_density(args.state)
    rho2 = m2.get_trans_density(args.state)

    print("\nStart scan...\n")
    print(f"{'Dist('+args.axis+')':>8} | {'J_Coul':>10} | {'J_Exch':>10} | {'J_Pterm':>11} | {'J_DipDip':>10} | {'J_Total':>10}")
    print("-" * 63)

    start, stop, step = args.range
    for dist in np.arange(start, stop + (step/10), step): # +(step/10) is just to allow the calculation at distance=stop
        
        
        # Creation translation vector
        trans_vector = np.array(args.offset)
        if args.axis == 'x':
            trans_vector[0] += dist
        elif args.axis == 'y':
            trans_vector[1] += dist
        elif args.axis == 'z':
            trans_vector[2] += dist

        m2.set_position(trans_vector)
        
        # Calcolo coupling
        coup = ElectronicCoupling(m1, m2, rho1, rho2)
        jc = coup.get_J(singlet=is_singlet)
        jk = coup.get_K()
        
        # P term is not finished yet
        try:
            jp = coup.get_P_term()
        except Exception:
            jp = 0.0
            
        jd = utils.dipole_dipole_J(m1, m2, trans_vector, args.state, singlet=is_singlet) 
        
        j_total = jc - jk - jp
        
        utils.print_row(dist, jc, jk, jp, jd, j_total)

    # 5. Report finale
    end_time = time.time()
    mem_final = utils.get_memory_usage()
            
    print("\nPERFORMANCE REPORT")
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")
    print(f"Initial Memory: {mem_start:.2f} MB")
    print(f"Peak Memory: {mem_final:.2f} MB")
    print(f"Memory Overhead: {mem_final - mem_start:.2f} MB")

if __name__ == "__main__":
    main()