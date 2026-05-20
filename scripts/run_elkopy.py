import argparse
import time
import pandas as pd
from elkopy.utils import io, system
from elkopy.engine.scanner import run_distance_scan
from elkopy.analysis.plot import generate_coupling_plot

def main():
    parser = argparse.ArgumentParser(description="Electronic Coupling Scan CLI Engine.")
    parser.add_argument("xyz_1", help="File .xyz for the donor")
    parser.add_argument("xyz_2", nargs='?', help="File .xyz for the acceptor")
    parser.add_argument("-s", "--state", type=int, default=1, help="Excited state index")
    parser.add_argument("--spin", choices=['singlet', 'triplet'], default='singlet', help="Spin multiplicity (singlet or triplet)")
    parser.add_argument("-b", "--basis", type=str, default='3-21g', help="Basis set")
    parser.add_argument("--xc", type=str, default=None, help="DFT functional")
    parser.add_argument("--axis", choices=['x', 'y', 'z'], default='z', help="Scanning axis")
    parser.add_argument("--range", type=float, nargs=3, default=[3.0, 13.0, 0.5], metavar=('START', 'STOP', 'STEP'))
    parser.add_argument("--offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=('X', 'Y', 'Z'))
    parser.add_argument("-o", "--output", default="coupling_scan_results.csv", help="Output CSV filename")
    parser.add_argument("--plot", action='store_true', help="Generate plot after scan")

    args = parser.parse_args()

    start_time = time.time()
    mem_start = system.get_memory_usage()

    io.print_input_recap(args.xyz_1, args.xyz_2, args.basis, args.state, args.spin, args.axis, args.range, args.offset)
    print('Start calculations on monomers...\n')  

    scanner_generator = run_distance_scan(
        xyz_1=args.xyz_1, xyz_2=args.xyz_2, state=args.state, spin=args.spin,
        basis=args.basis, xc=args.xc, axis=args.axis, scan_range=args.range, offset=args.offset
    )

    results = []
    for message in scanner_generator:
        if message["status"] == "init_done":
            io.print_td_table(message["m1_data"], args.xyz_1)
            if message['m2_data'] is not None: 
                io.print_td_table(message["m2_data"], args.xyz_2)
            
            print("\nStarting scan...\n")
            print(f"{'Dist('+args.axis+')':>8} | {'J_Coul':>11} | {'J_Exch':>11} | {'J_Pterm':>11} | {'J_DipDip':>11} | {'J_Total':>11}")
            print("-" * 80)
        
        elif message["status"] == "scan_point":
            row_data = message["data"]
            io.print_row_output(row_data[0], row_data[1], row_data[2], row_data[3], row_data[4], row_data[5])
            results.append(row_data)
    
    # 3. Save results to CSV
    columns = ['Distance', 'J_Coul', 'J_Exch', 'J_Pterm', 'J_DipDip', 'J_Total']
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(args.output, index=False)
    print(f"\nScan completed. Results saved to '{args.output}'.")

    # 4. Generate plot if requested
    if args.plot:
        try:
            generate_coupling_plot(input_csv=args.output)
            base_name = args.output.rsplit('.', 1)[0]
            print(f"Plot successfully saved to '{base_name}.png'.")
        except Exception as e:
            print(f"Warning: Could not generate plot automatically. Error: {str(e)}")

    # 5. Final report
    system.print_performance_report(start_time, mem_start)

if __name__ == "__main__":
    main()