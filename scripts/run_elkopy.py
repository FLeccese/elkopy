import argparse
from elkopy.engine.scanner import run_distance_scan

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
    args = parser.parse_args()
    run_distance_scan(
        xyz_1=args.xyz_1, xyz_2=args.xyz_2, state=args.state, spin=args.spin,
        basis=args.basis, xc=args.xc, axis=args.axis, scan_range=args.range, offset=args.offset, output_filename=args.output
    )

if __name__ == "__main__":
    main()