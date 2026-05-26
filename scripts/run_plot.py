import argparse
from elkopy.analysis.plot import generate_coupling_plot

def main():
    parser = argparse.ArgumentParser(description='Plot Elkopy Results CSV.')
    parser.add_argument('input', help='The generated scan_results.csv file')
    parser.add_argument('--logy', action='store_true', help='Set y-scale to log')
    parser.add_argument('--logx', action='store_true', help='Set x-scale to log')
    parser.add_argument('-o','--output', help='PNG plot name')
    parser.add_argument('--columns', '-c', nargs='+', help='Columns to plot')
    parser.add_argument('--cm', action='store_true', help='Convert coupling values from eV to cm^-1 in the plot')
    args = parser.parse_args()
    
    generate_coupling_plot(
        input_csv=args.input, logx=args.logx, logy=args.logy, output=args.output, columns_to_plot=args.columns, unit_cm=args.cm
    )

if __name__ == "__main__":
    main()