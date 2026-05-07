import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import io
import os

parser = argparse.ArgumentParser(description='Plot Electronic Coupling Scan.')
parser.add_argument('input', help='Elecoup output file.txt')
parser.add_argument('--logy', action='store_true', help='To set yscale=log')
parser.add_argument('--logx', action='store_true', help='To set xscale=log')
parser.add_argument('--output', help='PNG plot name (optional)')
parser.add_argument('--columns', '-c', nargs='+', help='What columns you want to plot (ex: J_Total J_Coul)')
args = parser.parse_args()

def main():

    if not args.output:
        base_name = os.path.splitext(args.input)[0] #remove extension txt
        output_file = f"{base_name}.png" #add extension png
    else:
        output_file = args.output

    table_data = []
    found_table = False

    try:
        with open(args.input, 'r') as f:
            for line in f:
                if "Dist" in line:
                    found_table = True
                    table_data.append(line.replace('|', ','))
                    continue
                
                if found_table:
                    if "---" in line:
                        continue
                    if not line.strip() or "PERFORMANCE REPORT" in line:
                        break
                    table_data.append(line.replace('|', ','))

        if not found_table:
            print("Error: table with coupling values not found")
            return

        df = pd.read_csv(io.StringIO("".join(table_data)))
        df.columns = df.columns.str.strip()
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df = df.apply(pd.to_numeric)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        x_col = df.columns[0]

        available_cols = df.columns[1:].tolist()
        if args.columns:   
            to_plot = [c for c in args.columns if c in available_cols]
            invalid = [c for c in args.columns if c not in available_cols]
            if invalid:
                print(f"Columns not found: {invalid}")
                print(f"Available columns: {available_cols}")
            if not to_plot:
                print("No valid columns selected. Default plotting")
                to_plot = available_cols
        else:
            to_plot = available_cols

        for col in to_plot:
            ax.plot(df[x_col], df[col], marker='o', label=col)

        def format_as_float(axis_obj):
            axis_obj.set_major_formatter(ticker.ScalarFormatter())
            axis_obj.get_major_formatter().set_scientific(False)
            axis_obj.get_major_formatter().set_useOffset(False)

        # Logarithmic scale?
        if args.logx:
            ax.set_xscale('log')
            ticks = [3, 4, 6, 8, 10, 15, 20]
            ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
            format_as_float(ax.xaxis)

        if args.logy:
            ax.set_yscale('log')
            format_as_float(ax.yaxis)


        plt.xlabel(x_col)
        plt.ylabel('Coupling/eV')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)

    except FileNotFoundError:
        print(f"Errore: Il file '{args.input}' non esiste.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()