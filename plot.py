import pandas as pd
import matplotlib.pyplot as plt
import argparse
import io
import os

def main():
    # 1. Configurazione Argparse
    parser = argparse.ArgumentParser(description='Plot Electronic Coupling Scan.')
    parser.add_argument('input', help='Elecoup output file.txt')
    parser.add_argument('--log', action='store_true', help='To set yscale=log')
    parser.add_argument('--output', help='PNG plot name (optional)')
    args = parser.parse_args()

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

        # 4. Plotting
        plt.figure(figsize=(10, 6))
        
        x_col = df.columns[0]
        for col in df.columns[1:]:
            plt.plot(df[x_col], df[col], marker='o', label=col)

        # Logarithmic scale?
        if args.log:
            plt.yscale('log')

        plt.xlabel(x_col)
        plt.ylabel('Coupling/eV')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)

    except FileNotFoundError:
        print(f"Errore: Il file '{args.input}' non esiste.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()