import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io
import os

def generate_coupling_plot(input_csv, logx=False, logy=False, output=None, columns_to_plot=None):

    if not output:
        base_name = os.path.splitext(input_csv)[0] #remove extension txt
        output_file = f"{base_name}.png" #add extension png
    else:
        output_file = output

    try:
        df = pd.read_csv(input_csv)
        fig, ax = plt.subplots(figsize=(10, 6))
        x_col = df.columns[0]
        available_cols = df.columns[1:].tolist()

        if columns_to_plot:   
            to_plot = [c for c in columns_to_plot if c in available_cols]
        else:
            to_plot = available_cols

        for col in to_plot:
            ax.plot(df[x_col], df[col], marker='o', label=col)

        def format_as_float(axis_obj):
            axis_obj.set_major_formatter(ticker.ScalarFormatter())
            axis_obj.get_major_formatter().set_scientific(False)
            axis_obj.get_major_formatter().set_useOffset(False)

        # Logarithmic scale?
        if logx:
            ax.set_xscale('log')
            ticks = [3, 4, 6, 8, 10, 15, 20]
            ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
            format_as_float(ax.xaxis)

        if logy:
            ax.set_yscale('log')
            format_as_float(ax.yaxis)

        plt.xlabel(x_col)
        plt.ylabel('Coupling/eV')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        print (f"Plot saved as {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_csv}' does not exist.")
    except Exception as e:
        print(f"Unexpected error: {e}")
