import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

def main():
    # Close the root Tkinter window
    Tk().withdraw()

    # Select input file
    input_file = askopenfilename(title="Select Input CSV File", filetypes=[("CSV Files", "*.csv")])
    if not input_file:
        print("No file selected. Exiting.")
        return

    # Load the CSV file
    df = pd.read_csv(input_file)

    # Clean CSV file
    columns_to_drop = ["description", "county"]
    df_dropped = df.drop(columns=columns_to_drop)
    df_cleaned = df_dropped.dropna()

    # Randomly select a subset
    subset = df_cleaned.sample(n=500, random_state=42)

    # Save the subset to a new file
    output_file = asksaveasfilename(title="Save Reduced CSV As", defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if output_file:
        subset.to_csv(output_file, index=False)
        print(f"Subset saved to {output_file}")
    else:
        print("No file selected for saving. Exiting.")

if __name__ == "__main__":
    main()