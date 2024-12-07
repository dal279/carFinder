import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

# Open a file dialog to select the input file
Tk().withdraw()  # Close the root window
input_file = askopenfilename(title="Select Input CSV File", filetypes=[("CSV Files", "*.csv")])

# Load the CSV file
df = pd.read_csv(input_file)

# Randomly select a subset
subset = df.sample(n=1000, random_state=42)

# Open a file dialog to save the output file
output_file = asksaveasfilename(title="Save Reduced CSV As", defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
subset.to_csv(output_file, index=False)

print(f"Subset saved to {output_file}")