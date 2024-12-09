import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def analyze_csv():
    """Analyze a CSV file and extract information about posting_date."""
    # Hide the Tkinter root window
    Tk().withdraw()

    # Prompt user to select a CSV file
    input_file = askopenfilename(title="Select CSV File to Analyze", filetypes=[("CSV Files", "*.csv")])
    if not input_file:
        print("No file selected. Exiting.")
        return

    # Load the CSV file into a DataFrame
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Print the total number of records
    print(f"Total number of records in the CSV: {len(df)}")

    # Check if the posting_date column exists
    if 'posting_date' not in df.columns:
        print("The selected CSV file does not contain a 'posting_date' column.")
        return

    # Convert posting_date to datetime
    try:
        df['posting_date'] = pd.to_datetime(df['posting_date'], errors='coerce')
    except Exception as e:
        print(f"Error converting 'posting_date' to datetime: {e}")
        return

    # Drop rows with invalid dates
    valid_dates = df.dropna(subset=['posting_date'])

    # Analyze the first and most recent records
    if not valid_dates.empty:
        first_date = valid_dates['posting_date'].min()
        last_date = valid_dates['posting_date'].max()

        print(f"Analysis of 'posting_date' in {input_file}:")
        print(f" - Earliest posting date: {first_date}")
        print(f" - Most recent posting date: {last_date}")
    else:
        print("No valid posting dates found in the dataset.")

if __name__ == "__main__":
    analyze_csv()