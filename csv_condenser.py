import pandas as pd
import numpy as np
import json
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

def clean_data(df):
    """Clean and preprocess the dataset."""
    # Drop irrelevant columns
    columns_to_drop = ['url', 'region_url', 'VIN', 'image_url', 'posting_date', 'state', 'lat', 'long', 'title_status', 'description', 'paint_color', 'drive']
    df = df.drop(columns=columns_to_drop, errors='ignore')  # Ignore columns not present

    # Handle missing values
    df['condition'].fillna('unknown', inplace=True)
    df['fuel'].fillna('unknown', inplace=True)
    df['price'].fillna(df['price'].median(), inplace=True)
    df['odometer'].fillna(df['odometer'].median(), inplace=True)

    # Remove outliers
    df = df[(df['price'] > 1000) & (df['price'] < 100000)]
    df = df[df['odometer'] < 500000]

    # Normalize text columns
    text_columns = ['manufacturer', 'model', 'condition', 'fuel', 'transmission', 'drive', 'type']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip()

    # Add derived features
    df['age'] = 2024 - df['year']
    bins = [0, 50000, 100000, 150000, 200000, np.inf]
    labels = ['low', 'medium', 'high', 'very high', 'extreme']
    df['odometer_bin'] = pd.cut(df['odometer'], bins=bins, labels=labels)

    # Drop duplicates
    df = df.drop_duplicates()

    return df


def extract_metadata(df):
    """Extract metadata for dropdowns and save to a JSON file."""
    dropdown_columns = ['manufacturer', 'model', 'fuel', 'transmission', 'drive', 'type']
    metadata = {}
    for column in dropdown_columns:
        if column in df.columns:
            metadata[column] = sorted(df[column].dropna().unique())
    return metadata


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

    # Ensure posting_date exists and is in datetime format
    if 'posting_date' in df.columns:
        df['posting_date'] = pd.to_datetime(df['posting_date'], errors='coerce')
        df = df.dropna(subset=['posting_date'])  # Drop rows with invalid dates
        df = df.sort_values(by='posting_date', ascending=False)  # Sort by recent first

    # Clean the data
    print("Cleaning the data...")
    df_cleaned = clean_data(df)

    # Split into most recent (70%) and oldest (30%)
    recent_records = df_cleaned.head(200000)
    oldest_records = recent_records.tail(int(0.3 * len(recent_records)))
    recent_records = recent_records.head(int(0.7 * len(recent_records)))

    # Randomly sample 70% from recent and 30% from oldest
    recent_sample = recent_records.sample(n=int(0.7 * 100000), random_state=42) if len(recent_records) > int(0.7 * 100000) else recent_records
    oldest_sample = oldest_records.sample(n=int(0.3 * 100000), random_state=42) if len(oldest_records) > int(0.3 * 100000) else oldest_records

    # Combine the samples
    subset = pd.concat([recent_sample, oldest_sample]).sample(frac=1, random_state=42)  # Shuffle the combined sample

    # Extract metadata
    metadata = extract_metadata(subset)

    # Save the subset to a new file
    output_file = asksaveasfilename(title="Save Reduced CSV As", defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if output_file:
        subset.to_csv(output_file, index=False)
        print(f"Subset saved to {output_file}")

        # Save metadata
        metadata_file = output_file.replace(".csv", "_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {metadata_file}")
    else:
        print("No file selected for saving. Exiting.")


if __name__ == "__main__":
    main()