import pandas as pd
import sqlite3
import csv
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

# # csv file condenser 
# # Open a file dialog to select the input file
# Tk().withdraw()  # Close the root window
# input_file = askopenfilename(title="Select Input CSV File", filetypes=[("CSV Files", "*.csv")])

# # Load the CSV file
# df = pd.read_csv(input_file)

# #clean CSV file
# columns_to_drop = ["description","county"]
# df_dropped = df.drop(columns=columns_to_drop)
# df_cleaned = df_dropped.dropna()


# # Randomly select a subset
# subset = df_cleaned.sample(n= 500, random_state=42)

# # Open a file dialog to save the output file
# output_file = asksaveasfilename(title="Save Reduced CSV As", defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
# subset.to_csv(output_file, index=False)

# print(f"Subset saved to {output_file}")

#sqlite setup 
# Define the CSV file and database file
csv_file = '/Users/danielli/Desktop/Rutgers/codes/dataManagement/carFinder.csv'  # Replace with your CSV file path
database_file = 'car_data.db'  # SQLite database file

# Connect to SQLite (creates the database file if it doesn't exist)
conn = sqlite3.connect(database_file)
cursor = conn.cursor()

# Create the table with the specified columns
table_name = 'car_listings'  # Replace with your table name
cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY,
        url TEXT,
        region TEXT,
        region_url TEXT,
        price REAL,
        year INTEGER,
        manufacturer TEXT,
        model TEXT,
        condition TEXT,
        cylinders TEXT,
        fuel TEXT,
        odometer REAL,
        title_status TEXT,
        transmission TEXT,
        VIN TEXT,
        drive TEXT,
        size TEXT,
        type TEXT,
        paint_color TEXT,
        image_url TEXT,
        state TEXT,
        lat REAL,
        long REAL,
        posting_date TEXT
    );
""")

# Open the CSV file and insert its data into the SQLite database
with open(csv_file, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)  # Use DictReader for column-based access

    # Prepare the insert statement dynamically based on CSV columns
    placeholders = ', '.join(['?' for _ in reader.fieldnames])
    columns = ', '.join(reader.fieldnames)
    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders});"

    # Insert rows into the table
    for row in reader:
        values = [row[col] if col in row else None for col in reader.fieldnames]
        cursor.execute(insert_query, values)

# Commit changes and close the connection
conn.commit()
conn.close()

print(f"Data from {csv_file} has been successfully imported into {database_file}.")

