import pandas as pd
import sqlite3
import csv
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

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

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the CSV file and database file
csv_file = os.path.join(script_dir, 'vehicleSample.csv')  # Dynamically find the CSV file in the same directory
database_file = os.path.join(script_dir, 'car_data.db')   # SQLite database file in the same directory

# Connect to SQLite (creates the database file if it doesn't exist)
conn = sqlite3.connect(database_file)
cursor = conn.cursor()

# Check if the table already exists
cursor.execute("""
    SELECT name FROM sqlite_master WHERE type='table' AND name='car_listings';
""")
table_exists = cursor.fetchone()

# Only create the table if it doesn't exist
if not table_exists:
    print("Initializing database...")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS car_listings (
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
    print("Table created.")
else:
    print("Database already initialized. Skipping table creation.")

# Open the CSV file and insert its data into the SQLite database
with open(csv_file, 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)  # Use DictReader for column-based access

    # Prepare the insert statement dynamically based on CSV columns
    placeholders = ', '.join(['?' for _ in reader.fieldnames])
    columns = ', '.join(reader.fieldnames)
    insert_query = f"INSERT OR IGNORE INTO car_listings ({columns}) VALUES ({placeholders});"

    # Insert rows into the table
    for row in reader:
        values = [row[col] if col in row else None for col in reader.fieldnames]
        cursor.execute(insert_query, values)

# Commit changes and close the connection
conn.commit()
conn.close()

print(f"Data from {csv_file} has been successfully imported into {database_file}.")

# Machine Learning - Linear Regression to predict price based on manufacturer

# Connect to SQLite database
conn = sqlite3.connect(database_file)
cursor = conn.cursor()

# Fetch unique manufacturers for user input
cursor.execute("SELECT DISTINCT manufacturer FROM car_listings WHERE manufacturer IS NOT NULL;")
manufacturers = [row[0] for row in cursor.fetchall()]
conn.close()

# Display options to the user
print("Available manufacturers:")
for i, manufacturer in enumerate(manufacturers, 1):
    print(f"{i}. {manufacturer}")

# Ask user to select a manufacturer
choice = int(input("Enter the number corresponding to the manufacturer you want: ")) - 1
selected_manufacturer = manufacturers[choice]
print(f"You selected: {selected_manufacturer}")

# Fetch data for the selected manufacturer
conn = sqlite3.connect(database_file)
query = f"""
    SELECT price, manufacturer FROM car_listings
    WHERE manufacturer = ? AND price IS NOT NULL;
"""
df = pd.read_sql_query(query, conn, params=(selected_manufacturer,))
conn.close()

# Ensure data is sufficient
if len(df) < 10:
    print("Not enough data available for this manufacturer to train a reliable model.")
else:
    # Prepare data for Linear Regression
    X = pd.get_dummies(df['manufacturer'], drop_first=True)  # One-hot encode manufacturer (though it's single-valued here)
    y = df['price']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    # Ask the user for prediction input
    print("Training complete. Ready to predict prices.")
    while True:
        prediction_input = input("Enter a value to estimate (or type 'exit' to quit): ").strip()
        if prediction_input.lower() == 'exit':
            break

        # Since we are one-hot encoding, predictions for this simple example are trivial
        # In a real-world scenario, the manufacturer choice is implicit
        prediction = model.predict([[1]])  # Hardcoded as only one manufacturer remains after filtering
        print(f"Estimated price for {selected_manufacturer}: ${prediction[0]:.2f}")