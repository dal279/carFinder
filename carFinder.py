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
    SELECT price, manufacturer, year, odometer FROM car_listings
    WHERE manufacturer = ? AND price IS NOT NULL;
"""
df = pd.read_sql_query(query, conn, params=(selected_manufacturer,))
conn.close()

print(df.head())  # Debug: Inspect the data
print(f"Number of rows: {len(df)}")

# Ensure data is sufficient
if len(df) < 10:
    print("Not enough data available for this manufacturer to train a reliable model.")
    exit()

# Prepare data for Linear Regression
X = df[['year', 'odometer']]  # Include additional features
y = df['price']

# Check for empty or invalid data
if X.empty or y.empty:
    print("No valid data for training.")
    exit()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Ask the user for prediction input
print("Training complete. Ready to predict prices.")
while True:
    prediction_input_year = input("Enter the year of the vehicle: ").strip()
    prediction_input_odometer = input("Enter the mileage of the vehicle: ").strip()

    try:
        prediction_input = pd.DataFrame({
            'year': [int(prediction_input_year)],
            'odometer': [float(prediction_input_odometer)]
        })

        prediction = model.predict(prediction_input)
        print(f"Estimated price: ${prediction[0]:.2f}")
    except Exception as e:
        print(f"Error: {e}")
        break