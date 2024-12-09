import pandas as pd
import sqlite3
import os
import hashlib
import json
from tkinter import Tk, Label, Entry, Button, StringVar, IntVar, ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from difflib import SequenceMatcher
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

# Paths for dataset and metadata
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, 'vehicleSample100k.csv')  # Cleaned CSV file
metadata_file = os.path.join(script_dir, 'vehicleSample100k_metadata.json')  # Metadata JSON file
database_file = os.path.join(script_dir, 'car_data.db')  # SQLite database file

# Function to calculate the hash of a file
def calculate_file_hash(file_path):
    """Calculate the hash of the file contents."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# Function to initialize the database
def initialize_database():
    """Initialize or update the database if the CSV file has changed."""
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # Check if the metadata table exists to store the hash
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """)

    # Calculate the hash of the CSV file
    csv_hash = calculate_file_hash(csv_file)

    # Check the stored hash in the metadata table
    cursor.execute("SELECT value FROM metadata WHERE key = 'csv_hash';")
    row = cursor.fetchone()
    stored_hash = row[0] if row else None

    if csv_hash != stored_hash:
        print("CSV file has changed. Recreating the table...")

        # Dynamically create the table from the CSV file
        df = pd.read_csv(csv_file)
        df.to_sql('car_listings', conn, if_exists='replace', index=False)

        # Update the stored hash in the metadata table
        cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('csv_hash', ?);", (csv_hash,))

        print(f"Table recreated and CSV data imported into {database_file}.")
    else:
        print("CSV file has not changed. Skipping table recreation.")

    conn.commit()
    conn.close()

# Function to fetch models for a specific manufacturer
def fetch_models(manufacturer):
    """Fetch models from the database for the selected manufacturer."""
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT model FROM car_listings WHERE manufacturer = ? AND model IS NOT NULL;
    """, (manufacturer,))
    models = sorted([row[0] for row in cursor.fetchall()])
    conn.close()
    return models

# Function to update the model dropdown based on the selected manufacturer
def update_model_dropdown(*args):
    """Update the model dropdown options when the manufacturer changes."""
    selected_manufacturer = manufacturer_var.get()
    if selected_manufacturer:
        models = fetch_models(selected_manufacturer)
        model_dropdown['values'] = models  # Update the dropdown values
        model_var.set('')  # Clear the current selection

# Load metadata for dropdowns
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

manufacturers = metadata['manufacturer']
fuels = metadata['fuel']
transmissions = metadata['transmission']
types = metadata['type']

# Tkinter GUI setup
root = Tk()
root.title("Car Price Predictor")
root.geometry("800x600")

# Input Variables
manufacturer_var = StringVar()
model_var = StringVar()
year_var = IntVar()
odometer_var = IntVar()
fuel_var = StringVar()
transmission_var = StringVar()
type_var = StringVar()

# Helper function to create labels and widgets
def create_dropdown(label_text, options, variable, row):
    Label(root, text=label_text).grid(row=row, column=0, sticky="w", padx=10, pady=5)
    dropdown = ttk.Combobox(root, textvariable=variable, values=options, state="readonly")
    dropdown.grid(row=row, column=1, padx=10, pady=5)
    return dropdown

def create_input(label_text, variable, row):
    Label(root, text=label_text).grid(row=row, column=0, sticky="w", padx=10, pady=5)
    entry = Entry(root, textvariable=variable)
    entry.grid(row=row, column=1, padx=10, pady=5)
    return entry

# Create GUI elements
create_dropdown("Manufacturer:", manufacturers, manufacturer_var, 0)
model_dropdown = create_dropdown("Model:", [], model_var, 1)  # Start with an empty list for models
create_input("Year:", year_var, 2)
create_input("Odometer (miles):", odometer_var, 4)
create_dropdown("Fuel:", fuels, fuel_var, 5)
create_dropdown("Transmission:", transmissions, transmission_var, 6)

# Bind the manufacturer dropdown to update the model dropdown
manufacturer_var.trace("w", update_model_dropdown)

def calculate_string_similarity(str1, str2):
        """Calculate string similarity using SequenceMatcher."""
        return SequenceMatcher(None, str1, str2).ratio()

def calculate_similarity_score(row, user_inputs):
    """Calculate a similarity score for each row based on user inputs."""
    score = 0
    weights = {
        "manufacturer": 1,
        "model": 3,  # Higher weight for model similarity
        "year": 2,
        "odometer": 2,
        "fuel": 1,
        "transmission": 1,
    }

    for key, value in user_inputs.items():
        if key in row and pd.notna(row[key]):
            if key == "model":
                # String similarity for model field
                similarity = calculate_string_similarity(str(row[key]).lower(), str(value).lower())
                score += weights[key] * similarity
            elif str(row[key]).lower() == str(value).lower():
                score += weights[key]

    return score

output_label = Label(root, text="", justify="left")
output_label.grid(row=10, column=0, columnspan=2, pady=10)

def predict_price():
    user_inputs = {
        "manufacturer": manufacturer_var.get(),
        "model": model_var.get(),
        "year": year_var.get(),
        "odometer": odometer_var.get(),
        "fuel": fuel_var.get(),
        "transmission": transmission_var.get(),
    }

    # Filter out empty inputs
    filtered_inputs = {key: value for key, value in user_inputs.items() if value}

    # Connect to SQLite and fetch data
    conn = sqlite3.connect(database_file)
    query = "SELECT * FROM car_listings WHERE price IS NOT NULL;"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Ensure sufficient data exists
    if len(df) < 10:
        output_label.config(text="Not enough data to create a reliable model.")
        return

    # Train the model based on similarity
    df['similarity_score'] = df.apply(calculate_similarity_score, axis=1, args=(filtered_inputs,))
    df = df.sort_values(by='similarity_score', ascending=False)

    top_similar_records = df[df['similarity_score'] > 0].head(100)
    if len(top_similar_records) < 10:
        output_label.config(text="Not enough similar data to make a reliable prediction.")
        return

    features = ['manufacturer', 'model', 'year', 'odometer', 'fuel', 'transmission']
    X = pd.get_dummies(top_similar_records[features], drop_first=True)
    y = top_similar_records['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    user_df = pd.DataFrame([filtered_inputs])
    user_df = pd.get_dummies(user_df, drop_first=True)
    user_df = user_df.reindex(columns=X_train.columns, fill_value=0)

    try:
        prediction = model.predict(user_df)
        similar_cars = top_similar_records[['manufacturer', 'model', 'year', 'price']].head(10)
        similar_cars_list = "\n".join(
            f"{row['year']} {row['manufacturer']} {row['model']}: ${row['price']:.2f}"
            for _, row in similar_cars.iterrows()
        )

        output_text = f"Estimated price: ${prediction[0]:.2f}\n\nSimilar cars:\n{similar_cars_list}"
        output_label.config(text=output_text)

        # Plot depreciation graph
        depreciation_years = np.arange(1, 11)
        depreciation_values = prediction[0] * (0.85 ** depreciation_years)

        fig, ax = plt.subplots()
        ax.plot(depreciation_years, depreciation_values, marker='o')
        ax.set_title("Depreciation Over 10 Years")
        ax.set_xlabel("Years")
        ax.set_ylabel("Price ($)")

        def format_coord(x, y):
            return f"Year: {int(x)}, Price: ${y:.2f}"

        ax.format_coord = format_coord

        for i, txt in enumerate(depreciation_values):
            ax.annotate(f"${txt:.0f}", (depreciation_years[i], depreciation_values[i]), textcoords="offset points", xytext=(0,5), ha='center')

        canvas = FigureCanvasTkAgg(fig, root)
        canvas.get_tk_widget().grid(row=11, column=0, columnspan=2, pady=10)
        canvas.draw()

    except Exception as e:
        output_label.config(text=f"Error during prediction: {e}")

# Add a Submit button
Button(root, text="Predict Price", command=predict_price).grid(row=9, column=0, columnspan=2, pady=20)

# Start the GUI loop
root.mainloop()
