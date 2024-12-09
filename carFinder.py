import pandas as pd
import sqlite3
import os
import hashlib
import json
from tkinter import Tk, Label, Entry, Button, StringVar, IntVar, ttk, Text, END
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from difflib import SequenceMatcher  # For string similarity

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
root.geometry("500x600")

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

# Function to find similar cars
def find_similar_cars(predicted_price, selected_model, car_type, df):
    """Find similar cars based on type and price, excluding the selected model."""
    similar_cars = df.loc[
        (df['type'] == car_type) &  # Same car type
        (df['model'].str.lower() != selected_model.lower()) &  # Exclude the same model
        (df['price'] >= predicted_price * 0.9) &
        (df['price'] <= predicted_price * 1.1),
        ['manufacturer', 'model', 'year', 'price']
    ]
    return similar_cars

# Output Area
output_text = Text(root, wrap="word", height=15, width=50)
output_text.grid(row=10, column=0, columnspan=2, padx=10, pady=10)

# Prediction function
def predict_price():
    output_text.delete("1.0", END)  # Clear previous output

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
        output_text.insert(END, "Error: Not enough data to create a reliable model.")
        return

    # Train the model based on similarity
    df['similarity_score'] = df.apply(calculate_similarity_score, axis=1, args=(filtered_inputs,))
    df = df.sort_values(by='similarity_score', ascending=False)

    top_similar_records = df[df['similarity_score'] > 0].head(100)
    if len(top_similar_records) < 10:
        output_text.insert(END, "Error: Not enough similar data to make a reliable prediction.")
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
        predicted_price = prediction[0]

        # Display prediction
        output_text.insert(END, f"Estimated price: ${predicted_price:.2f}\n\n")

        # Find and display similar cars
        car_type = df.loc[df['model'] == filtered_inputs['model'], 'type'].iloc[0] if 'model' in filtered_inputs else None
        if car_type:
            similar_cars = find_similar_cars(predicted_price, filtered_inputs['model'], car_type, df)
            if not similar_cars.empty:
                output_text.insert(END, "Similar cars:\n")
                for _, row in similar_cars.iterrows():
                    output_text.insert(END, f"{row['year']} {row['manufacturer']} {row['model']}: ${row['price']:.2f}\n")
            else:
                output_text.insert(END, "No similar cars found within the price range.\n")

    except Exception as e:
        output_text.insert(END, f"Error during prediction: {e}")

# Add a Submit button
Button(root, text="Predict Price", command=predict_price).grid(row=9, column=0, columnspan=2, pady=20)

# Start the GUI loop
root.mainloop()