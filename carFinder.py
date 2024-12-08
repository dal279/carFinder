import pandas as pd
import sqlite3
import os
import hashlib
from tkinter import Tk, Label, Entry, Button, StringVar, IntVar, ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to calculate the hash of the CSV file
def calculate_file_hash(file_path):
    """Calculate the hash of the file contents."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# Function to dynamically create a table from a CSV file
def create_table_from_csv(csv_file, conn, table_name):
    """Create a table in the database from a CSV file."""
    df = pd.read_csv(csv_file)
    # Write the DataFrame to the database as a new table
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f"Table '{table_name}' created or updated in the database.")

# SQLite database and CSV setup
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, 'vehicleSample10000.csv')  # CSV file path
database_file = os.path.join(script_dir, 'car_data.db')   # SQLite database file path

# Connect to SQLite database
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
    create_table_from_csv(csv_file, conn, 'car_listings')

    # Update the stored hash in the metadata table
    cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('csv_hash', ?);", (csv_hash,))

    print(f"Table recreated and CSV data imported into {database_file}.")
else:
    print("CSV file has not changed. Skipping table recreation.")

# Commit changes and close the connection
conn.commit()
conn.close()

# Fetch unique values for dropdowns
conn = sqlite3.connect(database_file)
cursor = conn.cursor()
cursor.execute("SELECT DISTINCT manufacturer FROM car_listings WHERE manufacturer IS NOT NULL;")
manufacturers = sorted([row[0] for row in cursor.fetchall()])
cursor.execute("SELECT DISTINCT model FROM car_listings WHERE model IS NOT NULL;")
models = sorted([row[0] for row in cursor.fetchall()])
cursor.execute("SELECT DISTINCT fuel FROM car_listings WHERE fuel IS NOT NULL;")
fuels = [row[0] for row in cursor.fetchall()]
cursor.execute("SELECT DISTINCT transmission FROM car_listings WHERE transmission IS NOT NULL;")
transmissions = [row[0] for row in cursor.fetchall()]
cursor.execute("SELECT DISTINCT drive FROM car_listings WHERE drive IS NOT NULL;")
drives = [row[0] for row in cursor.fetchall()]
cursor.execute("SELECT DISTINCT type FROM car_listings WHERE type IS NOT NULL;")
types = [row[0] for row in cursor.fetchall()]
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

# Tkinter GUI setup
root = Tk()
root.title("Car Price Predictor")
root.geometry("400x500")  # Adjusted for additional dropdown

# Input Variables
manufacturer_var = StringVar()
model_var = StringVar()
year_var = IntVar()
condition_var = StringVar()
odometer_var = IntVar()
fuel_var = StringVar()
transmission_var = StringVar()
drive_var = StringVar()
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
create_dropdown("Condition:", ["excellent", "new", "like new", "good", "fair"], condition_var, 3)
create_input("Odometer (miles):", odometer_var, 4)
create_dropdown("Fuel:", fuels, fuel_var, 5)
create_dropdown("Transmission:", transmissions, transmission_var, 6)
create_dropdown("Drive:", drives, drive_var, 7)
create_dropdown("Type:", types, type_var, 8)

# Bind the manufacturer dropdown to update the model dropdown
manufacturer_var.trace("w", update_model_dropdown)

# Prediction function (unchanged)
def predict_price():
    user_inputs = {
        "manufacturer": manufacturer_var.get(),
        "model": model_var.get(),
        "year": year_var.get(),
        "condition": condition_var.get(),
        "odometer": odometer_var.get(),
        "fuel": fuel_var.get(),
        "transmission": transmission_var.get(),
        "drive": drive_var.get(),
        "type": type_var.get(),
    }

    # Filter out empty inputs
    filtered_inputs = {key: value for key, value in user_inputs.items() if value}

    # Connect to SQLite and fetch data
    conn = sqlite3.connect(database_file)
    query = """
        SELECT price, manufacturer, model, year, condition, odometer, fuel, transmission, drive, type 
        FROM car_listings WHERE price IS NOT NULL;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Check if sufficient data exists
    if len(df) < 10:
        messagebox.showerror("Error", "Not enough data to create a reliable model.")
        return

    # Calculate similarity scores
    def calculate_similarity_score(row, user_inputs):
        score = 0
        for key, value in user_inputs.items():
            if key in row and pd.notna(row[key]) and str(row[key]).lower() == str(value).lower():
                score += 1
        return score

    df['similarity_score'] = df.apply(calculate_similarity_score, axis=1, args=(filtered_inputs,))
    df = df.sort_values(by='similarity_score', ascending=False)

    # Select top similar records
    top_similar_records = df[df['similarity_score'] > 0].head(100)
    if len(top_similar_records) < 10:
        messagebox.showerror("Error", "Not enough similar data to make a reliable prediction.")
        return

    # Train the model
    features = ['manufacturer', 'model', 'year', 'condition', 'odometer', 'fuel', 'transmission', 'drive', 'type']
    X = pd.get_dummies(top_similar_records[features], drop_first=True)
    y = top_similar_records['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prepare user input for prediction
    user_df = pd.DataFrame([filtered_inputs])
    user_df = pd.get_dummies(user_df, drop_first=True)
    user_df = user_df.reindex(columns=X_train.columns, fill_value=0)

    # Predict and show result
    try:
        prediction = model.predict(user_df)
        messagebox.showinfo("Prediction", f"Estimated price: ${prediction[0]:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Error during prediction: {e}")

# Add a Submit button
Button(root, text="Predict Price", command=predict_price).grid(row=9, column=0, columnspan=2, pady=20)

# Start the GUI loop
root.mainloop()