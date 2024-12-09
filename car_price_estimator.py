import pandas as pd
import sqlite3
import os
import hashlib
import json
from datetime import datetime
from tkinter import Tk, Label, Entry, Button, StringVar, IntVar, ttk, Text, END, Frame, Checkbutton  # Fixed imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from difflib import SequenceMatcher  # For string similarity
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors

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

# Tkinter GUI setup
root = Tk()
root.title("Car Price Estimator")
root.geometry("1200x600")  # Resized default window to fit both inputs and graph

# Main container frames
left_frame = Frame(root)  # For inputs and output text
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

right_frame = Frame(root)  # For the graph
right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

# Input Variables
manufacturer_var = StringVar()
model_var = StringVar()
year_var = IntVar()
odometer_var = IntVar()
fuel_var = StringVar()
transmission_var = StringVar()

# Helper function to create labels and widgets
def create_dropdown(parent, label_text, options, variable, row):
    Label(parent, text=label_text).grid(row=row, column=0, sticky="w", padx=10, pady=5)
    dropdown = ttk.Combobox(parent, textvariable=variable, values=options, state="readonly")
    dropdown.grid(row=row, column=1, padx=10, pady=5)
    return dropdown

def create_input(parent, label_text, variable, row):
    Label(parent, text=label_text).grid(row=row, column=0, sticky="w", padx=10, pady=5)
    entry = Entry(parent, textvariable=variable)
    entry.grid(row=row, column=1, padx=10, pady=5)
    return entry

# Create GUI elements in the left frame
create_dropdown(left_frame, "Manufacturer:", manufacturers, manufacturer_var, 0)
model_dropdown = create_dropdown(left_frame, "Model:", [], model_var, 1)  # Start with an empty list for models
create_input(left_frame, "Year:", year_var, 2)
create_input(left_frame, "Odometer (miles):", odometer_var, 3)
create_dropdown(left_frame, "Fuel:", fuels, fuel_var, 4)
create_dropdown(left_frame, "Transmission:", transmissions, transmission_var, 5)

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

# Output Area
output_text = Text(left_frame, wrap="word", height=15, width=50)
output_text.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

# Frame for the graph (in the right frame)
graph_frame = Frame(right_frame)
graph_frame.pack(fill="both", expand=True)

def show_depreciation_graph(estimated_price):
    """Show an interactive graph of depreciation."""
    # Clear the existing graph frame
    for widget in graph_frame.winfo_children():
        widget.destroy()

    # Get the selected manufacturer and model
    selected_manufacturer = manufacturer_var.get()
    selected_model = model_var.get()

    current_year = datetime.now().year
    years = [current_year + i for i in range(11)]
    depreciation_rates = [0.9 ** i for i in range(11)]  # Example depreciation rates
    prices = [estimated_price * rate for rate in depreciation_rates]

    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(years, prices, marker='o', label="Depreciation Curve")
    
    # Set dynamic title based on selected manufacturer and model
    ax.set_title(f"Depreciation of {selected_manufacturer} {selected_model} Over 10 Years")
    ax.set_xlabel("Year")
    ax.set_ylabel("Price ($)")
    ax.legend()

    # Interactive cursor with custom labels
    cursor = mplcursors.cursor(ax, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        # Update the hover label with x and y-axis labels
        x, y = sel.target
        sel.annotation.set_text(f"Year: {int(x)}\nPrice: ${y:.2f}")

    # Embed the graph into Tkinter
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Add variables for checkboxes
show_model_metrics = IntVar(value=1)  # Default to show Model Evaluation Metrics
show_cv_metrics = IntVar(value=1)    # Default to show Cross-Validation Metrics

# Checkboxes for toggling metrics
Label(left_frame, text="Options:").grid(row=8, column=0, sticky="w", padx=10)
Checkbutton(left_frame, text="Show Model Evaluation Metrics", variable=show_model_metrics).grid(row=9, column=0, sticky="w", padx=10)
Checkbutton(left_frame, text="Show Cross-Validation Metrics", variable=show_cv_metrics).grid(row=9, column=1, sticky="w", padx=10)

def estimate_price():
    output_text.delete("1.0", END)  # Clear previous output

    # Collect user inputs
    user_inputs = {
        "manufacturer": manufacturer_var.get(),
        "model": model_var.get(),
        "year": year_var.get(),
        "odometer": odometer_var.get(),
        "fuel": fuel_var.get(),
        "transmission": transmission_var.get(),
    }

    # Check for missing fields
    missing_fields = [key for key, value in user_inputs.items() if not value]
    if missing_fields:
        # Format missing fields for user display
        formatted_fields = ', '.join(missing_fields)
        output_text.insert(END, f"Error: Missing fields - {formatted_fields}\n")
        return

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
    df['similarity_score'] = df.apply(calculate_similarity_score, axis=1, args=(user_inputs,))
    df = df.sort_values(by='similarity_score', ascending=False)

    top_similar_records = df[df['similarity_score'] > 0].head(100)
    if len(top_similar_records) < 10:
        output_text.insert(END, "Error: Not enough similar data to make a reliable estimation.")
        return

    features = ['manufacturer', 'model', 'year', 'odometer', 'fuel', 'transmission']
    X = pd.get_dummies(top_similar_records[features], drop_first=True)
    y = top_similar_records['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    user_df = pd.DataFrame([user_inputs])
    user_df = pd.get_dummies(user_df, drop_first=True)
    user_df = user_df.reindex(columns=X_train.columns, fill_value=0)

    try:
        estimation = model.predict(user_df)
        output_text.insert(END, f"Estimated price: ${estimation[0]:.2f}\n\n")

        # Display Model Evaluation Metrics if checkbox is checked
        if show_model_metrics.get():
            train_mse = mean_squared_error(y_train, model.predict(X_train))
            test_mse = mean_squared_error(y_test, model.predict(X_test))
            train_r2 = model.score(X_train, y_train)
            test_r2 = model.score(X_test, y_test)
            output_text.insert(END, f"Model Evaluation Metrics:\n")
            output_text.insert(END, f"Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}\n")
            output_text.insert(END, f"Train R²: {train_r2:.2f}, Test R²: {test_r2:.2f}\n\n")

        # Display Cross-Validation Metrics if checkbox is checked
        if show_cv_metrics.get():
            mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
            r2_scores = cross_val_score(model, X, y, scoring='r2', cv=5)
            output_text.insert(END, f"Cross-Validation Metrics:\n")
            output_text.insert(END, f"Mean CV MSE: {abs(mse_scores.mean()):.2f}\n")
            output_text.insert(END, f"Mean CV R²: {r2_scores.mean():.2f}\n\n")

        # Find similar cars
        similar_cars = top_similar_records.loc[
            (top_similar_records['price'] >= estimation[0] * 0.9) &
            (top_similar_records['price'] <= estimation[0] * 1.1),
            ['manufacturer', 'model', 'year', 'price', 'odometer']
        ]

        output_text.insert(END, "Similar cars:\n")
        for _, row in similar_cars.iterrows():
            output_text.insert(
                END, 
                f"{row['year']} {row['manufacturer']} {row['model']} ({int(row['odometer'])} miles): ${row['price']:.2f}\n"
            )

    except Exception as e:
        output_text.insert(END, f"Error during estimation: {e}")

# Add buttons
Button(left_frame, text="Estimate Price", command=estimate_price).grid(row=7, column=0, pady=20, padx=10, sticky="w")
Button(left_frame, text="Predict Depreciation", command=lambda: show_depreciation_graph(10000)).grid(row=7, column=1, pady=20, padx=10, sticky="e")

# Start the GUI loop
root.mainloop()