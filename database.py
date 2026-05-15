import sqlite3

# Connect to database
conn = sqlite3.connect("database.db")

# Create cursor
cursor = conn.cursor()

# Create appointments table
cursor.execute("""
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT,
    age INTEGER,
    gender TEXT,
    disease TEXT,
    doctor TEXT,
    appointment_date TEXT
)
""")

# Save changes
conn.commit()

# Close connection
conn.close()

print("Database and table created successfully.")