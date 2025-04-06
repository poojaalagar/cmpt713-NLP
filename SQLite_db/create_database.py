import pandas as pd
import sqlite3

# Create (or connect to) an SQLite database file
db_path = "dnd_data.db"
conn = sqlite3.connect(db_path)

# List of your CSV files
csv_files = ["classes", "equipment", "monsters", "races", "spells"]

# Load each CSV into a corresponding SQLite table
for file in csv_files:
    df = pd.read_csv(f"D&D_data/{file}.csv")  # Update path if needed
    df.to_sql(file, conn, if_exists="replace", index=False)

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database created successfully.")