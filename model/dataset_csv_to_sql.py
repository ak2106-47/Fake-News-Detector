import pandas as pd
import sqlite3
import os
import random
from datetime import datetime

def load_entries_to_feedback_db(csv_path, db_path, num_entries=30, confidence_range=(3, 5)):

    df = pd.read_csv(csv_path)

    num_entries = min(num_entries, len(df))

    sampled_df = df.sample(n=num_entries)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_feedback'")
    table_exists = cursor.fetchone() is not None

    if not table_exists:
        cursor.execute('''
        CREATE TABLE user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_text TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            prediction_result TEXT,
            correct_classification TEXT,
            confidence_level INTEGER,
            ip_address TEXT
        )
        ''')

    entries_added = 0
    for _, row in sampled_df.iterrows():
        claim_text = row['title_text']
        class_val = row['class']
        correct_classification = "Genuine" if class_val == 1 else "Fake"

        confidence_level = random.randint(confidence_range[0], confidence_range[1])

        feedback_type = "classification"
        timestamp = datetime.now().isoformat()
        prediction_result = correct_classification
        ip_address = "127.0.0.1"
        
        try:
            cursor.execute(
                '''INSERT INTO user_feedback 
                   (claim_text, feedback_type, timestamp, prediction_result, 
                   correct_classification, confidence_level, ip_address) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (claim_text, feedback_type, timestamp, prediction_result, 
                 correct_classification, confidence_level, ip_address)
            )
            entries_added += 1
        except sqlite3.Error as e:
            print(f"Error inserting entry: {e}")

    conn.commit()
    conn.close()
    
    print(f"Successfully added {entries_added} entries from dataset.csv to feedback database.")
    return entries_added

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    CSV_PATH = os.path.join(BASE_DIR, "dataset.csv")
    DB_PATH = os.path.join(BASE_DIR, "feedback_database.db")

    num_entries = 30
    load_entries_to_feedback_db(CSV_PATH, DB_PATH, num_entries)