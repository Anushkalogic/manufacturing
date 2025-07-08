import sqlite3
import os

DB_PATH = "detections.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            volume_liters REAL,
            label TEXT,
            camera_name TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_detection_only(image_path, label, camera_name="1"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO images (image_path, volume_liters, label, camera_name)
        VALUES (?, ?, ?, ?)
    """, (image_path, None, label, camera_name))

    unique_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return unique_id, camera_name

def fetch_all_detections():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT image_path, label FROM images WHERE image_path IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

def update_latest_detection(image_path, volume_liters, label, unique_id, camera_name, height_cm, diameter_cm, severity):
    # You can log or update recent detection elsewhere if needed
    pass

def cleanup_null_entries():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM images WHERE image_path IS NULL OR label IS NULL")
    conn.commit()
    conn.close()
