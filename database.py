import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Anujit$321',
    'database': 'ManufacturingDetection'
}

def init_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_path VARCHAR(255),
                volume_liters FLOAT,
                label VARCHAR(100),
                camera_name VARCHAR(50)
            )
        ''')
        conn.commit()
        print("✅ Table 'images' created or already exists.")
    except Error as e:
        print(f"❌ Error (init_db): {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def insert_detection_only(image_path, label, camera_name="1"):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO images (image_path, volume_liters, label, camera_name)
            VALUES (%s, %s, %s, %s)
        """, (image_path, None, label, camera_name))

        conn.commit()
        unique_id = cursor.lastrowid
        print(f"✅ Inserted: ID={unique_id}, Path={image_path}, Label={label}, Camera={camera_name}")
        return unique_id, camera_name
    except Error as e:
        print(f"❌ Error (insert_detection_only): {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def fetch_all_detections():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT image_path, label FROM images WHERE image_path IS NOT NULL
        """)
        rows = cursor.fetchall()
        print(f"📦 Total Records Fetched: {len(rows)}")
        for row in rows:
            print(f" - Image: {row[0]}, Label: {row[1]}")
        return rows
    except Error as e:
        print(f"❌ Error (fetch_all_detections): {e}")
        return []
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def update_latest_detection(image_path, volume_liters, label, unique_id, camera_name, height_cm, diameter_cm, severity):
    # You can implement update logic here if needed
    pass

def cleanup_null_entries():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM images WHERE image_path IS NULL OR label IS NULL
        """)
        deleted = cursor.rowcount
        conn.commit()
        print(f"🧹 Deleted {deleted} null entries.")
    except Error as e:
        print(f"❌ Error (cleanup_null_entries): {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# ✅ TESTING (run only when executing this file directly)
if __name__ == "__main__":
    init_db()
    insert_detection_only("frame_001.jpg", "cow")
    insert_detection_only("frame_002.jpg", "cow")
    fetch_all_detections()
    cleanup_null_entries()
