import json
import sqlite3
from pathlib import Path


DB_PATH = Path("data/rag.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_name TEXT NOT NULL,
        chunk_text TEXT NOT NULL,
        embedding TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


def clear_chunks():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM chunks")

    conn.commit()
    conn.close()


def insert_chunk(source_name, chunk_text, embedding):
    embedding_json = json.dumps(embedding)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO chunks (source_name, chunk_text, embedding)
    VALUES (?, ?, ?)
    """, (source_name, chunk_text, embedding_json))

    conn.commit()
    conn.close()


def get_all_chunks():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, source_name, chunk_text, embedding FROM chunks")
    rows = cursor.fetchall()

    conn.close()

    chunks = []

    for row in rows:
        chunks.append({
            "id": row[0],
            "source_name": row[1],
            "chunk_text": row[2],
            "embedding": json.loads(row[3])
        })

    return chunks