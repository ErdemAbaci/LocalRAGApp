import json
import sqlite3
from pathlib import Path


DB_PATH = Path("data/rag.db")

CHUNK_METADATA_COLUMNS = {
    "source_type": "TEXT",
    "page_number": "INTEGER",
    "chunk_index": "INTEGER"
}


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
        source_type TEXT,
        page_number INTEGER,
        chunk_index INTEGER,
        chunk_text TEXT NOT NULL,
        embedding TEXT NOT NULL
    )
    """)

    ensure_chunk_metadata_columns(cursor)

    conn.commit()
    conn.close()


def ensure_chunk_metadata_columns(cursor):
    cursor.execute("PRAGMA table_info(chunks)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    for column_name, column_type in CHUNK_METADATA_COLUMNS.items():
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE chunks ADD COLUMN {column_name} {column_type}"
            )


def clear_chunks():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM chunks")

    conn.commit()
    conn.close()


def insert_chunk(
    source_name,
    chunk_text,
    embedding,
    source_type=None,
    page_number=None,
    chunk_index=None
):
    embedding_json = json.dumps(embedding)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO chunks (
        source_name,
        source_type,
        page_number,
        chunk_index,
        chunk_text,
        embedding
    )
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        source_name,
        source_type,
        page_number,
        chunk_index,
        chunk_text,
        embedding_json
    ))

    conn.commit()
    conn.close()


def get_all_chunks():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        id,
        source_name,
        source_type,
        page_number,
        chunk_index,
        chunk_text,
        embedding
    FROM chunks
    """)
    rows = cursor.fetchall()

    conn.close()

    chunks = []

    for row in rows:
        chunks.append({
            "id": row[0],
            "source_name": row[1],
            "source_type": row[2],
            "page_number": row[3],
            "chunk_index": row[4],
            "chunk_text": row[5],
            "embedding": json.loads(row[6])
        })

    return chunks


def get_chunk_stats():
    if not DB_PATH.exists():
        return {
            "db_path": str(DB_PATH),
            "total_chunks": 0,
            "source_count": 0
        }

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT source_name) FROM chunks")
    source_count = cursor.fetchone()[0]

    conn.close()

    return {
        "db_path": str(DB_PATH),
        "total_chunks": total_chunks,
        "source_count": source_count
    }
