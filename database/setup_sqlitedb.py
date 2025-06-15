#! /usr/bin/env python3
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

'''
Create SQLite DB

Creates the real_estate.sqlite database based on the schema.sql

'''

import sqlite3
import argparse
from pathlib import Path

def init_db(db_path: Path, schema_path: Path):
    """Initialize a new SQLite database from schema file"""
    if db_path.exists():
        raise FileExistsError(f"Database already exists at {db_path}")
    
    print(f"Creating new database at {db_path}")
    conn = sqlite3.connect(db_path)
    
    with open(schema_path, 'r') as f:
        schema_sql = f.read()
    
    try:
        conn.executescript(schema_sql)
        print("Schema loaded successfully")
        
        # Verify tables were created
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = cur.fetchall()
        print(f"Created tables: {[t[0] for t in tables]}")
        
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
        db_path.unlink()  # Clean up failed creation
        raise
    finally:
        conn.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Initialize SQLite database')
    parser.add_argument('--db', default='real_estate.sqlite', help='Database file path')
    args = parser.parse_args()
    
    db_path = Path(args.db)
    schema_path = Path(__file__).parent / 'schema.sql'
    
    try:
        init_db(db_path, schema_path)
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        exit(1)