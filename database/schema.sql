-- Initial database schema

BEGIN TRANSACTION;

-- Properties table
CREATE TABLE properties_raw (
    type TEXT,
    construction INTEGER,
    bedrooms INTEGER,
    parking_spaces INTEGER,
    pool BOOLEAN DEFAULT 0,
    address TEXT,
    price REAL,
    link TEXT,
    lat REAL,
    lon REAL,
    full_bathroom INTEGER,
    half_bathroom INTEGER,
    property_age TEXT, -- Integer after cleaning
    land REAL,
    county TEXT,
);

-- Venues table
CREATE TABLE venues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    address TEXT,
    lat REAL,
    lon REAL,
    venue_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better performance
/*CREATE INDEX idx_properties_county ON properties(county);
CREATE INDEX idx_properties_price ON properties(price);
CREATE INDEX idx_venues_county ON venues(county);*/

COMMIT;