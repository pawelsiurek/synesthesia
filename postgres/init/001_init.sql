-- enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- songs table
CREATE TABLE IF NOT EXISTS songs (
    id          SERIAL PRIMARY KEY,
    title       VARCHAR(255)    NOT NULL,
    artist      VARCHAR(255)    NOT NULL,
    genre       VARCHAR(100),
    mood        VARCHAR(100),
    energy      FLOAT,                          -- 0.0 to 1.0, from Spotify API
    valence     FLOAT,                          -- 0.0 to 1.0, musical positiveness
    tempo       FLOAT,                          -- BPM
    year        INTEGER,
    themes      TEXT,                           -- comma separated: "hollywood, nostalgia, road trip"
    cultural_tags TEXT,                         -- comma separated: "california, american dream"
    spotify_id  VARCHAR(100)    UNIQUE,
    preview_url   VARCHAR(500),
    document    TEXT            NOT NULL,       -- full enriched text description fed to CLIP
    embedding   vector(512)     NOT NULL,       -- CLIP ViT-B/32 output
    created_at  TIMESTAMP       DEFAULT NOW()
);

-- ivfflat index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS songs_embedding_idx
    ON songs
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
