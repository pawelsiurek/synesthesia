#!/usr/bin/env python3
"""
Populates the songs table with CLIP-encoded track data from Deezer.

Deezer's public API needs no auth and returns working 30s MP3 preview URLs,
unlike Spotify (which deprecated preview_url and audio-features in Nov 2024).

Run from project root with venv activated:
    python scripts/enrich_songs.py

When running outside Docker, temporarily set POSTGRES_HOST=localhost in .env
(Postgres is still served from Docker on port 5432).
"""

import sys
import time
import logging
from pathlib import Path

import httpx
import numpy as np
from sqlalchemy.dialects.postgresql import insert as pg_insert

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.clip_encoder import CLIPEncoder
from backend.db.database import SessionLocal
from backend.db.models import Song

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TARGET_TOTAL = 2000
DEEZER_SEARCH = "https://api.deezer.com/search"

# genre label (stored in DB) → visual descriptor fed to CLIP
GENRE_VISUALS: dict[str, str] = {
    "pop":         "stage lights crowd stadium bright colors pop art glossy commercial",
    "rock":        "electric guitar concert arena raw energy leather jacket amplifier",
    "hip-hop":     "urban street city nightlife microphone studio booth graffiti",
    "r-n-b":       "silhouette romantic low light smooth velvet warm intimate glow",
    "electronic":  "neon lights synthesizer rave laser grid futuristic digital pulse",
    "jazz":        "smoky lounge dim brass trumpet candlelit late night club bebop",
    "classical":   "concert hall orchestra grand piano marble columns formal elegant",
    "metal":       "dark stage fire pyrotechnics skull aggressive crowd moshing",
    "country":     "barn field sunset pickup truck dusty road open sky americana",
    "indie":       "bedroom lo-fi vintage film grain coffee shop hazy warm nostalgic",
    "latin":       "tropical vibrant beach carnival colorful dancers festive rhythm",
    "reggae":      "beach palm trees sunshine relaxed coastal island Rastafari",
    "soul":        "church choir warm spotlight deep feeling vintage soulful gospel",
    "blues":       "delta crossroads rain stormy night old guitar porch southern",
    "folk":        "campfire acoustic forest river cabin handmade wooden earthy",
    "punk":        "torn denim safety pin graffiti dive bar raw aggressive DIY",
    "ambient":     "vast landscape fog misty oceanic space stars ethereal floating",
    "k-pop":       "pastel idol group choreography colorful hair glossy K-pop",
    "afrobeat":    "vibrant African print percussion community celebration rhythm",
    "soundtracks": "cinematic epic orchestra sweeping wide shot dramatic score",
}

# Representative artists per genre — searched via artist:"Name"
GENRE_ARTISTS: dict[str, list[str]] = {
    "pop":         ["Taylor Swift", "Ed Sheeran", "Ariana Grande", "Dua Lipa", "Billie Eilish", "Bruno Mars", "Katy Perry", "Justin Bieber"],
    "rock":        ["Queen", "Foo Fighters", "The Beatles", "Nirvana", "Red Hot Chili Peppers", "AC/DC", "The Rolling Stones", "David Bowie"],
    "hip-hop":     ["Drake", "Kendrick Lamar", "Eminem", "Jay-Z", "Travis Scott", "J. Cole", "Nicki Minaj", "Lil Wayne"],
    "r-n-b":       ["The Weeknd", "Beyonce", "Usher", "Alicia Keys", "Frank Ocean", "SZA", "Daniel Caesar", "H.E.R."],
    "electronic":  ["Daft Punk", "Calvin Harris", "Avicii", "The Chainsmokers", "Marshmello", "Skrillex", "Kygo", "Zedd"],
    "jazz":        ["Miles Davis", "John Coltrane", "Dave Brubeck", "Thelonious Monk", "Bill Evans", "Herbie Hancock", "Stan Getz", "Chet Baker"],
    "classical":   ["Beethoven", "Mozart", "Bach", "Chopin", "Debussy", "Tchaikovsky", "Vivaldi", "Schubert"],
    "metal":       ["Metallica", "Black Sabbath", "Iron Maiden", "Slayer", "Megadeth", "Tool", "System of a Down", "Slipknot"],
    "country":     ["Johnny Cash", "Dolly Parton", "Luke Combs", "Morgan Wallen", "Chris Stapleton", "Carrie Underwood", "Garth Brooks", "Kenny Rogers"],
    "indie":       ["Arctic Monkeys", "The Strokes", "Vampire Weekend", "Tame Impala", "Bon Iver", "Arcade Fire", "Sufjan Stevens", "Beach House"],
    "latin":       ["Bad Bunny", "J Balvin", "Shakira", "Daddy Yankee", "Maluma", "Ozuna", "Rosalia", "Marc Anthony"],
    "reggae":      ["Bob Marley", "Jimmy Cliff", "Damian Marley", "Shaggy", "Sean Paul", "UB40", "Peter Tosh", "Toots and the Maytals"],
    "soul":        ["Aretha Franklin", "Marvin Gaye", "Sam Cooke", "Stevie Wonder", "Ray Charles", "Al Green", "Otis Redding", "James Brown"],
    "blues":       ["B.B. King", "Muddy Waters", "Eric Clapton", "John Lee Hooker", "Howlin Wolf", "Buddy Guy", "Robert Cray", "Gary Clark Jr."],
    "folk":        ["Bob Dylan", "Simon & Garfunkel", "Joni Mitchell", "Neil Young", "Nick Drake", "Mumford & Sons", "The Lumineers", "Iron & Wine"],
    "punk":        ["The Clash", "Ramones", "Green Day", "The Offspring", "Bad Religion", "Blink-182", "Sex Pistols", "Rise Against"],
    "ambient":     ["Brian Eno", "Aphex Twin", "Moby", "Sigur Ros", "Max Richter", "Nils Frahm", "Olafur Arnalds", "Tycho"],
    "k-pop":       ["BTS", "BLACKPINK", "EXO", "TWICE", "Red Velvet", "NCT 127", "Stray Kids", "aespa"],
    "afrobeat":    ["Fela Kuti", "Burna Boy", "Wizkid", "Davido", "Mr Eazi", "Tiwa Savage", "Adekunle Gold", "Tems"],
    "soundtracks": ["Hans Zimmer", "John Williams", "Ennio Morricone", "Howard Shore", "Danny Elfman", "Alexandre Desplat", "James Horner", "Alan Silvestri"],
}


def build_document(title: str, artist: str, genre_visual: str) -> str:
    return f"{title} by {artist}. {genre_visual}."


def fetch_tracks(client: httpx.Client) -> list[dict]:
    """Search Deezer by representative artist per genre, keeping tracks with a preview."""
    seen: set[str] = set()
    result: list[dict] = []
    per_genre = (TARGET_TOTAL // len(GENRE_ARTISTS)) + 60

    for genre_label, artists in GENRE_ARTISTS.items():
        genre_visual = GENRE_VISUALS[genre_label]
        genre_tracks: list[dict] = []

        for artist in artists:
            if len(genre_tracks) >= per_genre:
                break
            try:
                resp = client.get(DEEZER_SEARCH, params={"q": f'artist:"{artist}"', "limit": 100})
                resp.raise_for_status()
                items = resp.json().get("data", [])
            except Exception as exc:
                log.warning(f"Deezer search failed for artist={artist}: {exc}")
                continue

            for item in items:
                ext_id = str(item.get("id"))
                if not ext_id or ext_id in seen:
                    continue
                if not item.get("preview"):
                    continue
                seen.add(ext_id)
                genre_tracks.append({
                    "external_id":  ext_id,
                    "title":        item["title"],
                    "artist":       item["artist"]["name"],
                    "preview_url":  item["preview"],
                    "genre":        genre_label,
                    "genre_visual": genre_visual,
                })
            time.sleep(0.05)

        log.info(f"  {genre_label:12s} → {len(genre_tracks)} tracks with preview")
        result.extend(genre_tracks)

    return result


def main() -> None:
    log.info("Fetching tracks from Deezer (no auth required)...")
    with httpx.Client(timeout=20, headers={"User-Agent": "synesthesia/0.1"}) as client:
        tracks = fetch_tracks(client)
    log.info(f"Total unique tracks with preview: {len(tracks)}")

    if not tracks:
        log.error("No tracks fetched — check network access to api.deezer.com.")
        sys.exit(1)

    for t in tracks:
        t["document"] = build_document(t["title"], t["artist"], t["genre_visual"])

    log.info("Sample documents:")
    for t in tracks[:3]:
        log.info(f"  → {t['document']}")

    if len(tracks) > TARGET_TOTAL:
        tracks = tracks[:TARGET_TOTAL]
    log.info(f"Encoding {len(tracks)} documents with CLIP (a few minutes on CPU)...")

    encoder = CLIPEncoder()
    embeddings: list[np.ndarray] = []
    for i, t in enumerate(tracks):
        embeddings.append(encoder.encode_text(t["document"]))
        if (i + 1) % 200 == 0:
            log.info(f"  {i + 1}/{len(tracks)} encoded")

    log.info("Inserting into database...")
    db = SessionLocal()
    try:
        rows = [
            {
                "title":       t["title"],
                "artist":      t["artist"],
                "genre":       t["genre"],
                "spotify_id":  t["external_id"],   # external (Deezer) id; column rename pending
                "preview_url": t["preview_url"],
                "document":    t["document"],
                "embedding":   emb.tolist(),
            }
            for t, emb in zip(tracks, embeddings)
        ]

        stmt = pg_insert(Song).values(rows).on_conflict_do_nothing(index_elements=["spotify_id"])
        db.execute(stmt)
        db.commit()
        log.info(f"Done — {len(rows)} songs inserted.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
