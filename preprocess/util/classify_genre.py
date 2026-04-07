import pandas as pd
import numpy as np

# INITIAL IMPLEMENTATION
def categorize_genre(genre_string):
    # Standardize the text to lowercase to avoid case-sensitivity issues
    genre = str(genre_string).lower()
    
    # 1. Electronic / Dance
    if any(word in genre for word in ['house', 'techno', 'dubstep', 'edm', 'electro', 'trance', 'dance', 'phonk', 'bass', 'hardcore']):
        return 'Electronic & Dance'
        
    # 2. Hip Hop / Rap
    elif any(word in genre for word in ['rap', 'hip hop', 'drill', 'trap', 'grime', 'boom bap', 'crunk']):
        return 'Hip Hop & Rap'
        
    # 3. Metal / Hardcore
    # 4. Rock / Punk
    # APPENDED: Grouped Metal and Rock together to boost sample size
    elif any(word in genre for word in ['metal', 'screamo', 'djent', 'rock', 'punk', 'grunge', 'shoegaze', 'emo', 'britpop']):
        return 'Rock, Metal & Punk'
        
    # 5. Latin / Música Mexicana
    elif any(word in genre for word in ['reggaeton', 'corrido', 'banda', 'sierreño', 'norteño', 'latin', 'bachata', 'cumbia']):
        return 'Latin'
        
    # 6. R&B / Soul / Funk
    # 10. Jazz & Blues
    # APPENDED: Grouped R&B, Funk, Jazz, and Blues together
    elif any(word in genre for word in ['r&b', 'soul', 'funk', 'motown', 'jazz', 'blues', 'swing', 'big band']):
        return 'R&B, Soul & Jazz'
        
    # 7. Country / Americana
    # 11. Folk / Acoustic
    # APPENDED: Grouped Country and Folk together
    elif any(word in genre for word in ['country', 'bluegrass', 'americana', 'honky tonk', 'folk', 'acoustic', 'singer-songwriter', 'celtic', 'shanties']):
        return 'Country & Folk'
        
    # 14. Pop (Put this near the end)
    elif 'pop' in genre:
        return 'Pop'
        
    # 8. Reggae / Caribbean
    # 9. African
    # 12. Classical / Traditional
    # 13. Ambient / Chill
    # 15. Catch-all for what's left
    # APPENDED: Routed all low-support/zero-recall classes into the catch-all
    else:
        return 'Other'
    
    # genre_mapper.py

# Define the macro groups
MACRO_GROUPS = {
    "Soundtrack & Specialty": ["soundtrack", "medieval", "anime", "christmas", "nightcore", "classical", "neoclassical", "sea shanties", "chamber music", "musicals", "classical piano", "opera", "japanese vgm", "japanese classical", "gregorian chant", "comedy"],
    "Pop": ["pop", "soft pop", "art pop", "k-pop", "bedroom pop", "hyperpop", "baroque pop", "dream pop", "norwegian pop", "synthpop", "folk pop", "french pop", "europop", "dance pop", "c-pop", "indie pop", "j-pop", "pop rock"],
    "Hip Hop & Rap": ["rap", "hip hop", "emo rap", "cloud rap", "old school hip hop", "east coast hip hop", "rage rap", "melodic rap", "west coast hip hop", "gangster rap", "drill", "southern hip hop", "trap", "hardcore hip hop", "grime", "uk grime", "uk drill", "brooklyn drill", "new york drill", "crunk", "underground hip hop", "jazz rap", "chicago drill", "boom bap"],
    "Rock, Metal & Punk": ["rock", "nu metal", "classic rock", "grunge", "alternative rock", "alternative metal", "metal", "rap metal", "art rock", "industrial metal", "industrial rock", "indie", "psychedelic rock", "hard rock", "post-grunge", "emo", "garage rock", "soft rock", "punk", "indie rock", "progressive rock", "new wave", "shoegaze", "blues rock", "noise rock", "glam metal", "funk rock", "post-punk", "glam rock", "heavy metal", "pop punk", "skate punk", "thrash metal", "metalcore", "math rock", "djent"],
    "Country & Folk": ["country", "acoustic country", "classic country", "celtic", "pop country", "traditional country", "honky tonk", "outlaw country", "singer-songwriter", "folk", "americana", "bluegrass", "christian country", "red dirt", "texas country", "alt country", "country rock", "folk rock"],
    "Electronic & Dance": ["edm", "industrial", "slap house", "hypertechno", "tropical house", "electronic", "electro house", "witch house", "darkwave", "house", "big room", "dubstep", "synthwave", "electro", "electroclash", "future bass", "trance", "hard techno", "brazilian bass", "progressive house", "drum and bass", "deep house", "techno"],
    "R&B, Soul & Jazz": ["dark r&b", "r&b", "alternative r&b", "trap soul", "neo soul", "soul", "classic soul", "modern blues", "quiet storm", "jazz", "blues", "indie soul", "afro r&b", "new jack swing", "vocal jazz", "jazz blues"],
    "Ambient & Chill": ["lo-fi indie", "downtempo", "lo-fi", "lo-fi hip hop", "lo-fi beats", "drone", "vaporwave", "chillwave", "ambient", "chillstep", "lo-fi house", "space music"],
    "Latin": ["latin", "reggaeton", "urbano latino", "trap latino", "latin pop", "corridos bélicos", "corrido", "corridos tumbados", "música mexicana", "sierreño", "banda", "bachata", "sad sierreño", "electro corridos", "reggaeton mexa", "reggaeton chileno", "cumbia norteña", "norteño", "grupera", "tejano", "ranchera"],
    "MENA & South Asian": ["egyptian pop", "khaleeji", "mahraganat", "arabic hip hop", "egyptian hip hop", "sholawat", "egyptian shaabi", "desi hip hop", "malayalam hip hop", "hindi pop", "hindi indie", "desi", "desi pop", "bollywood", "moroccan pop", "gnawa"],
    "Afro & Caribbean": ["afrobeats", "afrobeat", "amapiano", "dancehall", "reggae", "roots reggae", "dub", "soca", "dembow", "dembow belico", "shatta", "azonto", "hiplife", "bongo flava", "gqom", "afro house", "afro tech", "afro adura", "afroswing", "nigerian drill"],
}

#  Invert dictionary globally so it only runs once when the file is imported
GENRE_MAP = {genre: macro for macro, genres in MACRO_GROUPS.items() for genre in genres}


# USE KEY INVERSIONS
def categorize_genre2(genre_string):
    """Takes a micro-genre string and returns the mapped macro-genre."""
    genre = str(genre_string).lower()

    # .get() looks up the key. If it doesn't exist, it safely returns the second argument ('Other')
    return GENRE_MAP.get(genre, "Other")

# genre_mapper.py



GENRE_MAP = {genre: macro for macro, genres in MACRO_GROUPS.items() for genre in genres}

# THIS ONE HSA THE BEST VALUE_COUNTS AND NORMALIZED SETUP
def categorize_genre3(genre_string):
    """Takes a micro-genre string and returns the mapped macro-genre using a hybrid approach."""
    genre = str(genre_string).lower()
    
    # STEP 1: Check the precise dictionary first
    if genre in GENRE_MAP:
        return GENRE_MAP[genre]
        
    # STEP 2: Keyword sweeper for the remaining long-tail micro-genres
    if any(word in genre for word in ['pop']):
        return 'Pop'
    elif any(word in genre for word in ['rock', 'metal', 'punk', 'emo', 'grunge', 'core']):
        return 'Rock, Metal & Punk'
    elif any(word in genre for word in ['rap', 'hip hop', 'drill', 'trap']):
        return 'Hip Hop & Rap'
    elif any(word in genre for word in ['house', 'techno', 'dubstep', 'electro', 'dance', 'edm', 'trance', 'bass']):
        return 'Electronic & Dance'
    elif any(word in genre for word in ['country', 'folk', 'acoustic']):
        return 'Country & Folk'
    elif any(word in genre for word in ['r&b', 'soul', 'jazz', 'blues', 'funk']):
        return 'R&B, Soul & Jazz'
    elif any(word in genre for word in ['latin', 'reggaeton', 'corrido', 'cumbia', 'mexa']):
        return 'Latin'
    elif any(word in genre for word in ['afro', 'dancehall', 'reggae', 'dub']):
        return 'Afro & Caribbean'
    elif any(word in genre for word in ['lo-fi', 'chill', 'ambient']):
        return 'Ambient & Chill'

    # STEP 3: True catch-all for completely un-guessable strings (e.g., 'shibuya-kei', 'mpb')
    return 'Other'