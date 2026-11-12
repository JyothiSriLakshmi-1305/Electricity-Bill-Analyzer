import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
HISTORICAL_DIR = DATA_DIR / "historical"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"

VECTOR_DB_DIR = BASE_DIR / "vector_db"

# --------------------------------------------------
# ENVIRONMENT CONFIGURATION
# --------------------------------------------------

load_dotenv(BASE_DIR / ".env")

APP_NAME = os.getenv(
    "APP_NAME",
    "AI Electricity Bill Analyzer & Energy Assistant"
)

DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# --------------------------------------------------
# GEMINI CONFIGURATION
# --------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

# --------------------------------------------------
# EMBEDDING CONFIGURATION
# --------------------------------------------------

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

EMBEDDING_DEVICE = "cpu"
EMBEDDING_DIMENSION = 384

# --------------------------------------------------
# DOCUMENT CHUNKING
# --------------------------------------------------

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --------------------------------------------------
# RETRIEVAL CONFIGURATION
# --------------------------------------------------

TOP_K = 3
SIMILARITY_THRESHOLD = 0.30

# --------------------------------------------------
# FILE CONFIGURATION
# --------------------------------------------------

HISTORICAL_DATA_FILE = PROCESSED_DIR / "historical_bills.json"

EXTRACTED_DATA_FILE = PROCESSED_DIR / "extracted_bills_draft.json"

FAISS_INDEX_FILE = VECTOR_DB_DIR / "electricity_bills.index"

CHUNKS_METADATA_FILE = VECTOR_DB_DIR / "chunks_metadata.json"

# --------------------------------------------------
# APPLICATION SETTINGS
# --------------------------------------------------

MAX_UPLOAD_SIZE_MB = 10

ALLOWED_FILE_TYPES = ["pdf"]

# --------------------------------------------------
# DIRECTORY INITIALIZATION
# --------------------------------------------------

for directory in [
    UPLOAD_DIR,
    PROCESSED_DIR,
    VECTOR_DB_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)