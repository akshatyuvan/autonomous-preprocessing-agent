"""
config.py — Central configuration loaded from environment variables.
Never hardcode API keys or paths. Always use this module.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file if present

# LLM
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
# Which endpoint serves the LLM. "github" = GitHub Models (free, rate-limited,
# OpenAI-compatible, same gpt-4o-mini); "openai" = OpenAI API (paid credits).
# Swapping provider is a .env change, never a code change.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")

# ChromaDB
CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "data_quality_patterns")

# Agent behaviour
MAX_CRITIC_ROUNDS: int = int(os.getenv("MAX_CRITIC_ROUNDS", "3"))

# Paths
DATA_DIR: str = os.getenv("DATA_DIR", "data")
RAW_DIR: str = f"{DATA_DIR}/raw"
CLEANED_DIR: str = f"{DATA_DIR}/cleaned"
CORRUPTED_DIR: str = f"{DATA_DIR}/corrupted"
