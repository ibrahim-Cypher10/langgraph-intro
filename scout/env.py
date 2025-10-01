from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables from .env.


SUPABASE_URL=os.getenv("SUPABASE_URL", None)
GROQ_API_KEY=os.getenv("GROQ_API_KEY", None)
GROQ_MODEL=os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_TEMPERATURE=float(os.getenv("GROQ_TEMPERATURE", "0.0"))
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY", None)
REDIS_URI=os.getenv("REDIS_URI", None)
DATABASE_URI=os.getenv("DATABASE_URI", None)


required_env_vars = [
    "SUPABASE_URL",
    "GROQ_API_KEY",
]

for var in required_env_vars:
    if not var:
        raise ValueError(f"Missing required environment variable: {var}")