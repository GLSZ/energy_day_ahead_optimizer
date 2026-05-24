from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).parent / "logs.env"

load_dotenv(env_path)

print(os.getenv("TEST"))