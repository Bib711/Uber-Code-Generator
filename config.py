# Configuration management with environment variables
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables"""
    
    # API Keys for AI Agents
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    
    # Google OAuth
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    
    # MongoDB
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "uber_code_generator")
    
    # Server
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "jwt-secret-key")
    PORT: int = int(os.getenv("PORT", "5000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # AI Model Configuration
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    MISTRAL_MODEL: str = "mistral-large-latest"
    QWEN_MODEL: str = "qwen-plus"
    
    @classmethod
    def get_api_key(cls, provider: str) -> str:
        """Get API key for a specific provider"""
        keys = {
            "groq": cls.GROQ_API_KEY,
            "mistral": cls.MISTRAL_API_KEY,
            "qwen": cls.QWEN_API_KEY,
        }
        return keys.get(provider.lower(), "")
    
    @classmethod
    def is_valid_key(cls, key: str) -> bool:
        """Check if an API key is valid (non-empty and reasonable length)"""
        return bool(key) and len(key) > 10 and key != "your-groq-api-key-here"


settings = Settings()
