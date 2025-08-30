from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key: str
    db_name: str

    class Config:
        env_file = ".env"  # Optional if you want local dev
        env_file_encoding = "utf-8"

settings = Settings()
