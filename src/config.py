from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLATFORM_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    source: str = "demo"
    api_url: str = "http://localhost:8000"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "platform.duckdb"


settings = Settings()

