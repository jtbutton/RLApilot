from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    environment: str = "dev"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Postgres
    postgres_dsn: str = "postgresql+psycopg2://user:password@localhost/radiolab"

    # Chroma
    chroma_path: str = "./chroma_db"

    # LLM
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    embedding_model_name: str = "text-embedding-3-large"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 200

    class Config:
        env_prefix = "RADIOLAB_ATLAS_"
        env_file = ".env"
