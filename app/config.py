from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    data_dir: str = "data/documents"
    index_dir: str = "data/index"
    top_k: int = 8
    rerank_top_k: int = 4
    min_evidence_chars: int = 80
    max_agent_attempts: int = 2

settings = Settings()
