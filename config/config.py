"""
Configuração centralizada do projeto Assistente Virtual Médico
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Configurações do sistema com validação Pydantic"""
    
    # API Keys
    huggingfacehub_api_token: str
    google_api_key: str
    
    # Model Configuration
    llama_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    gemini_model_name: str = "gemini-pro"
    
    # RAG Configuration
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_type: str = "faiss"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/assistente_medico.log"
    
    # Paths
    data_raw_path: str = "data/raw"
    data_processed_path: str = "data/processed"
    vectorstore_path: str = "data/vectorstore"
    
    # Data Source
    medquad_repo_url: str = "https://github.com/abachaa/MedQuAD/tree/master/1_CancerGov_QA"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Retorna instância única das configurações"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Caminhos do projeto
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
LOGS_DIR = BASE_DIR / "logs"
