"""Application configuration using pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Neo4jSettings(BaseSettings):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "temporal_kb_2026"

    model_config = {"env_prefix": "NEO4J_"}


class OpenAISettings(BaseSettings):
    api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_model: str = "gpt-4o-mini"

    model_config = {"env_prefix": "OPENAI_"}


class QwenSettings(BaseSettings):
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    max_new_tokens: int = 2048
    temperature: float = 0.7

    model_config = {"env_prefix": "QWEN_"}


class TrainingSettings(BaseSettings):
    learning_rate: float = 1e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    grpo_group_size: int = 4
    epochs: int = 3
    kl_coef: float = 0.001
    max_prompt_length: int = 4096
    max_completion_length: int = 2048
    gradient_checkpointing: bool = True
    output_dir: str = "checkpoints"

    model_config = {"env_prefix": "TRAIN_"}


class AppSettings(BaseSettings):
    neo4j: Neo4jSettings = Neo4jSettings()
    openai: OpenAISettings = OpenAISettings()
    qwen: QwenSettings = QwenSettings()
    training: TrainingSettings = TrainingSettings()

    log_level: str = "INFO"
    graph_memory_max_tokens: int = 2048
    similarity_threshold: float = 0.7
    max_search_results: int = 10

    model_config = {"env_prefix": "APP_"}


def get_settings() -> AppSettings:
    """Create settings instance loading from environment."""
    return AppSettings()
