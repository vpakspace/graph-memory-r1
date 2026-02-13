"""Shared fixtures for tests.

Requires:
- Neo4j running at bolt://localhost:7687
- OPENAI_API_KEY set in .env or environment
"""

from __future__ import annotations

import os
import socket
from unittest.mock import AsyncMock, MagicMock

import pytest
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from config import AppSettings, Neo4jSettings, OpenAISettings, QwenSettings, TrainingSettings


def _neo4j_available() -> bool:
    try:
        sock = socket.create_connection(("localhost", 7687), timeout=2)
        sock.close()
        return True
    except OSError:
        return False


def _openai_available() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key and key.startswith("sk-"))


skip_no_neo4j = pytest.mark.skipif(
    not _neo4j_available(), reason="Neo4j not available at localhost:7687"
)

skip_no_openai = pytest.mark.skipif(
    not _openai_available(), reason="OPENAI_API_KEY not configured"
)


@pytest.fixture(scope="session")
def neo4j_settings() -> Neo4jSettings:
    return Neo4jSettings()


@pytest.fixture(scope="session")
def openai_settings() -> OpenAISettings:
    return OpenAISettings()


@pytest.fixture(scope="session")
def qwen_settings() -> QwenSettings:
    return QwenSettings()


@pytest.fixture(scope="session")
def training_settings() -> TrainingSettings:
    return TrainingSettings()


@pytest.fixture(scope="session")
def app_settings() -> AppSettings:
    return AppSettings()


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for unit tests."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for unit tests."""
    client = MagicMock()
    # Mock embeddings
    embedding_response = MagicMock()
    embedding_data = MagicMock()
    embedding_data.embedding = [0.1] * 1536
    embedding_response.data = [embedding_data]
    client.embeddings.create.return_value = embedding_response
    # Mock chat completions
    chat_response = MagicMock()
    choice = MagicMock()
    choice.message.content = "Test answer"
    chat_response.choices = [choice]
    client.chat.completions.create.return_value = chat_response
    return client
