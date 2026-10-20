import ollama
from ai_learning.core.config import config
from ai_learning.core.logging import setup_logging

logger = setup_logging()

class OllamaClient:
    """Shared Ollama client wrapper used across different modules."""
    def __init__(self, host=config.OLLAMA_HOST):
        self.client = ollama.Client(host=host)
        logger.info(f"Initialized OllamaClient at {host}")

    def chat(self, model, messages):
        try:
            return self.client.chat(model=model, messages=messages)
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            raise

# Singleton instance for project-wide use
ollama_client = OllamaClient()
