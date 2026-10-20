import logging
import sys

def setup_logging(level=logging.INFO):
    """Sets up a standardized logger for the ai-learning project."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("ai_learning")
