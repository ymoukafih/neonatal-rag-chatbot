from src.config.logging_config import setup_logging
from src.config.settings import get_settings
from src.ui.app import launch

if __name__ == "__main__":
    settings = get_settings()
    setup_logging(settings.log_level)
    launch()