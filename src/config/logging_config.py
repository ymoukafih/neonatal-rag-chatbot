"""
Centralized logging configuration for the entire application.
Call setup_logging() once at startup — all modules inherit the config.
"""
import logging
import logging.config
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structured logging with console + rotating file output.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR. Read from settings.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": (
                    "%(asctime)s | %(levelname)-8s | "
                    "%(name)s:%(lineno)d | %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {
                "format": "%(levelname)s | %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "detailed",
                "filename": "logs/app.log",
                "maxBytes": 10 * 1024 * 1024,  # 10MB per file
                "backupCount": 5,               # keep last 5 files
                "encoding": "utf-8",
            },
        },
        "loggers": {
            # Your app — show everything
            "src": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            # Third-party — only warnings and above
            "langchain": {"level": "WARNING", "handlers": ["file"], "propagate": False},
            "chromadb":  {"level": "WARNING", "handlers": ["file"], "propagate": False},
            "httpx":     {"level": "WARNING", "handlers": ["file"], "propagate": False},
            "gradio":    {"level": "WARNING", "handlers": ["file"], "propagate": False},
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"],
        },
    }

    logging.config.dictConfig(config)
    logging.getLogger("src").info(
        "Logging initialized — level=%s, file=logs/app.log", log_level
    )