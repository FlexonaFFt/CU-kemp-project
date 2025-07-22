import logging
import warnings
import os


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def suppress_warnings():
    # Подавляет FutureWarning от transformers
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    
    # Подавляет UserWarning от torch о NumPy
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    
    # Подавляет urllib3 warnings
    warnings.filterwarnings("ignore", module="urllib3")
    
    # Альтернативно, можно полностью отключить все warnings
    # warnings.filterwarnings("ignore")


def setup_clean_environment():
    # Устанавливает переменные окружения для подавления warnings
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    
    # Подавляет warnings
    suppress_warnings()


def setup_logging(level: int = logging.INFO, clean_env: bool = True):
    """Настраивает логирование с опциональным подавлением warnings"""
    if clean_env:
        setup_clean_environment()
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    root_logger.addHandler(handler)
    return root_logger


app_logger = setup_logging()
