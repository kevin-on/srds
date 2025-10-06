"""
Logging utility for SRDS/SParareal algorithms.
Provides functionality to log to both console and file simultaneously.
"""

import logging
import sys
from datetime import datetime
from typing import Optional


class DualLogger:
    """Logger that writes to both console and file simultaneously."""

    def __init__(self, log_file: Optional[str] = None, log_level: int = logging.INFO):
        """
        Initialize dual logger.

        Args:
            log_file: Path to log file. If None, uses 'log.txt' in current directory.
            log_level: Logging level (default: INFO)
        """
        if log_file is None:
            log_file = "log.txt"

        self.log_file = log_file

        # Create logger
        self.logger = logging.getLogger("srds_logger")
        self.logger.setLevel(log_level)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create formatters
        console_formatter = logging.Formatter("%(message)s")
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Log session start
        self.info("=" * 60)
        self.info(f"New logging session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("=" * 60)

    def info(self, message: str):
        """Log info level message."""
        self.logger.info(message)

    def debug(self, message: str):
        """Log debug level message."""
        self.logger.debug(message)

    def warning(self, message: str):
        """Log warning level message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error level message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical level message."""
        self.logger.critical(message)


class TqdmDualLogger:
    """Wrapper for tqdm.write that also logs to file."""

    def __init__(self, dual_logger: DualLogger):
        self.dual_logger = dual_logger

    def write(self, message: str):
        """Write message using tqdm-compatible logging."""
        # For tqdm compatibility, we'll use the logger directly
        # Remove any trailing newlines as logger adds them
        message = message.rstrip("\n")
        self.dual_logger.info(message)


# Global logger instance
_global_logger: Optional[DualLogger] = None
_global_tqdm_logger: Optional[TqdmDualLogger] = None


def setup_logging(log_file: Optional[str] = None, log_level: int = logging.INFO) -> DualLogger:
    """
    Setup global logging configuration.

    Args:
        log_file: Path to log file. If None, uses 'log.txt' in current directory.
        log_level: Logging level

    Returns:
        DualLogger instance
    """
    global _global_logger, _global_tqdm_logger

    _global_logger = DualLogger(log_file, log_level)
    _global_tqdm_logger = TqdmDualLogger(_global_logger)

    return _global_logger


def get_logger() -> DualLogger:
    """Get the global logger instance."""
    if _global_logger is None:
        return setup_logging()
    return _global_logger


def get_tqdm_logger() -> TqdmDualLogger:
    """Get the global tqdm logger instance."""
    if _global_tqdm_logger is None:
        setup_logging()
    return _global_tqdm_logger


def log_info(message: str):
    """Convenience function to log info message."""
    get_logger().info(message)


def log_debug(message: str):
    """Convenience function to log debug message."""
    get_logger().debug(message)


def log_warning(message: str):
    """Convenience function to log warning message."""
    get_logger().warning(message)


def log_error(message: str):
    """Convenience function to log error message."""
    get_logger().error(message)


def log_critical(message: str):
    """Convenience function to log critical message."""
    get_logger().critical(message)


# Context manager for temporary log file
class LogToFile:
    """Context manager to temporarily redirect logging to a specific file."""

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.old_logger = None

    def __enter__(self):
        global _global_logger, _global_tqdm_logger
        self.old_logger = _global_logger
        setup_logging(self.log_file)
        return get_logger()

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _global_logger, _global_tqdm_logger
        _global_logger = self.old_logger
        if self.old_logger is not None:
            _global_tqdm_logger = TqdmDualLogger(self.old_logger)
        else:
            _global_tqdm_logger = None


if __name__ == "__main__":
    # Demo usage
    logger = setup_logging("demo_log.txt")

    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test convenience functions
    log_info("Using convenience function")

    # Test context manager
    with LogToFile("temp_log.txt"):
        log_info("This goes to temp_log.txt")

    log_info("Back to original log file")
