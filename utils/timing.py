"""
Timing utility for tracking wall-clock execution overhead.
Used to measure training duration and populate report artifacts.
"""

import time
from contextlib import contextmanager
from typing import Generator, Optional

class Timer:
    """Helper class to track elapsed wall-clock time."""
    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_seconds: float = 0.0

    def start(self) -> None:
        """Start or restart the timer."""
        self.start_time = time.time()
        self.end_time = 0.0
        self.elapsed_seconds = 0.0

    def stop(self) -> float:
        """Stop the timer and record elapsed seconds.

        Returns:
            Elapsed time in seconds.
        """
        self.end_time = time.time()
        self.elapsed_seconds = self.end_time - self.start_time
        return self.elapsed_seconds

    @property
    def elapsed_hours(self) -> float:
        """Returns elapsed time in wall-clock hours."""
        return self.elapsed_seconds / 3600.0

@contextmanager
def timer() -> Generator[Timer, None, None]:
    """Context manager yielding a started Timer instance.

    Yields:
        Active Timer instance.
    """
    t = Timer()
    t.start()
    try:
        yield t
    finally:
        t.stop()
