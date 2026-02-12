"""Retry with exponential backoff — no external dependencies."""

import logging
import time

logger = logging.getLogger(__name__)


def with_retry(fn, max_attempts=3, backoff_base=2.0, backoff_max=300.0, logger=logger):
    """Call fn() with exponential backoff on exception.

    Returns the result of fn() on success, or re-raises the last exception
    after max_attempts failures.
    """
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_attempts - 1:
                raise
            delay = min(backoff_base ** attempt, backoff_max)
            if logger:
                logger.warning(
                    "Attempt %d/%d failed: %s — retrying in %.0fs",
                    attempt + 1, max_attempts, exc, delay,
                )
            time.sleep(delay)
