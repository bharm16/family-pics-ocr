import logging
import time
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class OCRErrorHandler:
    """Handle common OCR errors and edge cases"""

    @staticmethod
    def handle_api_errors(func: Callable):
        """Decorator for API error handling"""
        def wrapper(*args, **kwargs):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    msg = str(e).lower()
                    if "rate_limit" in msg:
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    elif "timeout" in msg and attempt < max_retries - 1:
                        logger.warning(f"Timeout, retrying... (attempt {attempt + 1})")
                        continue
                    else:
                        logger.error(f"API error: {e}")
                        raise
            return None
        return wrapper

    @staticmethod
    def validate_extraction(result: Dict) -> bool:
        """Validate extraction results"""
        if 'error' in result:
            return False
        if not result.get('extracted_elements'):
            return False
        if not result['extracted_elements'].get('all_text_lines'):
            logger.warning(f"No text found in {result.get('source_file', 'unknown')}")
        return True

