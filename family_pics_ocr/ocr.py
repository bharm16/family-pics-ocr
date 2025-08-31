import base64
import io
import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageEnhance
import time, random
from dateutil import parser as date_parser
from datetime import datetime as dt

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image optimization for OCR processing"""

    @staticmethod
    def optimize_image(image_path: str, max_size: Tuple[int, int] = (2048, 2048)) -> bytes:
        try:
            with Image.open(image_path) as img:
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=95, optimize=True)
                return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise

    @staticmethod
    def encode_image(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode('utf-8')


class PatternLibrary:
    """Learns and stores patterns from processed photos"""

    def __init__(self):
        self.discovered_patterns = defaultdict(list)
        self.code_formats: List[str] = []
        self.date_formats: List[str] = []

    def analyze_and_learn(self, extracted_elements: Dict) -> Dict:
        analysis = {
            'identified_patterns': {},
            'confidence_scores': {},
            'suggested_categories': []
        }

        # Codes
        for code in extracted_elements.get('potential_codes', []):
            pattern = self._derive_pattern(code)
            if pattern:
                self.discovered_patterns['codes'].append(pattern)
                analysis['identified_patterns'][code] = {
                    'type': 'code',
                    'pattern': pattern,
                    'components': self._decompose_code(code)
                }

        # Dates
        for date_str in extracted_elements.get('potential_dates', []):
            parsed_date = self._flexible_date_parse(date_str)
            if parsed_date:
                analysis['identified_patterns'][date_str] = {
                    'type': 'date',
                    'parsed': parsed_date,
                    'original': date_str
                }

        return analysis

    def _derive_pattern(self, text: str) -> str:
        pattern = ""
        for char in text:
            if char.isalpha():
                pattern += 'A' if char.isupper() else 'a'
            elif char.isdigit():
                pattern += 'D'
            elif char in '<>()[]{}':
                pattern += char
            elif char in '-_/':
                pattern += char
            else:
                pattern += '?'
        return pattern

    def _decompose_code(self, code: str) -> List[str]:
        components = re.split(r'([<>\(\)\[\]\-/\s]+)', code)
        return [c for c in components if c.strip()]

    def _flexible_date_parse(self, date_str: str) -> Optional[Dict]:
        # Special: dd m'yy (e.g., 26 6'95)
        m = re.search(r"\b(\d{1,2})\s+(\d{1,2})'(\d{2})\b", date_str)
        if m:
            d, mo, yy = map(int, m.groups())
            century = 1900 if yy >= 30 else 2000
            year = century + yy
            try:
                parsed = dt(year, mo, d)
                return {'year': parsed.year, 'month': parsed.month, 'day': parsed.day, 'iso': parsed.isoformat()}
            except Exception:
                pass
        # General fallback
        try:
            parsed = date_parser.parse(date_str, fuzzy=True)
            return {'year': parsed.year, 'month': parsed.month or None, 'day': parsed.day or None, 'iso': parsed.isoformat()}
        except Exception:
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                year = int(year_match.group())
                return {'year': year, 'month': None, 'day': None, 'iso': f"{year}-01-01"}
        return None


class AdaptivePhotoOCR:
    """Flexible OCR engine that adapts to any text format found"""

    def __init__(self, client, model: str = "claude-sonnet-4-20250514", request_max_retries: int = 6,
                 backoff_base: float = 0.8, backoff_max: float = 30.0):
        self.client = client
        self.model = model
        self.preprocessor = ImagePreprocessor()
        self.pattern_library = PatternLibrary()
        self.request_max_retries = request_max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max

    def extract_text(self, image_path: str, side: str = "unknown") -> Dict:
        try:
            image_bytes = self.preprocessor.optimize_image(image_path)
            base64_image = self.preprocessor.encode_image(image_bytes)
            prompt = self._create_adaptive_prompt(side)

            # Use configured model with retry on overload/timeouts
            last_err = None
            response = None
            for attempt in range(self.request_max_retries):
                try:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=2048,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": base64_image
                                        }
                                    },
                                    {"type": "text", "text": prompt}
                                ]
                            }
                        ],
                        temperature=0.1
                    )
                    break
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
                    if any(tok in msg for tok in ("overloaded", "529", "timeout", "temporarily unavailable")) and attempt < self.request_max_retries - 1:
                        delay = min(self.backoff_max, self.backoff_base * (2 ** attempt))
                        delay = delay * (0.5 + random.random())
                        logger.info(f"Retrying after {delay:.2f}s due to transient error (attempt {attempt+1}/{self.request_max_retries})")
                        time.sleep(delay)
                        continue
                    raise
            if response is None:
                raise last_err if last_err else RuntimeError("Claude API call failed without error")

            raw_response = response.content[0].text if getattr(response, 'content', None) else str(response)
            result = {
                'source_file': image_path,
                'side': side,
                'raw_response': raw_response,
                'timestamp': datetime.now().isoformat(),
                'extracted_elements': self._parse_raw_response(raw_response)
            }
            return result
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            return {'source_file': image_path, 'side': side, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    def _create_adaptive_prompt(self, side: str) -> str:
        if side == "front":
            return (
                "Look at this photo and extract ALL visible text exactly as shown. Include:"
                "\n- Any captions or titles"
                "\n- Text on borders or frames"
                "\n- Stamps or watermarks"
                "\n- Any printed or handwritten text visible on the photo itself"
                "\n- Studio or photographer marks"
                "\n- Any numbers, codes, or identifiers\n\n"
                "Return ONLY text elements, no explanations or intros."
                " List each on a new line. Optional short location prefix like 'top left corner:' is allowed."
            )
        elif side == "back":
            return (
                "Look at this photo back and extract ALL visible text exactly as shown. Include:"
                "\n- Any handwritten notes or annotations"
                "\n- Printed stamps or marks"
                "\n- Codes, numbers, or identifiers (preserve exact brackets/dashes)"
                "\n- Dates in any format"
                "\n- Names or descriptions"
                "\n- Developer/processor marks\n\n"
                "Return ONLY text elements, no explanations or intros. List each on a new line."
            )
        else:
            return (
                "Extract ALL visible text exactly as shown, preserving characters and spacing."
                " Return ONLY text elements (no explanations). List each on a new line."
            )

    def _parse_raw_response(self, raw_response: str) -> Dict:
        lines = raw_response.strip().split('\n') if raw_response else []

        elements = {
            'all_text_lines': [],
            'location_tagged': defaultdict(list),
            'potential_codes': [],
            'potential_dates': [],
            'potential_names': [],
            'numeric_sequences': [],
            'special_patterns': []
        }

        disallowed_prefixes = [
            'here is the text', 'here’s the text', 'the rest of the image',
            'no other text is visible', 'that is the only', 'the image appears', 'this image'
        ]
        code_regexes = [
            re.compile(r"^[A-Z]{1,4}\d{2,4}-\d{2,4}\s+[<«][^>»]{1,4}[>»]\s+[A-Z]{2,4}$"),
            re.compile(r"^[A-Z]{2,4}\s+[«<][^>»]{1,4}[>»]\s+\d{2,4}-\d{2,4}$"),
        ]
        date_compact_re = re.compile(r"\b(\d{1,2})\s+(\d{1,2})'(\d{2})\b")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            line = line.lstrip('-• ').strip()
            if any(line.lower().startswith(pfx) for pfx in disallowed_prefixes):
                continue

            elements['all_text_lines'].append(line)

            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    location = parts[0].lower()
                    text = parts[1].strip()
                    elements['location_tagged'][location].append(text)

            if any(ch in line for ch in ['<', '>', '«', '»', '(', ')', '[', ']', '#']):
                elements['potential_codes'].append(line)
            elif re.search(r'[A-Z]{2,}.*\d+|^\d+[A-Z]+', line):
                elements['potential_codes'].append(line)

            if re.search(r'\d{1,4}[-/\s]\d{1,2}[-/\s]\d{1,4}|\b\d{4}\b', line):
                elements['potential_dates'].append(line)
            if date_compact_re.search(line):
                elements['potential_dates'].append(line)

            if re.search(r'\d{3,}', line):
                elements['numeric_sequences'].append(line)
            if re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', line):
                elements['potential_names'].append(line)

            for cre in code_regexes:
                if cre.match(line):
                    elements['potential_codes'].append(line)
                    elements['special_patterns'].append({'type': 'code', 'value': line})
                    break

        return elements
