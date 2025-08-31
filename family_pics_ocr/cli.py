import os
import json
import argparse
import logging

from .ocr import AdaptivePhotoOCR
from .processor import PhotoCollectionProcessor


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('photo_ocr.log'),
            logging.StreamHandler()
        ]
    )


def main():
    setup_logging()
    from anthropic import Anthropic

    parser = argparse.ArgumentParser(description="Family Pictures OCR with Claude API")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image to OCR")
    group.add_argument("--dir", type=str, help="Directory of images to process")

    parser.add_argument("--side", type=str, choices=["front", "back", "unknown"], default="unknown",
                        help="Side hint for single image OCR")
    parser.add_argument("--pairing", type=str, choices=["auto", "sequential", "single"], default="auto",
                        help="Pairing strategy when processing a directory")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022",
                        help="Claude model to use")
    parser.add_argument("--api-key", type=str, default=os.getenv("ANTHROPIC_API_KEY"),
                        help="Anthropic API key (or set ANTHROPIC_API_KEY)")

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Please provide ANTHROPIC_API_KEY via --api-key or environment variable.")

    client = Anthropic(api_key=args.api_key)
    ocr_engine = AdaptivePhotoOCR(client=client, model=args.model)
    processor = PhotoCollectionProcessor(ocr_engine)

    if args.image:
        result = ocr_engine.extract_text(args.image, side=args.side)
        if 'extracted_elements' in result:
            lines = result['extracted_elements'].get('all_text_lines', [])
            print("Extracted text lines:")
            for line in lines:
                print(f"  - {line}")
        else:
            print(json.dumps(result, indent=2))
        return

    # Directory mode
    df = processor.process_directory(args.dir, pairing_strategy=args.pairing)
    print(df.head().to_string(index=False))

