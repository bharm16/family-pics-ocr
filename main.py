import os
import json
import argparse
import logging
from pathlib import Path

from typing import Optional

from family_pics_ocr.ocr import AdaptivePhotoOCR
from family_pics_ocr.processor import PhotoCollectionProcessor


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
    from openai import OpenAI

    parser = argparse.ArgumentParser(description="Family Pictures OCR with OpenAI (gpt-4o) API")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image to OCR")
    group.add_argument("--dir", type=str, help="Directory of images to process")

    parser.add_argument("--side", type=str, choices=["front", "back", "unknown"], default="unknown",
                        help="Side hint for single image OCR")
    parser.add_argument("--pairing", type=str, choices=["auto", "sequential", "single"], default="auto",
                        help="Pairing strategy when processing a directory")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model to use (vision capable, e.g., gpt-4o, gpt-4o-mini)")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"),
                        help="OpenAI API key (or set OPENAI_API_KEY)")

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Please provide OPENAI_API_KEY via --api-key or environment variable.")

    client = OpenAI(api_key=args.api_key)
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


if __name__ == "__main__":
    main()
