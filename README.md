Family Pictures OCR (Claude API)

Overview
- Adaptive OCR pipeline to extract all visible text from family photo fronts/backs using OpenAI vision (gpt-4o family).
- Learns recurring patterns (codes, dates) and exports results to CSV/Excel/JSON with a pattern report.

Quick Start
- Python 3.9+
- Set env var `OPENAI_API_KEY` to your OpenAI API key.
- Option A (recommended): install package for CLI
  - `pip install -e .`
  - Run: `family-pics-ocr --image photos/test.jpg --side back`
- Option B: use scripts directly
  - Install deps: `pip install -r requirements.txt`
  - Run: `python main.py --dir photos --pairing auto`

CLI Usage
- Single image: `family-pics-ocr --image photos/test.jpg --side back`
- Directory (auto front/back pairing): `family-pics-ocr --dir photos --pairing auto`
- Sequential pairing: `family-pics-ocr --dir photos --pairing sequential`
- Single per file: `family-pics-ocr --dir photos --pairing single`

Outputs
- `ocr_results/photo_metadata.csv|xlsx`: flattened metadata per photo
- `ocr_results/detailed_results.json`: full raw + analysis per side
- `ocr_results/pattern_analysis.json`: discovered code/date patterns summary

Notes
- Optional OpenCV preprocessing can be enabled by installing `opencv-python` and `numpy`.
- The pipeline keeps prompts simple and open-ended to capture every text element without assumptions.
 
Environment
- Create `.env` or export: `export OPENAI_API_KEY=your-key`
- Default model: `gpt-4o` (configurable via `--model`, e.g., `gpt-4o-mini`)
