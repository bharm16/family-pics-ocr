import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .ocr import AdaptivePhotoOCR, PatternLibrary

logger = logging.getLogger(__name__)


class PhotoCollectionProcessor:
    """Process entire photo collections with pattern discovery"""

    def __init__(self, ocr_engine: AdaptivePhotoOCR):
        self.ocr_engine = ocr_engine
        self.pattern_library = PatternLibrary()
        self.results_db: List[Dict] = []

    def process_photo_pair(self, front_path: Optional[str], back_path: Optional[str],
                           photo_id: str = None) -> Dict:
        """
        Process both sides of a photo

        Args:
            front_path: Path to front image (optional)
            back_path: Path to back image (optional)
            photo_id: Identifier for this photo pair

        Returns:
            Combined results from both sides
        """
        if not photo_id:
            key = f"{front_path or ''}{back_path or ''}"
            photo_id = str(abs(hash(key)))[:8]

        result: Dict = {
            'photo_id': photo_id,
            'front': None,
            'back': None,
            'combined_metadata': {}
        }

        # Process front
        if front_path:
            logger.info(f"Processing front: {front_path}")
            front_result = self.ocr_engine.extract_text(front_path, side='front')
            front_analysis = self.pattern_library.analyze_and_learn(
                front_result.get('extracted_elements', {})
            )
            result['front'] = {
                'raw': front_result,
                'analysis': front_analysis
            }

        # Process back
        if back_path:
            logger.info(f"Processing back: {back_path}")
            back_result = self.ocr_engine.extract_text(back_path, side='back')
            back_analysis = self.pattern_library.analyze_and_learn(
                back_result.get('extracted_elements', {})
            )
            result['back'] = {
                'raw': back_result,
                'analysis': back_analysis
            }

        # Combine metadata from both sides
        result['combined_metadata'] = self._merge_metadata(result)

        return result

    def process_directory(self, directory: str, pairing_strategy: str = 'auto') -> pd.DataFrame:
        """
        Process all photos in a directory

        Args:
            directory: Path to directory containing photos
            pairing_strategy: How to pair fronts/backs: 'auto', 'sequential', 'single'

        Returns:
            DataFrame with all results
        """
        photo_files = self._scan_directory(directory)
        pairs = self._create_pairs(photo_files, pairing_strategy)

        results: List[Dict] = []
        for pair_info in tqdm(pairs, desc="Processing photos"):
            result = self.process_photo_pair(
                pair_info.get('front'),
                pair_info.get('back'),
                pair_info.get('id')
            )
            results.append(result)
            self.results_db.append(result)

        # Convert to DataFrame
        df = self._results_to_dataframe(results)

        # Save results
        self._save_results(df, directory)

        # Generate pattern report
        self._generate_pattern_report(directory)

        return df

    def _scan_directory(self, directory: str) -> List[str]:
        """Scan directory for image files"""
        extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif'}
        photo_files: List[str] = []

        for file in Path(directory).iterdir():
            name = file.name
            # Skip macOS resource fork files and hidden files
            if name.startswith('._') or name.startswith('.'):
                continue
            if file.suffix.lower() in extensions:
                photo_files.append(str(file))

        return sorted(photo_files)

    def _create_pairs(self, files: List[str], strategy: str) -> List[Dict]:
        """Create front/back pairs based on strategy"""
        pairs = []

        if strategy == 'auto':
            # Look for patterns like IMG_001_front.jpg, IMG_001_back.jpg
            grouped = defaultdict(dict)
            for file in files:
                base = Path(file).stem
                lower = base.lower()
                if 'front' in lower:
                    key = lower.replace('front', '').replace('_', '').strip()
                    grouped[key]['front'] = file
                elif 'back' in lower:
                    key = lower.replace('back', '').replace('_', '').strip()
                    grouped[key]['back'] = file
                else:
                    grouped[base]['single'] = file

            for key, group in grouped.items():
                if 'single' in group:
                    pairs.append({'id': key, 'front': group['single'], 'back': None})
                else:
                    pairs.append({'id': key, 'front': group.get('front'), 'back': group.get('back')})

        elif strategy == 'sequential':
            # Assume alternating front/back
            for i in range(0, len(files), 2):
                pairs.append({
                    'id': f"photo_{i//2:04d}",
                    'front': files[i],
                    'back': files[i+1] if i+1 < len(files) else None
                })

        else:  # single
            for i, file in enumerate(files):
                pairs.append({'id': f"photo_{i:04d}", 'front': file, 'back': None})

        return pairs

    def _merge_metadata(self, result: Dict) -> Dict:
        """Merge metadata from front and back"""
        metadata = {
            'all_text': [],
            'codes': [],
            'dates': [],
            'names': [],
            'locations': []
        }

        # Collect from both sides
        for side in ['front', 'back']:
            if result.get(side):
                elements = result[side]['raw'].get('extracted_elements', {})
                metadata['all_text'].extend(elements.get('all_text_lines', []))

                analysis = result[side]['analysis']
                for text, pattern_info in analysis.get('identified_patterns', {}).items():
                    if pattern_info['type'] == 'code':
                        metadata['codes'].append(pattern_info)
                    elif pattern_info['type'] == 'date':
                        metadata['dates'].append(pattern_info)

        return metadata

    def _results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        rows = []
        for result in results:
            row = {
                'photo_id': result['photo_id'],
                'has_front': result['front'] is not None,
                'has_back': result['back'] is not None,
                'all_text': ' | '.join(result['combined_metadata'].get('all_text', [])),
                'codes': json.dumps(result['combined_metadata'].get('codes', [])),
                'dates': json.dumps(result['combined_metadata'].get('dates', [])),
                'front_file': result['front']['raw']['source_file'] if result.get('front') else None,
                'back_file': result['back']['raw']['source_file'] if result.get('back') else None,
                'timestamp': datetime.now().isoformat()
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def _save_results(self, df: pd.DataFrame, directory: str):
        """Save results to multiple formats"""
        output_dir = Path(directory) / 'ocr_results'
        output_dir.mkdir(exist_ok=True)

        df.to_csv(output_dir / 'photo_metadata.csv', index=False)
        try:
            df.to_excel(output_dir / 'photo_metadata.xlsx', index=False)
        except Exception as e:
            logger.warning(f"Failed to write Excel file: {e}")

        with open(output_dir / 'detailed_results.json', 'w') as f:
            json.dump(self.results_db, f, indent=2, default=str)

        logger.info(f"Results saved to {output_dir}")

    def _generate_pattern_report(self, directory: str):
        """Generate report of discovered patterns"""
        output_dir = Path(directory) / 'ocr_results'

        report = {
            'total_photos_processed': len(self.results_db),
            'discovered_patterns': dict(self.pattern_library.discovered_patterns),
            'pattern_frequency': {},
            'recommendations': []
        }

        # Count pattern frequencies
        all_patterns: List[str] = []
        for pattern_list in self.pattern_library.discovered_patterns.values():
            all_patterns.extend(pattern_list)

        pattern_counts = Counter(all_patterns)
        report['pattern_frequency'] = dict(pattern_counts.most_common(20))

        if pattern_counts:
            report['recommendations'].append(
                f"Most common pattern: {pattern_counts.most_common(1)[0][0]}"
            )

        with open(output_dir / 'pattern_analysis.json', 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Pattern report saved to {output_dir / 'pattern_analysis.json'}")
