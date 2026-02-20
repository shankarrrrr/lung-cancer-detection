"""
Batch DICOM preprocessing script for all datasets.
"""
import argparse
from pathlib import Path
from loguru import logger
import sys

from src.data.dicom_converter import DICOMConverter
from src.config import config

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/preprocessing.log", rotation="500 MB")


def preprocess_dataset(
    dataset_name: str,
    input_dir: Path,
    output_dir: Path
) -> dict:
    """
    Preprocess a single dataset.
    
    Args:
        dataset_name: Name of the dataset
        input_dir: Input directory with DICOM files
        output_dir: Output directory for processed PNGs
        
    Returns:
        Statistics dictionary
    """
    logger.info(f"=" * 60)
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"=" * 60)
    
    converter = DICOMConverter()
    stats = converter.batch_convert(input_dir, output_dir)
    
    logger.info(f"\n{dataset_name} Statistics:")
    logger.info(f"  Success: {stats['success']}")
    logger.info(f"  Failed: {stats['failed']}")
    
    if stats['failed'] > 0:
        logger.warning(f"  Failed files: {stats['failed_files'][:10]}...")  # Show first 10
    
    return stats


def main():
    """Main preprocessing pipeline"""
    parser = argparse.ArgumentParser(description="Preprocess DICOM datasets")
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['nih', 'chexpert', 'vindr', 'mimic'],
        help='Datasets to process'
    )
    parser.add_argument(
        '--raw-dir',
        type=Path,
        default=config.data.raw_dir,
        help='Raw data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=config.data.processed_dir,
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting batch preprocessing...")
    logger.info(f"Datasets: {args.datasets}")
    
    all_stats = {}
    
    for dataset_name in args.datasets:
        input_dir = args.raw_dir / dataset_name
        output_dir = args.output_dir / dataset_name
        
        if not input_dir.exists():
            logger.warning(f"Skipping {dataset_name}: directory not found at {input_dir}")
            continue
        
        try:
            stats = preprocess_dataset(dataset_name, input_dir, output_dir)
            all_stats[dataset_name] = stats
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {str(e)}")
            all_stats[dataset_name] = {'success': 0, 'failed': 0, 'error': str(e)}
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    
    total_success = sum(s.get('success', 0) for s in all_stats.values())
    total_failed = sum(s.get('failed', 0) for s in all_stats.values())
    
    for dataset_name, stats in all_stats.items():
        logger.info(f"{dataset_name}: {stats.get('success', 0)} success, {stats.get('failed', 0)} failed")
    
    logger.info(f"\nTotal: {total_success} success, {total_failed} failed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
