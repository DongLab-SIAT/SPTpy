# -*- coding: utf-8 -*-
"""
@author Shasha Liao
@date 2025-11-29 14:20:56
@packageName
@className automated_spt_processor
@version 1.0.0
@describe TODO
"""
#!/usr/bin/env python3
"""
Automated SPT Processor for specific directory structure
Processes _tracked_table.txt files from specific directories and generates appropriate output files

Usage:
    python automated_spt_processor.py [options]
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
import logging
from datetime import datetime

# Import our analysis modules
from track_processor import TrackProcessor, TrackProcessorConfig
from msd_analyzer import MSDAnalyzer, MSDAnalyzerConfig
from radius_calculator import (
    RadiusOfConfinementCalculator,
    RadiusOfGyrationCalculator,
    RadiusCalculatorConfig,
    RadiusDataExporter,
)


@dataclass
class AutomatedProcessorConfig:
    """Configuration for automated processing"""
    # Directory mappings for different targets
    DIRECTORY_MAPPINGS = {
        # # ===== MED1 =====
        # r"G:\50ms_ng_dataset\MED1_after_seg\0h_auxin": ("MED1", "no_auxin", "rc_only"),
        # r"G:\50ms_ng_dataset\MED1_after_seg\6h_auxin": ("MED1", "auxin", "rc_only"),
        #
        # # ===== MED6 =====
        # r"G:\50ms_ng_dataset\MED6_after_seg\0h_auxin": ("MED6", "no_auxin", "rc_only"),
        # r"G:\50ms_ng_dataset\MED6_after_seg\6h_auxin": ("MED6", "auxin", "rc_only"),

        # ===== BRD4 =====
        r"G:\50ms_ng_dataset\20210129_RAD21_Halo-Brd4\test_sample_6h_01": ("BRD4", "auxin", "rc_only"),
        # r"G:\50ms_ng_dataset\Brd4_after_seg\no_auxin": ("BRD4", "no_auxin", "rc_only"),
        #
        # # ===== OCT4 =====
        # r"G:\50ms_ng_dataset\Oct4_after_seg\6h_auxin": ("OCT4", "auxin", "rc_only"),
        # r"G:\50ms_ng_dataset\Oct4_after_seg\no_auxin": ("OCT4", "no_auxin", "rc_only"),
        #
        # # ===== mTBP (TBP) =====
        # r"G:\50ms_ng_dataset\mTBP_after_seg\6h_auxin": ("TBP", "auxin", "rc_only"),
        # r"G:\50ms_ng_dataset\mTBP_after_seg\no_auxin": ("TBP", "no_auxin", "rc_only"),
        #
        # # ===== H2B =====
        # r"G:\50ms_ng_dataset\H2B_after_seg\1st\Ctrl": ("H2B", "no_auxin", "rc_only"),
        # r"G:\50ms_ng_dataset\H2B_after_seg\1st\RAD21": ("H2B", "auxin", "rc_only"),
        # r"G:\50ms_ng_dataset\H2B_after_seg\RAD21\0h_auxin": ("H2B", "no_auxin", "rc_only"),
        # r"G:\50ms_ng_dataset\H2B_after_seg\RAD21\6h_auxin": ("H2B", "auxin", "rc_only"),
        #
        # # ===== H2AZ =====
        # r"G:\50ms_ng_dataset\H2AZ_after_seg\Ctrl": ("H2AZ", "no_auxin", "rc_only"),
        # r"G:\50ms_ng_dataset\H2AZ_after_seg\Ctrl_2": ("H2AZ", "no_auxin", "rc_only"),
        # r"G:\50ms_ng_dataset\H2AZ_after_seg\RAD21": ("H2AZ", "auxin", "rc_only"),
        # r"G:\50ms_ng_dataset\H2AZ_after_seg\RAD21_2": ("H2AZ", "auxin", "rc_only"),
    }

    # Analysis parameters
    track_length_cutoff: int = 5
    acquisition_time: float = 0.05  # 50ms
    pixel_size: float = 0.109  # 109nm
    space_units: str = 'um'
    time_units: str = 's'

    # Processing options
    output_directory: str = None
    log_level: str = 'INFO'
    skip_existing: bool = True
    max_files_per_directory: int = None


class AutomatedSPTProcessor:
    """
    Automated processor for specific directory structure
    """

    def __init__(self, config: AutomatedProcessorConfig):
        self.config = config
        self.setup_logging()

        # Initialize analysis components
        self.track_config = TrackProcessorConfig(
            track_length_cutoff=config.track_length_cutoff,
            acquisition_time=config.acquisition_time,
            pixel_size=config.pixel_size
        )

        self.msd_config = MSDAnalyzerConfig()
        self.radius_config = RadiusCalculatorConfig()

        self.processor = TrackProcessor(self.track_config)
        self.msd_analyzer = MSDAnalyzer(self.msd_config)
        self.rc_calculator = RadiusOfConfinementCalculator(self.radius_config)
        self.rg_calculator = RadiusOfGyrationCalculator(self.radius_config)
        self.exporter = RadiusDataExporter()

        # Create output directory if specified
        if self.config.output_directory:
            os.makedirs(self.config.output_directory, exist_ok=True)

    def setup_logging(self):
        """Setup logging configuration - only console output, no log files"""
        log_level = getattr(logging, self.config.log_level.upper())

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Automated SPT processing started at {datetime.now()}")

    def find_tracked_files(self, directory: str) -> List[Path]:
        """
        Find all _tracked_table.txt files in a directory (including subdirectories)

        Parameters
        ----------
        directory : str
            Directory path to search

        Returns
        -------
        List[Path]
            List of tracked table files
        """
        directory_path = Path(directory)

        if not directory_path.exists():
            self.logger.warning(f"Directory does not exist: {directory}")
            return []

        # Only look for _table.txt files (input format)
        # Exclude _tracked_table.txt and _tracked_table_selected.txt (these are outputs)
        table_files = list(directory_path.rglob("*_table.txt"))

        # Filter out any files that are already processed outputs
        input_files = []
        for file_path in table_files:
            filename = file_path.name
            # Only include files that are NOT already processed outputs
            if not ("_tracked_table" in filename or "_RC.txt" in filename or "_RG_RC.txt" in filename):
                input_files.append(file_path)

        all_files = input_files

        # Limit number of files if specified
        if self.config.max_files_per_directory and len(all_files) > self.config.max_files_per_directory:
            all_files = all_files[:self.config.max_files_per_directory]
            self.logger.info(f"Limited to {self.config.max_files_per_directory} files in {directory}")

        self.logger.info(f"Found {len(all_files)} tracked table files in {directory} (including subdirectories)")

        # Log subdirectory structure for debugging
        if all_files:
            subdirs = set(f.parent for f in all_files)
            if len(subdirs) > 1:
                self.logger.info(f"Files found in {len(subdirs)} subdirectories:")
                for subdir in sorted(subdirs):
                    files_in_subdir = [f for f in all_files if f.parent == subdir]
                    self.logger.info(f"  {subdir}: {len(files_in_subdir)} files")

        return all_files

    def _save_filtered_tracks(self, tracks: List[np.ndarray], output_file: Path) -> None:
        """
        Save filtered tracks to _tracked_table_selected.txt format

        Parameters
        ----------
        tracks : List[np.ndarray]
            List of track arrays
        output_file : Path
            Output file path
        """
        try:
            # Convert tracks back to the format expected by the analysis
            # Each track is [time, x, y] and we need to add track_id and frame columns
            all_track_data = []

            for track_id, track in enumerate(tracks):
                # track is [time, x, y] - convert back to [x, y, frame, track_id]
                frames = (track[:, 0] / self.config.acquisition_time).astype(int)
                x_coords = track[:, 1] / self.config.pixel_size
                y_coords = track[:, 2] / self.config.pixel_size
                track_ids = np.full(len(track), track_id)

                # Stack as [x, y, frame, track_id]
                track_data = np.column_stack([x_coords, y_coords, frames, track_ids])
                all_track_data.append(track_data)

            # Combine all tracks
            if all_track_data:
                combined_data = np.vstack(all_track_data)

                # Save as tab-separated file
                np.savetxt(
                    output_file,
                    combined_data,
                    delimiter='\t',
                    fmt='%.6f',
                    header='x\ty\tframe\ttrack_id',
                    comments=''
                )

                self.logger.info(f"Saved {len(tracks)} filtered tracks to {output_file.name}")
            else:
                self.logger.warning(f"No tracks to save for {output_file.name}")

        except Exception as e:
            self.logger.error(f"Error saving filtered tracks to {output_file}: {e}")
            raise

    def process_single_file(self, file_path: Path, target: str, condition: str,
                            output_type: str) -> Dict:
        """
        Process a single _table.txt file through the complete workflow:
        1. _table.txt -> _tracked_table_selected.txt (filtering)
        2. _tracked_table_selected.txt -> RC/RG analysis

        Parameters
        ----------
        file_path : Path
            Path to the _table.txt file
        target : str
            Target name (MED1, MED6, BRD4, etc.)
        condition : str
            Condition (auxin, no_auxin)
        output_type : str
            Output type (rc_only, rg_rc)

        Returns
        -------
        Dict
            Processing results
        """
        start_time = time.time()

        try:
            self.logger.info(f"Processing: {file_path.name} (Target: {target}, Condition: {condition})")

            # Step 1: Generate intermediate _tracked_table_selected.txt filename
            base_name = file_path.stem.replace("_table", "")
            selected_file = file_path.parent / f"{base_name}_tracked_table_selected.txt"

            # Step 2: Generate final output filenames
            if self.config.output_directory:
                output_dir = self.config.output_directory
            else:
                output_dir = file_path.parent

            rc_output = os.path.join(output_dir, f"{base_name}_RC.txt")
            rg_rc_output = os.path.join(output_dir, f"{base_name}_RG_RC.txt")

            # Check if final output already exists
            if self.config.skip_existing:
                if output_type == "rc_only" and os.path.exists(rc_output):
                    self.logger.info(f"Skipping {file_path.name} (RC output already exists)")
                    return {
                        'file': str(file_path),
                        'target': target,
                        'condition': condition,
                        'output_type': output_type,
                        'status': 'skipped',
                        'processing_time': 0,
                        'tracks_processed': 0,
                        'rc_tracks': 0,
                        'rg_tracks': 0
                    }
                elif output_type == "rg_rc" and os.path.exists(rg_rc_output):
                    self.logger.info(f"Skipping {file_path.name} (RG+RC output already exists)")
                    return {
                        'file': str(file_path),
                        'target': target,
                        'condition': condition,
                        'output_type': output_type,
                        'status': 'skipped',
                        'processing_time': 0,
                        'tracks_processed': 0,
                        'rc_tracks': 0,
                        'rg_tracks': 0
                    }

            # Step 1: Load and filter _table.txt to generate _tracked_table_selected.txt
            self.logger.info(f"Step 1: Filtering {file_path.name} -> {selected_file.name}")
            tracks = self.processor.load_tracked_data(str(file_path))
            reconstructed_tracks = self.processor.reconstruct_tracks(tracks)

            if not reconstructed_tracks:
                self.logger.warning(f"No valid tracks found in {file_path.name}")
                return {
                    'file': str(file_path),
                    'target': target,
                    'condition': condition,
                    'output_type': output_type,
                    'status': 'no_tracks',
                    'processing_time': time.time() - start_time,
                    'tracks_processed': 0,
                    'rc_tracks': 0,
                    'rg_tracks': 0
                }

            # Save filtered tracks as _tracked_table_selected.txt
            self._save_filtered_tracks(reconstructed_tracks, selected_file)

            # Step 2: Process the filtered tracks for RC/RG analysis
            self.logger.info(f"Step 2: Analyzing {selected_file.name} for RC/RG")

            # Compute MSD
            msd_data = self.msd_analyzer.compute_msd(reconstructed_tracks)

            if not msd_data:
                self.logger.warning(f"No valid MSD data found in {file_path.name}")
                return {
                    'file': str(file_path),
                    'target': target,
                    'condition': condition,
                    'output_type': output_type,
                    'status': 'no_msd',
                    'processing_time': time.time() - start_time,
                    'tracks_processed': len(reconstructed_tracks),
                    'rc_tracks': 0,
                    'rg_tracks': 0
                }

            # Compute Radius of Confinement
            rc_data = self.rc_calculator.compute_radius_of_confinement(msd_data)

            if not rc_data:
                self.logger.warning(f"No valid RC data found in {file_path.name}")
                return {
                    'file': str(file_path),
                    'target': target,
                    'condition': condition,
                    'output_type': output_type,
                    'status': 'no_rc',
                    'processing_time': time.time() - start_time,
                    'tracks_processed': len(reconstructed_tracks),
                    'rc_tracks': 0,
                    'rg_tracks': 0
                }

            # Export results based on output type
            if output_type == "rc_only":
                # Generate RC only file
                self.exporter.export_rc_only(rc_data, rc_output)
                rg_tracks = 0

            elif output_type == "rg_rc":
                # Compute Radius of Gyration
                rg_data = self.rg_calculator.compute_radius_of_gyration(reconstructed_tracks)

                # Generate RC and RG file
                self.exporter.export_rc_rg(rc_data, rg_data, rg_rc_output)
                rg_tracks = len(rg_data)

            processing_time = time.time() - start_time

            self.logger.info(f"Completed {file_path.name} in {processing_time:.2f}s")
            self.logger.info(f"  Tracks: {len(reconstructed_tracks)}, RC: {len(rc_data)}, RG: {rg_tracks}")

            return {
                'file': str(file_path),
                'target': target,
                'condition': condition,
                'output_type': output_type,
                'status': 'success',
                'processing_time': processing_time,
                'tracks_processed': len(reconstructed_tracks),
                'rc_tracks': len(rc_data),
                'rg_tracks': rg_tracks
            }

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing {file_path.name}: {str(e)}")

            return {
                'file': str(file_path),
                'target': target,
                'condition': condition,
                'output_type': output_type,
                'status': 'error',
                'error': str(e),
                'processing_time': processing_time,
                'tracks_processed': 0,
                'rc_tracks': 0,
                'rg_tracks': 0
            }

    def process_all_directories(self) -> Dict:
        """
        Process all directories according to the mapping

        Returns
        -------
        Dict
            Processing results summary
        """
        start_time = time.time()

        all_results = []
        total_files = 0
        successful = 0
        failed = 0
        skipped = 0

        self.logger.info(f"Starting automated processing of {len(self.config.DIRECTORY_MAPPINGS)} directories")

        for directory, (target, condition, output_type) in self.config.DIRECTORY_MAPPINGS.items():
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Processing directory: {directory}")
            self.logger.info(f"Target: {target}, Condition: {condition}, Output: {output_type}")
            self.logger.info(f"{'=' * 60}")

            # Find tracked files in this directory
            tracked_files = self.find_tracked_files(directory)

            if not tracked_files:
                self.logger.warning(f"No tracked files found in {directory}")
                continue

            # Process each file
            for file_path in tracked_files:
                result = self.process_single_file(file_path, target, condition, output_type)
                all_results.append(result)
                total_files += 1

                if result['status'] == 'success':
                    successful += 1
                elif result['status'] == 'skipped':
                    skipped += 1
                else:
                    failed += 1

        total_time = time.time() - start_time

        # Log summary
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("AUTOMATED PROCESSING SUMMARY")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"Total files processed: {total_files}")
        self.logger.info(f"Successful: {successful}")
        self.logger.info(f"Failed: {failed}")
        self.logger.info(f"Skipped: {skipped}")
        self.logger.info(f"Total processing time: {total_time:.2f}s")
        if total_files > 0:
            self.logger.info(f"Average time per file: {total_time / total_files:.2f}s")

        return {
            'total_files': total_files,
            'successful_files': successful,
            'failed_files': failed,
            'skipped_files': skipped,
            'total_processing_time': total_time,
            'results': all_results
        }

    def export_summary(self, batch_results: Dict, output_file: str = None):
        """
        Export processing summary (only if output_file is specified)

        Parameters
        ----------
        batch_results : Dict
            Processing results
        output_file : str, optional
            Output file path for summary (if None, no file is created)
        """
        # Only create summary file if explicitly requested
        if output_file is None:
            self.logger.info("Summary not exported (no output file specified)")
            return

        with open(output_file, 'w') as f:
            f.write("AUTOMATED SPT PROCESSING SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Total files: {batch_results['total_files']}\n")
            f.write(f"Successful: {batch_results['successful_files']}\n")
            f.write(f"Failed: {batch_results['failed_files']}\n")
            f.write(f"Skipped: {batch_results['skipped_files']}\n")
            f.write(f"Total processing time: {batch_results['total_processing_time']:.2f}s\n")
            if batch_results['total_files'] > 0:
                f.write(
                    f"Average time per file: "
                    f"{batch_results['total_processing_time'] / batch_results['total_files']:.2f}s\n"
                )
            f.write("\nDETAILED RESULTS:\n")
            f.write("-" * 50 + "\n")

            for result in batch_results['results']:
                f.write(f"File: {result['file']}\n")
                f.write(f"Target: {result['target']}\n")
                f.write(f"Condition: {result['condition']}\n")
                f.write(f"Output Type: {result['output_type']}\n")
                f.write(f"Status: {result['status']}\n")
                f.write(f"Processing time: {result['processing_time']:.2f}s\n")
                f.write(f"Tracks processed: {result['tracks_processed']}\n")
                f.write(f"RC tracks: {result['rc_tracks']}\n")
                f.write(f"RG tracks: {result['rg_tracks']}\n")
                if 'error' in result:
                    f.write(f"Error: {result['error']}\n")
                f.write("\n")

        self.logger.info(f"Summary exported to {output_file}")


def main():
    """Main function for automated processing"""
    import argparse

    parser = argparse.ArgumentParser(description='Automated SPT Processor')
    parser.add_argument('--output-directory', help='Output directory for results')
    parser.add_argument('--track-length', type=int, default=5,
                        help='Minimum track length cutoff')
    parser.add_argument('--acquisition-time', type=float, default=0.05,
                        help='Acquisition time interval in seconds (default: 0.05 for 50ms)')
    parser.add_argument('--pixel-size', type=float, default=0.109,
                        help='Pixel size in micrometers (default: 0.109)')
    parser.add_argument('--max-files-per-directory', type=int,
                        help='Maximum number of files to process per directory')
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='Do not skip files that already have output')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')

    args = parser.parse_args()

    # Create configuration
    config = AutomatedProcessorConfig(
        output_directory=args.output_directory,
        track_length_cutoff=args.track_length,
        acquisition_time=args.acquisition_time,
        pixel_size=args.pixel_size,
        max_files_per_directory=args.max_files_per_directory,
        skip_existing=not args.no_skip_existing,
        log_level=args.log_level
    )

    # Initialize processor
    processor = AutomatedSPTProcessor(config)

    # Process all directories
    results = processor.process_all_directories()

    # Export summary
    processor.export_summary(results)

    print("\nAutomated processing completed!")
    if config.output_directory:
        print(f"Results saved to: {config.output_directory}")
    else:
        print("Results saved to original directories")


if __name__ == "__main__":
    main()
