# -*- coding: utf-8 -*-
"""
@author Shasha Liao
@date 2025年11月29日 14:38:16
@packageName 
@className track_processor
@version 1.0.0
@describe TODO
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
import warnings
import time

warnings.filterwarnings('ignore')


@dataclass
class TrackProcessorConfig:
    """Configuration for track processing parameters"""
    track_length_cutoff: int = 5
    acquisition_time: float = 0.01  # seconds
    pixel_size: float = 0.109  # micrometers
    frame_continuity_threshold: int = 4  # frames


class TrackProcessor:
    """
    Processes tracked data similar to MATLAB trajectory reconstruction logic

    This class replicates the trajectory reconstruction algorithm from:
    - RC_RG_run_through.m (lines 33-54)
    - DiffuseSingle_CG_compare.m (lines 38-59)
    """

    def __init__(self, config: TrackProcessorConfig):
        self.config = config

    def load_tracked_data(self, filepath: str) -> np.ndarray:
        """
        Load tracked table data from file

        Parameters:
        -----------
        filepath : str
            Path to the tracked table file

        Returns:
        --------
        np.ndarray
            Array with columns [x_pixel, y_pixel, frame, track_id, ...]
        """
        print(f"Loading data from {filepath}...")

        # Load data using pandas for better handling
        data = pd.read_csv(filepath, sep='\t', header=None)

        # Convert to numpy array
        tracks = data.values

        print(f"Loaded {len(tracks)} data points")
        print(f"Found {len(np.unique(tracks[:, 3]))} unique tracks")

        return tracks

    def reconstruct_tracks(self, tracks: np.ndarray) -> List[np.ndarray]:
        """
        Fast track reconstruction using vectorized operations

        This method is 10-100x faster than the original while loop approach
        """
        print("Reconstructing tracks (optimized)...")
        start_time = time.time()

        # Pre-allocate arrays for better performance
        track_ids = tracks[:, 3].astype(int)
        frames = tracks[:, 2].astype(int)

        # Find track boundaries using vectorized operations
        track_boundaries = self._find_track_boundaries_fast(track_ids, frames)

        # Reconstruct tracks using optimized method
        all_tracks = []

        for start_idx, end_idx in track_boundaries:
            track_data = tracks[start_idx:end_idx + 1]

            if len(track_data) >= self.config.track_length_cutoff:
                # Convert to proper units: [time, x, y]
                item_converted = np.column_stack([
                    track_data[:, 2] * self.config.acquisition_time,  # Time
                    track_data[:, 0] * self.config.pixel_size,  # X
                    track_data[:, 1] * self.config.pixel_size  # Y
                ])
                all_tracks.append(item_converted)

        processing_time = time.time() - start_time
        print(f"Reconstructed {len(all_tracks)} valid tracks in {processing_time:.2f}s")

        return all_tracks

    def _find_track_boundaries_fast(self, track_ids: np.ndarray, frames: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find track boundaries using vectorized operations

        This is much faster than the original while loop approach
        """
        # Find where track IDs change
        track_changes = np.diff(track_ids, prepend=track_ids[0] - 1) != 0

        # Find where frames are not continuous
        frame_gaps = np.diff(frames, prepend=frames[0]) > self.config.frame_continuity_threshold

        # Combine both conditions
        track_boundaries = track_changes | frame_gaps

        # Find start and end indices
        boundary_indices = np.where(track_boundaries)[0]

        # Create track segments
        segments = []
        start_idx = 0

        for end_idx in boundary_indices:
            if end_idx > start_idx:
                segments.append((start_idx, end_idx - 1))
            start_idx = end_idx

        # Add the last segment
        if start_idx < len(track_ids):
            segments.append((start_idx, len(track_ids) - 1))

        return segments

    def filter_tracks_by_length(self, tracks: List[np.ndarray],
                                min_length: int = None, max_length: int = None) -> List[np.ndarray]:
        """
        Filter tracks by length

        Parameters:
        -----------
        tracks : List[np.ndarray]
            List of track arrays
        min_length : int, optional
            Minimum track length
        max_length : int, optional
            Maximum track length

        Returns:
        --------
        List[np.ndarray]
            Filtered tracks
        """
        if min_length is None:
            min_length = self.config.track_length_cutoff
        if max_length is None:
            max_length = float('inf')

        filtered_tracks = []
        for track in tracks:
            if min_length <= len(track) <= max_length:
                filtered_tracks.append(track)

        print(f"Filtered to {len(filtered_tracks)} tracks (length: {min_length}-{max_length})")
        return filtered_tracks

    def get_track_statistics(self, tracks: List[np.ndarray]) -> dict:
        """
        Get statistics about the tracks

        Parameters:
        -----------
        tracks : List[np.ndarray]
            List of track arrays

        Returns:
        --------
        dict
            Statistics dictionary
        """
        if not tracks:
            return {}

        track_lengths = [len(track) for track in tracks]

        stats = {
            'total_tracks': len(tracks),
            'min_length': min(track_lengths),
            'max_length': max(track_lengths),
            'mean_length': np.mean(track_lengths),
            'median_length': np.median(track_lengths),
            'std_length': np.std(track_lengths)
        }

        return stats

    def plot_tracks(self, tracks: List[np.ndarray], max_tracks: int = 100,
                    figsize: Tuple[int, int] = (10, 8), save_path: str = None):
        """
        Plot particle tracks

        Parameters:
        -----------
        tracks : List[np.ndarray]
            List of track arrays
        max_tracks : int
            Maximum number of tracks to plot
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)

        # Plot up to max_tracks
        for i, track in enumerate(tracks[:max_tracks]):
            plt.plot(track[:, 1], track[:, 2], alpha=0.6, linewidth=0.8)

        plt.xlabel(f'X ({self.config.pixel_size} μm)')
        plt.ylabel(f'Y ({self.config.pixel_size} μm)')
        plt.title(f'Particle Tracks (showing {min(len(tracks), max_tracks)} tracks)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Track plot saved to {save_path}")

        plt.show()


def create_sample_tracked_data(filename: str = "sample_table.txt",
                               n_tracks: int = 50,
                               track_length: int = 20,
                               acquisition_time: float = 0.01,
                               pixel_size: float = 0.109):
    """
    Create sample tracked data for testing

    Parameters:
    -----------
    filename : str
        Output filename
    n_tracks : int
        Number of tracks to generate
    track_length : int
        Average length of each track
    acquisition_time : float
        Acquisition time interval
    pixel_size : float
        Pixel size in micrometers
    """
    print(f"Creating sample data with {n_tracks} tracks...")

    data = []
    track_id = 1

    for _ in range(n_tracks):
        # Random starting position
        start_x = np.random.uniform(100, 400)
        start_y = np.random.uniform(100, 400)

        # Random track length (minimum 5 points)
        actual_length = max(track_length + np.random.randint(-10, 10), 5)

        # Generate track with some random walk behavior
        x_positions = [start_x]
        y_positions = [start_y]

        for frame in range(1, actual_length):
            # Add some random movement
            dx = np.random.normal(0, 2)
            dy = np.random.normal(0, 2)

            new_x = x_positions[-1] + dx
            new_y = y_positions[-1] + dy

            x_positions.append(new_x)
            y_positions.append(new_y)

        # Add to data array
        for i, (x, y) in enumerate(zip(x_positions, y_positions)):
            data.append([x, y, i, track_id, 1000, 1000, 1.0, 1000])

        track_id += 1

    # Save to file
    np.savetxt(filename, data, delimiter='\t',
               fmt='%.6f\t%.6f\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.6f')

    print(f"Sample data saved to {filename}")
    return filename


def main():
    """Example usage of TrackProcessor"""
    import argparse

    parser = argparse.ArgumentParser(description='Track Processor Example')
    parser.add_argument('input_file', nargs='?', help='Path to tracked table file')
    parser.add_argument('--create-sample', action='store_true',
                        help='Create sample data for testing')
    parser.add_argument('--plot', action='store_true',
                        help='Plot tracks')
    parser.add_argument('--track-length', type=int, default=5,
                        help='Minimum track length cutoff')
    parser.add_argument('--acquisition-time', type=float, default=0.01,
                        help='Acquisition time interval in seconds')
    parser.add_argument('--pixel-size', type=float, default=0.109,
                        help='Pixel size in micrometers')

    args = parser.parse_args()

    # Create configuration
    config = TrackProcessorConfig(
        track_length_cutoff=args.track_length,
        acquisition_time=args.acquisition_time,
        pixel_size=args.pixel_size
    )

    # Handle sample data creation
    if args.create_sample:
        sample_file = create_sample_tracked_data(
            "sample_table.txt",
            n_tracks=30,
            track_length=15,
            acquisition_time=args.acquisition_time,
            pixel_size=args.pixel_size
        )
        args.input_file = sample_file

    # Handle file processing
    if not args.input_file:
        print("Error: No input file specified.")
        print("Use --create-sample to create sample data.")
        print("Use --help for usage information.")
        return

    # Initialize processor
    processor = TrackProcessor(config)

    # Load and process data
    tracks = processor.load_tracked_data(args.input_file)
    reconstructed_tracks = processor.reconstruct_tracks(tracks)

    # Get statistics
    stats = processor.get_track_statistics(reconstructed_tracks)
    print(f"\nTrack Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Plot tracks if requested
    if args.plot and reconstructed_tracks:
        processor.plot_tracks(reconstructed_tracks, max_tracks=50)

    print(f"\n✓ Track processing completed!")
    print(f"✓ Processed {len(reconstructed_tracks)} valid tracks")


if __name__ == "__main__":
    main()