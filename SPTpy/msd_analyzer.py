# -*- coding: utf-8 -*-
"""
@author Shasha Liao
@date 2025年11月29日 14:39:22
@packageName 
@className msd_analyzer
@version 1.0.0
@describe TODO
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


@dataclass
class MSDAnalyzerConfig:
    """Configuration for MSD analysis parameters"""
    space_units: str = 'um'
    time_units: str = 's'
    min_points_for_fit: int = 5
    max_time_delay_ratio: float = 0.8  # Maximum time delay as ratio of track length


class MSDAnalyzer:
    """
    Computes Mean Square Displacement similar to MATLAB msdanalyzer

    This class replicates the MSD computation algorithm from:
    - MATLAB msdanalyzer.computeMSD()
    - MATLAB msdanalyzer.fitMSD()
    - MATLAB msdanalyzer.fitLogLogMSD()
    """

    def __init__(self, config: MSDAnalyzerConfig = None):
        if config is None:
            config = MSDAnalyzerConfig()
        self.config = config
        self.msd_data = []

    def compute_msd(self, tracks: List[np.ndarray]) -> List[Dict]:
        """
        Compute MSD for all tracks

        Parameters:
        -----------
        tracks : List[np.ndarray]
            List of track arrays with columns [time, x, y]

        Returns:
        --------
        List[Dict]
            List of MSD data dictionaries
        """
        print("Computing Mean Square Displacement...")

        self.msd_data = []

        for i, track in enumerate(tracks):
            msd = self._calculate_msd_single_track(track)
            if msd is not None:
                self.msd_data.append({
                    'track_id': i,
                    'msd': msd,
                    'track_length': len(track)
                })

        print(f"Computed MSD for {len(self.msd_data)} tracks")
        return self.msd_data

    def _calculate_msd_single_track(self, track: np.ndarray) -> Optional[np.ndarray]:
        """
        Calculate MSD for a single track

        This method replicates the MATLAB MSD calculation:
        ```matlab
        for dt = 1:n_points-1
            msd_sum = 0;
            count = 0;
            for i = 1:n_points-dt
                displacement = track(i+dt, 1:2) - track(i, 1:2);
                msd_sum = msd_sum + sum(displacement.^2);
                count = count + 1;
            end
            msd_values(dt) = msd_sum / count;
        end
        ```

        Parameters:
        -----------
        track : np.ndarray
            Track data with columns [time, x, y]

        Returns:
        --------
        np.ndarray or None
            MSD data with columns [time_delay, msd, n_points, error]
        """
        n_points = len(track)
        if n_points < 2:
            return None

        time_delays = []
        msd_values = []
        n_points_list = []
        errors = []

        # Calculate MSD for different time delays
        max_dt = min(n_points - 1, int(n_points * self.config.max_time_delay_ratio))

        for dt in range(1, max_dt + 1):
            msd_sum = 0
            count = 0
            squared_displacements = []

            for i in range(n_points - dt):
                displacement = track[i + dt, 1:3] - track[i, 1:3]  # x, y displacement
                squared_disp = np.sum(displacement ** 2)
                msd_sum += squared_disp
                squared_displacements.append(squared_disp)
                count += 1

            if count > 0:
                time_delays.append(track[dt, 0] - track[0, 0])  # Time delay
                msd_values.append(msd_sum / count)  # Average MSD
                n_points_list.append(count)

                # Calculate standard error
                if len(squared_displacements) > 1:
                    error = np.std(squared_displacements) / np.sqrt(count)
                else:
                    error = 0
                errors.append(error)

        if not time_delays:
            return None

        return np.column_stack([
            time_delays,
            msd_values,
            n_points_list,
            errors
        ])

    def fit_msd_linear(self, msd_data: List[Dict], fit_ratio: float = 0.8) -> List[Dict]:
        """
        Fit MSD curves with linear diffusion model

        Parameters:
        -----------
        msd_data : List[Dict]
            List of MSD data dictionaries
        fit_ratio : float
            Ratio of MSD curve to use for fitting

        Returns:
        --------
        List[Dict]
            List of fitting results
        """
        print("Fitting MSD curves with linear diffusion model...")

        fit_results = []

        for track_data in msd_data:
            msd = track_data['msd']

            # Use only the first portion of the MSD curve
            n_points = len(msd)
            fit_points = int(n_points * fit_ratio)

            if fit_points < self.config.min_points_for_fit:
                continue

            time_delays = msd[:fit_points, 0]
            msd_values = msd[:fit_points, 1]

            try:
                # Linear fit: MSD = 4*D*t
                def linear_model(t, D):
                    return 4 * D * t

                popt, pcov = curve_fit(linear_model, time_delays, msd_values,
                                       p0=[0.01], bounds=([0], [np.inf]))

                diffusion_coeff = popt[0]
                r_squared = self._calculate_r_squared(msd_values,
                                                      linear_model(time_delays, diffusion_coeff))

                fit_results.append({
                    'track_id': track_data['track_id'],
                    'diffusion_coefficient': diffusion_coeff,
                    'r_squared': r_squared,
                    'fit_points': fit_points,
                    'model': 'linear'
                })

            except Exception as e:
                continue

        print(f"Fitted {len(fit_results)} tracks with linear model")
        return fit_results

    def fit_msd_loglog(self, msd_data: List[Dict], fit_ratio: float = 0.9) -> List[Dict]:
        """
        Fit MSD curves with log-log model

        Parameters:
        -----------
        msd_data : List[Dict]
            List of MSD data dictionaries
        fit_ratio : float
            Ratio of MSD curve to use for fitting

        Returns:
        --------
        List[Dict]
            List of fitting results
        """
        print("Fitting MSD curves with log-log model...")

        fit_results = []

        for track_data in msd_data:
            msd = track_data['msd']

            # Use only the first portion of the MSD curve
            n_points = len(msd)
            fit_points = int(n_points * fit_ratio)

            if fit_points < self.config.min_points_for_fit:
                continue

            time_delays = msd[:fit_points, 0]
            msd_values = msd[:fit_points, 1]

            # Filter out zero or negative values for log
            valid_mask = (time_delays > 0) & (msd_values > 0)
            if np.sum(valid_mask) < self.config.min_points_for_fit:
                continue

            log_time = np.log(time_delays[valid_mask])
            log_msd = np.log(msd_values[valid_mask])

            try:
                # Log-log fit: log(MSD) = log(a) + b*log(t)
                def loglog_model(log_t, log_a, b):
                    return log_a + b * log_t

                popt, pcov = curve_fit(loglog_model, log_time, log_msd,
                                       p0=[np.log(0.01), 1.0])

                a = np.exp(popt[0])
                b = popt[1]
                r_squared = self._calculate_r_squared(log_msd,
                                                      loglog_model(log_time, popt[0], b))

                fit_results.append({
                    'track_id': track_data['track_id'],
                    'a': a,
                    'b': b,
                    'r_squared': r_squared,
                    'fit_points': fit_points,
                    'model': 'loglog'
                })

            except Exception as e:
                continue

        print(f"Fitted {len(fit_results)} tracks with log-log model")
        return fit_results

    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared value"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    def plot_msd(self, msd_data: List[Dict], max_tracks: int = 50,
                 figsize: Tuple[int, int] = (12, 8), save_path: str = None):
        """
        Plot MSD curves

        Parameters:
        -----------
        msd_data : List[Dict]
            List of MSD data dictionaries
        max_tracks : int
            Maximum number of MSD curves to plot
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=figsize)

        for i, track_data in enumerate(msd_data[:max_tracks]):
            msd = track_data['msd']
            plt.plot(msd[:, 0], msd[:, 1], alpha=0.6, linewidth=0.8)

        plt.xlabel(f'Time delay ({self.config.time_units})')
        plt.ylabel(f'MSD ({self.config.space_units}²)')
        plt.title(f'Mean Square Displacement (showing {min(len(msd_data), max_tracks)} tracks)')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.xscale('log')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"MSD plot saved to {save_path}")

        plt.show()

    def plot_msd_histogram(self, fit_results: List[Dict],
                           figsize: Tuple[int, int] = (10, 6), save_path: str = None):
        """
        Plot histogram of diffusion coefficients

        Parameters:
        -----------
        fit_results : List[Dict]
            List of fitting results
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if not fit_results:
            print("No fitting results to plot")
            return

        plt.figure(figsize=figsize)

        # Extract diffusion coefficients
        if fit_results[0]['model'] == 'linear':
            d_values = [result['diffusion_coefficient'] for result in fit_results]
            plt.hist(np.log10(d_values), bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Log10(D (μm²/s))')
            plt.ylabel('Count')
            plt.title('Diffusion Coefficient Distribution')
        else:
            a_values = [result['a'] for result in fit_results]
            plt.hist(np.log10(a_values), bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Log10(a)')
            plt.ylabel('Count')
            plt.title('Log-log Fit Parameter Distribution')

        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Histogram plot saved to {save_path}")

        plt.show()

    def export_msd_data(self, msd_data: List[Dict], output_file: str):
        """
        Export MSD data to file

        Parameters:
        -----------
        msd_data : List[Dict]
            List of MSD data dictionaries
        output_file : str
            Output file path
        """
        print(f"Exporting MSD data to {output_file}...")

        # Flatten MSD data for export
        all_msd_data = []
        for track_data in msd_data:
            msd = track_data['msd']
            track_id = track_data['track_id']

            for row in msd:
                all_msd_data.append([track_id] + row.tolist())

        # Create DataFrame and save
        df = pd.DataFrame(all_msd_data,
                          columns=['track_id', 'time_delay', 'msd', 'n_points', 'error'])
        df.to_csv(output_file, sep='\t', index=False)

        print(f"MSD data exported to {output_file}")


def main():
    """Example usage of MSDAnalyzer"""
    import argparse
    from track_processor import TrackProcessor, TrackProcessorConfig

    parser = argparse.ArgumentParser(description='MSD Analyzer Example')
    parser.add_argument('input_file', nargs='?', help='Path to tracked table file')
    parser.add_argument('--create-sample', action='store_true',
                        help='Create sample data for testing')
    parser.add_argument('--plot', action='store_true',
                        help='Plot MSD curves')
    parser.add_argument('--export', help='Export MSD data to file')
    parser.add_argument('--track-length', type=int, default=5,
                        help='Minimum track length cutoff')
    parser.add_argument('--acquisition-time', type=float, default=0.01,
                        help='Acquisition time interval in seconds')
    parser.add_argument('--pixel-size', type=float, default=0.109,
                        help='Pixel size in micrometers')

    args = parser.parse_args()

    # Handle sample data creation
    if args.create_sample:
        from track_processor import create_sample_tracked_data
        sample_file = create_sample_tracked_data(
            "sample_tracked_table.txt",
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

    # Initialize components
    track_config = TrackProcessorConfig(
        track_length_cutoff=args.track_length,
        acquisition_time=args.acquisition_time,
        pixel_size=args.pixel_size
    )

    msd_config = MSDAnalyzerConfig()

    processor = TrackProcessor(track_config)
    msd_analyzer = MSDAnalyzer(msd_config)

    # Load and process data
    tracks = processor.load_tracked_data(args.input_file)
    reconstructed_tracks = processor.reconstruct_tracks(tracks)

    if not reconstructed_tracks:
        print("No valid tracks found!")
        return

    # Compute MSD
    msd_data = msd_analyzer.compute_msd(reconstructed_tracks)

    if not msd_data:
        print("No valid MSD data found!")
        return

    # Fit MSD curves
    linear_fits = msd_analyzer.fit_msd_linear(msd_data)
    loglog_fits = msd_analyzer.fit_msd_loglog(msd_data)

    print(f"\nMSD Analysis Results:")
    print(f"  Total tracks: {len(reconstructed_tracks)}")
    print(f"  Valid MSD tracks: {len(msd_data)}")
    print(f"  Linear fits: {len(linear_fits)}")
    print(f"  Log-log fits: {len(loglog_fits)}")

    # Plot results
    if args.plot:
        msd_analyzer.plot_msd(msd_data, max_tracks=30)
        if linear_fits:
            msd_analyzer.plot_msd_histogram(linear_fits)

    # Export data
    if args.export:
        msd_analyzer.export_msd_data(msd_data, args.export)

    print(f"\n✓ MSD analysis completed!")


if __name__ == "__main__":
    main()
