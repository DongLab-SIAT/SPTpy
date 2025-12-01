# -*- coding: utf-8 -*-
"""
@author Shasha Liao
@date 2025年11月29日 14:40:18
@packageName 
@className radius_calculator
@version 1.0.0
@describe TODO
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


@dataclass
class RadiusCalculatorConfig:
    """Configuration for radius calculation parameters"""
    min_points_for_fit: int = 5
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6


class RadiusOfConfinementCalculator:
    """
    Calculates Radius of Confinement using confined diffusion model

    This class replicates the MATLAB RadiusOfConfinement.m algorithm:
    ```matlab
    function RC = RadiusOfConfinement(obj, indices)
        ft = fittype( 'a*(1-exp(-4*b*x/a))+c', 'independent', 'x', 'dependent', 'y' );
        opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
        opts.Algorithm = 'Trust-Region';
        opts.Display = 'Off';
        opts.Robust = 'Bisquare';
        opts.Lower= [0 0 0];
        opts.StartPoint = [0.14 0.02 0];
        [fitresult, gof] = fit(t, X_msd, ft, opts );
        coef=coeffvalues(fitresult);
        RC(i,1) = sqrt(coef(1));
        RC(i,2) = coef(2);
        RC(i,3) = gof.sse/size(track_msd,1);
    end
    ```
    """

    def __init__(self, config: RadiusCalculatorConfig = None):
        if config is None:
            config = RadiusCalculatorConfig()
        self.config = config

    def compute_radius_of_confinement(self, msd_data: List[Dict]) -> List[Dict]:
        """
        Compute Radius of Confinement using confined diffusion model

        Parameters:
        -----------
        msd_data : List[Dict]
            List of MSD data dictionaries

        Returns:
        --------
        List[Dict]
            List of radius of confinement results
        """
        print("Computing Radius of Confinement...")

        radius_confinement = []

        for track_data in msd_data:
            msd = track_data['msd']

            # Filter out zero MSD values
            valid_mask = msd[:, 1] > 0
            if np.sum(valid_mask) < self.config.min_points_for_fit:
                continue

            time_delays = msd[valid_mask, 0]
            msd_values = msd[valid_mask, 1]

            try:
                # Fit confined diffusion model: MSD = a*(1-exp(-4*b*t/a)) + c
                popt, pcov = curve_fit(
                    self._confined_diffusion_model,
                    time_delays,
                    msd_values,
                    p0=[0.14, 0.02, 0],  # Initial guess (same as MATLAB)
                    bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                    method='trf',
                    maxfev=self.config.max_iterations
                )

                # Calculate radius of confinement = sqrt(a)
                rc = np.sqrt(popt[0])
                diffusion_coeff = popt[1]

                # Calculate fitting error (SSE normalized by number of points)
                predicted = self._confined_diffusion_model(time_delays, *popt)
                sse = np.sum((msd_values - predicted) ** 2)
                mse = sse / len(msd_values)

                radius_confinement.append({
                    'track_id': track_data['track_id'],
                    'radius_confinement': rc,
                    'diffusion_coefficient': diffusion_coeff,
                    'fitting_error': mse
                })

            except Exception as e:
                # Skip tracks that fail to fit
                continue

        print(f"Computed Radius of Confinement for {len(radius_confinement)} tracks")
        return radius_confinement

    def _confined_diffusion_model(self, t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """
        Confined diffusion model: MSD = a*(1-exp(-4*b*t/a)) + c

        This replicates the MATLAB fittype:
        ```matlab
        ft = fittype( 'a*(1-exp(-4*b*x/a))+c', 'independent', 'x', 'dependent', 'y' );
        ```

        Parameters:
        -----------
        t : np.ndarray
            Time delays
        a : float
            Confinement radius squared
        b : float
            Diffusion coefficient
        c : float
            Offset

        Returns:
        --------
        np.ndarray
            Predicted MSD values
        """
        return a * (1 - np.exp(-4 * b * t / a)) + c


class RadiusOfGyrationCalculator:
    """
    Calculates Radius of Gyration using geometric analysis

    This class replicates the MATLAB RadiusOfGyration.m algorithm:
    ```matlab
    function RG = RadiusOfGyration(obj)
        for i=1:n_tracks
            Tr = Tracks{i,1};
            X = sum(Tr(:,2:3))/n;
            RGn=zeros(n,1);
            for j=1:n
                RGn(j)=sqrt(pdist([X; Tr(j,2:3)], 'euclidean'));
            end
            RG(i) = max(RGn);
        end
    end
    """

    def __init__(self, config: RadiusCalculatorConfig = None):
        if config is None:
            config = RadiusCalculatorConfig()
        self.config = config

    def compute_radius_of_gyration(self, tracks: List[np.ndarray]) -> List[Dict]:
        """
        Compute Radius of Gyration for all tracks

        Parameters:
        -----------
        tracks : List[np.ndarray]
            List of track arrays with columns [time, x, y]

        Returns:
        --------
        List[Dict]
            List of radius of gyration results
        """
        print("Computing Radius of Gyration...")

        radius_gyration = []

        for i, track in enumerate(tracks):
            if len(track) < 2:
                continue

            # Calculate centroid (same as MATLAB: X = sum(Tr(:,2:3))/n)
            centroid = np.mean(track[:, 1:3], axis=0)  # x, y centroid

            # Calculate distances from centroid (same as MATLAB pdist logic)
            distances = []
            for point in track[:, 1:3]:
                dist = np.sqrt(np.sum((point - centroid) ** 2))
                distances.append(dist)

            # Radius of gyration is the maximum distance from centroid
            # (same as MATLAB: RG(i) = max(RGn))
            rg = np.max(distances)

            radius_gyration.append({
                'track_id': i,
                'radius_gyration': rg
            })

        print(f"Computed Radius of Gyration for {len(radius_gyration)} tracks")
        return radius_gyration

    def compute_radius_of_gyration_average(self, tracks: List[np.ndarray]) -> List[Dict]:
        """
        Compute Radius of Gyration using average method

        This replicates the MATLAB RadiusOfGyration_average.m algorithm:
        ```matlab
        function RG = RadiusOfGyration_average(Tracks)
            for i=1:m
                Tr = Tracks{i};
                X = Tr(:,2:3);
                item = sqrt(sum(pdist(X, 'euclidean').^2)/2/(n^2));
                RG=[RG;item];
            end
        end
        ```

        Parameters:
        -----------
        tracks : List[np.ndarray]
            List of track arrays with columns [time, x, y]

        Returns:
        --------
        List[Dict]
            List of radius of gyration results (average method)
        """
        print("Computing Radius of Gyration (average method)...")

        radius_gyration = []

        for i, track in enumerate(tracks):
            if len(track) < 2:
                continue

            # Extract x, y coordinates
            X = track[:, 1:3]  # x, y coordinates
            n = len(X)

            # Calculate all pairwise distances
            distances = pdist(X, 'euclidean')

            # Calculate average radius of gyration
            # Same as MATLAB: sqrt(sum(pdist(X, 'euclidean').^2)/2/(n^2))
            rg_avg = np.sqrt(np.sum(distances ** 2) / (2 * n ** 2))

            radius_gyration.append({
                'track_id': i,
                'radius_gyration': rg_avg
            })

        print(f"Computed Radius of Gyration (average) for {len(radius_gyration)} tracks")
        return radius_gyration


class RadiusDataExporter:
    """Exports radius data to files similar to MATLAB dlmwrite"""

    def __init__(self):
        pass

    def export_rc_only(self, rc_data: List[Dict], output_file: str):
        """
        Export only RC data (like RC_RG_run_through.m)

        Parameters:
        -----------
        rc_data : List[Dict]
            List of radius of confinement results
        output_file : str
            Output file path
        """
        print(f"Exporting RC data to {output_file}...")

        # Create DataFrame
        rc_df = pd.DataFrame(rc_data)

        # Save to file (same format as MATLAB dlmwrite)
        rc_df.to_csv(output_file, sep='\t', index=False,
                     columns=['track_id', 'radius_confinement', 'diffusion_coefficient', 'fitting_error'])

        print(f"RC data exported to {output_file}")

    def export_rc_rg(self, rc_data: List[Dict], rg_data: List[Dict], output_file: str):
        """
        Export RC and RG data (like DiffuseSingle_CG_compare.m)

        Parameters:
        -----------
        rc_data : List[Dict]
            List of radius of confinement results
        rg_data : List[Dict]
            List of radius of gyration results
        output_file : str
            Output file path
        """
        print(f"Exporting RC and RG data to {output_file}...")

        # Create DataFrames
        rc_df = pd.DataFrame(rc_data)
        rg_df = pd.DataFrame(rg_data)

        # Merge results (same as MATLAB horzcat)
        results = pd.merge(rc_df, rg_df, on='track_id', how='outer')

        # Save to file
        results.to_csv(output_file, sep='\t', index=False,
                       columns=['track_id', 'radius_confinement', 'diffusion_coefficient',
                                'fitting_error', 'radius_gyration'])

        print(f"RC and RG data exported to {output_file}")

    def plot_radius_distributions(self, rc_data: List[Dict], rg_data: List[Dict] = None,
                                  figsize: Tuple[int, int] = (15, 5), save_path: str = None):
        """
        Plot radius distributions

        Parameters:
        -----------
        rc_data : List[Dict]
            List of radius of confinement results
        rg_data : List[Dict], optional
            List of radius of gyration results
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if rg_data is None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        else:
            fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Radius of Confinement
        rc_values = [r['radius_confinement'] for r in rc_data]
        axes[0].hist(rc_values, bins=30, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Radius of Confinement (μm)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Radius of Confinement Distribution')
        axes[0].grid(True, alpha=0.3)

        if rg_data is not None:
            # Radius of Gyration
            rg_values = [r['radius_gyration'] for r in rg_data]
            axes[1].hist(rg_values, bins=30, alpha=0.7, edgecolor='black', color='orange')
            axes[1].set_xlabel('Radius of Gyration (μm)')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Radius of Gyration Distribution')
            axes[1].grid(True, alpha=0.3)

            # Scatter plot
            rc_df = pd.DataFrame(rc_data)
            rg_df = pd.DataFrame(rg_data)
            merged = pd.merge(rc_df, rg_df, on='track_id', how='inner')

            axes[2].scatter(merged['radius_confinement'], merged['radius_gyration'],
                            alpha=0.6, s=20)
            axes[2].set_xlabel('Radius of Confinement (μm)')
            axes[2].set_ylabel('Radius of Gyration (μm)')
            axes[2].set_title('RC vs RG Correlation')
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Radius distribution plot saved to {save_path}")

        plt.show()


def main():
    """Example usage of Radius Calculator"""
    import argparse
    from track_processor import TrackProcessor, TrackProcessorConfig
    from msd_analyzer import MSDAnalyzer, MSDAnalyzerConfig

    parser = argparse.ArgumentParser(description='Radius Calculator Example')
    parser.add_argument('input_file', nargs='?', help='Path to tracked table file')
    parser.add_argument('--create-sample', action='store_true',
                        help='Create sample data for testing')
    parser.add_argument('--plot', action='store_true',
                        help='Plot radius distributions')
    parser.add_argument('--export-rc', help='Export RC data to file')
    parser.add_argument('--export-rg-rc', help='Export RC and RG data to file')
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

    # Initialize components
    track_config = TrackProcessorConfig(
        track_length_cutoff=args.track_length,
        acquisition_time=args.acquisition_time,
        pixel_size=args.pixel_size
    )

    msd_config = MSDAnalyzerConfig()
    radius_config = RadiusCalculatorConfig()

    processor = TrackProcessor(track_config)
    msd_analyzer = MSDAnalyzer(msd_config)
    rc_calculator = RadiusOfConfinementCalculator(radius_config)
    rg_calculator = RadiusOfGyrationCalculator(radius_config)
    exporter = RadiusDataExporter()

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

    # Compute Radius of Confinement
    rc_data = rc_calculator.compute_radius_of_confinement(msd_data)

    if not rc_data:
        print("No valid RC data found!")
        return

    # Compute Radius of Gyration
    rg_data = rg_calculator.compute_radius_of_gyration(reconstructed_tracks)

    print(f"\nRadius Analysis Results:")
    print(f"  Total tracks: {len(reconstructed_tracks)}")
    print(f"  Valid MSD tracks: {len(msd_data)}")
    print(f"  RC tracks: {len(rc_data)}")
    print(f"  RG tracks: {len(rg_data)}")

    # Plot results
    if args.plot:
        exporter.plot_radius_distributions(rc_data, rg_data)

    # Export data
    if args.export_rc:
        exporter.export_rc_only(rc_data, args.export_rc)

    if args.export_rg_rc:
        exporter.export_rc_rg(rc_data, rg_data, args.export_rg_rc)

    print(f"\n✓ Radius analysis completed!")


if __name__ == "__main__":
    main()
