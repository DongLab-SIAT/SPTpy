#!/usr/bin/env python3
"""
Python 3.10 implementation of RC_RG_combined.m with automated directory processing
Added: support for an 'after_seg' set of directories which are plotted as solid lines.
Export: all targets (before/after) ΔC into a single Excel sheet for GraphPad Prism.
Usage:
    python rc_rg_combined_auto_with_export.py --save delta_c_plot.png --excel deltaC_all_targets.xlsx
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union, Sequence
import argparse

try:
    from sklearn.neighbors import KernelDensity
except ImportError:
    KernelDensity = None


class RCRGCombinedAnalyzer:
    """Analyzer for RC/RG combined data with automated directory processing (with after_seg export)"""

    def __init__(self, bandwidth: float = 0.01):
        self.bandwidth = bandwidth

        # ---------- Original (pre-seg) data directories ----------
        self.target_directories: Dict[str, Union[str, List[str]]] = {
            # MED1 / MED6 (generate RG+RC)
            "MED1_auxin": r"G:\50ms_ng_dataset\20191213_MED1_MED6_spt\MED1\6h_auxin",
            "MED1_no_auxin": r"G:\50ms_ng_dataset\20191213_MED1_MED6_spt\MED1\no_auxin",
            "MED6_auxin": r"G:\50ms_ng_dataset\20191213_MED1_MED6_spt\MED6\6h_auxin",
            "MED6_no_auxin": r"G:\50ms_ng_dataset\20191213_MED1_MED6_spt\MED6\no_auxin",

            # BRD4 (generate RC only)
            "BRD4_auxin": r"G:\50ms_ng_dataset\20210129_RAD21_Halo-Brd4\6h_auxin",
            "BRD4_no_auxin": r"G:\50ms_ng_dataset\20210129_RAD21_Halo-Brd4\no_auxin",

            # OCT4 (generate RC only)
            "OCT4_auxin": r"G:\50ms_ng_dataset\20210212_RAD21_Halo-Oct4\6h_auxin",
            "OCT4_no_auxin": r"G:\50ms_ng_dataset\20210212_RAD21_Halo-Oct4\no_auxin",

            # mTBP (TBP) (generate RC only)
            "TBP_auxin": r"G:\50ms_ng_dataset\20210222_RAD21_Halo-mTBP\6h_auxin",
            "TBP_no_auxin": r"G:\50ms_ng_dataset\20210222_RAD21_Halo-mTBP\no_auxin",

            # H2B (generate RC only) <- merged from two sources
            "H2B_auxin": [
                r"G:\50ms_ng_dataset\H2B_50ms\1st\RAD21",
            ],
            "H2B_no_auxin": [
                r"G:\50ms_ng_dataset\H2B_50ms\1st\Ctrl",
            ],

            # H2AZ (generate RC only)
            "H2AZ_auxin": [
                r"G:\50ms_ng_dataset\H2AZ_50ms\TIF\RAD21_2",
            ],
            "H2AZ_no_auxin": [
                r"G:\50ms_ng_dataset\H2AZ_50ms\TIF\Ctrl_2",
            ],
        }

        # ---------- after_seg (post-segmentation) data directories ----------
        self.afterseg_directories: Dict[str, Union[str, List[str]]] = {
            # MED1
            "MED1_afterseg_auxin": r"G:\50ms_ng_dataset\MED1_after_seg\6h_auxin",
            "MED1_afterseg_no_auxin": r"G:\50ms_ng_dataset\MED1_after_seg\0h_auxin",

            # MED6
            "MED6_afterseg_auxin": r"G:\50ms_ng_dataset\MED6_after_seg\6h_auxin",
            "MED6_afterseg_no_auxin": r"G:\50ms_ng_dataset\MED6_after_seg\0h_auxin",

            # BRD4
            "BRD4_afterseg_auxin": r"G:\50ms_ng_dataset\Brd4_after_seg\6h_auxin",
            "BRD4_afterseg_no_auxin": r"G:\50ms_ng_dataset\Brd4_after_seg\no_auxin",

            # OCT4
            "OCT4_afterseg_auxin": r"G:\50ms_ng_dataset\Oct4_after_seg\6h_auxin",
            "OCT4_afterseg_no_auxin": r"G:\50ms_ng_dataset\Oct4_after_seg\no_auxin",

            # TBP
            "TBP_afterseg_auxin": r"G:\50ms_ng_dataset\mTBP_after_seg\6h_auxin",
            "TBP_afterseg_no_auxin": r"G:\50ms_ng_dataset\mTBP_after_seg\no_auxin",

            # H2B (list)
            "H2B_afterseg_auxin": [
                r"G:\50ms_ng_dataset\H2B_after_seg\1st\RAD21",
            ],
            "H2B_afterseg_no_auxin": [
                r"G:\50ms_ng_dataset\H2B_after_seg\1st\Ctrl",
            ],

            # H2AZ (list)
            "H2AZ_afterseg_auxin": [
                r"G:\50ms_ng_dataset\H2AZ_after_seg\RAD21_2",
            ],
            "H2AZ_afterseg_no_auxin": [
                r"G:\50ms_ng_dataset\H2AZ_after_seg\Ctrl_2",
            ],
        }

        # Target groups for main analysis
        self.target_groups: Dict[str, Tuple[str, str]] = {
            "MED1": ("MED1_auxin", "MED1_no_auxin"),
            "MED6": ("MED6_auxin", "MED6_no_auxin"),
            "BRD4": ("BRD4_auxin", "BRD4_no_auxin"),
            "OCT4": ("OCT4_auxin", "OCT4_no_auxin"),
            "TBP": ("TBP_auxin", "TBP_no_auxin"),
            "H2B": ("H2B_auxin", "H2B_no_auxin"),
            "H2AZ": ("H2AZ_auxin", "H2AZ_no_auxin"),
        }

        # after_seg mapping
        self.afterseg_groups: Dict[str, Tuple[str, str]] = {
            "MED1": ("MED1_afterseg_auxin", "MED1_afterseg_no_auxin"),
            "MED6": ("MED6_afterseg_auxin", "MED6_afterseg_no_auxin"),
            "BRD4": ("BRD4_afterseg_auxin", "BRD4_afterseg_no_auxin"),
            "OCT4": ("OCT4_afterseg_auxin", "OCT4_afterseg_no_auxin"),
            "TBP": ("TBP_afterseg_auxin", "TBP_afterseg_no_auxin"),
            "H2B": ("H2B_afterseg_auxin", "H2B_afterseg_no_auxin"),
            "H2AZ": ("H2AZ_afterseg_auxin", "H2AZ_afterseg_no_auxin"),
        }

        # Colors for plotting
        self.colors = {
            'MED1': '#e41a1c',  # red
            'MED6': '#ff7f00',  # orange
            'BRD4': '#ff33cc',  # magenta
            'OCT4': '#00bcd4',  # cyan
            'TBP': '#377eb8',  # blue
            'H2B': '#000000',  # black
            'H2AZ': '#808080',
        }

    def find_data_files(self, directory_or_dirs: Union[str, List[str]]) -> List[str]:
        """Find RC, RG+RC, or simple RC files in one or multiple directories"""
        dirs: List[str] = [directory_or_dirs] if isinstance(directory_or_dirs, str) else list(directory_or_dirs)
        all_files: List[str] = []

        for directory in dirs:
            if not os.path.exists(directory):
                print(f"Directory does not exist: {directory}")
                continue

            rc_files = glob.glob(os.path.join(directory, "**", "*_table_RC.txt"), recursive=True)
            rg_rc_files = glob.glob(os.path.join(directory, "**", "*_table_RG_RC.txt"), recursive=True)
            simple_rc_files = glob.glob(os.path.join(directory, "**", "*_RC.txt"), recursive=True)
            simple_rc_files = [
                f for f in simple_rc_files
                if not (f.endswith("_table_RC.txt") or f.endswith("_table_RG_RC.txt"))
            ]

            files_in_dir = rc_files + rg_rc_files + simple_rc_files
            all_files.extend(files_in_dir)

            print(f"Found {len(files_in_dir)} files in {directory}")
            if rc_files:
                print(f"  RC files: {len(rc_files)}")
            if rg_rc_files:
                print(f"  RG+RC files: {len(rg_rc_files)}")
            if simple_rc_files:
                print(f"  Simple RC files: {len(simple_rc_files)}")

        all_files = sorted(set(all_files))
        return all_files

    def load_and_filter_data(self, files: List[str]) -> np.ndarray:
        """Load and filter data from RC, RG+RC, or simple RC files"""
        all_data = []
        print(f"\nDEBUG: Processing {len(files)} files...")

        for file_path in files:
            print(f"\nProcessing: {os.path.basename(file_path)}")
            try:
                print("  Reading file...")
                df = pd.read_csv(file_path, sep='\t', engine='python')
                print(f"  File loaded: {df.shape[0]} rows, {df.shape[1]} columns")

                if df.empty or df.shape[1] < 3:
                    print("  Warning: skipping empty file or file with insufficient columns")
                    continue

                print("  Checking headers...")
                try:
                    first0 = str(df.iloc[0, 0])
                    first1 = str(df.iloc[0, 1])
                except Exception:
                    first0, first1 = '', ''
                print(f"    First row: {first0} | {first1}")
                if first0 == 'track_id' or 'radius_confinement' in first1:
                    print("  Header detected, skipping first row")
                    df = df.iloc[1:].reset_index(drop=True)
                    print(f"  After header removal: {df.shape[0]} rows")
                else:
                    print("  No header detected, using all data")

                print("  Converting to numeric...")
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                nan_counts = df.isna().sum()
                print(f"  NaN counts: {nan_counts.to_dict()}")

                print("  Removing rows with NaN in RC or D columns...")
                rc_col = df.columns[1]  # radius_confinement
                d_col = df.columns[2]   # diffusion_coefficient
                before_dropna = len(df)
                df = df.dropna(subset=[rc_col, d_col])
                after_dropna = len(df)
                print(f"  Before dropna: {before_dropna} rows, After: {after_dropna} rows")

                if df.empty:
                    print("  Warning: no data left after removing NaN values")
                    continue

                print("  Data ranges before filtering:")
                print(f"    RC ({rc_col}): {df[rc_col].min():.6f} to {df[rc_col].max():.6f}")
                print(f"    D ({d_col}):  {df[d_col].min():.6f} to {df[d_col].max():.6f}")

                if "_table_RC.txt" in file_path or file_path.endswith("_RC.txt"):
                    print("  File type: RC file")
                    mask = (df[rc_col] > 0.001) & (df[rc_col] < 20.0) & \
                           (df[d_col] > 0) & (df[d_col] < 0.1)
                elif "_table_RG_RC.txt" in file_path:
                    print("  File type: RG+RC file")
                    mask = (df[rc_col] > 0.001) & (df[rc_col] < 20.0) & \
                           (df[d_col] > 0) & (df[d_col] < 0.1)
                else:
                    print("  Warning: unknown file type, skipping")
                    continue

                print(f"  Filtered rows: {mask.sum()} / {len(df)}")
                rc_data = df.loc[mask, rc_col].to_numpy(dtype=float)

                if len(rc_data) > 0:
                    print(f"  Loaded {len(rc_data)} RC values, range {rc_data.min():.6f}–{rc_data.max():.6f}")
                    all_data.extend(rc_data)
                else:
                    print("  Warning: no RC values after filtering")

            except Exception as e:
                print(f"  Error loading {os.path.basename(file_path)}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\nDEBUG SUMMARY: Total RC values collected: {len(all_data)}")
        return np.array(all_data)

    def kde_pdf_epanechnikov(self, x_values: np.ndarray, samples: np.ndarray) -> np.ndarray:
        """Estimate PDF using Epanechnikov kernel"""
        if KernelDensity is None:
            raise RuntimeError("scikit-learn is required for KDE. Please install scikit-learn.")

        if samples.size == 0:
            return np.zeros_like(x_values)

        x_grid = x_values.reshape(-1, 1)
        ss = samples.reshape(-1, 1)
        kde = KernelDensity(kernel='epanechnikov', bandwidth=self.bandwidth)
        kde.fit(ss)
        log_dens = kde.score_samples(x_grid)
        pdf = np.exp(log_dens)

        dx = np.mean(np.diff(x_values)) if x_values.size > 1 else 1.0
        area = np.sum(pdf) * dx
        if area > 0:
            pdf = pdf / area

        return pdf

    def to_cdf(self, pdf: np.ndarray) -> np.ndarray:
        """Convert PDF to CDF"""
        if pdf.size == 0:
            return pdf
        cdf = np.cumsum(pdf)
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]
        return cdf

    def process_single_target(self, target_name: str,
                              auxin_dir: Union[str, List[str]],
                              no_auxin_dir: Union[str, List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single target (auxin vs no_auxin)"""
        print(f"\nProcessing {target_name}...")

        auxin_files = self.find_data_files(auxin_dir)
        auxin_data = self.load_and_filter_data(auxin_files)
        print(f"  Auxin: {len(auxin_data)} RC values")

        no_auxin_files = self.find_data_files(no_auxin_dir)
        no_auxin_data = self.load_and_filter_data(no_auxin_files)
        print(f"  No auxin: {len(no_auxin_data)} RC values")

        if len(auxin_data) == 0 or len(no_auxin_data) == 0:
            print(f"  Warning: insufficient data for {target_name}")
            return np.array([]), np.array([])

        return auxin_data, no_auxin_data

    def generate_delta_c_plot_and_export(self, save_path: Optional[str] = None,
                                         excel_path: str = "deltaC_all_targets.xlsx"):
        """
        Generate ΔC plot for all targets and export results to a single-sheet Excel file.
        Pre-seg (self.target_directories) plotted as dashed (--).
        After-seg (self.afterseg_directories) plotted as solid (-).
        """
        print("\nGenerating ΔC plot for all targets and exporting to Excel...")

        x_values = np.arange(0.02, 0.300 + 1e-12, 0.002)
        x_nm = x_values * 1000.0

        # For Excel export
        export_dict = {"RoC (nm)": x_nm.copy()}

        plt.figure(figsize=(12, 6))

        for target_name, (auxin_key, no_auxin_key) in self.target_groups.items():
            # Pre-seg (dashed)
            pre_auxin_dir = self.target_directories.get(auxin_key)
            pre_noaux_dir = self.target_directories.get(no_auxin_key)
            pre_auxin_data, pre_noaux_data = self.process_single_target(target_name, pre_auxin_dir, pre_noaux_dir)

            delta_pre = None
            delta_after = None

            if len(pre_auxin_data) > 0 and len(pre_noaux_data) > 0:
                pdf_pre_auxin = self.kde_pdf_epanechnikov(x_values, pre_auxin_data)
                pdf_pre_noaux = self.kde_pdf_epanechnikov(x_values, pre_noaux_data)
                cdf_pre_auxin = self.to_cdf(pdf_pre_auxin)
                cdf_pre_noaux = self.to_cdf(pdf_pre_noaux)
                delta_pre = 0.5 * (cdf_pre_noaux - cdf_pre_auxin)
                try:
                    from scipy.ndimage import gaussian_filter1d
                    delta_pre = gaussian_filter1d(delta_pre, sigma=2.5)
                except Exception:
                    pass
                delta_pre[0] = 0.0
                delta_pre[-1] = 0.0
                color = self.colors.get(target_name, '#444444')
                plt.plot(
                    x_nm, delta_pre, '--', color=color,
                    label=f"{target_name} (before_seg)", linewidth=1.6, alpha=0.95
                )
            else:
                print(f"  Pre-seg data insufficient: {target_name} (will be NaN in Excel)")

            # After-seg (solid)
            after_keys = self.afterseg_groups.get(target_name)
            if after_keys:
                after_auxin_key, after_noauxin_key = after_keys
                after_auxin_dir = self.afterseg_directories.get(after_auxin_key)
                after_no_auxin_dir = self.afterseg_directories.get(after_noauxin_key)
                if after_auxin_dir and after_no_auxin_dir:
                    after_auxin_files = self.find_data_files(after_auxin_dir)
                    after_noaux_files = self.find_data_files(after_no_auxin_dir)
                    after_auxin_data = self.load_and_filter_data(after_auxin_files)
                    after_noaux_data = self.load_and_filter_data(after_noaux_files)

                    if len(after_auxin_data) > 0 and len(after_noaux_data) > 0:
                        pdf_a_auxin = self.kde_pdf_epanechnikov(x_values, after_auxin_data)
                        pdf_a_noaux = self.kde_pdf_epanechnikov(x_values, after_noaux_data)
                        cdf_a_auxin = self.to_cdf(pdf_a_auxin)
                        cdf_a_noaux = self.to_cdf(pdf_a_noaux)
                        delta_after = 0.5 * (cdf_a_noaux - cdf_a_auxin)
                        try:
                            from scipy.ndimage import gaussian_filter1d
                            delta_after = gaussian_filter1d(delta_after, sigma=2.5)
                        except Exception:
                            pass
                        delta_after[0] = 0.0
                        delta_after[-1] = 0.0
                        color = self.colors.get(target_name, '#444444')
                        if target_name == 'TBP':
                            plt.plot(
                                x_nm, delta_after, '-', color=color,
                                label=f"{target_name} (after_seg)",
                                linewidth=2.5, marker='o', markevery=8, markersize=4
                            )
                        else:
                            plt.plot(
                                x_nm, delta_after, '-', color=color,
                                label=f"{target_name} (after_seg)",
                                linewidth=2.5
                            )
                    else:
                        print(f"  after_seg data insufficient: {target_name} (will be NaN in Excel)")
                else:
                    print(f"  afterseg_directories not configured or path does not exist for {target_name}")

            # Collect columns for Excel (fill with NaN if no data)
            n_points = x_nm.size
            if delta_pre is None:
                export_dict[f"{target_name}_before_seg"] = np.full(n_points, np.nan)
            else:
                export_dict[f"{target_name}_before_seg"] = delta_pre

            if delta_after is None:
                export_dict[f"{target_name}_after_seg"] = np.full(n_points, np.nan)
            else:
                export_dict[f"{target_name}_after_seg"] = delta_after

        # Figure formatting
        plt.axhline(0.0, color='k', linewidth=0.8, alpha=0.5)
        plt.xlim([0, 300])
        plt.ylim([-0.03, 0.10])

        plt.xlabel('RoC (nm)')
        plt.ylabel(r'Differential cumulative probability ($\Delta$C)', fontsize=12)
        plt.grid(False)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=10)
        # legend placed on the left outside the plot area
        ax.legend(loc='center left', bbox_to_anchor=(-0.22, 0.5), frameon=False, fontsize=8, ncol=1)

        plt.tight_layout()

        # Export Excel (single sheet)
        try:
            df_export = pd.DataFrame(export_dict)
            cols = ["RoC (nm)"]
            for t in self.target_groups.keys():
                cols.append(f"{t}_before_seg")
                cols.append(f"{t}_after_seg")
            df_export = df_export[cols]
            df_export.to_excel(excel_path, index=False)
            print(f"Saved Excel: {excel_path}")
        except Exception as e:
            print(f"Failed to export Excel: {e}")

        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved figure: {save_path}")

        plt.show()

    def run_analysis(self, save_path: Optional[str] = None, excel_path: str = "deltaC_all_targets.xlsx"):
        """Run complete analysis and export"""
        print("RC/RG Combined Analysis - Automated Processing")
        print("=" * 60)

        missing_dirs: List[str] = []
        for key, directory in self.target_directories.items():
            if isinstance(directory, list):
                missing = [d for d in directory if not os.path.exists(d)]
                if missing:
                    missing_dirs.append(f"{key}: {missing}")
            else:
                if not os.path.exists(directory):
                    missing_dirs.append(f"{key}: {directory}")

        if missing_dirs:
            print("Warning: Missing directories:")
            for missing in missing_dirs:
                print(f"  {missing}")
            print()

        self.generate_delta_c_plot_and_export(save_path=save_path, excel_path=excel_path)

        print("\nAnalysis completed!")


def plot_delta_c_from_two_conditions(
    condA_rc_files: Sequence[str],
    condB_rc_files: Sequence[str],
    label: str = "MyTarget",
    subtract_mode: str = "B-A",   # "B-A" or "A-B"
    bandwidth: float = 0.01,
    save_path: Optional[str] = None,
    excel_path: Optional[str] = None,
):
    """
    Generate a single ΔC curve from two sets of RC files and optionally
    plot it and export to an Excel file.

    condA_rc_files: list of RC files for condition A
    condB_rc_files: list of RC files for condition B
    subtract_mode:  "B-A" means ΔC = CDF(B) - CDF(A)
                    "A-B" means ΔC = CDF(A) - CDF(B)
    label: curve name (used in legend and Excel column name)
    """

    analyzer = RCRGCombinedAnalyzer(bandwidth=bandwidth)

    # Use the analyzer's existing load + filter logic
    condA_data = analyzer.load_and_filter_data(list(condA_rc_files))
    condB_data = analyzer.load_and_filter_data(list(condB_rc_files))

    if condA_data.size == 0 or condB_data.size == 0:
        print("RC data for condition A or B is empty; cannot compute ΔC.")
        return

    # x-axis grid
    x_values = np.arange(0.02, 0.300 + 1e-12, 0.002)
    x_nm = x_values * 1000.0

    # KDE & CDF
    pdf_A = analyzer.kde_pdf_epanechnikov(x_values, condA_data)
    pdf_B = analyzer.kde_pdf_epanechnikov(x_values, condB_data)
    cdf_A = analyzer.to_cdf(pdf_A)
    cdf_B = analyzer.to_cdf(pdf_B)

    if subtract_mode == "B-A":
        delta_c = 0.5 * (cdf_B - cdf_A)
    else:  # "A-B"
        delta_c = 0.5 * (cdf_A - cdf_B)

    # Optional smoothing
    try:
        from scipy.ndimage import gaussian_filter1d
        delta_c = gaussian_filter1d(delta_c, sigma=2.5)
    except Exception:
        pass

    # Force endpoints to zero
    delta_c[0] = 0.0
    delta_c[-1] = 0.0

    # Plot curve
    plt.figure(figsize=(8, 5))
    color = analyzer.colors.get(label, '#444444')
    plt.plot(x_nm, delta_c, '-', color=color, label=label, linewidth=2.0)

    plt.axhline(0.0, color='k', linewidth=0.8, alpha=0.5)
    plt.xlim([0, 300])
    plt.ylim([-0.03, 0.10])
    plt.xlabel('RoC (nm)')
    plt.ylabel(r'Differential cumulative probability ($\Delta$C)', fontsize=12)
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved ΔC figure: {save_path}")

    # Save Excel
    if excel_path:
        df_export = pd.DataFrame({
            "RoC (nm)": x_nm,
            f"{label}_deltaC": delta_c
        })
        df_export.to_excel(excel_path, index=False)
        print(f"Saved ΔC Excel: {excel_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Automated RC/RG combined analysis (with after_seg export)')
    parser.add_argument('--bandwidth', type=float, default=0.01,
                        help='KDE bandwidth (default: 0.01)')
    parser.add_argument('--save', type=str, default=None,
                        help='Output figure path (e.g., delta_c_plot.png)')
    parser.add_argument('--excel', type=str, default="deltaC_all_targets.xlsx",
                        help='Output Excel path (e.g., deltaC_all_targets.xlsx)')

    args = parser.parse_args()

    analyzer = RCRGCombinedAnalyzer(bandwidth=args.bandwidth)
    analyzer.run_analysis(save_path=args.save, excel_path=args.excel)


if __name__ == '__main__':
    main()
