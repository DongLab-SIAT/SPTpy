#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:Liao Shasha
@file: SLIMfast_v16.py
@institute:SIAT
@location:Shenzhen,China
@time: 2025/08/16
这个版本是修改并行和串行的定位问题
nuclear segmentation的问题已经解决了
"""
from pathlib import Path

# -*- coding: utf-8 -*-
import torch
from scipy.ndimage import gaussian_filter1d
from torch import nn, optim
from torch.utils.data import DataLoader
import cv2
import imageio
import lmfit
from PIL.ImageDraw import ImageDraw
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from scipy.special import erfc
from numba import njit
import math
from matplotlib.patches import Polygon, Rectangle
import glob
import tkinter as tk
import time
import warnings
from datetime import datetime
from tkinter import messagebox, filedialog, Canvas, ttk, simpledialog, \
    colorchooser
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.io import loadmat
from scipy.stats import multivariate_normal, gaussian_kde
from scipy.optimize import minimize, curve_fit
from skimage.io import imread
from PIL import Image, ImageTk, ImageFilter, ImageDraw
import scipy.io as sio
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import numpy as np
from scipy.stats import chi2
import os
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, KDTree
from tifffile import tifffile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
from tqdm import tqdm
from collections import defaultdict, Counter
import itertools
import statsmodels.api as sm
from sklearn.neighbors import KDTree
import cProfile, pstats
from numpy.fft import fft2, ifft2, fftshift
from unet_attention_model import MultiFusionAttentionUNet,DualInputDataset,train_transform

TFHM  = None
TFHGC = None
SGC2  = None
from automated_spt_processor import AutomatedSPTProcessor, AutomatedProcessorConfig
from rc_rg_combined_auto import plot_delta_c_from_two_conditions
from sklearn.neighbors import KernelDensity

# ===== Motion trajectory 4 categories =====
CATEGORIES = [
    'Free trajectory',
    'Condensate trajectory',
    'Condensate to free',
    'Free to condensate',
]




class SlimFastApp:
    def __init__(self, master):
        self.master = master
        master.title("SPTpy")
        master.geometry("250x200")


        # Initialize data storage using a dictionary
        self.image_bin = {
            'is_own': 1,
            'is_loaded': 0,
            'is_superstack': 0,
            'is_track': 0,
            'frame': 1,
            'view_mode': 'monoView',
            'loc_start': 1,
            'loc_end': float('inf'),
            'error_rate': -6,
            'w2d': 7,
            'dfltn_loops': 0,
            'min_int': 0,
            'loc_parallel': 0,
            'n_cores': 1,
            'spatial_correction': 0,
            'is_radius_tol': 0,
            'radius_tol': 50,
            'pos_tol': 1.5,
            'max_optim_iter': 50,
            'term_tol': -2,
            'is_scalebar': 0,
            'micron_bar_length': 1000,
            'is_colormap': 0,
            'is_timestamp': 0,
            'timestamp_size': 2,
            'timestamp_increment': 0.032,
            'colormap_width': 10,
            'exf_old': 1,
            'exf_new': 5,
            'r_w': 50,
            'r_step': 10,
            'r_start': 1,
            'r_end': float('inf'),
            'r_live': 0,
            'fps': 25,
            'mov_compression': 1,
            'conv_mode': 1,
            'int_weight': 0.1,
            'size_fac': 1,
            'is_cumsum': 0,
            'is_thresh_loc_prec': 0,
            'min_loc': 0,
            'max_loc': float('inf'),
            'is_thresh_snr': 0,
            'min_snr': 0,
            'max_snr': float('inf'),
            'is_thresh_density': 0,
            'cluster_mode': 1,
            'px_size': 0.16,
            'cnts_per_photon': 20.2,
            'em_wavelength': 590,
            'na': 1.49,
            'psf_scale': 1.35,
            'psf_std': 1.03,
            'pathname': '',
            'filename': '',
            'width': 0,###示例值有些是100
            'height': 0,###示例值有些是100,这几个值估计之后要改
            'roi': [0, 0, 0, 0],
            'h_roi': None,
            'intensity_range': [0,255],
            'image_name': '',
            'image_bin': {},
            'settings': {},
            'tracks': [],
            'n_tracks': 0,
            'search_path': '',
            'stack': [],
            'image': None,
            'h_image': None,
            'stack_size': [],
            'roi_x0': 0,
            'roi_y0': 0,
            'hROI': None,
            'ctrsN':[],
            'nImCh': 1,  # Number of image channels
            'imCh': [0, 1],  # Example channel indices
            'ctrsX': 0,  # Example X coordinates
            'ctrsY':0,
            'drawing': False,
            'roi_coords': [],
            'locParallel': False,
            'nCores': 1,
            'errorRate': -6,
            'dfltnLoops': 0,
            'minInt': 0,
            'isRadiusTol': False,
            'radiusTol': 10,
            'posTol': 1.5,
            'maxOptimIter': 50,
            'termTol': -2,
            'psfStd': 1.03,
            'spatialCorrection': False,
            'tMat': None,  # Placeholder for spatial transformation matrix
            'locStart':1,
            'locEnd': float('inf'),
        }
        self.track_list = []

        ###图像显示窗口引用
        self.img_window = None
        self.roi_canvas = None
        self.current_img_tk = None

        #把optics面板里的子Frame存在这里
        self.optics_panels = {}

        # Create menu
        self.menu = tk.Menu(master)
        master.config(menu=self.menu)


        # Load menu
        self.load_menu = tk.Menu(self.menu,tearoff=0)
        self.menu.add_cascade(label="Load", menu=self.load_menu)
        self.load_menu.add_command(label="Imagestack", command=self.load_imagestack)
        # self.load_menu.add_command(label="Superstack", command=self.load_superstack)
        self.load_menu.add_command(label="Batch Localization", command=self.batch_localization)

        self.particle_data_menu = tk.Menu(self.load_menu,tearoff=0)
        self.load_menu.add_cascade(label="Particle Data", menu=self.particle_data_menu)
        self.particle_data_menu.add_command(label="SLIMfast", command=self.load_slimfast)
        # self.particle_data_menu.add_command(label="FPALM", command=self.load_data, state='disabled')
        self.load_menu.add_command(label="Tracking Data", command=self.load_tracking_txt)

        # Channel menu
        self.channel_menu = tk.Menu(self.menu,tearoff=0)
        self.menu.add_cascade(label="Analysis", menu=self.channel_menu)
        self.channel_menu.add_command(label="spot on", command=self.load_spoton_core)
        self.channel_menu.add_command(label="RoC",command=self.open_roc_param_window)
        self.channel_menu.add_command(label="Motion trajectory", command=self.open_motion_trajectory_window)

        # Extras menu
        self.extras_menu = tk.Menu(self.menu,tearoff=0)
        self.menu.add_cascade(label="Extras", menu=self.extras_menu)
        self.extras_menu.add_command(label="Help", command=self.show_help, state='disabled')
        self.extras_menu.add_command(label="Credits", command=self.show_credits)
        self.extras_menu.add_command(label="Changelog", command=self.show_change_log)

        # Canvas for drawing
        self.canvas = tk.Canvas(master, bg='white', width=300, height=300)
        self.canvas.grid(row=0, column=0)

        # Add more widgets and layout as needed

    def open_motion_trajectory_window(self):
        """Motion trajectory entry: select files → open parameter window"""
        paths = filedialog.askopenfilenames(
            title="Select trajectory table files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not paths:
            return

        self.motion_files = list(paths)
        self.open_motion_param_window()

    def open_motion_param_window(self):
        """弹出参数调整窗口"""
        win = tk.Toplevel(self.master)
        win.title("Motion trajectory parameters")

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill="both", expand=True)

        # 简单提示一下选了多少文件
        ttk.Label(frame, text=f"Selected files: {len(self.motion_files)}").grid(row=0, column=0, columnspan=2,
                                                                                sticky="w")
        ttk.Label(frame, text=os.path.basename(self.motion_files[0])).grid(row=1, column=0, columnspan=2, sticky="w")

        # 参数变量（默认值按你脚本里的）
        self.mt_pixel_var = tk.DoubleVar(value=109.0)
        self.mt_radius_var = tk.DoubleVar(value=80.0)
        self.mt_count_var = tk.IntVar(value=60)

        ttk.Label(frame, text="Pixel size (nm/px)").grid(row=2, column=0, sticky="e", pady=2)
        ttk.Entry(frame, textvariable=self.mt_pixel_var, width=10).grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(frame, text="Neighborhood radius (nm)").grid(row=3, column=0, sticky="e", pady=2)
        ttk.Entry(frame, textvariable=self.mt_radius_var, width=10).grid(row=3, column=1, sticky="w", pady=2)

        ttk.Label(frame, text="Neighbor threshold").grid(row=4, column=0, sticky="e", pady=2)
        ttk.Entry(frame, textvariable=self.mt_count_var, width=10).grid(row=4, column=1, sticky="w", pady=2)

        ttk.Button(
            frame,
            text="Analyze",
            command=lambda: self.run_motion_trajectory_analysis(win)
        ).grid(row=5, column=0, columnspan=2, pady=10)

    def run_motion_trajectory_analysis(self, param_window):
        """读取参数并对选中的文件做分类统计"""
        # 关掉参数窗口
        param_window.destroy()

        pixel_size_nm = float(self.mt_pixel_var.get())
        threshold_nm = float(self.mt_radius_var.get())
        count_threshold = int(self.mt_count_var.get())

        threshold_um = threshold_nm / 1000.0

        global_counter = Counter({k: 0 for k in CATEGORIES})
        per_file_results = []

        for path in self.motion_files:
            labels = self._mt_label_in_out(
                path,
                pixel_size_nm=pixel_size_nm,
                threshold_um=threshold_um,
                count_threshold=count_threshold
            )
            n_traj = len(labels)

            cats = [self._mt_classify_traj(states) for states in labels.values()]
            counts = Counter(cats)

            # 记录每个文件的结果，方便在结果窗口里展示
            file_info = {
                "file": os.path.basename(path),
                "Ntraj": n_traj,
                "counts": {},
                "percentages": {},
            }
            for cat in CATEGORIES:
                cnt = counts.get(cat, 0)
                pct = (cnt * 100.0 / n_traj) if n_traj > 0 else 0.0
                file_info["counts"][cat] = cnt
                file_info["percentages"][cat] = pct

            per_file_results.append(file_info)

            # 累加到全局
            global_counter.update({k: counts.get(k, 0) for k in CATEGORIES})

        # 展示结果
        self.show_motion_results_window(per_file_results, global_counter)

    def _mt_label_in_out(self, path, pixel_size_nm, threshold_um, count_threshold):
        """
        对单个文件返回 {track_id: bool_array}，逻辑与脚本中 label_in_out 相同
        """
        data = np.loadtxt(path, usecols=(0, 1, 2, 3))  # x, y, frame, track_id

        # px → μm
        data[:, 0] *= pixel_size_nm / 1000.0
        data[:, 1] *= pixel_size_nm / 1000.0

        coords = data[:, :2]
        tree = cKDTree(coords)
        neighbors = tree.query_ball_point(coords, r=threshold_um)
        counts = np.array([len(nb) for nb in neighbors])
        inside = counts > count_threshold

        from collections import defaultdict
        traj_indices = defaultdict(list)
        for idx, (_, _, frame, tid) in enumerate(data):
            traj_indices[int(tid)].append(idx)

        traj_labels = {}
        for tid, idxs in traj_indices.items():
            idxs_sorted = sorted(idxs, key=lambda i: data[i, 2])  # 按 frame 排序
            traj_labels[tid] = inside[idxs_sorted]

        return traj_labels

    def _mt_classify_traj(self, states: np.ndarray) -> str:
        """把一条轨迹的布尔 in/out 数组分类为四种情况"""
        if states.size == 0:
            return 'Other'
        if states.all():
            return 'Condensate trajectory'
        if not states.any():
            return 'Free trajectory'
        if states[0] and not states[-1]:
            return 'Condensate to free'
        if not states[0] and states[-1]:
            return 'Free to condensate'
        return 'Other'

    def show_motion_results_window(self, per_file_results, global_counter):
        """在新窗口显示每个文件和全局的分类比例"""
        win = tk.Toplevel(self.master)
        win.title("Motion trajectory results")

        frame = ttk.Frame(win, padding=10)
        frame.pack(fill="both", expand=True)

        # 可滚动文本框
        text = tk.Text(frame, width=90, height=30)
        text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)

        # ----- 全局汇总 -----
        total = sum(global_counter[c] for c in CATEGORIES) or 1

        text.insert("end", "=== Global summary (all files) ===\n")
        text.insert("end", f"{'Category':<25s} {'%':>7s} {'(n)':>8s}\n")
        for cat in CATEGORIES:
            n = global_counter[cat]
            pct = n * 100.0 / total
            text.insert("end", f"{cat:<25s} {pct:6.1f}% {n:8d}\n")

        text.insert("end", "\n=== Per file results ===\n")
        for info in per_file_results:
            text.insert("end", f"\n{info['file']}  (N = {info['Ntraj']})\n")

            # 用“非 Other 的总数”做分母，跟全局一致
            total_valid = sum(info["counts"][cat] for cat in CATEGORIES) or 1

            for cat in CATEGORIES:
                cnt = info["counts"][cat]
                pct = cnt * 100.0 / total_valid
                text.insert("end", f"  {cat:<25s} {pct:6.1f}% ({cnt:4d})\n")

        text.config(state="disabled")

        # 让 text 区域可以自动拉伸
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    def open_roc_param_window(self):
        """Popup window for RoC parameters + two buttons."""
        param_win = tk.Toplevel(self.master)
        param_win.title("RoC Parameters")

        # Acquisition time & pixel size
        acq_time_var = tk.StringVar(value="10")  # seconds
        pixel_size_var = tk.StringVar(value="0.109")  # µm/px (109 nm)

        row = 0
        tk.Label(param_win, text="Acquisition time Δt (s):").grid(row=row, column=0, sticky="e", padx=5, pady=3)
        tk.Entry(param_win, textvariable=acq_time_var, width=10).grid(row=row, column=1, sticky="w", padx=5, pady=3)
        row += 1

        tk.Label(param_win, text="Pixel size (µm/px):").grid(row=row, column=0, sticky="e", padx=5, pady=3)
        tk.Entry(param_win, textvariable=pixel_size_var, width=10).grid(row=row, column=1, sticky="w", padx=5, pady=3)
        row += 1

        def run_generate_rc():
            """Triggered when clicking 'generate rc file':
               1) read parameters
               2) let user choose *_table.txt
               3) generate *_RC.txt
            """
            # 1. Parameters
            try:
                acq_time = float(acq_time_var.get())
                pixel_size = float(pixel_size_var.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid acquisition time or pixel size.")
                return

            # 2. Choose tracked table files
            fnames = filedialog.askopenfilenames(
                title="Select tracked TXT files (*_table.txt)",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if not fnames:
                return

            # 3. Create config & processor
            config = AutomatedProcessorConfig(
                output_directory=None,
                track_length_cutoff=5,
                acquisition_time=acq_time,
                pixel_size=pixel_size,
                max_files_per_directory=None,
                skip_existing=True,
                log_level='INFO',
            )
            processor = AutomatedSPTProcessor(config)

            n_success = 0
            n_fail = 0

            for fname in fnames:
                path = Path(fname)
                print(f"[RoC] Processing: {path.name}")
                result = processor.process_single_file(
                    file_path=path,
                    target="TARGET",
                    condition="cond",
                    output_type="rc_only",
                )

                if result["status"] in ("success", "skipped"):
                    n_success += 1
                else:
                    n_fail += 1

            messagebox.showinfo(
                "Done",
                f"RC generation completed!\n\n"
                f"Total files: {len(fnames)}\n"
                f"Success/Exist: {n_success}\n"
                f"Failed: {n_fail}\n\n"
                f"RC files saved to the same folder as original tracked files."
            )
            # param_win.destroy()

        # Buttons in one row
        btn_frame = tk.Frame(param_win)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)

        tk.Button(btn_frame, text="Generate RC file", width=18, command=run_generate_rc).grid(
            row=0, column=0, padx=8
        )

        tk.Button(btn_frame, text="Generate ΔC plot", width=18, command=self.open_deltaC_window).grid(
            row=0, column=1, padx=8
        )

    def open_deltaC_window(self):
        """Full-feature ΔC GUI:
           - Add unlimited curves
           - Each curve has: label + RC files for Condition A & Condition B
           - Generate one ΔC plot containing all curves
        """
        win = tk.Toplevel(self.master)
        win.title("ΔC Analysis (from RC files)")

        dir_var = tk.StringVar(value="B_minus_A")

        row = 0

        tk.Label(
            win,
            text="For each curve: select Label + RC files of Condition A & Condition B.\n"
                 "Finally click 'Generate ΔC plot'.",
            justify="left"
        ).grid(row=row, column=0, columnspan=4, sticky="w", padx=5, pady=5)
        row += 1

        curves_frame = tk.Frame(win)
        curves_frame.grid(row=row, column=0, columnspan=4, padx=5, pady=3, sticky="w")
        row += 1

        curves = []  # list of dict: {"label_var", "files_A", "files_B"}

        def add_curve_row():
            """Add one curve block."""
            idx = len(curves)
            row_local = idx * 2

            label_var = tk.StringVar(value=f"Curve {idx + 1}")
            files_A = []
            files_B = []
            selA_var = tk.StringVar(value="None")
            selB_var = tk.StringVar(value="None")

            def choose_A():
                fns = filedialog.askopenfilenames(
                    parent=win,
                    title=f"Select RC files for Condition A (Curve {idx + 1})",
                    filetypes=[("RC files", "*_RC.txt"), ("Text files", "*.txt"), ("All files", "*.*")]
                )
                if fns:
                    files_A.clear()
                    files_A.extend(fns)
                    selA_var.set(f"{len(fns)} files")

            def choose_B():
                fns = filedialog.askopenfilenames(
                    parent=win,
                    title=f"Select RC files for Condition B (Curve {idx + 1})",
                    filetypes=[("RC files", "*_RC.txt"), ("Text files", "*.txt"), ("All files", "*.*")]
                )
                if fns:
                    files_B.clear()
                    files_B.extend(fns)
                    selB_var.set(f"{len(fns)} files")

            tk.Label(curves_frame, text=f"Curve {idx + 1} label:").grid(
                row=row_local, column=0, sticky="e", padx=5, pady=2
            )
            tk.Entry(curves_frame, textvariable=label_var, width=28).grid(
                row=row_local, column=1, sticky="w", padx=5, pady=2
            )

            tk.Button(curves_frame, text="Condition A: select RC", command=choose_A).grid(
                row=row_local, column=2, sticky="w", padx=5, pady=2
            )
            tk.Label(curves_frame, textvariable=selA_var).grid(
                row=row_local, column=3, sticky="w", padx=5, pady=2
            )

            tk.Label(curves_frame, text="").grid(row=row_local + 1, column=0)
            tk.Button(curves_frame, text="Condition B: select RC", command=choose_B).grid(
                row=row_local + 1, column=2, sticky="w", padx=5, pady=2
            )
            tk.Label(curves_frame, textvariable=selB_var).grid(
                row=row_local + 1, column=3, sticky="w", padx=5, pady=2
            )

            curves.append({
                "label_var": label_var,
                "files_A": files_A,
                "files_B": files_B,
            })

        add_curve_row()

        tk.Label(win, text="ΔC direction:").grid(row=row, column=0, sticky="e", padx=5, pady=3)
        tk.Radiobutton(win, text="ΔC = B − A", variable=dir_var, value="B_minus_A").grid(
            row=row, column=1, sticky="w", padx=5, pady=3
        )
        tk.Radiobutton(win, text="ΔC = A − B", variable=dir_var, value="A_minus_B").grid(
            row=row, column=2, sticky="w", padx=5, pady=3
        )
        row += 1

        tk.Button(win, text="➕ Add another curve", command=add_curve_row).grid(
            row=row, column=0, columnspan=4, pady=5
        )
        row += 1

        def run_generate_deltac():
            """Generate ΔC for all curves."""
            if not curves:
                messagebox.showwarning("Warning", "Please add at least one curve.")
                return

            if KernelDensity is None:
                messagebox.showerror("Error", "scikit-learn missing. Cannot compute KDE.")
                return

            x_values = np.arange(0.02, 0.300 + 1e-12, 0.002)
            x_nm = x_values * 1000.0
            self.delta_export_dict = {"RoC (nm)": x_nm.copy()}

            plt.figure(figsize=(8, 4))
            direction = dir_var.get()
            any_plotted = False

            for idx, cfg in enumerate(curves, start=1):
                label = cfg["label_var"].get().strip()
                files_A = cfg["files_A"]
                files_B = cfg["files_B"]

                if not files_A or not files_B:
                    print(f"[ΔC] Curve {idx}: Condition A or B file missing. Skipped.")
                    continue

                if not label:
                    label = f"Curve {idx}"

                try:
                    self.compute_deltaC_and_plot(
                        label=label,
                        files_A=files_A,
                        files_B=files_B,
                        direction=direction,
                    )
                    any_plotted = True
                except Exception as e:
                    print(f"[ΔC] Curve {idx} error: {e}")

            if not any_plotted:
                messagebox.showwarning("Warning", "No valid curve plotted.")
                return

            plt.xlabel("RoC (nm)")
            plt.ylabel(r"Differential cumulative probability ($\Delta$C)")
            plt.axhline(0, color="k", linewidth=0.8, alpha=0.5)
            plt.grid(False)
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(-0.03, 0.1)
            ax.tick_params(axis='both', labelsize=10)
            plt.legend(frameon=False, fontsize=8)
            plt.tight_layout()
            plt.show()

        tk.Button(win, text="Generate ΔC plot", command=run_generate_deltac).grid(
            row=row, column=0, columnspan=4, pady=10
        )

    def _load_rc_values_from_files(self, files):
        """Load RC column from RC/RG_RC files."""
        all_data = []

        for file_path in files:
            try:
                df = pd.read_csv(file_path, sep='\t', engine='python')
                if df.empty or df.shape[1] < 2:
                    continue

                try:
                    first0 = str(df.iloc[0, 0])
                    first1 = str(df.iloc[0, 1])
                except Exception:
                    first0, first1 = '', ''

                if first0 == 'track_id' or 'radius_confinement' in first1:
                    df = df.iloc[1:].reset_index(drop=True)

                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                if df.shape[1] < 3:
                    continue

                rc_col = df.columns[1]
                d_col = df.columns[2]

                df = df.dropna(subset=[rc_col, d_col])
                if df.empty:
                    continue

                mask = (df[rc_col] > 0.001) & (df[rc_col] < 20.0) & \
                       (df[d_col] > 0) & (df[d_col] < 0.1)
                rc_data = df.loc[mask, rc_col].to_numpy(dtype=float)

                if rc_data.size > 0:
                    all_data.extend(rc_data.tolist())

            except Exception as e:
                print(f"[ΔC] Failed loading RC file: {file_path}, error={e}")

        return np.array(all_data, dtype=float)

    def _kde_pdf_epanechnikov(self, x_values, samples):
        if KernelDensity is None:
            raise RuntimeError("scikit-learn missing.")

        if samples.size == 0:
            return np.zeros_like(x_values)

        x_grid = x_values.reshape(-1, 1)
        ss = samples.reshape(-1, 1)

        kde = KernelDensity(kernel='epanechnikov', bandwidth=0.01)
        kde.fit(ss)
        log_d = kde.score_samples(x_grid)
        pdf = np.exp(log_d)

        dx = np.mean(np.diff(x_values))
        area = np.sum(pdf) * dx
        if area > 0:
            pdf /= area

        return pdf

    def _to_cdf(self, pdf):
        if pdf.size == 0:
            return pdf
        cdf = np.cumsum(pdf)
        if cdf[-1] > 0:
            cdf /= cdf[-1]
        return cdf

    def compute_deltaC_and_plot(self, label, files_A, files_B, direction):

        x_values = np.arange(0.02, 0.300 + 1e-12, 0.002)
        x_nm = x_values * 1000.0

        rc_A = self._load_rc_values_from_files(files_A)
        rc_B = self._load_rc_values_from_files(files_B)

        if rc_A.size == 0 or rc_B.size == 0:
            print(f"[ΔC] Curve {label}: invalid RC data.")
            return

        pdf_A = self._kde_pdf_epanechnikov(x_values, rc_A)
        pdf_B = self._kde_pdf_epanechnikov(x_values, rc_B)

        cdf_A = self._to_cdf(pdf_A)
        cdf_B = self._to_cdf(pdf_B)

        if direction == "B_minus_A":
            delta = cdf_B - cdf_A
        else:
            delta = cdf_A - cdf_B

        delta=delta*0.5

        # smoothing
        if gaussian_filter1d is not None:
            delta = gaussian_filter1d(delta, sigma=2.5)
        else:
            print("[ΔC] scipy not installed: no smoothing applied.")

        delta[0] = 0.0
        delta[-1] = 0.0

        plt.plot(x_nm, delta, '-', linewidth=2.0, label=label)

        if hasattr(self, "delta_export_dict"):
            self.delta_export_dict[label] = delta.copy()

    def load_spoton_core(self):
        # 1) 选择跟踪文件
        fnames = filedialog.askopenfilenames(
            title="选择 TXT 追踪文件",
            filetypes=[("Text files", "*.txt")]
        )
        # 2) 读取原始数据 [x, y, frame, track_id]
        all_raw = []
        for fname in fnames:
            try:
                data = np.genfromtxt(
                    fname,
                    delimiter='\t',
                    usecols=(0, 1, 2, 3),
                    comments='#',  # 跳过以 # 开头的注释
                    invalid_raise=False
                )
                if data.ndim == 1:
                    data = data[np.newaxis, :]  # 保证至少二维
                all_raw.append(data)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {os.path.basename(fname)}: {e}")
                return

        # 3) 弹出参数设置窗口
        win = tk.Toplevel(self.master)
        win.title("Parameter Settings")

        # ─────── 在这里一次性定义所有的 Var ───────
        var_fr = tk.DoubleVar(value=10.0)  # 帧间隔(毫秒)
        var_px = tk.DoubleVar(value=109.0)  # 像素大小(nm/px)
        var_binwidth = tk.DoubleVar(value=0.010)
        var_timepoints = tk.IntVar(value=8)
        var_jumps = tk.IntVar(value=4)
        var_use_entire = tk.BooleanVar(value=False)
        var_maxjump = tk.DoubleVar(value=3)

        # 帧间隔
        row1 = tk.Frame(win)
        row1.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(row1, text="Frame interval (ms):").pack(side=tk.LEFT)
        ent_fr = tk.Entry(row1, textvariable=var_fr)
        ent_fr.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # 像素大小
        row2 = tk.Frame(win)
        row2.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(row2, text="Pixel size (nm/pixel):").pack(side=tk.LEFT)
        ent_px = tk.Entry(row2, textvariable=var_px)
        ent_px.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # 确认按钮 (点击后再创建后续所有面板)
        confirm_btn = tk.Button(win, text="Confirm", width=12)
        confirm_btn.pack(pady=(10, 5))

        # --- 先创建变量 ---
        var_states = tk.IntVar(value=2)  # 2 or 3 states
        var_db_min = tk.DoubleVar(value=0.0005)
        var_db_max = tk.DoubleVar(value=0.08)
        var_ds_min = tk.DoubleVar(value=0.15)
        var_ds_max = tk.DoubleVar(value=5)
        var_df_min = tk.DoubleVar(value=0.15)
        var_df_max = tk.DoubleVar(value=25)
        var_dfast_min= tk.DoubleVar(value=0.15)
        var_dfast_max = tk.DoubleVar(value=25)
        var_fbound_min = tk.DoubleVar(value=0.0)
        var_fbound_max = tk.DoubleVar(value=1.0)
        var_ffast_min = tk.DoubleVar(value=0.0)
        var_ffast_max = tk.DoubleVar(value=1.0)
        var_dz = tk.DoubleVar(value=0.7)
        var_fit_error = tk.BooleanVar(value=False)  # Fit localization error?
        var_error = tk.DoubleVar(value=0.035)
        var_zcorr = tk.BooleanVar(value=True)
        var_model_fit = tk.StringVar(value="CDF")  # "PDF" or "CDF"
        var_single = tk.BooleanVar(value=False)
        var_iters = tk.IntVar(value=3)

        def _on_confirm():
            # 1) 读取参数并校验
            try:
                fr_ms = float(ent_fr.get())
                px_nm = float(ent_px.get())
            except ValueError:
                messagebox.showerror("输入错误", "帧间隔和像素大小都必须是数字。")
                return

            # 2) 隐藏前两行和 Confirm 按钮
            row1.pack_forget()
            row2.pack_forget()
            confirm_btn.pack_forget()

            # 3) 创建全局统计区
            win.stats_frame = tk.LabelFrame(win, text="Global statistics")
            win.stats_frame.pack(fill='x', padx=8, pady=(0, 8))

            # --- 全局统计参数初始化 ---
            total_frames = 0
            num_detect = 0
            n_traces = 0
            num_tr3 = 0
            jump_lengths = []  # 存放所有“相邻点”的空间位移
            lengths = []  # 存放每条轨迹的长度（帧数）
            particles_per_frame = []  # 存放每个帧的粒子数量
            longest_gaps = []  # 存放每条轨迹的最长帧间隙

            for arr in all_raw:
                # 1) 缩放坐标 & 基本累加
                data = arr.copy()
                data[:, 0] *= px_nm
                data[:, 1] *= px_nm
                frames = data[:, 2].astype(int)

                # total_frames += np.max(frames)
                total_frames +=(frames.max()-frames.min()+1)
                num_detect += len(data)

                # 找到最小帧号和最大帧号
                f_min = int(frames.min())
                f_max = int(frames.max())
                # 建一个长度 = f_max - f_min + 1 的全零数组
                counts_full = np.zeros(f_max - f_min + 1, dtype=int)

                # 把出现过的帧号都填进去
                unique_f, cts = np.unique(frames, return_counts=True)
                # 用帧号偏移来做索引
                for f, c in zip(unique_f.astype(int), cts):
                    counts_full[f - f_min] = c

                particles_per_frame.extend(counts_full.tolist())

                # 3) 按 track_id 分组
                td = defaultdict(list)
                for x, y, frame, tid in data:
                    td[int(tid)].append((x, y, int(frame)))

                # 4) 对每条轨迹分别统计
                for pts in td.values():
                    L = len(pts)
                    n_traces += 1
                    lengths.append(L)
                    if L >= 3:
                        num_tr3 += 1

                    # —— 计算本条轨迹的最长帧间隙 ——
                    # 按帧号排序，diff(frame) - 1
                    pts.sort(key=lambda p: p[2])
                    frame_list = [p[2] for p in pts]
                    gaps = np.diff(frame_list) - 1
                    longest_gaps.append(int(np.max(gaps)) if gaps.size > 0 else 0)

                    # —— 计算空间跳距 ——
                    for i in range(L - 1):
                        x1, y1, f1 = pts[i]
                        x2, y2, f2 = pts[i + 1]
                        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        jump_lengths.append(dist)

            # 5) 汇总统计
            max_gap = max(longest_gaps) if longest_gaps else 0
            num_jumps = len(jump_lengths)

            jump_lengths_um = [d / 1000.0 for d in jump_lengths]

            median_length = np.median(lengths)
            mean_length = np.mean(lengths)
            median_pps = np.median(particles_per_frame)
            mean_pps = np.mean(particles_per_frame)
            median_jump_length = float(np.median(jump_lengths_um)) if jump_lengths_um else 0
            mean_jump_length = float(np.mean(jump_lengths_um)) if jump_lengths_um else 0

            # 6) 把所有统计输出到界面
            stats = [
                f"Number of traces: {n_traces}",
                f"Number of frames: {total_frames}",
                f"Number of detections: {num_detect}",
                f"Longest gap (frames): {max_gap}",  # ← 重新加回来了
                f"Traces ≥3 detections: {num_tr3}",
                f"Number of jumps: {num_jumps}",
                f"Length of trajectories (frames): median={median_length:.1f}, mean={mean_length:.3f}",
                f"Particles per frame: median={median_pps:.1f}, mean={mean_pps:.3f}",
                f"Jump length (µm): median={median_jump_length:.3f}, mean={mean_jump_length:.3f}",
            ]

            # 清空旧统计并显示新统计
            for w in win.stats_frame.winfo_children():
                w.destroy()
            for txt in stats:
                tk.Label(win.stats_frame, text=txt, anchor='w').pack(anchor='w')

            # 4) 创建参数面板：左右两列
            param_panels = tk.Frame(win)
            param_panels.pack(fill='x', padx=8, pady=(0, 8))

            # 4.1 左侧：Jump length distribution
            jld_panel = tk.LabelFrame(param_panels, text="Jump length distribution")
            jld_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
            # — 在这里粘进你的各个 Spinbox/Checkbutton 代码 —
            # Bin width (µm)
            row_bw = tk.Frame(jld_panel)
            row_bw.pack(fill='x', padx=5, pady=2)
            tk.Label(row_bw, text="Bin width (µm):").pack(side='left')
            var_binwidth = tk.DoubleVar(value=0.010)
            tk.Spinbox(
                row_bw,
                from_=0.0, to=10.0, increment=0.001,
                textvariable=var_binwidth,
                format="%.3f",
                width=8
            ).pack(side='left', fill='x', expand=True, padx=(5, 0))

            # Number of timepoints
            row_nt = tk.Frame(jld_panel)
            row_nt.pack(fill='x', padx=5, pady=2)
            tk.Label(row_nt, text="Number of timepoints:").pack(side='left')
            var_timepoints = tk.IntVar(value=8)
            tk.Spinbox(
                row_nt,
                from_=1, to=100, increment=1,
                textvariable=var_timepoints,
                width=8
            ).pack(side='left', fill='x', expand=True, padx=(5, 0))

            # Jumps to consider
            row_jc = tk.Frame(jld_panel)
            row_jc.pack(fill='x', padx=5, pady=2)
            tk.Label(row_jc, text="Jumps to consider:").pack(side='left')
            var_jumps = tk.IntVar(value=4)
            tk.Spinbox(
                row_jc,
                from_=1, to=100, increment=1,
                textvariable=var_jumps,
                width=8
            ).pack(side='left', fill='x', expand=True, padx=(5, 0))

            # Use entire trajectories?
            row_entire = tk.Frame(jld_panel)
            row_entire.pack(fill='x', padx=5, pady=2)
            var_use_entire = tk.BooleanVar(value=False)
            tk.Checkbutton(
                row_entire,
                text="Use entire trajectories?",
                variable=var_use_entire
            ).pack(side='left')

            # Max jump (µm)
            row_mj = tk.Frame(jld_panel)
            row_mj.pack(fill='x', padx=5, pady=2)
            tk.Label(row_mj, text="Max jump (µm):").pack(side='left')
            var_maxjump = tk.DoubleVar(value=3)
            tk.Spinbox(
                row_mj,
                from_=0.0, to=10.0, increment=0.010,
                textvariable=var_maxjump,
                format="%.3f",
                width=8
            ).pack(side='left', fill='x', expand=True, padx=(5, 0))

            # 4.2 右侧：Model fitting
            mf_panel = tk.LabelFrame(param_panels, text="Model fitting")
            mf_panel.pack(side='left', fill='both', expand=True)
            # — 在这里粘进你的 Model fitting 的控件代码 —

            # ------ 新增：先创建一个容器，只用来装2s/3s面板 ------
            models_frame = tk.Frame(mf_panel)
            models_frame.pack(fill='x', padx=5, pady=5)

            # --- Kinetic model 选择 ---
            row = tk.Frame(mf_panel)
            row.pack(fill='x', padx=5, pady=2)
            tk.Label(row, text="Kinetic model:").pack(side='left')

            # --- 2-state 参数区 ---
            frame_2s = tk.Frame(models_frame)

            def _build_2s():
                # D_bound
                r = tk.Frame(frame_2s)
                r.pack(fill='x', padx=5, pady=2)
                tk.Label(r, text="D_bound (µm²/s):").pack(side='left')
                tk.Spinbox(r, from_=0.0, to=10.0, increment=0.0001,
                           textvariable=var_db_min, format="%.4f", width=8).pack(side='left')
                tk.Label(r, text="max").pack(side='left', padx=(5, 0))
                tk.Spinbox(r, from_=0.0, to=100.0, increment=0.0001,
                           textvariable=var_db_max, format="%.4f", width=8).pack(side='left')

                # D_free
                r = tk.Frame(frame_2s)
                r.pack(fill='x', padx=5, pady=2)
                tk.Label(r, text="D_free (µm²/s):").pack(side='left')
                tk.Spinbox(r, from_=0.0, to=100.0, increment=0.01,
                           textvariable=var_df_min, format="%.2f", width=8).pack(side='left')
                tk.Label(r, text="max").pack(side='left', padx=(5, 0))
                tk.Spinbox(r, from_=0.0, to=100.0, increment=0.01,
                           textvariable=var_df_max, format="%.2f", width=8).pack(side='left')

                # F_bound
                r = tk.Frame(frame_2s)
                r.pack(fill='x', padx=5, pady=2)
                tk.Label(r, text="F_bound:").pack(side='left')
                tk.Spinbox(r, from_=0.0, to=1.0, increment=0.01,
                           textvariable=var_fbound_min, format="%.2f", width=8).pack(side='left')
                tk.Label(r, text="max").pack(side='left', padx=(5, 0))
                tk.Spinbox(r, from_=0.0, to=1.0, increment=0.01,
                           textvariable=var_fbound_max, format="%.2f", width=8).pack(side='left')

            _build_2s()

            # --- 3-state 参数区 ---
            frame_3s = tk.Frame(models_frame)

            def _build_3s():
                # D_bound
                r = tk.Frame(frame_3s)
                r.pack(fill='x', padx=5, pady=2)
                tk.Label(r, text="D_bound (µm²/s):").pack(side='left')
                tk.Spinbox(r, from_=0.0, to=10.0, increment=0.0001,
                           textvariable=var_db_min, format="%.4f", width=8).pack(side='left')
                tk.Label(r, text="max").pack(side='left', padx=(5, 0))
                tk.Spinbox(r, from_=0.0, to=100.0, increment=0.0001,
                           textvariable=var_db_max, format="%.4f", width=8).pack(side='left')

                # D_med
                r = tk.Frame(frame_3s)
                r.pack(fill='x', padx=5, pady=2)
                tk.Label(r, text="D_slow (µm²/s):").pack(side='left')
                tk.Spinbox(r, from_=0.0, to=100.0, increment=0.01,
                           textvariable=var_ds_min, format="%.2f", width=8).pack(side='left')
                tk.Label(r, text="max").pack(side='left', padx=(5, 0))
                tk.Spinbox(r, from_=0.0, to=100.0, increment=0.01,
                           textvariable=var_ds_max, format="%.2f", width=8).pack(side='left')

                # D_fast
                r = tk.Frame(frame_3s)
                r.pack(fill='x', padx=5, pady=2)
                tk.Label(r, text="D_fast (µm²/s):").pack(side='left')
                tk.Spinbox(r, from_=0.0, to=100.0, increment=0.01,
                           textvariable=var_dfast_min, format="%.2f", width=8).pack(side='left')
                tk.Label(r, text="max").pack(side='left', padx=(5, 0))
                tk.Spinbox(r, from_=0.0, to=100.0, increment=0.01,
                           textvariable=var_dfast_max, format="%.2f", width=8).pack(side='left')

                # F_bound
                r = tk.Frame(frame_3s)
                r.pack(fill='x', padx=5, pady=2)
                tk.Label(r, text="F_bound:").pack(side='left')
                tk.Spinbox(r, from_=0.0, to=1.0, increment=0.01,
                           textvariable=var_fbound_min, format="%.2f", width=8).pack(side='left')
                tk.Label(r, text="max").pack(side='left', padx=(5, 0))
                tk.Spinbox(r, from_=0.0, to=1.0, increment=0.01,
                           textvariable=var_fbound_max, format="%.2f", width=8).pack(side='left')

                # F_fast
                r = tk.Frame(frame_3s)
                r.pack(fill='x', padx=5, pady=2)
                tk.Label(r, text="F_fast:").pack(side='left')
                tk.Spinbox(r, from_=0.0, to=1.0, increment=0.01,
                           textvariable=var_ffast_min, format="%.2f", width=8).pack(side='left')
                tk.Label(r, text="max").pack(side='left', padx=(5, 0))
                tk.Spinbox(r, from_=0.0, to=1.0, increment=0.01,
                           textvariable=var_ffast_max, format="%.2f", width=8).pack(side='left')

            _build_3s()

            # --- show / hide 2s or 3s ---
            def _switch_model():
                if var_states.get() == 2:
                    frame_3s.forget()
                    frame_2s.pack(fill='x', expand=True)
                else:
                    frame_2s.forget()
                    frame_3s.pack(fill='x', expand=True)

            # 把切换按钮也绑定上
            tk.Radiobutton(models_frame, text="2-State", variable=var_states, value=2,
                           command=_switch_model).pack(side='left', padx=5)
            tk.Radiobutton(models_frame, text="3-State", variable=var_states, value=3,
                           command=_switch_model).pack(side='left')

            _switch_model()


            # --- Localization error and fitting options ---
            row = tk.Frame(mf_panel)
            row.pack(fill='x', padx=5, pady=2)
            tk.Checkbutton(row, text="Fit localization error", variable=var_fit_error).pack(side='left')
            tk.Spinbox(row, from_=0.0, to=1.0, increment=0.001, textvariable=var_error,
                       format="%.3f", width=8, state=('normal' if not var_fit_error.get() else 'disabled')
                       ).pack(side='left', padx=(5, 0))

            row = tk.Frame(mf_panel)
            row.pack(fill='x', padx=5, pady=2)
            tk.Label(row, text="dZ (µm):").pack(side='left')
            tk.Spinbox(row, from_=0.0, to=10.0, increment=0.01, textvariable=var_dz,
                       format="%.2f", width=8).pack(side='left', padx=(5, 0))
            tk.Checkbutton(row, text="Use Z correction", variable=var_zcorr).pack(side='left', padx=5)

            row = tk.Frame(mf_panel)
            row.pack(fill='x', padx=5, pady=2)
            tk.Label(row, text="Model fit:").pack(side='left')
            tk.OptionMenu(row, var_model_fit, "PDF", "CDF").pack(side='left', padx=5)

            row = tk.Frame(mf_panel)
            row.pack(fill='x', padx=5, pady=2)
            tk.Checkbutton(row, text="Single cell fit", variable=var_single).pack(side='left')
            tk.Label(row, text="Iterations:").pack(side='left', padx=(10, 0))
            tk.Spinbox(row, from_=1, to=100, increment=1, textvariable=var_iters, width=5).pack(side='left')

            # 下面真正触发拟合的 Button，绑定到你的 _on_fit()
            fit_btn = tk.Button(mf_panel, text="Fit kinetic model", command=_on_fit)
            fit_btn.pack(pady=10)

        confirm_btn.configure(command=_on_confirm)

        def _on_fit():
            t0 = time.time()
            print("[INFO] Tracking analysis started ...")
            # 1) 读取参数并校验
            try:
                fr_ms = float(ent_fr.get())
                px_nm = float(ent_px.get())
            except ValueError:
                messagebox.showerror("输入错误", "帧间隔和像素大小都必须是数字。")
                return

            # --- 全局统计参数初始化 ---
            total_frames = 0
            num_detect = 0
            n_traces = 0
            num_tr3 = 0
            jump_lengths = []  # 存放所有“相邻点”的空间位移
            lengths = []  # 存放每条轨迹的长度（帧数）
            particles_per_frame = []  # 存放每个帧的粒子数量
            longest_gaps = []  # 存放每条轨迹的最长帧间隙

            for arr in all_raw:
                # 1) 缩放坐标 & 基本累加
                data = arr.copy()
                data[:, 0] *= px_nm
                data[:, 1] *= px_nm
                frames = data[:, 2].astype(int)

                total_frames += np.max(frames)
                num_detect += len(data)

                # 找到最小帧号和最大帧号
                f_min = int(frames.min())
                f_max = int(frames.max())
                # 建一个长度 = f_max - f_min + 1 的全零数组
                counts_full = np.zeros(f_max - f_min + 1, dtype=int)

                # 把出现过的帧号都填进去
                unique_f, cts = np.unique(frames, return_counts=True)
                # 用帧号偏移来做索引
                for f, c in zip(unique_f.astype(int), cts):
                    counts_full[f - f_min] = c

                particles_per_frame.extend(counts_full.tolist())

                # 3) 按 track_id 分组
                td = defaultdict(list)
                for x, y, frame, tid in data:
                    td[int(tid)].append((x, y, int(frame)))

                # 4) 对每条轨迹分别统计
                for pts in td.values():
                    L = len(pts)
                    n_traces += 1
                    lengths.append(L)
                    if L >= 3:
                        num_tr3 += 1

                    # —— 计算本条轨迹的最长帧间隙 ——
                    # 按帧号排序，diff(frame) - 1
                    pts.sort(key=lambda p: p[2])
                    frame_list = [p[2] for p in pts]
                    gaps = np.diff(frame_list) - 1
                    longest_gaps.append(int(np.max(gaps)) if gaps.size > 0 else 0)

                    # —— 计算空间跳距 ——
                    for i in range(L - 1):
                        x1, y1, f1 = pts[i]
                        x2, y2, f2 = pts[i + 1]
                        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        jump_lengths.append(dist)

            # --- 7) 计算每条轨迹的 MSD 并拟合限域 RoC 模型 ---
            msd_fits = []  # 用于存放 (样本索引, track_id, popt)
            for sample_idx, arr in enumerate(all_raw):
                # 先复制并做单位换算：像素坐标 × px_nm → nm，再 ÷1000 → µm
                data = arr.copy()
                data[:, 0] = data[:, 0] * px_nm / 1000.0  # x (µm)
                data[:, 1] = data[:, 1] * px_nm / 1000.0  # y (µm)
                # 按轨迹 ID 分组
                td = defaultdict(list)
                for x_nm, y_nm, frame, tid in data:
                    td[int(tid)].append((x_nm, y_nm, int(frame)))

                for track_id, pts in td.items():
                    pts = np.array(pts, dtype=float)
                    # 将 x,y 从 nm 转为 µm
                    xy = pts[:, :2] * 1e-3
                    frames = pts[:, 2].astype(int)
                    L = len(frames)
                    if L < 2:
                        continue

                    # 计算滞后时间 (秒) 和对应的 MSD(τ)
                    lags = np.arange(1, L)
                    t = lags * (fr_ms / 1000.0)  # fr_ms 单位 ms → 秒
                    msd = np.array([
                        np.mean(np.sum((xy[lag:] - xy[:-lag]) ** 2, axis=1))
                        for lag in lags
                    ])

                    # 拟合限域模型：msd_confined(t; Rc, D, sigma)
                    try:
                        # 初始猜测：Rc ~ sqrt(max(msd)), D ~ 0.1, sigma ~ sqrt(msd[0])/2
                        p0 = [np.sqrt(np.max(msd)), 0.1, np.sqrt(msd[0]) / 2]
                        popt, _ = curve_fit(
                            self.msd_confined, t, msd, p0=p0,
                            bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
                        )
                    except Exception:
                        popt = [np.nan, np.nan, np.nan]

                    msd_fits.append((sample_idx, track_id, popt))

            # msd_fits 中现在包含每条轨迹对应的 (Rc, D, σ)
            # --- 8) 构造 trackedPar 以进行跳距分析 ---
            trackedPar = []
            for arr in all_raw:
                data = arr.copy()
                data[:, 0] = data[:, 0] * px_nm / 1000.0  # x (µm)
                data[:, 1] = data[:, 1] * px_nm / 1000.0  # y (µm)
                td = defaultdict(list)
                for x_nm, y_nm, frame, tid in data:
                    td[int(tid)].append((
                        x_nm ,
                        y_nm ,
                        int(frame)
                    ))
                for pts in td.values():
                    pts = np.array(pts, dtype=float)
                    xy = pts[:, :2]  # shape=(N,2)，单位 µm
                    frames = pts[:, 2].astype(int)
                    # compute_jump_length_distribution 需要
                    # trackedPar[i][0] = xy
                    # trackedPar[i][2][0] = frames
                    # 把 frames 包装成一个 numpy 数组，shape = (1, N)
                    trackedPar.append([xy,
                                       None,
                                       np.array([frames], dtype=int)  # shape = (1, len(frames))
                                       ])


            # （b）直接调用完整版 compute_jump_length_distribution，
            #     同时拿到 PDF 与（可选的）CDF
            if var_model_fit.get() == "CDF":
                # CDF 模式：直接让函数帮我们生成细化后的 CDF 网格和 JumpProbCDF
               data=self.compute_jump_length_distribution(
                        trackedPar,
                        CDF=True,
                        useEntireTraj=var_use_entire.get(),
                        TimePoints=var_timepoints.get(),
                        GapsAllowed=1,  # 或者你可以再加个面板让用户设
                        JumpsToConsider=var_jumps.get(),
                        MaxJump=var_maxjump.get(),
                        BinWidth=var_binwidth.get()
                    )
               HistVecJumps = data[2]
               JumpProb = data[3]
               HistVecJumpsCDF = data[0]
               JumpProbCDF = data[1]
            else:
                # PDF 模式：只需要 PDF，自己再去累加算 CDF
                data = self.compute_jump_length_distribution(
                        trackedPar,
                        CDF=False,
                        useEntireTraj=var_use_entire.get(),
                        TimePoints=var_timepoints.get(),
                        GapsAllowed=1,
                        JumpsToConsider=var_jumps.get(),
                        MaxJump=var_maxjump.get(),
                        BinWidth=var_binwidth.get()
                    )
                HistVecJumps = data[0]
                JumpProb = data[1]
                HistVecJumpsCDF = data[0]
                JumpProbCDF = data[1]


            # —— 9.2 并行提交拟合 —— #
            dT = fr_ms / 1000.0
            # 下面的 a,b 还是保存在 image_bin 里的
            dZ = var_dz.get()
            gaps_allowed = 1

            # —— 先查找到最匹配的 a, b —— #
            Z_corr_a, Z_corr_b, matched_dT, matched_dZ = self.match_z_corr_coeff(
                dT, dZ,
                gaps_allowed=gaps_allowed,
                # mat_path="MonteCarloParams_1_gap.mat"  # .mat 所在目录
            )

            # 从“Model fitting”面板读所有参数：
            states = var_states.get()  # 2 or 3
            db_min = var_db_min.get()
            db_max = var_db_max.get()
            df_min = var_df_min.get()
            df_max = var_df_max.get()
            dfast_min = var_dfast_min.get()
            dfast_max = var_dfast_max.get()
            ds_min = var_ds_min.get()
            ds_max = var_ds_max.get()
            ffast_min = var_ffast_min.get()
            ffast_max = var_ffast_max.get()
            fbound_min = var_fbound_min.get()
            fbound_max = var_fbound_max.get()

            use_error = var_fit_error.get()  # True/False是否启用fit localization error
            error_val = var_error.get()   #####fit localization error 0.035
            zcorr = var_zcorr.get() ###0.7
            fit_mode = 1 if var_model_fit.get() == "PDF" else 2
            iterations = var_iters.get()

            # 构造上下限列表：
            if states == 2:
                sigma_bound = [0.01, 0.075]
                LB = [df_min, db_min, fbound_min,sigma_bound[0]]
                UB = [df_max, db_max, fbound_max,sigma_bound[1]]
                params = {
                          'UB': UB,
                          'LB': LB,
                          'LocError': None,  # Manually input the localization error in um: 35 nm = 0.035 um.
                          'iterations': iterations,  # Manually input the desired number of fitting iterations:
                          'dT': dT,  # Time between frames in seconds
                          'dZ': dZ,  # The axial illumination slice: measured to be roughly 700 nm
                          'ModelFit': fit_mode,
                          'fit2states': True,
                          'fitSigma': True,
                          'a': Z_corr_a,
                          'b': Z_corr_b,
                          'useZcorr': True
                          }


                fit = self.fit_jump_length_distribution(JumpProb, JumpProbCDF, HistVecJumps, HistVecJumpsCDF,
                                                            **params)
                y = self.generate_jump_length_distribution(fit.params,
                                                               JumpProb=JumpProbCDF, r=HistVecJumpsCDF,
                                                               LocError=fit.params['sigma'].value,
                                                               dT=params['dT'], dZ=params['dZ'],
                                                               a=params['a'], b=params['b'], norm=True,
                                                               useZcorr=params['useZcorr'])
                ## Normalization does not work for PDF yet (see commented line in fastspt.py)
                if True:
                    y *= float(len(HistVecJumpsCDF)) / float(len(HistVecJumps))
                    fig_jump_1 = plt.figure(figsize=(10, 5))  # Initialize the plot
                    # self.plot_histogram(HistVecJumps, JumpProb)  ## Read the documentation of this function to learn how to populate
            else:
                sigma_bound = [0.01, 0.075]
                LB = [dfast_min, ds_min, db_min, ffast_min,fbound_min, sigma_bound[0]]
                UB = [dfast_max, ds_max, db_max, ffast_max,fbound_max, sigma_bound[1]]
                params = {'UB': UB,
                          'LB': LB,
                          'LocError': None if use_error else error_val,  # Manually input the localization error in um: 35 nm = 0.035 um.
                          'iterations': iterations,  # Manually input the desired number of fitting iterations:
                          'dT': dT,  # Time between frames in seconds
                          'dZ': dZ,  # The axial illumination slice: measured to be roughly 700 nm
                          'ModelFit': fit_mode,
                          'fit2states': False,
                          'fitSigma': True,
                          'a': Z_corr_a,
                          'b': Z_corr_b,
                          'useZcorr': True,
                          # （可选）放宽 solver precision，使结果和 MATLAB 默认更接近
                          'solverparams': {'ftol': 1e-8, 'xtol': 1e-8, 'maxfev': 100000},
                          }
                ## Perform the fit
                fit = self.fit_jump_length_distribution(JumpProb, JumpProbCDF, HistVecJumps, HistVecJumpsCDF,
                                                            **params)
                y = self.generate_jump_length_distribution(fit.params,
                                                               JumpProb=JumpProbCDF, r=HistVecJumpsCDF,
                                                               LocError=fit.params['sigma'].value,
                                                               dT=params['dT'], dZ=params['dZ'],
                                                               a=params['a'], b=params['b'],
                                                               fit2states=params['fit2states'],
                                                               useZcorr=params['useZcorr'])
                ## Normalize it
                norm_y = np.zeros_like(y)
                for i in range(y.shape[0]):  # Normalize y as a PDF
                    norm_y[i, :] = y[i, :] / y[i, :].sum()
                scaled_y = (float(len(HistVecJumpsCDF)) / len(
                    HistVecJumps)) * norm_y  # scale y for plotting next to histograms
                fig_jump_2 = plt.figure(figsize=(10, 5))  # Initialize the plot

            # 回调只画一幅
            def _draw_fit_one(fit):
                # —— 1) 先生成参数统计区 —— #
                hist_frame = tk.LabelFrame(plot_win, text="Jump length histograms")
                hist_frame.pack(fill='x', padx=5, pady=(5, 0))

                # 左半列
                col1 = tk.Frame(hist_frame)
                col1.pack(side='left', fill='both', expand=True, padx=(5, 2), pady=5)
                tk.Label(col1, text="Fit parameters for cell 1.").pack(anchor='w')

                vals = fit.params
                lines1 = [
                    f"D_bound : {vals['D_bound'].value:.3f} ± {vals['D_bound'].stderr or 0:.3f}",
                ]
                if states == 2:
                    lines1 += [
                        f"D_free  : {vals['D_free'].value:.3f} ± {vals['D_free'].stderr or 0:.3f}",
                        f"F_bound : {vals['F_bound'].value:.3f} ± {vals['F_bound'].stderr or 0:.3f}",
                    ]
                else:
                    lines1 += [
                        f"D_slow  : {vals['D_med'].value:.3f} ± {vals['D_med'].stderr or 0:.3f}",
                        f"D_fast  : {vals['D_fast'].value:.3f} ± {vals['D_fast'].stderr or 0:.3f}",
                        f"F_bound : {vals['F_bound'].value:.3f} ± {vals['F_bound'].stderr or 0:.3f}",
                        f"F_slow  : {1 - vals['F_fast'].value - vals['F_bound'].value:.3f} ± 0.000",
                        # 将 F_slow 放在 F_fast 前面
                        f"F_fast  : {vals['F_fast'].value:.3f} ± {vals['F_fast'].stderr or 0:.3f}",
                    ]
                # 通用项
                lines1 += [
                    f"l₂ error: {fit.params.ssq2:.6f}",
                    f"AIC: {fit.aic:.2f}, BIC: {fit.bic:.2f}",
                ]
                for txt in lines1:
                    tk.Label(col1, text=f"- {txt}", anchor='w').pack(anchor='w')

                # 右半列
                col2 = tk.Frame(hist_frame)
                col2.pack(side='left', fill='both', expand=True, padx=(2, 5), pady=5)
                tk.Label(col2, text="Global fit parameters for cells [1].").pack(anchor='w')
                for txt in lines1:
                    tk.Label(col2, text=f"- {txt}", anchor='w').pack(anchor='w')

                # 设置标题并根据 states 值和 len(all_raw) 决定调用的参数
                if states == 2:
                    title = "2-State Fit"
                    y_to_use =  y
                elif states == 3:
                    title = "3-State Fit"
                    y_to_use = scaled_y

                fig=self.plot_histogram(
                    HistVecJumps=HistVecJumps, emp_hist=JumpProb, HistVecJumpsCDF=HistVecJumpsCDF,
                    sim_hist=y_to_use  # 根据条件选择 y_m, y, y_n 或 scaled_y
                )
                return fig

            # —— 弹窗 + Notebook 三标签页 —— #
            plot_win = tk.Toplevel(self.master)
            plot_win.title("Tracking Analysis")

            notebook = ttk.Notebook(plot_win)
            notebook.pack(fill='both', expand=True, padx=5, pady=5)

            fig =_draw_fit_one(fit)  # 直接绘制拟合结果，而不需要回调

            # —— 在绘图前，先重建 tracks 列表 —— #
            # tracks 中每个元素是一个形状为 (N,3) 的 ndarray，列依次是 [x(µm), y(µm), frame]
            tracks = []
            px_um = px_nm / 1000.0  # 先把像素单位从 nm 转成 µm
            for arr in all_raw:
                data = arr.copy()
                # 前面你已经对 data 进行了单位转换，这里如果没做就再做一次：
                data[:, 0] *= px_um
                data[:, 1] *= px_um

                td = defaultdict(list)
                for x, y, frame, tid in data:
                    td[int(tid)].append((x, y, int(frame)))

                for pts in td.values():
                    pts = np.array(pts, dtype=float)  # shape=(N,3)
                    tracks.append(pts)

            # —— 绘制限域 MSD 拟合图 —— #
            # 只对轨迹长度 >=5 的轨迹进行示例
            valid_tracks = [tr for tr in tracks if len(tr) >= 5]
            if not valid_tracks:
                valid_tracks = tracks
            example_idx = len(valid_tracks) // 2
            msd_ex = self.compute_msd(valid_tracks[example_idx])
            t_ex = np.arange(1, len(msd_ex) + 1) * (fr_ms / 1000.0)
            popt_ex, _ = curve_fit(
                self.msd_confined,
                t_ex, msd_ex,
                p0=[np.sqrt(msd_ex[-1]), 0.1, 0.03],
                bounds=([0, 0, 0], [np.inf, np.inf, np.inf])
            )
            fig_confined = plt.figure(figsize=(5, 4))
            ax = fig_confined.add_subplot(111)
            ax.scatter(t_ex, msd_ex, label="MSD data")
            ax.plot(t_ex, self.msd_confined(t_ex, *popt_ex), 'r-', label=f"Rc={popt_ex[0]:.2f} μm")
            ax.set_xlabel("Δt (s)")
            ax.set_ylabel("MSD (μm²)")
            ax.legend()
            # 取消坐标轴偏移，使用普通数字
            ax.ticklabel_format(useOffset=False, style='plain')

            # —— 先定义 popts —— #
            popts = [popt for (_, _, popt) in msd_fits]
            # —— 绘制 RoC 分布直方图 —— #
            Rc_vals = [p[0] for p in popts  # p[0] 就是 Rc
                       if p is not None and not np.isnan(p[0])]
            mean_Rc = np.mean(Rc_vals) if Rc_vals else 0
            std_Rc = np.std(Rc_vals) if Rc_vals else 0
            fig_roc = plt.figure(figsize=(5, 4))
            axr = fig_roc.add_subplot(111)
            axr.hist(Rc_vals, bins=20, density=True)
            axr.set_xlabel("RoC (μm)")
            axr.set_ylabel("Density")
            axr.set_title(f"RoC Distribution")

            # Tab1: 跳距分布
            tab1 = ttk.Frame(notebook)
            notebook.add(tab1, text="Jump-length")

            # # 根据states值选择不同的fig对象
            fig = fig_jump_1 if states == 2 else fig_jump_2

            canvas1 = FigureCanvasTkAgg(fig, master=tab1)
            canvas1.draw()
            canvas1.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

            # —— 新增：保存图像按钮 —— #
            def save_current_figure():
                # 弹出对话框让用户选择保存路径
                fn = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG 图像", "*.png"),
                               ("TIFF 图像", "*.tif"),
                               ("JPEG 图像", "*.jpg")],
                    title="保存当前图像"
                )
                if not fn:
                    return
                try:
                    # dpi=300, bbox_inches='tight' 去掉周围多余空白
                    fig.savefig(fn, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("保存成功", f"图像已保存到：\n{fn}")
                except Exception as e:
                    messagebox.showerror("保存失败", str(e))

            # 放一个按钮到 tab1 底部
            btn_save = ttk.Button(tab1, text="save", command=save_current_figure)
            btn_save.pack(side='bottom', anchor='center', padx=10, pady=5)
            t1 = time.time()
            elapsed = t1 - t0
            print(f"[INFO] Tracking analysis completed in {elapsed:.2f} seconds.")

            # # Tab2: 限域 MSD
            # tab2 = ttk.Frame(notebook)
            # notebook.add(tab2, text="Confined MSD")
            # canvas2 = FigureCanvasTkAgg(fig_confined, master=tab2)
            # canvas2.draw()
            # canvas2.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
            #
            # # Tab3: RoC 分布 & 统计
            # tab3 = ttk.Frame(notebook)
            # notebook.add(tab3, text="RoC Distribution")
            # canvas3 = FigureCanvasTkAgg(fig_roc, master=tab3)
            # canvas3.draw()
            # canvas3.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

            # # 添加平均值和标准差文本
            # stats = ttk.Frame(tab3)
            # stats.pack(fill='x', pady=(5, 0))
            # ttk.Label(stats, text=f"Average RoC: {mean_Rc:.4f} μm").pack(side='left', padx=8)
            # ttk.Label(stats, text=f"Std Dev: {std_Rc:.4f} μm").pack(side='left')


    def compute_msd(self,track):
        # track: numpy 数组，shape=(N_frames, >=3)，列为 [x, y, frame, ...]
        max_lag = track.shape[0] - 1
        msd = np.zeros(max_lag)
        for tau in range(1, max_lag + 1):
            diffs = track[tau:, :2] - track[:-tau, :2]
            msd[tau - 1] = np.mean(np.sum(diffs ** 2, axis=1))
        return msd

    def msd_confined(self,t, Rc, D, sigma):
        M = 10
        term = np.zeros_like(t)
        for n in range(1, M + 1):
            k = 2 * n - 1
            term += (1.0 / k ** 2) * np.exp(-k ** 2 * np.pi ** 2 * D * t / (4 * Rc ** 2))
        return Rc ** 2 * (1 - 8 / np.pi ** 2 * term) + 4 * sigma ** 2

    def match_z_corr_coeff(self,dT, dZ, gaps_allowed=1, mat_path=None):
        """
        在 MonteCarloParams_1_gap.mat 中，根据给定的 dT (s) 和 dZ (µm)
        查找最接近的 Z 校正系数 a、b。

        参数
        ----
        dT : float
            帧间隔 (秒)
        dZ : float
            轴向切片深度 (微米)
        gaps_allowed : int, optional
            允许的轨迹丢帧数，当前仅实现 ==1 时加载 MonteCarloParams_1_gap.mat
        mat_path : str, optional
            .mat 文件所在路径（不含文件名），默认与脚本同目录

        返回
        ----
        Z_corr_a : float
        Z_corr_b : float
        matched_dT : float
        matched_dZ : float
        """
        # 1) 选择 .mat 文件
        if mat_path is None:
            mat_path = ''  # 默认当前目录
        if gaps_allowed == 1:
            fname = f"{mat_path}MonteCarloParams_1_gap.mat"
        else:
            raise ValueError(f"GapsAllowed = {gaps_allowed} not supported.")

        # 2) 加载 .mat
        mat = loadmat(fname)
        # mat 包含：a (N×1), b (N×1), matrix_dTdZ (N×2)，其余键为 metadata，可以忽略

        a_array = mat['a'].ravel()  # shape (N,)
        b_array = mat['b'].ravel()  # shape (N,)
        dTdZ_mat = mat['matrix_dTdZ']  # shape (N,2), 列0是 dT (ms), 列1是 dZ (µm)

        # mat 中的 dT 单位是“秒”还是“毫秒”？MATLAB 里使用 num2str(dT*1000) 打印时，
        # 说明 matrix_dTdZ 存的是以秒为单位的 dT；display 时乘 1000 打成 ms。
        # 这里假设它存的是秒，如果读取后看数值很小（<0.1），则无须转换。

        # 3) 构建 KD 树并查询
        tree = KDTree(dTdZ_mat, leaf_size=40)
        dist, idx = tree.query([[dT, dZ]], k=1)
        i = idx[0, 0]

        Z_corr_a = float(a_array[i])
        Z_corr_b = float(b_array[i])
        matched_dT = float(dTdZ_mat[i, 0])
        matched_dZ = float(dTdZ_mat[i, 1])

        # 4) 打印匹配信息（可选）
        print("===== finding Z-correction coefficients =====")
        print(f"User supplied dT: {dT * 1000:.1f} ms; matched dT: {matched_dT * 1000:.1f} ms")
        print(f"User supplied dZ: {dZ * 1000:.1f} nm; matched dZ: {matched_dZ * 1000:.1f} nm")
        print(f"Using Z_corr_a = {Z_corr_a:.5f}; Z_corr_b = {Z_corr_b:.5f}")
        print("===== done =====\n")

        return Z_corr_a, Z_corr_b, matched_dT, matched_dZ

    def generate_jump_length_distribution(self,fitparams, JumpProb, r,
                                          LocError, dT, dZ, a, b, fit2states=True,
                                          norm=False, useZcorr=True):
        """
        This function has no docstring. This is bad
        """
        if fit2states:
            D_free = fitparams['D_free']
            D_bound = fitparams['D_bound']
            F_bound = fitparams['F_bound']
        else:
            D_fast = fitparams['D_fast']
            D_med = fitparams['D_med']
            D_bound = fitparams['D_bound']
            F_fast = fitparams['F_fast']
            F_bound = fitparams['F_bound']

        y = np.zeros((JumpProb.shape[0], r.shape[0]))
        # Z_corr = np.zeros(JumpProb.shape[0]) # Assume ABSORBING BOUNDARIES

        # Calculate the axial Z-correction
        if useZcorr:
            if fit2states:
                DeltaZ_use = dZ + a * D_free ** .5 + b  # HalfDeltaZ_use = DeltaZ_use/2
            else:
                DeltaZ_useFAST = dZ + a * D_fast ** .5 + b  # HalfDeltaZ_use = DeltaZ_use/2
                DeltaZ_useMED = dZ + a * D_med ** .5 + b  # HalfDeltaZ_use = DeltaZ_use/2
        else:
            DeltaZ_use = None
            DeltaZ_useFAST = None
            DeltaZ_useMED = None

        for iterator in range(JumpProb.shape[0]):
            # Calculate the jump length distribution of the parameters for each
            # time-jump
            curr_dT = (iterator + 1) * dT

            if fit2states:
                y[iterator, :] = self._compute_2states(D_free, D_bound, F_bound,
                                                  curr_dT, r, DeltaZ_use, LocError, useZcorr)
            else:
                y[iterator, :] = self._compute_3states(D_fast, D_med, D_bound,
                                                  F_fast, F_bound,
                                                  curr_dT, r, DeltaZ_useFAST,
                                                  DeltaZ_useMED, LocError, useZcorr)

            if norm:
                norm_y = np.zeros_like(y)
                for i in range(y.shape[0]):  # Normalize y as a PDF
                    norm_y[i, :] = y[i, :] / y[i, :].sum()

                # scaled_y = (float(len(HistVecJumpsCDF))/len(HistVecJumps))*norm_y #scale y for plotting next to histograms
                y = norm_y
        return y

    def compute_jump_length_distribution(self,trackedPar,
                                         CDF=False, useEntireTraj=False, TimePoints=8,
                                         GapsAllowed=1, JumpsToConsider=4,
                                         MaxJump=1.25, BinWidth=0.010,
                                         useAllTraj=None):
        """Function that takes a series of translocations and computes an histogram of
        jump lengths. Returns both

        Arguments:
        - trackedPar: an object containing the trajectories
        - CDF (bool): compute the cumulative distribution function (CDF) instead of the probability distribution function (PDF)
        - useEntireTraj (bool): True if we should use all trajectories to compute the histogram. This can lead to an overestimate of the bound fraction (see paper), but useful for troubleshooting
        - TimePoints (int): how many jump lengths to use for the fitting: 3 timepoints, yields 2 jumps
        - GapsAllowed (int): number of missing frames that are allowed in a single trajectory
        - JumpsToConsider (int): if `UseAllTraj` is False, then use no more than 3 jumps.
        - TimeGap (float): time between frames in milliseconds;
        - MaxJump (float): for PDF fitting and plotting
        - BinWidth (float): for PDF fitting and plotting
        - useAllTraj (bool): DEPRECATED alias for useEntireTraj

        Returns:
        - An histogram at various \Delta t values.
        """

        if useAllTraj == None:
            useAllTraj = useEntireTraj
        else:
            print("WARNING: the useAllTraj parameter is deprecated, use useEntireTraj instead")

        PDF = not CDF
        tic = time.time()  # Start the timer

        # Find total frames using a slight ad-hoc way: find the last frame with
        # a localization and round it. This is not an elegant solution, but it
        # works for your particle density:

        ## /!\ TODO MW: check this critical part of the code
        TempLastFrame = np.max([np.max(i[2]) for i in trackedPar])  # TempLastFrame = max(trackedPar(1,end).Frame)
        CellFrames = 100 * round(TempLastFrame / 100)
        CellLocs = sum([i[2].shape[1] for i in trackedPar])  # for counting the total number of localizations

        print("Number of frames: {}, number of localizations: {}".format(CellFrames, CellLocs))

        ##
        ## ==== Compile histograms for each jump lengths
        ##
        Min3Traj = 0  # for counting number of min3 trajectories;
        CellJumps = 0  # for counting the total number of jumps
        TransFrames = TimePoints + GapsAllowed * (TimePoints - 1)
        TransLengths = []

        for i in range(TransFrames):  # Initialize TransLengths
            TransLengths.append({"Step": []})  # each iteration is a different number of timepoints

        if useAllTraj:  ## Use all of the trajectory
            for i in range(len(trackedPar)):  # 1:length(trackedPar)
                CurrTrajLength = trackedPar[i][0].shape[0]  # size(trackedPar(i).xy,1);

                if CurrTrajLength >= 3:  # save lengths
                    Min3Traj += 1

                # Now loop through the trajectory. Keep in mind that there are
                # missing timepoints in the trajectory, so some gaps may be for
                # multiple timepoints.

                # Figure out what the max jump to consider is:
                HowManyFrames = min(TimePoints - 1, CurrTrajLength)
                if CurrTrajLength > 1:
                    CellJumps = CellJumps + CurrTrajLength - 1  # for counting all the jumps
                    for n in range(1, HowManyFrames + 1):  # 1:HowManyFrames
                        for k in range(CurrTrajLength - (n + 1)):  # =1:CurrTrajLength-n
                            # Find the current XY coordinate and frames between
                            # timepoints
                            CurrXY_points = np.vstack((trackedPar[i][0][k, :],
                                                       trackedPar[i][0][k + n, :]))
                            # vertcat(trackedPar(i).xy[k,:], trackedPar(i).xy[k+n,:])
                            CurrFrameJump = trackedPar[i][2][0][k + n] - \
                                            trackedPar[i][2][0][k]
                            # trackedPar(i).Frame(k+n) - trackedPar(i).Frame(k);

                            # Calculate the distance between the pair of points
                            TransLengths[CurrFrameJump - 1]["Step"].append(self.pdist(CurrXY_points))

        elif not useAllTraj:  ## Use only the first JumpsToConsider timepoints
            for i in range(len(trackedPar)):  # 1:length(trackedPar)
                CurrTrajLength = trackedPar[i][0].shape[0]  # size(trackedPar(i).xy,1);
                if CurrTrajLength >= 3:
                    Min3Traj += 1

                # Loop through the trajectory. If it is a short trajectory, you
                # need to make sure that you do not overshoot. So first figure out
                # how many jumps you can consider.

                # Figure out what the max jump to consider is:
                HowManyFrames = min([TimePoints - 1, CurrTrajLength])
                if CurrTrajLength > 1:
                    CellJumps = CellJumps + CurrTrajLength - 1  # for counting all the jumps
                    for n in range(1, HowManyFrames + 1):  # 1:HowManyFrames
                        FrameToStop = min([CurrTrajLength, n + JumpsToConsider])
                        for k in range(FrameToStop - n):  # =1:FrameToStop-n
                            # Find the current XY coordinate and frames between
                            # timepoints
                            CurrXY_points = np.vstack((trackedPar[i][0][k, :],
                                                       trackedPar[i][0][k + n, :]))

                            CurrFrameJump = trackedPar[i][2][0][k + n] - \
                                            trackedPar[i][2][0][k]
                            # trackedPar(i).Frame(k+n) - trackedPar(i).Frame(k);
                            # Compute the distance between the pair of points
                            TransLengths[CurrFrameJump - 1]["Step"].append(self.pdist(CurrXY_points))

        ## Calculate the PDF histograms (required for CDF)
        HistVecJumps = np.arange(0, MaxJump + BinWidth, BinWidth)  # jump lengths in micrometers
        JumpProb = np.zeros((TimePoints - 1,
                             len(HistVecJumps)))  ## second -1 added when converting to Python due to inconsistencies between the histc and histogram function
        # TODO MW: investigate those differences
        for i in range(JumpProb.shape[0]):
            JumpProb[i, :] = np.float64(
                np.histogram(TransLengths[i]["Step"],
                             bins=np.hstack((HistVecJumps, HistVecJumps[-1])))[0]) / len(TransLengths[i]["Step"])

        if CDF:  ## CALCULATE THE CDF HISTOGRAMS:
            BinWidthCDF = 0.001
            HistVecJumpsCDF = np.arange(0, MaxJump + BinWidthCDF, BinWidthCDF)  # jump lengths in micrometers
            JumpProbFine = np.zeros((TimePoints - 1, len(HistVecJumpsCDF)))
            for i in range(JumpProb.shape[0]):
                JumpProbFine[i, :] = np.float64(
                    np.histogram(TransLengths[i]["Step"],
                                 bins=np.hstack((HistVecJumpsCDF, HistVecJumpsCDF[-1])))[0]) / len(
                    TransLengths[i]["Step"])

            JumpProbCDF = np.zeros((TimePoints - 1,
                                    len(HistVecJumpsCDF)))  ## second -1 added when converting to Python due to inconsistencies between the histc and histogram function
            for i in range(JumpProbCDF.shape[0]):  # 1:size(JumpProbCDF,1)
                for j in range(2, JumpProbCDF.shape[1] + 1):  # =2:size(JumpProbCDF,2)
                    JumpProbCDF[i, j - 1] = sum(JumpProbFine[i, :j])

        toc = time.time()

        if PDF:
            return [HistVecJumps, JumpProb, {'time': toc - tic}]
        elif CDF:
            return [HistVecJumpsCDF, JumpProbCDF, HistVecJumps, JumpProb,
                    {'time': toc - tic}]



    def _compute_2states(self,D_free, D_bound, F_bound,
                         curr_dT, r, DeltaZ, LocError, useZcorr):
        """
        Compute the PDF contribution for the 2-state (bound/unbound) model
        at a given Δt (curr_dT) over radii r.

        D_free, D_bound: diffusion coefficients (µm²/s)
        F_bound: fraction bound
        curr_dT: time lag (s)
        r: array of radii (µm)
        DeltaZ: axial detection range (µm) after z-correction
        LocError: localization error (µm)
        useZcorr: whether to apply z-correction
        """
        # z-correction factor
        if useZcorr:
            halfZ = DeltaZ / 2.0
            stp = DeltaZ / 200.0
            xint = np.linspace(-halfZ, halfZ, 200)
            yint = [self.C_AbsorBoundAUTO(x, curr_dT, D_free, halfZ) * stp for x in xint]
            Zcorr = np.sum(yint) / DeltaZ
        else:
            Zcorr = 1.0

        # free component
        var_free = 2 * (D_free * curr_dT + LocError ** 2)
        y_free = Zcorr * (1 - F_bound) * (r / var_free) * np.exp(-r ** 2 / var_free)

        # bound component
        var_bound = 2 * (D_bound * curr_dT + LocError ** 2)
        y_bound = F_bound * (r / var_bound) * np.exp(-r ** 2 / var_bound)

        return y_free + y_bound

    def _compute_3states(self,D_fast, D_med, D_bound, F_fast, F_bound,
                         curr_dT, r, DeltaZ_fast, DeltaZ_med, LocError, useZcorr):
        """
        Compute the PDF contribution for the 3-state (fast/medium/bound) model
        at a given Δt over radii r.
        """
        # z-correction for fast
        if useZcorr:
            halfZf = DeltaZ_fast / 2.0
            xintf = np.linspace(-halfZf, halfZf, int(DeltaZ_fast / 0.08))
            yintf = [self.C_AbsorBoundAUTO(x, curr_dT, D_fast, halfZf) * 0.08 for x in xintf]
            Zf = np.sum(yintf) / DeltaZ_fast
        else:
            Zf = 1.0

        # z-correction for med
        if useZcorr:
            halfZm = DeltaZ_med / 2.0
            xintm = np.linspace(-halfZm, halfZm, int(DeltaZ_med / 0.08))
            yintm = [self.C_AbsorBoundAUTO(x, curr_dT, D_med, halfZm) * 0.08 for x in xintm]
            Zm = np.sum(yintm) / DeltaZ_med
        else:
            Zm = 1.0

        # bound component
        var_b = 2 * (D_bound * curr_dT + LocError ** 2)
        y_b = F_bound * (r / var_b) * np.exp(-r ** 2 / var_b)

        # fast component
        var_f = 2 * (D_fast * curr_dT + LocError ** 2)
        y_f = Zf * F_fast * (r / var_f) * np.exp(-r ** 2 / var_f)

        # medium component
        F_med = 1.0 - F_fast - F_bound
        var_m = 2 * (D_med * curr_dT + LocError ** 2)
        y_m = Zm * F_med * (r / var_m) * np.exp(-r ** 2 / var_m)

        return y_b + y_f + y_m

    def C_AbsorBoundAUTO(self,z, CurrTime, D, halfZ):
        """
        Corrected reflecting‐boundary survival probability using the
        'method of images', stopping when contributions fall below 1e-10.
        """
        WhenToStop = 1e-10
        n = 0
        f = np.inf
        h = 1.0

        while abs(f) > WhenToStop:
            if CurrTime > 0:
                z1 = ((2 * n + 1) * halfZ - z) / np.sqrt(4 * D * CurrTime)
                z2 = ((2 * n + 1) * halfZ + z) / np.sqrt(4 * D * CurrTime)
            else:
                # instantaneous case
                z1 = np.inf if (2 * n + 1) * halfZ - z >= 0 else -np.inf
                z2 = np.inf
            f = ((-1) ** n) * (erfc(z1) + erfc(z2))
            h -= f
            n += 1

        return h

    def simulate_jump_length_distribution(self,parameter_guess, JumpProb,
                                          HistVecJumpsCDF, HistVecJump,
                                          dT, dZ, LocError, PDF_or_CDF, a, b,
                                          fit2states=True, useZcorr=True, verbose=True):
        """Function 'SS_2State_model_Z_corr_v4' actually returns a distribution
        given the parameter_guess input. This function is to be used inside a
        least square fitting method, such as Matlab's `lsqcurvefit` or
        Python's `lmfit`.

        Note that this function assumes some *global variables* that are provided
        by the main script: LocError dT HistVecJumps dZ HistVecJumpsCDF PDF_or_CDF
        """

        # ==== Initialize stuff
        HistVecJumps = HistVecJump.copy()
        HistVecJumps += HistVecJumps[1] / 2.
        r = HistVecJumpsCDF.copy()
        r += r[1] / 2.
        y = np.zeros((JumpProb.shape[0], len(r)))
        Binned_y_PDF = np.zeros((JumpProb.shape[0], JumpProb.shape[1]))

        if fit2states:
            D_FREE = parameter_guess[0]
            D_BOUND = parameter_guess[1]
            F_BOUND = parameter_guess[2]
        else:
            D_FAST = parameter_guess[0]
            D_MED = parameter_guess[1]
            D_BOUND = parameter_guess[2]
            F_FAST = parameter_guess[3]
            F_BOUND = parameter_guess[4]

        # ==== Precompute stuff
        # Calculate the axial Z-correction
        # First calculate the corrected DeltaZ:
        ##DeltaZ_use = dZ + 0.15716  * D_FREE**.5 + 0.20811 # See CHANGELOG_fit
        ##DeltaZ_use = dZ + 0.24472 * D_FREE**.5 + 0.19789
        if useZcorr:
            if fit2states:
                DeltaZ_use = dZ + a * D_FREE ** .5 + b  # HalfDeltaZ_use = DeltaZ_use/2
            else:
                DeltaZ_useFAST = dZ + a * D_FAST ** .5 + b  # HalfDeltaZ_use = DeltaZ_use/2
                DeltaZ_useMED = dZ + a * D_MED ** .5 + b  # HalfDeltaZ_use = DeltaZ_use/2
        else:
            DeltaZ_use = None
            DeltaZ_useFAST = None
            DeltaZ_useMED = None

        for iterator in range(JumpProb.shape[0]):
            # Calculate the jump length distribution of the parameters for each
            # time-jump
            curr_dT = (iterator + 1) * dT
            if verbose:
                print("-- computing dT = {} ({}/{})".format(curr_dT, iterator + 1, JumpProb.shape[0]))
            if fit2states:
                y[iterator, :] = self._compute_2states(D_FREE, D_BOUND, F_BOUND,
                                                  curr_dT, r, DeltaZ_use, LocError, useZcorr)
            else:
                y[iterator, :] = self._compute_3states(D_FAST, D_MED, D_BOUND,
                                                  F_FAST, F_BOUND,
                                                  curr_dT, r, DeltaZ_useFAST,
                                                  DeltaZ_useMED, LocError, useZcorr)

        if PDF_or_CDF == 1:
            # Now bin the output y so that it matches the JumpProb variable:
            for i in range(JumpProb.shape[0]):  # 1:size(JumpProb,1)
                for j in range(JumpProb.shape[1]):  # =1:size(JumpProb,2)
                    if j == (JumpProb.shape[1] - 1):
                        Binned_y_PDF[i, j] = y[i, maxIndex:].mean()
                    else:
                        minIndex = np.argmin(np.abs(r - HistVecJumps[j]))
                        maxIndex = np.argmin(np.abs(r - HistVecJumps[j + 1]))
                        Binned_y_PDF[i, j] = y[i, minIndex:maxIndex].mean()
            for i in range(JumpProb.shape[0]):  # 1:size(JumpProb,1) ## Normalize
                Binned_y_PDF[i, :] = Binned_y_PDF[i, :] / sum(Binned_y_PDF[i, :]);
            Binned_y = Binned_y_PDF  # You want to fit to a histogram, so no need to calculate the CDF
            return Binned_y

        elif PDF_or_CDF == 2:
            # You want to fit to a CDF function, so first we must calculate the CDF
            # from the finely binned PDF
            Binned_y_CDF = np.zeros((JumpProb.shape[0], JumpProb.shape[1]))

            ## Normalize the PDF
            for i in range(Binned_y_CDF.shape[0]):
                Binned_y_PDF[i, :] = y[i, :] / y[i, :].sum()

            ## calculate the CDF
            for i in range(Binned_y_CDF.shape[0]):  # 1:size(Binned_y_CDF,1):
                Binned_y_CDF[i, :] = np.cumsum(Binned_y_PDF[i, :])
                # for j in range(1, Binned_y_CDF.shape[1]): #=2:size(Binned_y_CDF,2):
                #    Binned_y_CDF[i,j] = Binned_y_PDF[i,:j].sum()
            Binned_y = Binned_y_CDF  ##Output the final variable

        return Binned_y

    def fit_jump_length_distribution(self,JumpProb, JumpProbCDF,
                                     HistVecJumps, HistVecJumpsCDF,
                                     LB, UB,
                                     LocError, iterations, dT, dZ, ModelFit, a, b,
                                     fit2states=True, fitSigma=False,
                                     verbose=True, init=None, useZcorr=True,
                                     solverparams={'ftol': 1e-20,
                                                   'xtol': 1e-20,
                                                   'maxfev': 100000,
                                                   }):
        """Fits a kinetic model to an empirical jump length distribution.
        This applies a non-linear least squared fitting procedure.
        """
        ## Lower and Upper parameter bounds
        diff = np.array(UB) - np.array(LB)  # difference: used for initial parameters guess
        best_ssq2 = 5e10  # initial error

        # Need to ensure that the x-input is the same size as y-output
        if ModelFit == 1:
            ModelHistVecJumps = np.zeros((JumpProb.shape[0], len(HistVecJumps)))
            for i in range(JumpProb.shape[0]):  # 1:size(JumpProb,1)
                ModelHistVecJumps[i, :] = HistVecJumps
            y = JumpProb
            x = np.repeat(HistVecJumps[:-1], JumpProb.shape[1])

        elif ModelFit == 2:
            ModelHistVecJumps = np.zeros((JumpProb.shape[0], len(HistVecJumpsCDF)))
            for i in range(JumpProb.shape[0]):  # 1:size(JumpProb,1)
                ModelHistVecJumps[i, :] = HistVecJumpsCDF
            y = JumpProbCDF
            x = np.repeat(HistVecJumpsCDF[:-1], JumpProbCDF.shape[1])

        params = {"dT": dT,
                  "dZ": dZ,
                  "HistVecJumps": HistVecJumps,
                  "HistVecJumpsCDF": HistVecJumpsCDF,
                  "PDF_or_CDF": ModelFit,
                  "JumpProb": JumpProb,
                  "JumpProbCDF": JumpProbCDF,
                  "LB": LB,
                  "UB": UB,
                  "a": a,
                  "b": b,
                  "useZcorr": useZcorr}

        ## ==== Get ready for the fitting
        def wrapped_jump_length_2states(x, D_free, D_bound, F_bound, sigma):
            """Wrapper for the main fit function (assuming global variables)"""
            if params['PDF_or_CDF'] == 1:  # PDF fit
                dis = self.simulate_jump_length_distribution(
                    (D_free, D_bound, F_bound),
                    params['JumpProb'],
                    params['HistVecJumpsCDF'],
                    params['HistVecJumps'],
                    params['dT'],
                    params['dZ'],
                    sigma,  # params['LocError'],
                    params['PDF_or_CDF'],
                    params['a'],
                    params['b'], fit2states=True, useZcorr=params['useZcorr'], verbose=False)
            elif params['PDF_or_CDF'] == 2:  # CDF fit
                dis = self.simulate_jump_length_distribution(
                    (D_free, D_bound, F_bound),
                    params['JumpProbCDF'],
                    params['HistVecJumpsCDF'],
                    params['HistVecJumps'],
                    params['dT'],
                    params['dZ'],
                    sigma,  # params['LocError'],
                    params['PDF_or_CDF'],
                    params['a'],
                    params['b'], fit2states=True, useZcorr=params['useZcorr'], verbose=False)
            return dis.flatten()

        def wrapped_jump_length_3states(x, D_fast, D_med, D_bound,
                                        F_fast, F_bound, sigma):
            """Wrapper for the main fit function (assuming global variables)"""
            if params['PDF_or_CDF'] == 1:  # PDF fit
                dis = self.simulate_jump_length_distribution(
                    (D_fast, D_med, D_bound, F_fast, F_bound),
                    params['JumpProb'],
                    params['HistVecJumpsCDF'],
                    params['HistVecJumps'],
                    params['dT'],
                    params['dZ'],
                    sigma,  # params['LocError'],
                    params['PDF_or_CDF'],
                    params['a'],
                    params['b'], fit2states=False, useZcorr=params['useZcorr'], verbose=False)
            elif params['PDF_or_CDF'] == 2:  # CDF fit
                dis = self.simulate_jump_length_distribution(
                    (D_fast, D_med, D_bound, F_fast, F_bound),
                    params['JumpProbCDF'],
                    params['HistVecJumpsCDF'],
                    params['HistVecJumps'],
                    params['dT'],
                    params['dZ'],
                    sigma,  # params['LocError'],
                    params['PDF_or_CDF'],
                    params['a'],
                    params['b'], fit2states=False, useZcorr=params['useZcorr'], verbose=False)
            if F_fast + F_bound < 1:
                return np.hstack((dis.flatten(), 0))
            else:
                return np.hstack((dis.flatten(), 10000 * (1 - F_fast - F_bound)))

        if fit2states:  # Instantiate model
            jumplengthmodel = lmfit.Model(wrapped_jump_length_2states)
            pars = jumplengthmodel.make_params()  ## Create an empty set of params
            y_init = y.flatten()
        else:
            jumplengthmodel = lmfit.Model(wrapped_jump_length_3states)
            pars = jumplengthmodel.make_params()  ## Create an empty set of params
            y_init = np.hstack((y.flatten(), 0))

        if init == None:
            init = []
            init1 = np.random.rand(len(LB)) * diff + LB
            while not fit2states and len(init) < iterations:
                init1 = np.random.rand(len(LB)) * diff + LB
                if init1[3] + init1[4] < 1:
                    init.append(init1)
            if fit2states:
                init = [np.random.rand(len(LB)) * diff + LB for i in range(iterations)]
        elif len(init) != iterations:
            print("'iterations' variable ignored because 'init' is provided and has length {}".format(len(init)))

        for (i, guess) in enumerate(init):
            eps = 1e-8
            if fit2states:
                if (guess.shape[0] != 3 and not fitSigma) or (guess.shape[0] != 4 and fitSigma):
                    print("init value has a wrong number of elements")
                pars['D_free'].set(min=LB[0], max=UB[0], value=guess[0])
                pars['D_bound'].set(min=LB[1], max=UB[1], value=guess[1])
                pars['F_bound'].set(min=LB[2], max=UB[2], value=guess[2])
                if fitSigma:
                    pars['sigma'].set(min=LB[3], max=UB[3], value=guess[3])
                else:
                    pars['sigma'].set(value=LocError, vary=False)
                if abs(LB[0] - UB[0]) < eps:
                    pars['D_free'].set(value=LB[0], vary=False)
                if abs(LB[1] - UB[1]) < eps:
                    pars['D_bound'].set(value=LB[1], vary=False)
            else:
                if (guess.shape[0] != 5 and not fitSigma) or (guess.shape[0] != 6 and fitSigma):
                    print("init value has a wrong number of elements")

                pars['D_fast'].set(min=LB[0], max=UB[0], value=guess[0])
                pars['D_med'].set(min=LB[1], max=UB[1], value=guess[1])
                pars['D_bound'].set(min=LB[2], max=UB[2], value=guess[2])
                pars['F_bound'].set(min=LB[4], max=UB[4], value=guess[4])
                pars['F_fast'].set(min=LB[3], max=UB[3], value=guess[3])
                if fitSigma:
                    pars['sigma'].set(min=LB[5], max=UB[5], value=guess[5])
                else:
                    pars['sigma'].set(value=LocError, vary=False)
                if abs(LB[0] - UB[0]) < eps:
                    pars['D_fast'].set(value=LB[0], vary=False)
                if abs(LB[1] - UB[1]) < eps:
                    pars['D_med'].set(value=LB[1], vary=False)
                if abs(LB[2] - UB[2]) < eps:
                    pars['D_bound'].set(value=LB[2], vary=False)

            out = jumplengthmodel.fit(y_init, x=x, params=pars, fit_kws=solverparams)
            ssq2 = (out.residual[:-1] ** 2).sum() / (out.residual.shape[0] - 1)
            out.params.ssq2 = ssq2

            ## See if the current fit is an improvement:
            if ssq2 < best_ssq2:
                best_vals = out.params
                best_ssq2 = ssq2
                if verbose:
                    print('==================================================')
                    print('Improved fit on iteration {}'.format(i + 1))
                    print('Improved error is {}'.format(ssq2))
                    print(out.params.pretty_print(columns=['value', 'min', 'max', 'stderr']))
                    print('==================================================')
            else:
                print('Iteration {} did not yield an improved fit'.format(i + 1))
        return out



    def plot_histogram(self,HistVecJumps, emp_hist, HistVecJumpsCDF=None, sim_hist=None,
                       TimeGap=None, SampleName=None, CellNumb=None,
                       len_trackedPar=None, Min3Traj=None, CellLocs=None,
                       CellFrames=None, CellJumps=None, ModelFit=None,
                       D_free=None, D_bound=None, F_bound=None):
        """Function that plots an empirical histogram of jump lengths,
        with an optional overlay of simulated/theoretical histogram of
        jump lengths"""

        ## Parameter parsing for text labels
        if CellLocs != None and CellFrames != None:
            locs_per_frame = round(CellLocs / CellFrames * 1000) / 1000
        else:
            locs_per_frame = 'na'
        if SampleName == None:
            SampleName = 'na'
        if CellNumb == None:
            CellNumb = 'na'
        if len_trackedPar == None:
            len_trackedPar = 'na'
        if Min3Traj == None:
            Min3Traj = 'na'
        if CellLocs == None:
            CellLocs = 'na'
        if CellFrames == None:
            CellFrames = 'na'
        if CellJumps == None:
            CellJumps = 'na'
        if ModelFit == None:
            ModelFit = 'na'
        if D_free == None:
            D_free = 'na'
        if D_bound == None:
            D_bound = 'na'
        if F_bound == None:
            F_bound = 'na'

        histogram_spacer = 0.055
        number = emp_hist.shape[0]
        cmap = plt.get_cmap('viridis')
        colour = [cmap(i) for i in np.linspace(0, 1, number)]

        Nbins = len(HistVecJumps)
        for i in range(emp_hist.shape[0] - 1, -1, -1):
            new_level = i * histogram_spacer
            # 画基础线
            plt.plot(HistVecJumps, [new_level] * Nbins, 'k-', linewidth=1)

            for j in range(Nbins):
                y1 = new_level
                y2 = emp_hist[i, j] + new_level
                if j < Nbins - 1:
                    x1 = HistVecJumps[j]
                    x2 = HistVecJumps[j + 1]
                else:
                    # 最后一 bin：往后延一个 bin 宽度
                    bin_w = HistVecJumps[-1] - HistVecJumps[-2]
                    x1 = HistVecJumps[-1]
                    x2 = HistVecJumps[-1] + bin_w
                plt.fill([x1, x1, x2, x2], [y1, y2, y2, y1], color=colour[i])

            if type(sim_hist) != type(None):  ## HistVecJumpsCDF should also be provided
                plt.plot(HistVecJumpsCDF, sim_hist[i, :] + new_level, 'k-', linewidth=2)
            if TimeGap != None:
                plt.text(0.8,  # 改成 0.8（80% x 轴长度）
                         new_level + 0.3 * histogram_spacer,
                         r'$\Delta t$ : {} ms'.format(TimeGap * (i + 1)))
                # plt.text(0.6 * max(HistVecJumps), new_level + 0.3 * histogram_spacer,
                #          '$\Delta t$ : {} ms'.format(TimeGap * (i + 1)))
            else:
                # plt.text(0.6 * max(HistVecJumps), new_level + 0.3 * histogram_spacer, '${} \Delta t$'.format(i + 1))
                plt.text(0.8,
                         new_level + 0.3 * histogram_spacer,
                         r'${}\,\Delta t$'.format(i + 1))

        # plt.xlim(0, HistVecJumps.max())
        plt.xlim(0,1)
        plt.ylabel('Probability')
        plt.xlabel('jump length ($\mu m$)')
        # if type(sim_hist) != type(None):
        #     plt.title(
        #         '{}; Cell number {}; Fit Type = {}; Dfree = {}; Dbound = {}; FracBound = {}, Total trajectories: {}; => Length 3 trajectories: {}, \nLocs = {}, Locs/Frame = {}; jumps: {}'
        #         .format(
        #             SampleName, CellNumb, ModelFit,
        #             D_free, D_bound, F_bound,
        #             len_trackedPar, Min3Traj, CellLocs,
        #             locs_per_frame,
        #             CellJumps))
        # else:
        #     plt.title(
        #         '{}; Cell number {}; Total trajectories: {}; => Length 3 trajectories: {}, \nLocs = {}, Locs/Frame = {}; jumps: {}'
        #         .format(
        #             SampleName, CellNumb,
        #             len_trackedPar, Min3Traj, CellLocs,
        #             locs_per_frame,
        #             CellJumps))
        plt.yticks([])
        return plt.gcf()

    def pdist(self,m):
        """Euclidean distance between two 2D points in a (2×2) array."""
        return np.hypot(m[0, 0] - m[1, 0], m[0, 1] - m[1, 1])

    def load_imagestack(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("TIF files", "*.tif")])
        if file_paths:
            full_path = file_paths[0]
            self.image_bin['image_path'] = full_path  # 完整路径
            self.image_bin['pathname'] = os.path.dirname(full_path)  # 目录
            self.image_bin['filename'] = os.path.basename(full_path)  # 文件名

            try:
                image = Image.open(full_path)
                self.image_bin['total_frames'] = getattr(image, 'n_frames', 1)
                self.image_bin['width'], self.image_bin['height'] = image.size
                self.image_bin['roi'] = [0, 0, self.image_bin['width'], self.image_bin['height']]
                self.image_bin['h_roi'] = 0
                self.image_bin['image_name'] = full_path
                self.image_bin['raw_image'] = image  # 可留着预览
                self.display_image(image, 1)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {e}")

    def load_slimfast(self):
        """从 TXT 文件加载 SLIMfast 数据，并按原始 TIFF 尺寸渲染。"""
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not filename:
            return

        # 读取 .txt
        try:
            localization_data = np.loadtxt(filename, delimiter='\t', skiprows=1)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            return

        # —— 新建窗口 & 工具栏 —— #
        menu_fig = tk.Toplevel(self.master)
        menu_fig.title("SPTpy Data Viewer")
        menu_fig.geometry("600x600")

        toolbar = tk.Frame(menu_fig)
        toolbar.grid(row=0, column=0, sticky='ew')

        buttons = [
            ("Options", self.set_options),
            ("Colormap", self.color_map_editor),
            ("Int Dist", self.particle_intensity_hist),
            ("Prec Dist", self.loc_prec),
            ("s2n Dist", self.snr_hist),
            ("Gen Traj", self.build_tracks),
            ("LOC Dist", self.detection_trace),
            # ("GEN Movie", self.render_movie),
            ("GEN Movie", self.export_three_snapshots),
            ("Scatter Plot", self.scatter_plot),
            ("Nuclear Seg", self.open_nuclear_segmentation)
        ]
        for index, (text, command) in enumerate(buttons):
            tk.Button(toolbar, text=text, command=command).grid(row=0, column=index, padx=1, pady=1)
        toolbar.grid_columnconfigure(tuple(range(len(buttons))), weight=1)

        # —— 初始化 image_bin 的必要字段 —— #
        self.image_bin = {
            'pathname': filename,
            'filename': os.path.basename(filename),
            'is_loaded': True,
            'is_superstack': False,
            'is_track': 0,
            'view_mode': 'monoView',
            'loc_start': 1,
            'loc_end': int(localization_data[-1, 1]),

            # 算法&光学参数（保持你原值）
            'error_rate': -6,
            'w2d': 7,
            'dfltn_loops': 0,
            'min_int': 0.0,
            'loc_parallel': False,
            'n_cores': 1,
            'spatial_correction': False,
            'is_radius_tol': False,
            'radius_tol': 50,
            'pos_tol': 1.5,
            'max_optim_iter': 50,
            'term_tol': -2,

            'micron_bar_length': 10.0,
            'pxSize': 0.16,
            'px_size':0.16,
            'frameSize': 40,
            'frame_size':40,
            'cntsPerPhoton': 20.2,
            'cnts_per_photon': 20.2,
            'emWvlnth': 590,
            'em_wvlnth': 590,
            'NA': 1.49,
            'na': 1.49,
            'psfScale': 1.35,
            'psf_scale': 1.35,
            'psfStd': 1.03,
            'psf_std': 1.03,

            'rW': 20,
            'rStep': 1,

            # 渲染选项
            'r_start': 1,
            'r_end': localization_data[-1, 1],
            'r_live': 1,
            'fps': 10,
            'mov_compression': 'none',
            'conv_mode': 1,  # 1=Fixed, 2=Dynamic
            'int_weight': 2000,
            'size_fac': 8,
            'is_cumsum': False,
            'exf_new': 5,  # 超分辨放大倍率
        }

        # 其他选项初始化
        self.image_bin['is_scalebar'] = 0
        self.image_bin['is_colormap'] = 0
        self.image_bin['is_timestamp'] = 0
        self.image_bin['is_thresh_loc_prec'] = 0
        self.image_bin['is_thresh_snr'] = 0
        self.image_bin['is_thresh_density'] = 0
        self.image_bin['cluster_mode'] = 1

        # 渲染选项
        self.image_bin['r_start'] = 1
        self.image_bin['r_end'] = localization_data[-1, 1]
        self.image_bin['r_live'] = 1
        self.image_bin['fps'] = 10
        self.image_bin['mov_compression'] = 'none'
        self.image_bin['conv_mode'] = 1
        self.image_bin['int_weight'] = 2000
        self.image_bin['size_fac'] = 8
        self.image_bin['is_cumsum'] = False
        self.image_bin['exf_new'] = 5

        # 阈值选项
        self.image_bin['isThreshLocPrec'] = 0
        self.image_bin['minLoc'] = 0
        self.image_bin['maxLoc'] = float('inf')
        self.image_bin['isThreshSNR'] = 0
        self.image_bin['minSNR'] = 0
        self.image_bin['maxSNR'] = float('inf')
        self.image_bin['isThreshDensity'] = 0
        self.image_bin['cluster_mode'] = 1

        # —— 从 txt 取数据列 —— #
        # 明确列索引（你给出的 _locs.txt 顺序）：
        # 0: particle_idx, 1: frame_idx, 2: y, 3: x, 4: alpha, 5: Sig2, 6: offset, 7: r, 8: result_ok
        IDX_PARTICLE = 0
        IDX_FRAME = 1
        IDX_Y = 2
        IDX_X = 3
        IDX_ALPHA = 4
        IDX_SIG2 = 5   # Sig^2 (noise^2) as in your description
        IDX_OFFSET = 6
        IDX_R = 7

        # Defensive: 如果列数小于预期则抛错
        if localization_data.shape[1] < 8:
            messagebox.showerror("Error", f"Unexpected .txt format: need >=8 cols, got {localization_data.shape[1]}")
            return

        # frame 保留为 int （1-based，跟 MATLAB 保持一致）
        self.image_bin['frame'] = localization_data[:, IDX_FRAME].astype(int)

        # x,y — 注意：MATLAB 里常用 (i -> row = y, j -> col = x)，在 Python 里我们只保持命名清晰
        self.image_bin['ctrsX'] = localization_data[:, IDX_X].astype(float)  # x_coord (可能为像素或 μm，取决你的文件)
        self.image_bin['ctrsY'] = localization_data[:, IDX_Y].astype(float)

        # radius 直接取 r 列
        self.image_bin['radius'] = localization_data[:, IDX_R].astype(float)

        # noise: 在你的描述中文件给出 Sig2（方差），所以 noise = sqrt(max(Sig2,0))
        self.image_bin['noise'] = np.maximum(localization_data[:, IDX_SIG2].astype(float), 0.0)

        # signal/alpha: 你的 _locs 中 alpha 是峰值强度（或已定义值）。
        # 这里使用你原来代码里和 MATLAB 等价的变换： signal = alpha / (sqrt(pi) * radius)
        # 但我们做防御性保护 radius 非零
        eps_r = 1e-12
        self.image_bin['signal'] = localization_data[:, IDX_ALPHA].astype(float) / (np.sqrt(np.pi) * np.maximum(self.image_bin['radius'], eps_r))

        # # 打印前 5 行，便于你核对列映射是否正确
        # print("[DBG] locs head (frame, x, y, alpha, sig2, r, computed signal, noise):")
        # for i in range(min(5, localization_data.shape[0])):
        #     print(i, int(self.image_bin['frame'][i]), self.image_bin['ctrsX'][i], self.image_bin['ctrsY'][i],
        #           localization_data[i, IDX_ALPHA], localization_data[i, IDX_SIG2], self.image_bin['radius'][i],
        #           self.image_bin['signal'][i], self.image_bin['noise'][i])

        # 光学选项
        self.image_bin['pxSize'] = 0.16
        self.image_bin['frameSize'] = 40
        self.image_bin['cntsPerPhoton'] = 20.2
        self.image_bin['emWvlnth'] = 590
        self.image_bin['NA'] = 1.49
        self.image_bin['psfScale'] = 1.35
        self.image_bin['psfStd'] = 1.03
        # 光子数/精度/SNR（保持原逻辑）
        self.image_bin['photons'] = (
                self.image_bin['signal'] * 2 * np.pi * self.image_bin['radius'] ** 2 / self.image_bin['cntsPerPhoton']
        )
        self.image_bin['precision'] = (
                np.sqrt(
                    self.image_bin['psfStd'] * self.image_bin['pxSize'] ** 2 + (self.image_bin['pxSize'] ** 2) / 12.0
                / np.maximum(self.image_bin['photons'], 1e-12)
                + (8 * np.pi * self.image_bin['psfStd'] * self.image_bin['pxSize'] ** 4 * self.image_bin['noise'] ** 2)
                / (self.image_bin['pxSize'] ** 2  * np.maximum(self.image_bin['photons'], 1e-12) ** 2))
        )
        self.image_bin['snr'] = self.image_bin['signal'] / np.maximum(self.image_bin['noise'], 1e-12)

        # —— 统计每帧粒子数（保持你原函数） —— #
        ctrsN, total_particles = self.compute_frame_particle_counts(filename)
        self.image_bin['ctrsN'] = ctrsN
        self.image_bin['total_particles'] = total_particles

        # —— 关键：按原始 TIFF 尺寸设置 width/height/roi —— #
        tif_path = os.path.splitext(filename)[0].replace('_locs', '') + '.tif'
        try:
            import tifffile
            with tifffile.TiffFile(tif_path) as tif:
                H, W = tif.pages[0].shape  # (rows, cols) = (height, width)
        except Exception:
            H, W = 320, 320  # 兜底

        self.image_bin['width'] = W
        self.image_bin['height'] = H
        self.image_bin['roi'] = [0, 0, W, H]  # x0, y0, w, h

        # 跟踪选项
        self.image_bin['trackStart'] = 1
        self.image_bin['trackEnd'] = float('inf')
        self.image_bin['Dmax'] = 0.5
        self.image_bin['searchExpFac'] = 1.2
        self.image_bin['stat_win'] = 10
        self.image_bin['maxComp'] = 1
        self.image_bin['max_off_time'] = 2
        self.image_bin['intLawWeight'] = 0.9
        self.image_bin['diffLawWeight'] = 0.5

        # ——（可选）做 ROI 预处理 —— #
        if hasattr(self, "_preprocess_by_roi"):
            self._preprocess_by_roi()

        # —— 渲染 —— #
        data = {
            'ctrsX': self.image_bin['ctrsX'],
            'ctrsY': self.image_bin['ctrsY'],
            'photons': self.image_bin['photons'],
            'precision': self.image_bin['precision'],
        }
        roi = self.image_bin['roi']

        rI, new_w, new_h, N = self.render_image(
            data=data,
            roi=roi,
            exf=self.image_bin['exf_new'],
            width=self.image_bin['width'],
            height=self.image_bin['height'],
            int_weight=self.image_bin['int_weight'],
            size_fac=self.image_bin['size_fac'],
            px_size=self.image_bin['pxSize'],
            mode=self.image_bin['conv_mode']
        )
        # 存下 float 图，方便后续处理/保存
        self.image_bin['render'] = rI

        # —— 图像区域（跟随窗口拉伸，无留白） —— #
        image_frame = tk.Frame(menu_fig)
        image_frame.grid(row=1, column=0, sticky='nsew')
        menu_fig.grid_rowconfigure(1, weight=1)
        menu_fig.grid_columnconfigure(0, weight=1)
        image_frame.grid_rowconfigure(0, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)

        self.fig = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.imshow(rI.astype(np.uint8), cmap="viridis", aspect="auto")
        self.ax.axis('off')

        canvas = FigureCanvasTkAgg(self.fig, master=image_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # 底部保存按钮
        btn_frame = tk.Frame(menu_fig)
        btn_frame.grid(row=2, column=0, pady=5, sticky='ew')
        tk.Button(btn_frame, text="Save", command=self.save_with_matplotlib).pack(side='left', expand=True, padx=5)

        messagebox.showinfo("Success", f"Loaded {self.image_bin['filename']} successfully.")

    def _preprocess_by_roi(self):
        # 原始数据（像素单位）
        x = self.image_bin['ctrsX']
        y = self.image_bin['ctrsY']
        r = self.image_bin['radius']
        n = self.image_bin['noise']
        frame = self.image_bin['frame'].astype(int)  # TXT 基本是 1-based
        signal = self.image_bin['signal']

        x0, y0, w, h = self.image_bin['roi']
        roi_mask = (x >= x0) & (x < x0 + w) & (y >= y0) & (y < y0 + h)

        # 仅保留 ROI 内的点 —— 之后 build_tracks 只用这些
        self.data = {
            'frame': frame[roi_mask],
            'ctrsX': x[roi_mask],
            'ctrsY': y[roi_mask],
            'radius': r[roi_mask],
            'noise': n[roi_mask],
            'signal': signal[roi_mask],  # 注意：alpha = signal * sqrt(pi) * radius
        }

        # 重算每帧计数（par_per_frame）
        max_f = int(self.data['frame'].max()) if self.data['frame'].size else 0
        ctrsN = np.bincount(self.data['frame'], minlength=max_f + 1)  # 档位 0 丢弃
        self.image_bin['ctrsN'] = ctrsN[1:].astype(int)  # 1..max_f

    def open_nuclear_segmentation(self):
        seg_win = tk.Toplevel(self.master)
        seg_win.title("Nuclear Segmentation")
        seg_win.geometry("800x600")

        self.raw_img = None
        self.render_img = None
        self.scatter_img = None

        # ---------------- 左侧显示图像 ----------------
        left_frame = tk.Frame(seg_win)
        left_frame.grid(row=0, column=0, sticky='nsew')
        seg_win.grid_columnconfigure(0, weight=0)
        seg_win.grid_rowconfigure(0, weight=0)

        self.seg_canvas_frame = tk.Frame(left_frame, width=320, height=320)
        self.seg_canvas_frame.grid(row=0, column=0, sticky='n')
        self.seg_canvas_frame.grid_propagate(False)

        # ---------------- 右侧工具栏 ----------------
        right_frame = tk.Frame(seg_win, padx=10, pady=10)
        right_frame.grid(row=0, column=1, sticky='nsew')
        seg_win.grid_columnconfigure(1, weight=1)

        # === 第一部分：Segmentation ===
        tk.Label(right_frame, text="Segmentation", font=("Arial", 12, "bold")).pack(anchor='w', pady=(0, 5))

        # 模型选择区
        seg_options_frame = tk.Frame(right_frame)
        seg_options_frame.pack(fill='x', pady=(0, 5))

        self.use_gpu_var = tk.BooleanVar(value=True)
        tk.Checkbutton(seg_options_frame, text="Use GPU", variable=self.use_gpu_var).grid(row=0, column=0, padx=(0, 10))

        self.model_path_var = tk.StringVar(value="")  # 存储模型路径
        tk.Button(seg_options_frame, text="Select Model File", command=self.select_model_file).grid(row=0, column=1,
                                                                                                    sticky='ew')
        seg_options_frame.grid_columnconfigure(1, weight=1)

        # 加载图像按钮（新位置）
        img_load_frame = tk.Frame(right_frame)
        img_load_frame.pack(pady=(5, 5), fill='x')

        tk.Button(img_load_frame, text="Load Raw Image", command=self.load_raw_image).pack(fill='x', pady=2)
        tk.Button(img_load_frame, text="Load Render Image", command=self.load_render_image).pack(fill='x', pady=2)
        tk.Button(img_load_frame, text="Load Scatter Image", command=self.load_scatter_image).pack(fill='x', pady=2)

        # Run 按钮
        tk.Button(right_frame, text="Run", width=10, anchor='center',
                  command=self.run_segmentation).pack(pady=5, fill='x')
        tk.Button(right_frame, text="gen traj from ROI", width=20,
                  command=self.run_gen_traj_roi).pack(pady=(5,10),fill='x')

        # === 第二部分：Train New Model ===
        tk.Label(right_frame, text="Train New Model", font=("Arial", 12, "bold")).pack(anchor='w', pady=(10, 5))

        tk.Label(right_frame, text="Learning Rate:").pack(anchor='w')
        self.learning_rate_var = tk.StringVar(value="1e-05")
        tk.Entry(right_frame, textvariable=self.learning_rate_var).pack(fill='x')

        tk.Label(right_frame, text="Epochs:").pack(anchor='w')
        self.epoch_var = tk.StringVar(value="200")
        tk.Entry(right_frame, textvariable=self.epoch_var).pack(fill='x')

        tk.Label(right_frame, text="Model Name:").pack(anchor='w')
        self.train_model_name_var = tk.StringVar(value="my_model")
        tk.Entry(right_frame, textvariable=self.train_model_name_var).pack(fill='x')

        tk.Button(right_frame, text="Train", width=10, anchor='center',
                  command=self.train_new_model).pack(pady=5, fill='x')



    def run_gen_traj_roi(self):
        if not hasattr(self, 'pred_mask') or self.pred_mask is None:
            messagebox.showerror("Error", "Please run segmentation first.")
            return

        # 保存预测的 mask（bool）
        self.predicted_roi_mask = self.pred_mask.astype(bool)
        self.build_tracks_from_roi()

    def build_tracks_from_roi(self):
        if not hasattr(self, 'predicted_roi_mask') or self.predicted_roi_mask is None:
            messagebox.showerror("Error", "No predicted ROI found.")
            return

        mask = self.predicted_roi_mask  # shape: (256, 256)

        # 读取所有粒子位置（像素级）
        x_all = np.array(self.image_bin['ctrsX'])  # 列坐标
        y_all = np.array(self.image_bin['ctrsY'])  # 行坐标
        frame_all = np.array(self.image_bin['frame'])
        signal_all = np.array(self.image_bin['signal'])
        noise_all = np.array(self.image_bin['noise'])
        radius_all = np.array(self.image_bin['radius'])

        # 注意粒子坐标是否经过 resize（需要配套 mask 尺寸）→ 可适配你的代码格式
        if self.raw_img.shape != mask.shape:
            scale_x = mask.shape[1] / self.raw_img.shape[1]
            scale_y = mask.shape[0] / self.raw_img.shape[0]
            x = (x_all * scale_x).astype(int)
            y = (y_all * scale_y).astype(int)
        else:
            x = x_all.astype(int)
            y = y_all.astype(int)

        # 限定范围防止越界
        valid = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])

        x = x[valid]
        y = y[valid]
        frame_all = frame_all[valid]
        signal_all = signal_all[valid]
        noise_all = noise_all[valid]
        radius_all = radius_all[valid]

        keep_idx = mask[y, x]

        # 再次限定数组长度
        x_keep = x_all[valid][keep_idx]
        y_keep = y_all[valid][keep_idx]
        frame_keep = frame_all[keep_idx]
        signal_keep = signal_all[keep_idx]
        noise_keep = noise_all[keep_idx]
        radius_keep = radius_all[keep_idx]

        self.image_bin['ctrsX'] = x_keep
        self.image_bin['ctrsY'] = y_keep
        self.image_bin['frame'] = frame_keep
        self.image_bin['signal'] = signal_keep
        self.image_bin['noise'] = noise_keep
        self.image_bin['radius'] = radius_keep

        # 更新 ctrsN（每帧粒子数）
        frame = self.image_bin['frame']
        max_frame = int(np.max(frame)) + 1 if len(frame) > 0 else 0
        ctrsN = np.zeros(max_frame, dtype=int)
        for f in frame:
            ctrsN[int(f)] += 1
        self.image_bin['ctrsN'] = ctrsN

        print("[INFO] Running tracking on ROI-filtered detections:", len(x_keep))
        self.build_tracks()

    def select_model_file(self):
        path = filedialog.askopenfilename(title="Select Model (.pth)", filetypes=[("PyTorch Model", "*.pth")])
        if path:
            self.model_path_var.set(path)
            print(f"[INFO] Model selected: {path}")

    def load_raw_image(self):
        path = filedialog.askopenfilename(title="Select Raw Image")
        if path:
            self.raw_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def load_render_image(self):
        path = filedialog.askopenfilename(title="Select Render Image")
        if path:
            self.render_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def load_scatter_image(self):
        path = filedialog.askopenfilename(title="Select Scatter Image")
        if path:
            self.scatter_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def run_segmentation(self):
        use_gpu = self.use_gpu_var.get()
        if self.raw_img is None or self.render_img is None or self.scatter_img is None:
            messagebox.showerror("Error", "Please load all three images first.")
            return

        import numpy as np, cv2, torch
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt

        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")

        # ---------- 关键：把图像先拉伸成训练时的“可视化 8-bit” ----------
        def to_uint8_view(img):
            arr = np.asarray(img)
            if arr.dtype == np.uint16:
                # 根据有效位深或分位数拉伸到 0–255（避免简单 >>8 导致全黑）
                v1, v2 = np.percentile(arr, (1, 99))
                if v2 <= v1:
                    v1, v2 = float(arr.min()), float(arr.max() or 1)
                out = np.clip((arr - v1) * 255.0 / (v2 - v1), 0, 255).astype(np.uint8)
            elif np.issubdtype(arr.dtype, np.integer):
                v1, v2 = np.percentile(arr, (1, 99))
                if v2 <= v1:
                    v1, v2 = float(arr.min()), float(arr.max() or 1)
                out = np.clip((arr - v1) * 255.0 / (v2 - v1), 0, 255).astype(np.uint8)
            else:
                # 浮点：用分位数到 0–255
                a = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                v1, v2 = np.percentile(a, (1, 99))
                if v2 <= v1:
                    v1, v2 = float(a.min()), float(a.max() or 1)
                out = np.clip((a - v1) * 255.0 / (v2 - v1), 0, 255).astype(np.uint8)
            return out

        raw_v = to_uint8_view(self.raw_img)
        rend_v = to_uint8_view(self.render_img)
        scat_v = to_uint8_view(self.scatter_img)
        scat_v = np.flipud(scat_v)

        # 记录原尺寸，用于回贴
        H0, W0 = raw_v.shape[:2]

        # 统一到 320×320（与训练一致）
        target = (320, 320)
        t1 = cv2.resize(raw_v, target, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        t2 = cv2.resize(rend_v, target, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        t3 = cv2.resize(scat_v, target, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0


        # 拼成 [1,3,320,320]
        input_tensor = torch.from_numpy(np.stack([t1, t2, t3], axis=0)).unsqueeze(0).to(device)

        # 加载模型
        model = MultiFusionAttentionUNet().to(device)
        model_path = self.model_path_var.get()
        if not os.path.isfile(model_path):
            messagebox.showerror("Error", "Please select a valid model file (.pth)")
            return
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()

        # 推理
        with torch.no_grad():
            logits = model(input_tensor)  # (1,1,320,320)
            probs = torch.sigmoid(logits)[0, 0]  # (320,320)
            prob_np = probs.detach().cpu().numpy()

            # 阈值可调；若太小可先用 0.3 看看
            thr = 0.3
            mask_320 = (prob_np > thr).astype(np.uint8)

        # 还原到原尺寸
        mask_full = cv2.resize(mask_320, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)

        # —— 显示：把 mask 叠在“可视化 raw”上（不是生 raw），更直观 —— #
        fig = plt.Figure(figsize=(max(W0 / 200, 3), max(H0 / 200, 3)), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(raw_v, cmap='gray')
        ax.imshow(mask_full, cmap='Reds', alpha=0.4)
        # ax.set_aspect('equal')
        ax.axis('off')

        for w in self.seg_canvas_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.seg_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True)

        self.pred_mask = mask_full.astype(np.float32)

    def train_new_model(self):
        learning_rate = float(self.learning_rate_var.get())
        epochs = int(self.epoch_var.get())
        model_name = self.train_model_name_var.get()
        use_gpu = self.use_gpu_var.get()  # <—— 获取是否使用 GPU

        folder = filedialog.askdirectory(title="Select training folder")
        if folder:
            messagebox.showinfo("Training Started", f"Training model: {model_name}\nGPU: {use_gpu}")
            try:
                self.train_model(
                    train_dir=folder,
                    model_name=model_name,
                    lr=learning_rate,
                    n_epochs=epochs,
                    use_gpu=use_gpu
                )
                messagebox.showinfo("Training Complete", f"Model '{model_name}' training finished successfully!")
            except Exception as e:
                messagebox.showerror("Training Error", f"An error occurred:\n{str(e)}")

    def train_model(self,train_dir, model_name, lr, n_epochs, use_gpu=True):
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {device}")

        save_dir = os.path.join(train_dir, "trained_models", model_name)
        os.makedirs(save_dir, exist_ok=True)

        train_ds = DualInputDataset(train_dir, transform=train_transform)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=1, pin_memory=True)

        model = MultiFusionAttentionUNet().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        best_val_loss = float("inf")
        model_save_path = os.path.join(save_dir, "best_model.pth")

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= len(train_ds)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    val_loss += criterion(logits, y).item() * x.size(0)
            val_loss /= len(train_ds)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"[Epoch {epoch}] Best model saved: val_loss = {val_loss:.4f}")

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch}/{n_epochs} Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Time: {elapsed:.1f}s")

        print(f"[DONE] Final model saved to: {model_save_path}")

    def save_with_matplotlib(self):
        """用 Matplotlib 保存当前 figure：
           1) 对话框自动带默认文件名：<原TXT名>_render_image.png
           2) 若用户改名也会强制在扩展名前追加 _render_image
           3) 若已存在则询问是否覆盖
        """
        # 1) 取默认目录和默认基名（基于已加载的 txt 路径）
        src_path = self.image_bin.get('pathname', '')  # 你在 load_slimfast 里设置过
        default_dir = os.path.dirname(src_path) if src_path else os.getcwd()
        base = os.path.splitext(os.path.basename(src_path))[0] if src_path else "render"
        default_ext = ".png"
        default_name = f"{base}_render_image{default_ext}"

        # 2) 打开保存对话框，自动带上默认目录与文件名
        filepath = filedialog.asksaveasfilename(
            initialdir=default_dir,
            initialfile=default_name,
            defaultextension=default_ext,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF", "*.tif")],
            title="保存为图像文件"
        )
        if not filepath:
            return

        # 3) 无论用户怎么改名，都确保在扩展名前带有 _render_image
        root, ext = os.path.splitext(filepath)
        if not root.endswith("_render_image"):
            filepath = f"{root}_render_image{ext or default_ext}"

        # 4) 覆盖确认
        if os.path.exists(filepath):
            overwrite = messagebox.askyesno(
                "文件已存在",
                f"文件 {os.path.basename(filepath)} 已存在。\n是否覆盖？"
            )
            if not overwrite:
                return

        # 5) 保存（去掉边缘留白）
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0)
        messagebox.showinfo("保存成功", f"图像已保存到：\n{filepath}")

    def compute_frame_particle_counts(self, filename_or_frames):
        # 假设 self.image_bin['frame'] 已经是 0-based (见我们之前的约定)
        frames = self.image_bin['frame']
        if frames.size == 0:
            return np.array([], dtype=int), 0
        maxf = int(np.max(frames))
        nframes = maxf + 1
        counts = np.zeros(nframes, dtype=int)
        for f in frames:
            counts[int(f)] += 1
        return counts, int(counts.sum())

    def color_map_editor(self):
        """颜色映射编辑器的回调函数，利用 colorchooser 弹出颜色选择对话框"""
        print("Opening color map editor...")
        # 弹出颜色选择对话框
        color = colorchooser.askcolor(title="请选择颜色映射")
        # 如果用户选择了颜色，color[1] 返回 16 进制字符串，否则为 None
        if color[1]:
            print("Selected color:", color[1])
            # 在这里可以根据需要对颜色进行处理，例如更新某个控件或内部状态
            # 例如，设置主窗口背景色：这里matlab函数并没有指明在那里做改变，后期考虑一下删除
            self.img_window.config(bg=color[1])
        else:
            print("No color selected.")
        print("Opening color map editor...")


    def add_stack(self, h_list):
        """Add a new image stack to the list."""
        stack_names = filedialog.askopenfilenames(
            title='Select Imagestacks',
            filetypes=[("TIF files", "*.tif")],
            initialdir=self.image_bin['search_path']
        )
        if stack_names:
            self.image_bin['search_path'] = stack_names[0]  # Update search path with the first selected file
            current_content = list(h_list.get(0, tk.END))  # Get current content as a list
            new_content = [name for name in stack_names]  # Ensure new content is a list
            h_list.delete(0, tk.END)  # Clear existing
            h_list.insert(tk.END, *(current_content + new_content))  # Insert new items

    def build_superstack(self,h_list):
        """Build the superstack from the selected images."""
        content = h_list.get(0, tk.END)
        if not content:
            messagebox.showwarning("Warning", "No files selected.")
            return

        # Initialize variables
        pathname = []
        filename = []
        for idx in range(len(content)):
            pathname_, filename_ = content[idx].rsplit('/', 1)
            pathname.append(pathname_)
            filename.append(filename_)

        self.image_bin['pathname'] = pathname
        self.image_bin['filename'] = filename
        self.image_bin['is_loaded'] = 0
        self.image_bin['is_superstack'] = 1
        self.image_bin['frame'] = 1
        self.image_bin['stack'] = 1
        self.image_bin['stack_size'] = []  # To be filled later

        # Set additional parameters related to localization, rendering, filtering, and optics
        self.image_bin['loc_start'] = 1
        self.image_bin['loc_end'] = float('inf')
        self.image_bin['error_rate'] = -6
        self.image_bin['w2d'] = 9
        self.image_bin['dfltn_loops'] = 0
        self.image_bin['min_int'] = 0
        self.image_bin['loc_parallel'] = 0
        self.image_bin['n_cores'] = 1
        self.image_bin['spatial_correction'] = 0

        # Load the first image to get dimensions
        I = Image.open(content[0])
        self.image_bin['width'], self.image_bin['height'] = I.size
        self.image_bin['roi'] = [0, 0, self.image_bin['width'], self.image_bin['height']]
        self.image_bin['intensity_range'] = []

        # Close any open file dialog
        # (In Tkinter, we don't manage dialogs like MATLAB, so this is a placeholder)

        messagebox.showinfo("Success", "Superstack built successfully.")

    def load_tracking_txt(self):
        """从 TXT 文件加载跟踪数据，并生成轨迹列表。"""
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not filename:
            return  # 用户取消选择

        self.image_bin['pathname'] = filename
        self.image_bin['filename'] = os.path.basename(filename)
        try:
            # 假设 TXT 文件中数据以空格或制表符分隔
            tracking_data = np.loadtxt(filename, delimiter='\t')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            return

        # tracking_data 的列含义：
        # 列0: x 坐标, 列1: y 坐标, 列2: 帧号, 列3: 轨迹编号（全局编号，不重置）
        # 使用字典根据第四列（轨迹编号）进行分组
        tracks_dict = defaultdict(list)
        for row in tracking_data:
            track_id = int(row[3])
            tracks_dict[track_id].append(row)

        # 将每个轨迹列表转换为 numpy 数组，并过滤掉长度小于 5 的轨迹
        tracks = [np.array(tracks_dict[tid]) for tid in sorted(tracks_dict.keys()) if len(tracks_dict[tid]) >= 5]

        # 更新图像字典
        self.image_bin['tracks'] = tracks
        self.image_bin['n_tracks'] = len(tracks)
        nTracks = len(self.image_bin['tracks'])
        r_end = tracking_data[-1, 2]
        self.image_bin['r_end'] = r_end

        cmap = plt.get_cmap('viridis')
        colors = [cmap(i / (nTracks - 1))[:3] for i in range(nTracks)]
        self.image_bin['trackColor'] = colors

        # 统计每个轨迹的长度
        track_lengths = [len(track) for track in tracks]

        # 打印轨迹统计信息
        print(f"Number of tracks: {self.image_bin['n_tracks']}")
        print(f"Track lengths: {track_lengths}")

        # 计算轨迹中点数的最小值、最大值和平均值
        min_track_length = min(track_lengths)
        max_track_length = max(track_lengths)
        average_track_length = np.mean(track_lengths)

        print(f"Minimum track length: {min_track_length}")
        print(f"Maximum track length: {max_track_length}")
        print(f"Average track length: {average_track_length:.2f}")

        # 根据帧号（第三列）计算每一帧的检测点数量
        frames = tracking_data[:, 2].astype(int)
        max_frame = frames.max()
        par_per_frame = np.zeros(max_frame, dtype=int)
        for f in frames:
            if f > 0:  # 跳过帧号为0的情况
                par_per_frame[f - 1] += 1
        self.image_bin['ctrsN'] = par_per_frame

        # 设置其他跟踪参数
        self.image_bin['is_track'] = 1
        self.image_bin['isLoaded'] = 1
        self.image_bin['isSuperstack'] = 0
        self.image_bin['frame'] = 1
        self.image_bin['viewMode'] = 'monoView'

        # 假设图像尺寸固定为 320x320，ROI 同样固定
        self.image_bin['width'] = 320
        self.image_bin['height'] = 320
        self.image_bin['roi'] = [0, 0, 320, 320]
        I = np.zeros((self.image_bin['height'], self.image_bin['width']))
        self.image_bin['intensity_range'] = [0, 255]

        # 显示跟踪数据
        self.display_tracking(I)

        # 生成 AVI 视频
        # output_filename = os.path.splitext(self.image_bin['filename'])[0] + '_tracking.avi'
        self.create_avi_from_tracking()

        messagebox.showinfo("Success", f"Loaded {self.image_bin['filename']} tracking data successfully.")

    def display_tracking(self, image):
        img_h = self.image_bin['height']
        img_w = self.image_bin['width']
        scale = 2

        # 1）准备放大后的底图
        pil = Image.fromarray(image.astype('uint8')).convert('RGB')
        big_pil = pil.resize((img_w * scale, img_h * scale), Image.NEAREST)
        bg = np.array(big_pil)

        # 2）新建顶层窗口
        win = tk.Toplevel(self.master)
        win.title("Interactive Tracking View")
        win.geometry("800x800")

        # 3）创建 Matplotlib Figure & Axes
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.set_position([0, 0, 1, 1])
        ax.imshow(bg, origin='upper', aspect='auto')
        ax.axis('off')

        # 4）在同一个 Axes 上叠加轨迹
        tracks = self.image_bin['tracks']
        colors = self.image_bin['trackColor']
        tw = self.image_bin.get('trackWidth', 2)
        min_t = self.image_bin.get('minTracksize', 1)

        for tid, tr in enumerate(tracks):
            if tr.shape[0] < min_t:
                continue
            xs = tr[:, 0] * scale
            ys = tr[:, 1] * scale
            ax.plot(xs, ys,
                    color=colors[tid],
                    linewidth=tw,
                    solid_capstyle='round')

        # 5）将 Figure 嵌入到 Tkinter
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()

        # 6）创建 Matplotlib 自带的导航工具栏
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()

        # —— 下面开始用 grid 布局：window 分三行 —— #
        # 行 0：工具栏（不拉伸）
        # 行 1：Canvas（可拉伸）
        # 行 2：Save 按钮（不拉伸）
        win.grid_rowconfigure(0, weight=0)  # 工具栏行高度固定
        win.grid_rowconfigure(1, weight=1)  # Canvas 行可拉伸
        win.grid_rowconfigure(2, weight=0)  # 按钮行高度固定
        win.grid_columnconfigure(0, weight=1)

        # 7）把工具栏放在 row=0, col=0
        toolbar.grid(row=0, column=0, sticky='ew')

        # 8）把 Canvas 放在 row=1, col=0，让它填满整行
        canvas_widget.grid(row=1, column=0, sticky='nsew')

        # 9）添加“Save”按钮到 row=2
        def save_image():
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("TIFF", "*.tif"), ("JPEG", "*.jpg")],
                title="Save image"
            )
            if not path:
                return

            # 尝试直接保存 big_pil（128位图像）
            try:
                fig.savefig(path, dpi=300, bbox_inches='tight')
            except Exception:
                # 如果遇到错误，再次用 Matplotlib 重新绘制并保存
                fig2, ax2 = plt.subplots(figsize=(5, 5), dpi=100)
                ax2.imshow(big_pil, origin='upper')
                ax2.axis('off')

                for tid, tr in enumerate(tracks):
                    if tr.shape[0] < min_t:
                        continue
                    xs = tr[:, 0] * scale
                    ys = tr[:, 1] * scale
                    ax2.plot(xs, ys,
                             color=colors[tid],
                             linewidth=tw,
                             solid_capstyle='round')

                fig2.savefig(path, dpi=300, bbox_inches='tight')
                plt.close(fig2)

            messagebox.showinfo("Saved", f"Image saved to:\n{path}")

        btn_save = tk.Button(win, text="Save", command=save_image)
        btn_save.grid(row=2, column=0,  pady=5)

        # 显示窗口
        win.deiconify()

    def create_avi_from_tracking(self):
        # ————————— 1) 选择保存路径 —————————
        movie_path = filedialog.asksaveasfilename(
            defaultextension=".avi", title="保存电影为"
        )
        if not movie_path:
            return

        # ————————— 2) 读取参数 —————————
        fps = self.image_bin.get('fps', 10)
        r_start = int(self.image_bin.get('r_start', 1))
        r_end = int(self.image_bin['r_end'])

        # 限制演示帧数不超过 2 分钟
        max_demo = fps * 120
        total = r_end - r_start + 1
        if total > max_demo:
            frame_indices = np.linspace(r_start, r_end, max_demo, dtype=int)
        else:
            frame_indices = np.arange(r_start, r_end + 1)

        # ————————— 3) 打开视频写入器 —————————
        writer = imageio.get_writer(
            movie_path, fps=fps, codec='libx264',
            pixelformat='yuv420p', mode='I'
        )

        # ————————— 4) 缩放系数 & 基本参数 —————————
        scale = 2
        H = int(self.image_bin['height'])
        W = int(self.image_bin['width'])
        bigW, bigH = W * scale, H * scale

        tracks = self.image_bin['tracks']
        colors = self.image_bin['trackColor']  # n_tracks×3 的 0–1
        tw = int(self.image_bin.get('trackWidth', 2))
        min_size = int(self.image_bin.get('minTracksize', 1))
        max_size = float(self.image_bin.get('maxTracksize', float('inf')))

        # ————————— 5) 逐帧渲染 —————————
        for frame_idx in frame_indices:
            # a) 新建一张放大后的黑底
            big_img = Image.new('RGB', (bigW, bigH), (0, 0, 0))
            draw = ImageDraw.Draw(big_img)

            # b) 对每条轨迹：
            for tid, tr in enumerate(tracks):
                # 1) 先筛轨迹长度
                L = tr.shape[0]
                if L < min_size or L > max_size:
                    continue

                # 2) 如果当前帧已经超过这条轨迹的最后一帧，就不画它
                track_end = int(tr[:, 2].max())
                if frame_idx > track_end:
                    continue

                # 3) 收集所有在当前帧之前出现过的点
                pts = [
                    (int(pt[0] * scale), int(pt[1] * scale))
                    for pt in tr
                    if int(pt[2]) <= frame_idx
                ]
                if len(pts) < 2:
                    continue

                # 4) 转颜色到 0–255
                c = colors[tid]
                col = (int(255 * c[0]), int(255 * c[1]), int(255 * c[2]))

                # 5) 画线
                draw.line(pts, fill=col, width=tw * scale, joint="curve")

            # c) 写入视频
            writer.append_data(np.array(big_img))

        writer.close()
        messagebox.showinfo("Success", f"跟踪动画已保存到：\n{movie_path}")

    def convert_to_image(self, data):
        """
        将 NumPy 数组转换为 Tkinter 可显示的图像。
        这里先将数据归一化到 0-255 后转换为 uint8 类型的 PIL Image，再转换为 PhotoImage 对象。
        """
        # 防止所有像素值相同导致除 0
        if np.max(data) == np.min(data):
            normalized_data = np.zeros_like(data)
        else:
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
        img = Image.fromarray(normalized_data.astype('uint8'))
        return ImageTk.PhotoImage(img)

    ####从这边开始后面的函数都没有回调函数，需要后期根据需要勾连上回调函数

    def log_in_image(self):
        """Log in the current image and update the menu."""
        if self.image_bin['is_loaded']:
            # Create or update menu items based on the loaded image
            filename = self.image_bin['filename']
            if self.image_bin['is_track']:
                # Update track channel menu
                pass  # Update tracks logic here
            else:
                # Update image channel menu
                pass  # Update image logic here

    def log_out_image(self):
        """Log out the current image and remove it from the menu."""
        if self.image_bin['is_loaded']:
            # Remove the image from the menu logic
            pass  # Implement the actual logic here

    def show_raw_stack(self, event=None):
        """安全访问播放状态"""
        # 动态初始化 playing 属性
        if not hasattr(self, 'playing'):
            self.playing = False

        # 切换播放状态
        self.playing = not self.playing

        def play_sequence():
            # 确保 playing 属性存在
            if not hasattr(self, 'playing'):
                self.playing = False

            if self.playing:
                try:
                    end_of_stack = self.next_frame()
                    if not end_of_stack:
                        self.master.after(50, play_sequence)  # 500ms间隔
                    else:
                        self.playing = False
                except Exception as e:
                    self.playing = False
                    messagebox.showerror("错误", f"播放失败: {str(e)}")

            # 更新按钮状态
            self._update_button_state()

        # 启动播放循环
        if self.playing:
            play_sequence()
        else:
            self._update_button_state()

    def previous_frame(self):
        """Load the previous frame from the stack."""
        frame = self.image_bin['frame'] - 1
        pathname = self.image_bin['pathname']
        filename = self.image_bin['filename']

        try:
            if self.image_bin['is_superstack']:
                if frame > 0:
                    # 加载当前堆栈的前一帧
                    image = self.load_image(pathname[self.image_bin['stack'] - 1],
                                            filename[self.image_bin['stack'] - 1], frame)
                else:
                    if self.image_bin['stack'] > 1:
                        # 切换到上一个堆栈的最后一帧
                        self.image_bin['stack'] -= 1
                        frame = self.image_bin['stack_size'][self.image_bin['stack'] - 1]
                        image = self.load_image(pathname[self.image_bin['stack'] - 1],
                                                filename[self.image_bin['stack'] - 1], frame)
            else:
                if frame > 0:
                    # 加载普通堆栈的前一帧
                    image = self.load_image(pathname, filename, frame)
                else:
                    # 如果已经是第一帧，不做任何操作
                    return

            # 更新当前帧编号
            self.image_bin['frame'] = frame

            # 如果成功加载图像，更新显示
            if image:
                self.update_frame(image, frame)

        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")

    def update_frame(self, image, frame):
        """更新当前窗口中的图像和帧信息"""
        if not hasattr(self, 'img_window') or not self.img_window.winfo_exists():
            # 如果窗口不存在，重新初始化
            self.create_image_window()

        # 将图像转换为 Tkinter 兼容格式
        img_tk = ImageTk.PhotoImage(image)

        # 更新画布中的图像
        self.roi_canvas.delete("all")  # 清空画布
        self.roi_canvas.create_image(0, 0, anchor='nw', image=img_tk)
        self.roi_canvas.image = img_tk  # 保持引用，避免垃圾回收

        # 更新窗口标题
        self.img_window.title(f"Frame {frame} of TIFF")  # 更新标题

    # 修改next_frame方法（优化显示逻辑）
    def next_frame(self):
        """加载下一帧并更新显示"""
        try:
            current_frame = self.image_bin.get('frame', 0)
            new_frame = current_frame + 1
            total_frames = self.image_bin.get('total_frames', 0)
            # 加载新帧图像
            image = self.load_frame(new_frame)  # 假设load_frame方法正确加载图像
            if image:
                # 更新帧编号
                self.image_bin['frame'] = new_frame
                # 更新原始图像，确保后续检测使用的是当前帧数据
                self.image_bin['raw_image'] = image
                # 显示图像
                self.display_image(image, new_frame)
            # 假设total_frames已通过其他方法设置（如在load_imagestack中）
            return current_frame >= total_frames - 1  # 0-based索引判断
        except Exception as e:
            messagebox.showerror("Error", f"Error loading frame: {str(e)}")
            raise

    def _update_button_state(self):
        """安全的按钮状态更新"""
        # 动态检查button_frame是否存在
        if hasattr(self, 'button_frame') and self.button_frame.winfo_exists():
            # 查找播放按钮（通过文本内容）
            for widget in self.button_frame.winfo_children():
                if isinstance(widget, tk.Button) and "Play Movie" in widget.cget('text'):
                    new_text = "Pause" if getattr(self, 'playing', False) else "Play Movie"
                    widget.config(text=new_text)
                    break

    def load_frame(self, frame_number):
        """封装帧加载逻辑（完整版）"""
        try:
            if self.image_bin.get('is_superstack', False):
                # ==================================================================
                # 超栈处理逻辑（多文件堆栈）
                # ==================================================================
                current_stack = self.image_bin.get('stack', 1) - 1  # 当前堆栈索引（0-based）
                total_stacks = len(self.image_bin.get('filename', []))

                # 验证堆栈索引有效性
                if current_stack >= total_stacks:
                    raise IndexError("Current stack index out of range")

                # 获取当前堆栈元数据
                stack_path = self.image_bin['pathname'][current_stack]
                stack_file = self.image_bin['filename'][current_stack]
                stack_size = self.image_bin['stack_size'][current_stack]

                # 处理元组/列表包装
                if isinstance(stack_path, (tuple, list)):
                    stack_path = stack_path[0]
                if isinstance(stack_file, (tuple, list)):
                    stack_file = stack_file[0]

                # 检查是否超出当前堆栈范围
                if frame_number > stack_size:
                    # 切换到下一个堆栈
                    if current_stack + 1 >= total_stacks:
                        raise IndexError("End of superstack reached")

                    # 更新堆栈索引
                    self.image_bin['stack'] = current_stack + 2  # 转换为1-based
                    new_stack = current_stack + 1

                    # 获取新堆栈元数据
                    new_stack_path = self.image_bin['pathname'][new_stack]
                    new_stack_file = self.image_bin['filename'][new_stack]
                    new_stack_size = self.image_bin['stack_size'][new_stack]

                    # 处理新堆栈的帧重置
                    if frame_number - stack_size > new_stack_size:
                        raise IndexError("Requested frame exceeds new stack size")

                    # 加载新堆栈首帧
                    return self.load_single_stack(
                        new_stack_path,
                        new_stack_file,
                        1  # 新堆栈从第一帧开始
                    )
                else:
                    # 加载当前堆栈指定帧
                    return self.load_single_stack(stack_path, stack_file, frame_number)

            else:
                # ==================================================================
                # 普通堆栈处理（单文件）
                # ==================================================================
                pathname = self.image_bin['pathname']
                filename = self.image_bin['filename']

                # 处理可能的元组/列表包装
                if isinstance(pathname, (tuple, list)):
                    pathname = pathname[0]
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]


                # 确保 pathname 是目录路径
                if os.path.isfile(pathname):
                    # 如果 pathname 已经是文件路径，直接使用
                    return self.load_single_stack(os.path.dirname(pathname), os.path.basename(pathname), frame_number)
                else:
                    # 否则，正常拼接路径
                    return self.load_single_stack(pathname, filename, frame_number)

        except IndexError as e:
            messagebox.showwarning("Stack Boundary", f"{str(e)}")
            return None
        except Exception as e:
            messagebox.showerror("Loading Error", f"Failed to load frame: {str(e)}")
            return None

    def load_single_stack(self, pathname, filename, frame_number):
        """加载单个堆栈的指定帧"""
        try:
            # 确保 pathname 是目录路径
            if os.path.isfile(pathname):
                # 如果 pathname 已经是文件路径，直接使用
                image_path = pathname
            else:
                # 否则，正常拼接路径
                image_path = os.path.join(pathname, filename)


            # 加载图像
            image = Image.open(image_path)
            image.seek(frame_number - 1)  # 移动到指定帧
            return image

        except Exception as e:
            raise Exception(f"Error loading {filename} frame {frame_number}: {str(e)}")

    # 修改load_image方法（移除display_image调用）
    def load_image(self, pathname, filename, frame):
        """加载图像但不直接显示"""
        try:
            # 确保 pathname 是目录路径
            if os.path.isfile(pathname):
                # 如果 pathname 已经是文件路径，直接使用
                image_path = pathname
            else:
                # 否则，正常拼接路径
                image_path = os.path.join(pathname, filename)


            # 加载图像
            image = Image.open(image_path)
            image.seek(frame - 1)  # 移动到指定帧
            self.image_bin['image'] = image
            return image  # 返回图像对象而不是直接显示

        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
            return None

    def display_image(self, image, frame):
        """更新显示窗口而不是创建新窗口，并更新和返回 self.image_bin['h_image'] 属性"""
        # 如果窗口不存在则创建
        if self.img_window is None or not self.img_window.winfo_exists():
            self.create_image_window()

        # 更新窗口标题
        self.img_window.title(f"Frame: {frame} - {self.image_bin.get('filename', '')}")

        # ——— 1. 将 PIL Image 转为 float32 numpy 数组 ———
        arr = np.array(image, dtype=np.float32)

        # ——— 2. 动态拉伸：最暗变 0，最亮变 1，再映射到 0–255 ———
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr = (arr * 255).astype(np.uint8)

        # 转回 PIL Image
        pil_img = Image.fromarray(arr)

        # 将处理后的数组转换回 PIL Image 对象
        pil_img = Image.fromarray(arr)
        scale = 2
        new_w, new_h = pil_img.width * scale, pil_img.height * scale
        pil_img = pil_img.resize((new_w, new_h), resample=Image.LANCZOS)
        pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        # 转换为 Tkinter 可显示的图像对象
        img_tk = ImageTk.PhotoImage(pil_img)

        # 清空 Canvas 并显示图像
        self.roi_canvas.delete("all")
        self.roi_canvas.create_image(0, 0, anchor='nw', image=img_tk)
        self.current_img_tk = img_tk

        # 更新 self.image_bin['h_image'] 属性
        self.image_bin['h_image'] = img_tk
        self.current_pil_img = pil_img.copy()  # 保留一份处理后图像，用于保存

        # === 关键：让 Canvas 跟图像一样大 ===
        self.roi_canvas.config(width=new_w, height=new_h)
        self.roi_canvas.config(scrollregion=(0, 0, new_w, new_h))

        # 让布局先计算出按钮高度
        self.img_window.update_idletasks()
        btn_h = self.button_frame.winfo_height() if hasattr(self, 'button_frame') else 0

        # === 关键：设置窗口几何尺寸 = 图像高 + 按钮条高 ===
        self.img_window.geometry(f"{new_w}x{new_h + btn_h}")

        # 使用 grid 布局调整画布尺寸
        # self.roi_canvas.config(scrollregion=(0, 0, image.width, image.height))
        self.roi_canvas.config(scrollregion=(0, 0, new_w, new_h))
        self.roi_canvas.grid(row=0, column=0, sticky='nsew')
        self.img_window.grid_rowconfigure(0, weight=1)
        self.img_window.grid_columnconfigure(0, weight=1)

        # 返回更新后的图像句柄
        return self.image_bin['h_image']

    def create_image_window(self):
        """创建图像显示窗口（只执行一次）"""
        self.img_window = tk.Toplevel(self.master)
        self.img_window.title("TIFF Viewer")
        # self.img_window.geometry("640x700")  # 给按钮留点高度

        # 网格权重：第0行画布可伸展，第1行按钮行固定
        self.img_window.grid_rowconfigure(0, weight=0)
        self.img_window.grid_rowconfigure(1, weight=0)
        self.img_window.grid_columnconfigure(0, weight=1)

        self.roi_canvas = tk.Canvas(
            self.img_window,
            bg='black',  # 建议用黑底，微弱荧光更清晰
            highlightthickness=0
        )
        self.roi_canvas.grid(row=0, column=0, sticky="nsew")

        # 按钮容器（占第1行）
        self.button_frame = ttk.Frame(self.img_window)
        self.button_frame.grid(row=1, column=0, sticky="ew")
        self.button_frame.grid_columnconfigure(0, weight=1)

        # 事件绑定（一次即可）
        self.roi_canvas.bind("<Button-1>", self.on_button_press)
        self.roi_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.roi_canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # 创建控制按钮 + Save 按钮放在 button_frame 里
        self.create_control_buttons()

        # —— 重要：初始化 ROI 交互状态，避免 KeyError —— #
        self.image_bin.setdefault('drawing', False)
        self.image_bin.setdefault('roi_coords', [])  # [(x0,y0), (x1,y1)]
        self.image_bin.setdefault('roi_values', [])  # 四角 (可选)
        self.image_bin.setdefault('roi_rect', None)  # Canvas 中的矩形句柄

        # 保存显示缩放（你现在是2）
        self.display_scale = 2

    def save_displayed_image(self):
        """
        从原始 TIFF 直接读第一帧，按位深做“标准不拉伸”转换并保存为 8-bit 灰度 PNG。
        - uint16: 使用高 8 位 (>>8)
        - uint8 : 原样
        - float : 若在[0,1]则*255，否则裁剪到[0,255]
        - 其它整数: 按本类型范围线性映射到0..255（尽量接近标准转换）
        不做任何锐化/缩放，尺寸与原图一致。
        """
        # 找原始路径（优先 image_path；其次 pathname+filename）
        src_path = self.image_bin.get('image_path')
        if not src_path:
            pn, fn = self.image_bin.get('pathname', ''), self.image_bin.get('filename', '')
            if pn and fn:
                src_path = os.path.join(pn, fn)

        if not src_path or not os.path.isfile(src_path):
            messagebox.showerror("错误", "找不到原始 TIFF 路径")
            return

        base_dir = os.path.dirname(src_path)
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        save_path = os.path.join(base_dir, f"{base_name}_frame1_raw.png")

        try:
            import tifffile, numpy as np
            from PIL import Image

            # 1) 直接从文件读取第1帧，确保拿到“真原始数据”
            with tifffile.TiffFile(src_path) as tif:
                arr = tif.pages[0].asarray()

            # 2) 如果是多维（如(H,W,C)或(C,H,W)），取第0通道做灰度
            if arr.ndim == 3:
                if arr.shape[0] in (3, 4):  # (C,H,W)
                    arr = arr[0]
                elif arr.shape[-1] in (3, 4):  # (H,W,C)
                    arr = arr[..., 0]
                else:
                    arr = arr[0]  # 不明确时取第0片
            elif arr.ndim > 3:
                while arr.ndim > 2:
                    arr = arr[0]

            # 3) 位深安全的“标准”8-bit 转换（不做直方图拉伸）
            if arr.dtype == np.uint8:
                arr8 = arr
            elif arr.dtype == np.uint16:
                arr8 = (arr >> 8).astype(np.uint8)  # 取高8位
            elif arr.dtype == np.int16:
                a = arr.astype(np.int32) - np.iinfo(np.int16).min  # 移到[0,65535]
                arr8 = (np.clip(a, 0, 65535) >> 8).astype(np.uint8)
            elif np.issubdtype(arr.dtype, np.floating):
                a = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
                if a.size and a.min() >= 0.0 and a.max() <= 1.0:
                    arr8 = np.clip(a * 255.0, 0, 255).astype(np.uint8)
                else:
                    arr8 = np.clip(a, 0, 255).astype(np.uint8)
            elif np.issubdtype(arr.dtype, np.integer):
                info = np.iinfo(arr.dtype)
                a = np.clip(arr.astype(np.int64), info.min, info.max).astype(np.float32)
                # 线性映射到0..255（不是自适应拉伸，只是类型范围到8位）
                arr8 = ((a - info.min) * (255.0 / (info.max - info.min))).astype(np.uint8)
            else:
                # 兜底：当成浮点裁剪
                arr8 = np.clip(arr.astype(np.float32), 0, 255).astype(np.uint8)

            # 4) 保存（不缩放，尺寸=原图）
            Image.fromarray(arr8, mode='L').save(save_path)
            messagebox.showinfo("保存成功", f"原始第一帧图像已保存为：\n{save_path}")

        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def create_control_buttons(self):
        """创建控制按钮面板"""
        buttons = [
            ("Previous", self.previous_frame),
            ("Play Movie", self.show_raw_stack),
            ("Next", self.next_frame),
            ("ROI", self.set_roi),
            ("OPT", self.set_options),
            ("MAX PROJ", self.max_projection),
            ("RVE PROJ", self.mean_projection),
            ("COLOR MAP", self.color_map_editor),
            ("INT DIST", self.pixel_intensity_hist),
            ("LOC TEST", self.show_loc_preview),
            ("LOC ALL", self.localization),
        ]

        # 计算列数：功能按钮 + Save 按钮
        n_cols = len(buttons) + 1
        for c in range(n_cols):
            self.button_frame.grid_columnconfigure(c, weight=1)

        # 放功能按钮
        self.play_button_widget = None  # 如需后续访问按钮本身
        for idx, (text, cmd) in enumerate(buttons):
            btn = tk.Button(self.button_frame, text=text, command=cmd)
            btn.grid(row=0, column=idx, padx=2, pady=4, sticky='ew')
            if text == "Play Movie":
                self.play_button_widget = btn  # 真正的按钮控件

        # Save 按钮作为最后一列，放在同一行里
        save_btn = tk.Button(self.button_frame, text="Save", command=self.save_displayed_image)
        save_btn.grid(row=0, column=len(buttons), padx=2, pady=4, sticky='ew')

    def set_options(self):
        """选项设置窗口"""
        opt_win = tk.Toplevel()
        opt_win.title("Options")
        opt_win.geometry("450x400")

        # 创建右键菜单
        context_menu = tk.Menu(opt_win, tearoff=0)
        context_menu.add_command(label="Save Settings", command=self.save_settings)
        context_menu.add_command(label="Load Settings", command=self.load_settings)
        opt_win.bind("<Button-3>", lambda e: context_menu.tk_popup(e.x_root, e.y_root))

        # 使用Notebook管理标签页
        notebook = ttk.Notebook(opt_win)
        notebook.pack(expand=True, fill='both')

        # 创建各选项页
        self.create_localization_tab(notebook)
        self.create_rendering_tab(notebook)
        self.create_filters_tab(notebook)
        self.create_scalebar_tab(notebook)
        self.create_acquisition_tab(notebook)
        self.create_tracking_tab(notebook)
        self.create_trajectories_tab(notebook)

        # 添加其他标签页...

    def create_trajectories_tab(self, notebook):
        """创建独立的轨迹显示参数标签页"""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Trajectories")

        # 滚动容器
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 主容器
        main_frame = ttk.Frame(scroll_frame)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # 动态生成通道设置
        for ch in range(1, self.image_bin.get('nTrackCh', 1) + 1):
            self.create_channel_trajectory_section(main_frame, ch)

        return tab

    def create_channel_trajectory_section(self, parent, channel):
        """创建单个通道的轨迹显示参数区块"""
        ch_frame = ttk.LabelFrame(
            parent,
            text=f"Channel {channel} Display Settings",
            padding=(10, 5)
        )
        ch_frame.grid(row=channel - 1, column=0, sticky="ew", pady=10)

        # 初始化显示参数变量
        self.init_channel_trajectory_params(channel)

        # ========== 显示参数 ==========
        display_frame = ttk.LabelFrame(ch_frame, text="Visualization Parameters")
        display_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

        row = 0
        # 帧范围
        ttk.Label(display_frame, text="Frame Range:").grid(row=row, column=0, padx=5, pady=2, sticky='e')
        ttk.Entry(display_frame, textvariable=self.image_bin[f'ch{channel}_rStart'], width=6).grid(row=row, column=1,
                                                                                                   sticky='w')
        ttk.Entry(display_frame, textvariable=self.image_bin[f'ch{channel}_rEnd'], width=6).grid(row=row, column=2,
                                                                                                 sticky='w')
        row += 1

        # 颜色编码
        ttk.Label(display_frame, text="Color Mode:").grid(row=row, column=0, padx=5, pady=2, sticky='e')
        color_mode = ttk.Combobox(
            display_frame,
            textvariable=self.image_bin[f'ch{channel}_trackColorMode'],
            values=['ensemble', 'individual'],
            state='readonly',
            width=10
        )
        color_mode.grid(row=row, column=1, padx=5, sticky='w')

        # 颜色选择按钮
        color_btn = tk.Canvas(display_frame, width=20, height=20,
                              bg=self.image_bin[f'ch{channel}_trackColor'].get())
        color_btn.grid(row=row, column=2, padx=5, sticky='w')
        color_btn.bind("<Button-1>",
                       lambda e, ch=channel: self.change_track_color(ch, color_btn))
        row += 1

        # 轨迹厚度
        ttk.Label(display_frame, text="Line Thickness:").grid(row=row, column=0, padx=5, pady=2, sticky='e')
        ttk.Entry(display_frame, textvariable=self.image_bin[f'ch{channel}_trackWidth'],
                  width=8).grid(row=row, column=1, sticky='w')
        row += 1

        # 轨迹长度过滤
        ttk.Checkbutton(
            display_frame,
            text="Track Length Filter:",
            variable=self.image_bin[f'ch{channel}_isTracksizeThresh']
        ).grid(row=row, column=0, padx=5, pady=2, sticky='e')
        ttk.Entry(display_frame, textvariable=self.image_bin[f'ch{channel}_minTracksize'],
                  width=5).grid(row=row, column=1, sticky='w')
        ttk.Label(display_frame, text="to").grid(row=row, column=1, padx=45, sticky='e')
        ttk.Entry(display_frame, textvariable=self.image_bin[f'ch{channel}_maxTracksize'],
                  width=5).grid(row=row, column=2, sticky='w')
        row += 1

        # 显示窗口设置
        ttk.Label(display_frame, text="Window Size (frames):").grid(row=row, column=0, padx=5, pady=2, sticky='e')
        ttk.Entry(display_frame, textvariable=self.image_bin[f'ch{channel}_rW'],
                  width=8).grid(row=row, column=1, sticky='w')
        row += 1

        # 高级显示选项
        ttk.Checkbutton(
            display_frame,
            text="Accumulative Display",
            variable=self.image_bin[f'ch{channel}_isCumsum']
        ).grid(row=row, column=0, columnspan=2, padx=5, pady=2, sticky='w')
        row += 1

        ttk.Checkbutton(
            display_frame,
            text="Always Visible",
            variable=self.image_bin[f'ch{channel}_trackVisibility']
        ).grid(row=row, column=0, columnspan=2, padx=5, pady=2, sticky='w')

        # 配置布局权重
        display_frame.columnconfigure(0, weight=1)
        ch_frame.columnconfigure(0, weight=1)

    def init_channel_trajectory_params(self, channel):
        """初始化轨迹显示参数"""
        params = {
            f'ch{channel}_rStart': tk.IntVar(value=1),
            f'ch{channel}_rEnd': tk.IntVar(value='inf'),
            f'ch{channel}_trackColorMode': tk.StringVar(value='ensemble'),
            f'ch{channel}_trackColor': tk.StringVar(value='#FF0000'),
            f'ch{channel}_trackWidth': tk.DoubleVar(value=1.0),
            f'ch{channel}_isTracksizeThresh': tk.BooleanVar(value=False),
            f'ch{channel}_minTracksize': tk.IntVar(value=0),
            f'ch{channel}_maxTracksize': tk.IntVar(value=100),
            f'ch{channel}_rW': tk.IntVar(value=5),
            f'ch{channel}_isCumsum': tk.BooleanVar(value=False),
            f'ch{channel}_trackVisibility': tk.BooleanVar(value=True),
        }
        self.image_bin.update(params)

    def change_track_color(self, channel, color_canvas):
        """修改轨迹颜色"""
        color_code = colorchooser.askcolor(title=f"Channel {channel} Track Color")
        if color_code[1]:
            self.image_bin[f'ch{channel}_trackColor'].set(color_code[1])
            color_canvas.config(bg=color_code[1])

    def create_tracking_tab(self, notebook):
        """创建支持多通道的跟踪参数标签页"""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Tracking")

        # 滚动容器
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 主容器
        main_frame = ttk.Frame(scroll_frame)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # 动态生成通道设置
        for ch in range(1, self.image_bin.get('nImCh', 1) + 1):
            self.create_channel_tracking_section(main_frame, ch)

        return tab

    def create_channel_tracking_section(self, parent, channel):
        """创建单个通道的跟踪参数区块"""
        ch_frame = ttk.LabelFrame(
            parent,
            text=f"Channel {channel} Tracking Settings",
            padding=(10, 5)
        )
        ch_frame.grid(row=channel - 1, column=0, sticky="ew", pady=10)

        # 初始化通道参数
        self.init_channel_tracking_params(channel)

        # ========== 运动参数 ==========
        motion_frame = ttk.LabelFrame(ch_frame, text="Motion Parameters")
        motion_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

        row = 0
        # 最大扩散系数
        ttk.Label(motion_frame, text="Max Diffusion (μm²/s):", wraplength=200).grid(
            row=row, column=0, padx=5, pady=2, sticky='e')
        ttk.Entry(motion_frame, textvariable=self.image_bin[f'ch{channel}_max_diffusion_coeff'],
                  width=8).grid(row=row, column=1, sticky='w')
        row += 1

        # 搜索扩展因子
        ttk.Label(motion_frame, text="Search Expansion Factor:", wraplength=200).grid(
            row=row, column=0, padx=5, pady=2, sticky='e')
        ttk.Entry(motion_frame, textvariable=self.image_bin[f'ch{channel}_search_exp_factor'],
                  width=8).grid(row=row, column=1, sticky='w')
        row += 1

        # ========== 关联参数 ==========
        association_frame = ttk.LabelFrame(ch_frame, text="Association Parameters")
        association_frame.grid(row=1, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

        row = 0
        # 统计窗口
        ttk.Label(association_frame, text="Statistics Window (frames):").grid(
            row=row, column=0, padx=5, pady=2, sticky='e')
        ttk.Entry(association_frame, textvariable=self.image_bin[f'ch{channel}_stat_win_frames'],
                  width=8).grid(row=row, column=1, sticky='w')
        row += 1

        # 最大竞争者
        ttk.Label(association_frame, text="Max Competitors:").grid(
            row=row, column=0, padx=5, pady=2, sticky='e')
        ttk.Entry(association_frame, textvariable=self.image_bin[f'ch{channel}_max_competitors'],
                  width=8).grid(row=row, column=1, sticky='w')
        row += 1

        # 最大离线时间
        ttk.Label(association_frame, text="Max OFF-Time (frames):").grid(
            row=row, column=0, padx=5, pady=2, sticky='e')
        ttk.Entry(association_frame, textvariable=self.image_bin[f'ch{channel}_max_off_time'],
                  width=8).grid(row=row, column=1, sticky='w')
        row += 1

        # ========== 权重参数 ==========
        weight_frame = ttk.LabelFrame(ch_frame, text="Weighting Parameters")
        weight_frame.grid(row=2, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

        row = 0
        # 强度波动权重
        ttk.Label(weight_frame, text="Intensity Fluctuation Weight:").grid(
            row=row, column=0, padx=5, pady=2, sticky='e')
        ttk.Entry(weight_frame, textvariable=self.image_bin[f'ch{channel}_int_fluc_weight'],
                  width=8).grid(row=row, column=1, sticky='w')
        row += 1

        # 扩散权重比例
        ttk.Label(weight_frame, text="Diffusion Weight Ratio:", wraplength=200).grid(
            row=row, column=0, padx=5, pady=2, sticky='e')
        ttk.Entry(weight_frame, textvariable=self.image_bin[f'ch{channel}_diffusion_weight'],
                  width=8).grid(row=row, column=1, sticky='w')

        # 配置列权重
        ch_frame.columnconfigure(0, weight=1)

    def init_channel_tracking_params(self, channel):
        """初始化通道跟踪参数"""
        params = {
            f'ch{channel}_max_diffusion_coeff': tk.DoubleVar(value=3),
            f'ch{channel}_search_exp_factor': tk.DoubleVar(value=1.5),
            f'ch{channel}_stat_win_frames': tk.IntVar(value=5),
            f'ch{channel}_max_competitors': tk.IntVar(value=3),
            f'ch{channel}_max_off_time': tk.IntVar(value=3),
            f'ch{channel}_int_fluc_weight': tk.DoubleVar(value=0.5),
            f'ch{channel}_diffusion_weight': tk.DoubleVar(value=0.7)
        }
        self.image_bin.update(params)

    def create_localization_tab(self, notebook):
        """创建支持多通道的定位设置标签页"""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Localization")

        # 创建滚动容器
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 主容器布局参数
        main_padx = 15
        main_pady = 10
        row = 0

        # 动态生成通道
        for ch in range(1, 2):
            # 每个通道的容器框架
            ch_frame = ttk.LabelFrame(
                scrollable_frame,
                text=f"Channel {ch} Settings",
                padding=(10, 5))
            ch_frame.grid(row=row, column=0, padx=main_padx, pady=main_pady, sticky="ew")
            row += 1

            # ========== 通道参数存储 ==========
            self.image_bin[f'ch{ch}'] = {
                'loc_start': tk.StringVar(value="1"),
                'loc_end': tk.StringVar(value="inf"),
                'error_rate': tk.DoubleVar(value=-6),
                'w2d': tk.IntVar(value=9),
                'dfltn_loops': tk.IntVar(value=0),
                'min_int': tk.IntVar(value=0),
                'loc_parallel': tk.BooleanVar(value=True),
                'n_cores': tk.StringVar(value='max'),
                'spatial_correction': tk.BooleanVar(value=False),
                'r_live': tk.BooleanVar(value=True),
                'max_optim_iter': tk.IntVar(value=50),
                'term_tol': tk.DoubleVar(value=-2),
                'is_radius_tol': tk.BooleanVar(value=True),
                'radius_tol': tk.DoubleVar(value=50),
                'pos_tol': tk.DoubleVar(value=1.5)
            }

            # ========== 通道控件布局 ==========
            # 第一行：帧范围
            ttk.Label(ch_frame, text="Framerange:").grid(
                row=0, column=0, padx=5, pady=5, sticky='e')
            ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{ch}']['loc_start'],
                      width=8).grid(row=0, column=1, sticky='w')
            ttk.Label(ch_frame, text="-").grid(row=0, column=2)
            ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{ch}']['loc_end'],
                      width=8).grid(row=0, column=3, sticky='w')

            # 第二行：错误率
            ttk.Label(ch_frame, text="Error Rate [10^]:").grid(
                row=1, column=0, padx=5, pady=5, sticky='e')
            ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{ch}']['error_rate'],
                      width=12).grid(row=1, column=1, columnspan=3, sticky='w')

            # 第三行：检测窗口
            ttk.Label(ch_frame, text="Detection Box [px]:").grid(
                row=2, column=0, padx=5, pady=5, sticky='e')
            ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{ch}']['w2d'],
                      width=12).grid(row=2, column=1, columnspan=3, sticky='w')

            # 第四行：迭代次数
            ttk.Label(ch_frame, text="Deflation Loops:").grid(
                row=3, column=0, padx=5, pady=5, sticky='e')
            ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{ch}']['dfltn_loops'],
                      width=12).grid(row=3, column=1, columnspan=3, sticky='w')

            # 第五行：强度阈值
            ttk.Label(ch_frame, text="Intensity Thresh [cnts]:").grid(
                row=4, column=0, padx=5, pady=5, sticky='e')
            ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{ch}']['min_int'],
                      width=12).grid(row=4, column=1, columnspan=3, sticky='w')

            # 第六行：并行处理
            ttk.Label(ch_frame, text="Parallel Processing:").grid(
                row=5, column=0, padx=5, pady=5, sticky='e')
            ttk.Checkbutton(ch_frame, variable=self.image_bin[f'ch{ch}']['loc_parallel']
                            ).grid(row=5, column=1, sticky='w')
            ttk.Combobox(ch_frame,
                         textvariable=self.image_bin[f'ch{ch}']['n_cores'],
                         values=('max', '2', '3', '4'),
                         width=4).grid(row=5, column=2, sticky='w')

            # 第七行：空间校正
            ttk.Label(ch_frame, text="Spatial Correction:").grid(
                row=6, column=0, padx=5, pady=5, sticky='e')
            ttk.Checkbutton(ch_frame,
                            variable=self.image_bin[f'ch{ch}']['spatial_correction'],
                            command=lambda c=ch: self.set_spatial_corr_path(c)
                            ).grid(row=6, column=1, sticky='w')

            # 第八行：实时定位
            ttk.Label(ch_frame, text="Live Localization:").grid(
                row=7, column=0, padx=5, pady=5, sticky='e')
            ttk.Checkbutton(ch_frame,
                            variable=self.image_bin[f'ch{ch}']['r_live']
                            ).grid(row=7, column=1, sticky='w')

            # 第九行：优化参数
            ttk.Label(ch_frame, text="Max Iterations:").grid(
                row=8, column=0, padx=5, pady=5, sticky='e')
            ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{ch}']['max_optim_iter'],
                      width=12).grid(row=8, column=1, columnspan=3, sticky='w')

            # 第十行：终止容差
            ttk.Label(ch_frame, text="Termination Tol [1e-]:").grid(
                row=9, column=0, padx=5, pady=5, sticky='e')
            ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{ch}']['term_tol'],
                      width=12).grid(row=9, column=1, columnspan=3, sticky='w')

            # 第十一行：半径容差
            ttk.Label(ch_frame, text="Radius Tolerance:").grid(
                row=10, column=0, padx=5, pady=5, sticky='e')
            ttk.Checkbutton(ch_frame,
                            variable=self.image_bin[f'ch{ch}']['is_radius_tol']
                            ).grid(row=10, column=1, sticky='w')
            ttk.Entry(ch_frame,
                      textvariable=self.image_bin[f'ch{ch}']['radius_tol'],
                      width=6).grid(row=10, column=2, sticky='w')
            ttk.Label(ch_frame, text="%").grid(row=10, column=3, sticky='w')

            # 第十二行：位置精修
            ttk.Label(ch_frame, text="Position Tolerance [px]:").grid(
                row=11, column=0, padx=5, pady=5, sticky='e')
            ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{ch}']['pos_tol'],
                      width=12).grid(row=11, column=1, columnspan=3, sticky='w')

            # 配置列权重
            ch_frame.columnconfigure(1, weight=1)

        return tab

    def set_spatial_corr_path(self, channel):
        """处理空间校正路径选择 (对应 MATLAB 的 setSpatialCorrPath)"""
        # 获取当前通道的校正状态变量
        correction_var = self.image_bin[f'ch{channel}_spatial_correction']

        if correction_var.get():
            # 获取上次访问路径，默认为当前工作目录
            initial_dir = self.image_bin.get('search_path', '.')

            # 弹出文件选择对话框
            file_path = filedialog.askopenfilename(
                title='Select spatial Transformation matrix',
                initialdir=initial_dir,
                filetypes=[('MAT files', '*.mat')]
            )

            if file_path:
                try:
                    # 加载 MAT 文件
                    mat_data = loadmat(file_path)
                    t_mat = mat_data.get('tMat', None)

                    if t_mat is not None:
                        # 保存到参数库
                        self.image_bin[f'ch{channel}_tMat'] = t_mat
                        self.image_bin['search_path'] = file_path.rsplit('/', 1)[0]
                        print(f"Loaded tMat for Channel {channel}")
                    else:
                        raise ValueError("tMat not found in the selected file")

                except Exception as e:
                    correction_var.set(False)  # 出错时取消勾选
                    self.show_error(f"加载失败: {str(e)}")
        else:
            # 清除相关数据
            if f'ch{channel}_tMat' in self.image_bin:
                del self.image_bin[f'ch{channel}_tMat']
            print(f"Disabled spatial correction for Channel {channel}")


    def create_channel_settings(self, parent):
        """动态生成通道设置"""
        n_im_ch = self.image_bin.get('n_im_ch', 1)

        for ch in range(n_im_ch):
            frame = ttk.LabelFrame(parent, text=f"Channel {ch + 1}")
            frame.grid(row=3 + ch, column=0, columnspan=3, padx=5, pady=5, sticky='ew')

            # 定位参数
            ttk.Label(frame, text="Detection Box (px):").grid(row=0, column=0)
            ttk.Entry(frame, width=8).grid(row=0, column=1)

            ttk.Label(frame, text="Deflation Loops:").grid(row=0, column=2)
            ttk.Entry(frame, width=8).grid(row=0, column=3)

    def create_rendering_tab(self, notebook):
        """创建支持多通道的渲染设置标签页"""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Rendering")

        # 滚动容器
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 主容器
        main_frame = ttk.Frame(scroll_frame)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # 获取通道数量
        n_im_ch = self.image_bin.get('nImCh', 1)

        # 动态生成通道设置
        for ch in range(1, n_im_ch + 1):
            self.create_channel_render_section(main_frame, ch)

        return tab

    def create_channel_render_section(self, parent, channel):
        """创建单个通道的渲染设置区块"""
        ch_frame = ttk.LabelFrame(
            parent,
            text=f"Channel {channel} Rendering Settings",
            padding=(10, 5)
        )
        ch_frame.grid(row=channel - 1, column=0, sticky="ew", pady=10)

        # 初始化通道参数
        self.init_channel_render_params(channel)

        # ========== 基本渲染参数 ==========
        row = 0
        padx = 5
        pady = 3

        # 帧范围
        ttk.Label(ch_frame, text="Framerange:").grid(row=row, column=0, padx=padx, pady=pady, sticky='e')
        ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{channel}_r_start'], width=6).grid(row=row, column=1,
                                                                                               sticky='w')
        ttk.Label(ch_frame, text="-").grid(row=row, column=2)
        ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{channel}_r_end'], width=6).grid(row=row, column=3,
                                                                                             sticky='w')
        row += 1

        # 扩展因子
        ttk.Label(ch_frame, text="Expansion Factor:").grid(row=row, column=0, padx=padx, pady=pady, sticky='e')
        ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{channel}_exf_new'], width=10).grid(row=row, column=1,
                                                                                                columnspan=3,
                                                                                                sticky='w')
        row += 1

        # 卷积模式
        ttk.Label(ch_frame, text="Convolution Mode:").grid(row=row, column=0, padx=padx, pady=pady, sticky='e')
        ttk.Combobox(
            ch_frame,
            textvariable=self.image_bin[f'ch{channel}_conv_mode'],
            values=('fixed', 'dynamic', 'none'),
            width=8,
            state='readonly'
        ).grid(row=row, column=1, columnspan=3, sticky='w')
        row += 1

        # 强度权重
        ttk.Label(ch_frame, text="Intensity Weight:").grid(row=row, column=0, padx=padx, pady=pady, sticky='e')
        ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{channel}_int_weight'], width=10).grid(row=row, column=1,
                                                                                                   columnspan=3,
                                                                                                   sticky='w')
        row += 1

        # 尺寸因子
        ttk.Label(ch_frame, text="Size Factor:").grid(row=row, column=0, padx=padx, pady=pady, sticky='e')
        ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{channel}_size_fac'], width=10).grid(row=row, column=1,
                                                                                                 columnspan=3,
                                                                                                 sticky='w')
        row += 1

        # ========== 电影设置 ==========
        ttk.Label(ch_frame, text="Movie Settings", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=4,
                                                                                    pady=10, sticky='w')
        row += 1

        # 步长
        ttk.Label(ch_frame, text="Stepsize:").grid(row=row, column=0, padx=padx, pady=pady, sticky='e')
        ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{channel}_r_step'], width=10).grid(row=row, column=1,
                                                                                               columnspan=3, sticky='w')
        row += 1

        # 帧率
        ttk.Label(ch_frame, text="FPS:").grid(row=row, column=0, padx=padx, pady=pady, sticky='e')
        ttk.Entry(ch_frame, textvariable=self.image_bin[f'ch{channel}_fps'], width=10).grid(row=row, column=1,
                                                                                            columnspan=3, sticky='w')
        row += 1

        # 压缩格式
        ttk.Label(ch_frame, text="Compression:").grid(row=row, column=0, padx=padx, pady=pady, sticky='e')
        ttk.Combobox(
            ch_frame,
            textvariable=self.image_bin[f'ch{channel}_mov_compression'],
            values=('RLE', 'MSVC', 'none'),
            width=8,
            state='readonly'
        ).grid(row=row, column=1, columnspan=3, sticky='w')
        row += 1

        # 累积电影
        ttk.Checkbutton(
            ch_frame,
            text="Accumulative Movie",
            variable=self.image_bin[f'ch{channel}_is_cumsum']
        ).grid(row=row, column=0, columnspan=4, sticky='w', pady=5)

        # 配置列权重
        ch_frame.columnconfigure(1, weight=1)

    def init_channel_render_params(self, channel):
        """初始化通道渲染参数"""
        params = {
            f'ch{channel}_exf_new': tk.DoubleVar(value=1.0),
            f'ch{channel}_conv_mode': tk.StringVar(value='fixed'),
            f'ch{channel}_int_weight': tk.DoubleVar(value=1.0),
            f'ch{channel}_size_fac': tk.DoubleVar(value=1.0),
            f'ch{channel}_r_start': tk.IntVar(value=1),
            f'ch{channel}_r_end': tk.IntVar(value=100),
            f'ch{channel}_r_step': tk.IntVar(value=1),
            f'ch{channel}_fps': tk.DoubleVar(value=30.0),
            f'ch{channel}_mov_compression': tk.StringVar(value='RLE'),
            f'ch{channel}_is_cumsum': tk.BooleanVar(value=False)
        }
        self.image_bin.update(params)

    def create_filters_tab(self, notebook):
        """创建支持多通道的过滤器设置标签页"""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Filters")

        # 滚动容器
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 主容器
        main_frame = ttk.Frame(scroll_frame)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # 动态生成通道设置
        for ch in range(1, self.image_bin.get('nImCh', 1) + 1):
            self.create_channel_filter_section(main_frame, ch)

        return tab

    def create_channel_filter_section(self, parent, channel):
        """创建单个通道的过滤器设置"""
        ch_frame = ttk.LabelFrame(
            parent,
            text=f"Channel {channel} Filters",
            padding=(10, 5))
        ch_frame.grid(row=channel - 1, column=0, sticky="ew", pady=10)

        # 初始化通道参数
        self.init_channel_filter_params(channel)

        # ========== 定位精度过滤 ==========
        loc_frame = ttk.LabelFrame(ch_frame, text="Localization Precision (nm)")
        loc_frame.grid(row=0, column=0, columnspan=5, sticky='ew', padx=5, pady=5)

        # 启用复选框
        ttk.Checkbutton(loc_frame,
                        text="Enable",
                        variable=self.image_bin[f'ch{channel}_is_thresh_loc_prec']
                        ).grid(row=0, column=0)

        # 最小值
        ttk.Label(loc_frame, text="Min:").grid(row=0, column=1, padx=(10, 2))
        ttk.Entry(loc_frame,
                  textvariable=self.image_bin[f'ch{channel}_min_loc'],
                  width=8).grid(row=0, column=2)

        # 最大值
        ttk.Label(loc_frame, text="Max:").grid(row=0, column=3, padx=(10, 2))
        ttk.Entry(loc_frame,
                  textvariable=self.image_bin[f'ch{channel}_max_loc'],
                  width=8).grid(row=0, column=4)

        # ========== 信噪比过滤 ==========
        snr_frame = ttk.LabelFrame(ch_frame, text="Signal to Noise Ratio")
        snr_frame.grid(row=1, column=0, columnspan=5, sticky='ew', padx=5, pady=5)

        ttk.Checkbutton(snr_frame,
                        text="Enable",
                        variable=self.image_bin[f'ch{channel}_is_thresh_snr']
                        ).grid(row=0, column=0)

        ttk.Label(snr_frame, text="Min:").grid(row=0, column=1, padx=(10, 2))
        ttk.Entry(snr_frame,
                  textvariable=self.image_bin[f'ch{channel}_min_snr'],
                  width=8).grid(row=0, column=2)

        ttk.Label(snr_frame, text="Max:").grid(row=0, column=3, padx=(10, 2))
        ttk.Entry(snr_frame,
                  textvariable=self.image_bin[f'ch{channel}_max_snr'],
                  width=8).grid(row=0, column=4)

        # ========== 检测密度过滤 ==========
        density_frame = ttk.LabelFrame(ch_frame, text="Detection Density")
        density_frame.grid(row=2, column=0, columnspan=5, sticky='ew', padx=5, pady=5)

        ttk.Checkbutton(density_frame,
                        text="Enable",
                        variable=self.image_bin[f'ch{channel}_is_thresh_density']
                        ).grid(row=0, column=0)

        ttk.Label(density_frame, text="Mode:").grid(row=0, column=1, padx=(10, 2))
        ttk.Combobox(density_frame,
                     textvariable=self.image_bin[f'ch{channel}_cluster_mode'],
                     values=('inclusive', 'exclusive'),
                     width=10,
                     state='readonly').grid(row=0, column=2, columnspan=3)

        # 配置列权重
        ch_frame.columnconfigure(0, weight=1)

    def init_channel_filter_params(self, channel):
        """初始化通道过滤器参数"""
        params = {
            f'ch{channel}_is_thresh_loc_prec': tk.BooleanVar(value=False),
            f'ch{channel}_min_loc': tk.DoubleVar(value=0.0),
            f'ch{channel}_max_loc': tk.DoubleVar(value=100.0),
            f'ch{channel}_is_thresh_snr': tk.BooleanVar(value=False),
            f'ch{channel}_min_snr': tk.DoubleVar(value=2.0),
            f'ch{channel}_max_snr': tk.DoubleVar(value=10.0),
            f'ch{channel}_is_thresh_density': tk.BooleanVar(value=False),
            f'ch{channel}_cluster_mode': tk.StringVar(value='inclusive')
        }
        self.image_bin.update(params)

    def create_scalebar_tab(self, notebook):
        """创建比例尺和时间戳设置标签页"""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Bars")

        # 参数存储结构
        self.image_bin.update({
            'is_colormap': tk.BooleanVar(value=False),
            'colormap_width': tk.IntVar(value=10),
            'is_scalebar': tk.BooleanVar(value=True),
            'micron_bar_length': tk.DoubleVar(value=1000),
            'is_timestamp': tk.BooleanVar(value=True),
            'timestamp_inkrement': tk.DoubleVar(value=0.032),
            'timestamp_size': tk.IntVar(value=2)
        })

        # 主布局框架
        main_frame = ttk.Frame(tab, padding=10)
        main_frame.pack(fill='both', expand=True)

        row = 0
        # ================= 颜色图设置 =================
        colormap_frame = ttk.LabelFrame(main_frame, text="Colormap Settings")
        colormap_frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)

        ttk.Checkbutton(colormap_frame,
                        text="Colormap",
                        variable=self.image_bin['is_colormap']).grid(row=0, column=0, sticky='w')

        ttk.Label(colormap_frame, text="[px]:").grid(row=0, column=1, padx=5)
        ttk.Entry(colormap_frame,
                  textvariable=self.image_bin['colormap_width'],
                  width=8).grid(row=0, column=2)
        row += 1

        # ================= 比例尺设置 =================
        scalebar_frame = ttk.LabelFrame(main_frame, text="Scalebar Settings")
        scalebar_frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)

        ttk.Checkbutton(scalebar_frame,
                        text="Scalebar",
                        variable=self.image_bin['is_scalebar']).grid(row=0, column=0, sticky='w')

        ttk.Label(scalebar_frame, text="[μm]:").grid(row=0, column=1, padx=5)
        ttk.Entry(scalebar_frame,
                  textvariable=self.image_bin['micron_bar_length'],
                  width=8).grid(row=0, column=2)
        row += 1

        # ================= 时间戳设置 =================
        timestamp_frame = ttk.LabelFrame(main_frame, text="Timestamp Settings")
        timestamp_frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=5)

        ttk.Checkbutton(timestamp_frame,
                        text="Timestamp",
                        variable=self.image_bin['is_timestamp']).grid(row=0, column=0, sticky='w')

        ttk.Label(timestamp_frame, text="Interval (s):").grid(row=0, column=1, padx=5)
        ttk.Entry(timestamp_frame,
                  textvariable=self.image_bin['timestamp_inkrement'],
                  width=8).grid(row=0, column=2)

        ttk.Label(timestamp_frame, text="Charactersize:").grid(row=1, column=1, padx=5)
        ttk.Entry(timestamp_frame,
                  textvariable=self.image_bin['timestamp_size'],
                  width=8).grid(row=1, column=2)
        row += 1

        # 配置列权重
        main_frame.columnconfigure(0, weight=1)

        return tab

    def create_acquisition_tab(self, notebook):
        """创建支持多通道的采集参数标签页"""
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Acquisition")

        # 滚动容器
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 主容器
        main_frame = ttk.Frame(scroll_frame)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # 动态生成通道设置
        for ch in range(1, self.image_bin.get('nImCh', 1) + 1):
            self.create_channel_acq_section(main_frame, ch)

        return tab

    def create_channel_acq_section(self, parent, channel):
        """创建单个通道的采集参数区块"""
        ch_frame = ttk.LabelFrame(
            parent,
            text=f"Channel {channel} Acquisition Settings",
            padding=(10, 5)
        )
        ch_frame.grid(row=channel - 1, column=0, sticky="ew", pady=10)

        # 初始化通道参数
        self.init_channel_acq_params(channel)

        # ========== 光学参数 ==========
        optics_frame = ttk.LabelFrame(ch_frame, text="Optical Parameters")
        optics_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

        row = 0
        # 像素尺寸
        ttk.Label(optics_frame, text="Pixel Size (μm):").grid(row=row, column=0, sticky='e', padx=5)
        px_entry = ttk.Entry(optics_frame, textvariable=self.image_bin[f'ch{channel}_px_size'], width=10)
        px_entry.grid(row=row, column=1, sticky='w')
        self.create_tooltip(px_entry, "Physical pixel size / Total magnification\n"
                                      "Examples:\n60x1.0: 0.267μm\n60x1.6: 0.167μm\n150x1.0: 0.107μm")
        row += 1

        # 发射波长
        ttk.Label(optics_frame, text="Emission WL (nm):").grid(row=row, column=0, sticky='e', padx=5)
        wl_entry = ttk.Entry(optics_frame, textvariable=self.image_bin[f'ch{channel}_emission_wl'], width=10)
        wl_entry.grid(row=row, column=1, sticky='w')
        self.create_tooltip(wl_entry, "Emission wavelength in nanometers")
        row += 1

        # 数值孔径
        ttk.Label(optics_frame, text="N.A.:").grid(row=row, column=0, sticky='e', padx=5)
        na_entry = ttk.Entry(optics_frame, textvariable=self.image_bin[f'ch{channel}_na'], width=10)
        na_entry.grid(row=row, column=1, sticky='w')
        self.create_tooltip(na_entry, "Numerical aperture of the objective")
        row += 1

        # ========== PSF参数 ==========
        psf_frame = ttk.LabelFrame(ch_frame, text="PSF Parameters")
        psf_frame.grid(row=1, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

        row = 0
        # PSF缩放
        ttk.Label(psf_frame, text="PSF Scaling:").grid(row=row, column=0, sticky='e', padx=5)
        scale_entry = ttk.Entry(psf_frame, textvariable=self.image_bin[f'ch{channel}_psf_scale'], width=10)
        scale_entry.grid(row=row, column=1, sticky='w')
        self.create_tooltip(scale_entry, "PSF scaling factor for simulation")
        # **绑定事件，当用户修改 `PSF Scaling` 时自动计算 `PSF Std`**
        scale_entry.bind("<KeyRelease>", lambda event: self.calc_psf(channel))
        row += 1

        # PSF标准差
        ttk.Label(psf_frame, text="PSF Std [px]:").grid(row=row, column=0, sticky='e', padx=5)
        std_entry = ttk.Entry(psf_frame, textvariable=self.image_bin[f'ch{channel}_psf_std'], width=10,
                              state='readonly')
        std_entry.grid(row=row, column=1, sticky='w')
        self.create_tooltip(std_entry, "Calculated PSF standard deviation (read-only)")
        row += 1

        # ========== 探测器参数 ==========
        detector_frame = ttk.LabelFrame(ch_frame, text="Detector Parameters")
        detector_frame.grid(row=2, column=0, columnspan=3, sticky='ew', padx=5, pady=5)

        row = 0
        # 光子计数
        ttk.Label(detector_frame, text="Counts/Photon:").grid(row=row, column=0, sticky='e', padx=5)
        cnt_entry = ttk.Entry(detector_frame, textvariable=self.image_bin[f'ch{channel}_counts_per_photon'], width=10)
        cnt_entry.grid(row=row, column=1, sticky='w')
        self.create_tooltip(cnt_entry, "Photon to digital count conversion factor")
        row += 1

        # 延迟时间
        ttk.Label(detector_frame, text="Lag Time [ms]:").grid(row=row, column=0, sticky='e', padx=5)
        lag_entry = ttk.Entry(detector_frame, textvariable=self.image_bin[f'ch{channel}_lag_time'], width=10)
        lag_entry.grid(row=row, column=1, sticky='w')
        self.create_tooltip(lag_entry, "Camera lag time between frames")
        row += 1

        # 帧尺寸
        ttk.Label(detector_frame, text="Frame Size [px]:").grid(row=row, column=0, sticky='e', padx=5)
        size_entry = ttk.Entry(detector_frame, textvariable=self.image_bin[f'ch{channel}_frame_size'], width=10)
        size_entry.grid(row=row, column=1, sticky='w')
        self.create_tooltip(size_entry, "Image frame dimensions (square)")

        # 配置列权重
        ch_frame.columnconfigure(0, weight=1)

    def init_channel_acq_params(self, channel):
        """初始化通道采集参数"""
        params = {
            f'ch{channel}_px_size': tk.DoubleVar(value=0.16),
            f'ch{channel}_emission_wl': tk.DoubleVar(value=590.0),
            f'ch{channel}_na': tk.DoubleVar(value=1.49),
            f'ch{channel}_psf_scale': tk.DoubleVar(value=1.35),
            f'ch{channel}_psf_std': tk.DoubleVar(value=1.03),
            f'ch{channel}_counts_per_photon': tk.DoubleVar(value=20.2),
            f'ch{channel}_lag_time': tk.DoubleVar(value=50.0),
            f'ch{channel}_frame_size': tk.IntVar(value=256)
        }
        self.image_bin.update(params)

    def create_tooltip(self, widget, text):
        """创建简易工具提示"""
        widget.bind("<Enter>", lambda e: self.show_tooltip(e.widget, text))
        widget.bind("<Leave>", lambda e: self.hide_tooltip())

    def show_tooltip(self, widget, text):
        """显示工具提示"""
        x = widget.winfo_rootx() + 20
        y = widget.winfo_rooty() + 20
        self.tooltip = tk.Toplevel()
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{x}+{y}")
        ttk.Label(self.tooltip, text=text, background="#ffffe0",
                  relief="solid", borderwidth=1, padding=5).pack()

    def hide_tooltip(self):
        """隐藏工具提示"""
        if hasattr(self, 'tooltip'):
            self.tooltip.destroy()

    def calc_psf(self, src_widget, channel, fieldname):
        """计算 PSF 标准差并更新显示 (对应 MATLAB 的 calcPSF 函数)"""
        try:
            # 将输入值转换为浮点数并保存到参数库
            input_value = float(src_widget.get())
            self.image_bin[f'ch{channel}_{fieldname}'].set(input_value)

            # 从参数库获取相关参数
            psf_scale = self.image_bin[f'ch{channel}_psfScale'].get()
            em_wavelength = self.image_bin[f'ch{channel}_emWvlnth'].get()
            na_value = self.image_bin[f'ch{channel}_NA'].get()
            px_size = self.image_bin[f'ch{channel}_pxSize'].get()

            # 避免除零错误
            if na_value == 0 or px_size == 0:
                self.image_bin[f'ch{channel}_psf_std'].set("Error")
                return

            # PSF 计算公式
            psf_std = psf_scale * 0.55 * (em_wavelength / 1000)
            psf_std /= na_value * 1.17 * 2 * px_size

            # 更新显示控件（假设有对应的显示变量）
            if hasattr(self, 'psf_std_display'):
                self.psf_std_display.set(f"{psf_std:.2f}")

            # 保存计算结果到参数库
            self.image_bin[f'ch{channel}_psfStd'].set(psf_std)

        except ValueError:
            # 处理无效输入
            self.show_error("Invalid input value")

    def show_error(self, message):
        """统一错误提示"""
        messagebox.showerror(
            "Input Error",
            f"{message}\nPlease enter a valid number",
            parent=self.root
        )

    def save_settings(self):
        """保存设置到文件"""
        try:
            settings = {k: v.get() for k, v in self.image_bin.items()
                        if isinstance(v, (tk.StringVar, tk.IntVar, tk.DoubleVar, tk.BooleanVar))}
            with open('settings.json', 'w') as f:
                json.dump(settings, f)
            messagebox.showinfo("Success", "Settings saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")

    def load_settings(self):
        """从文件加载设置"""
        try:
            with open('settings.json') as f:
                settings = json.load(f)
                for k, v in settings.items():
                    if k in self.image_bin:
                        self.image_bin[k].set(v)
            messagebox.showinfo("Success", "Settings loaded successfully!")
        except FileNotFoundError:
            messagebox.showerror("Error", "Settings file not found!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load settings: {str(e)}")


    ##############################################

    def set_roi(self):
        """Toggle the ROI drawing mode."""
        if self.image_bin['h_roi'] is not None:
            # 如果存在 ROI，清除它
            self.roi_canvas.delete(self.image_bin['h_roi'])
            self.image_bin['h_roi'] = None
            self.image_bin['drawing'] = False  # 停止绘制模式
        else:
            # 启用绘制模式
            self.image_bin['drawing'] = True
            self.image_bin['roi_coords'] = []  # 重置 ROI 坐标

    def on_button_press(self, event):
        """Start drawing the rectangle ROI."""
        # 开始绘制
        self.image_bin['drawing'] = True
        self.image_bin['roi_coords'] = [(event.x, event.y)]

        # 如有旧矩形，先删
        h_old = self.image_bin.get('h_roi')
        if h_old:
            self.roi_canvas.delete(h_old)

        # 新建矩形（起点=终点）
        self.image_bin['h_roi'] = self.roi_canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline='red', fill='', width=2
        )

    def on_mouse_drag(self, event):
        """Update the rectangle ROI as the mouse is dragged."""
        if self.image_bin.get('drawing') and self.image_bin.get('h_roi'):
            x0, y0 = self.image_bin['roi_coords'][0]
            self.roi_canvas.coords(self.image_bin['h_roi'], x0, y0, event.x, event.y)

    ####记录这些矩形顶点坐标
    def on_button_release(self, event):
        """Finalize the rectangle ROI shape and record corner coordinates."""
        if not self.image_bin.get('drawing'):
            return

        start_x, start_y = self.image_bin['roi_coords'][0]
        end_x, end_y = event.x, event.y

        # 规范化为左上(x0,y0)、右下(x1,y1)
        x0, x1 = sorted([start_x, end_x])
        y0, y1 = sorted([start_y, end_y])

        # 更新画布上的矩形为规范化后的坐标
        h_roi = self.image_bin.get('h_roi')
        if h_roi:
            self.roi_canvas.coords(h_roi, x0, y0, x1, y1)

        # 保存四个角（画布坐标）
        self.image_bin['roi_values'] = [
            (x0, y0),  # 左上
            (x0, y1),  # 左下
            (x1, y0),  # 右上
            (x1, y1)  # 右下
        ]
        # 保存两点（画布坐标）
        self.image_bin['roi_coords'] = [(x0, y0), (x1, y1)]
        self.image_bin['drawing'] = False

        # —— 画布坐标 → 原图坐标（scale=2）——
        scale = getattr(self, 'display_scale', 2)
        ix0 = int(round(x0 / scale))
        iy0 = int(round(y0 / scale))
        ix1 = int(round(x1 / scale))
        iy1 = int(round(y1 / scale))

        # 尺寸为半开区间宽高
        w = max(ix1 - ix0, 1)
        h = max(iy1 - iy0, 1)

        # 越界修正到原图范围
        W = int(self.image_bin.get('width', 0))
        H = int(self.image_bin.get('height', 0))
        if W and H:
            ix0 = max(0, min(ix0, W - 1))
            iy0 = max(0, min(iy0, H - 1))
            ix1 = max(0, min(ix1, W))
            iy1 = max(0, min(iy1, H))
            w = max(ix1 - ix0, 1)
            h = max(iy1 - iy0, 1)

        # 写回“原图坐标系”的 ROI：[x0, y0, w, h]
        self.image_bin['roi'] = [ix0, iy0, w, h]

    def adjust_roi(self):
        """Adjust the position and size of the ROI."""
        if self.image_bin['h_roi']:
            move_x = simpledialog.askinteger("Move ROI", "Enter X offset:")
            move_y = simpledialog.askinteger("Move ROI", "Enter Y offset:")
            if move_x is not None and move_y is not None:
                coords = self.canvas.coords(self.image_bin['h_roi'])
                self.canvas.coords(self.image_bin['h_roi'],
                                   coords[0] + move_x, coords[1] + move_y,
                                   coords[2] + move_x, coords[3] + move_y)

    def sync_roi(self):
        """Sync the ROI with another figure (placeholder)."""
        # This function is a placeholder for syncing logic
        messagebox.showinfo("Info", "Sync ROI functionality not implemented.")

    def show_loc_preview(self, frame=None):
        """
        显示基于 ROI 的检测位置预览。
        如果传入 frame，则更新当前帧，否则使用 self.image_bin['frame']。
        该函数重新调用检测函数 detect_et_estime_part_1vue_deflt，
        并在新的 Tkinter 窗口中嵌入 matplotlib 图形显示检测结果。
        """
        try:
            # 更新当前帧
            if frame is not None:
                self.image_bin['frame'] = frame
            else:
                frame = self.image_bin.get('frame', 0)


            # 检查图像堆栈或当前帧图像是否加载
            if 'raw_image' not in self.image_bin or self.image_bin['raw_image'] is None:
                messagebox.showwarning("Warning", "No image loaded.")
                return

            # 获取当前帧的原始图像（假定为 PIL Image 对象）
            I = self.image_bin['raw_image']
            if hasattr(I, 'size'):
                width, height = I.size  # PIL 返回 (width, height)
                default_roi = [0, 0, width, height]

            else:
                default_roi = [0, 0, I.shape[1], I.shape[0]]


            # 获取 ROI 参数（如果未设置则使用整个图像区域）
            roi = np.ceil(self.image_bin.get('roi', default_roi)).astype(int)


            # 如果存在 'hROI'（例如用户选定的 ROI 对象），则裁剪图像
            if self.image_bin.get('hROI') is not None:
                I_cropped = I.crop((roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]))
            else:
                I_cropped = I

            # 将裁剪后的图像转换为 NumPy 数组，确保为 float 类型以便后续检测
            I_cropped_np = np.array(I_cropped, dtype=float)

            # 构造检测参数（优化参数等）
            optim = [
                self.image_bin.get('maxOptimIter'),
                self.image_bin.get('termTol'),
                self.image_bin.get('isRadiusTol'),
                self.image_bin.get('radiusTol'),
                self.image_bin.get('posTol')
            ]


            error_rate = self.image_bin.get('errorRate', -6)
            pfa = 1.0 / (10 ** (-error_rate))  # 1/10^( -error_rate )
            prob = 1.0 - pfa
            if prob < 0:
                prob = 0.0
            threshold = chi2.ppf(prob, 1)

            w2d = self.image_bin.get('w2d')
            psf_std = self.image_bin.get('psfStd')
            dfltn_loops = self.image_bin.get('dfltnLoops')
            min_int = self.image_bin.get('minInt')

            width_roi = roi[2]
            height_roi = roi[3]

            # 调用检测函数
            list_arr, dfltI, mask, good = self.detect_et_estime_part_1vue_deflt(
                I_cropped_np,
                w2d,
                psf_std,
                threshold,
                dfltn_loops,
                width_roi,  # ROI 宽度
                height_roi,  # ROI 高度
                min_int,
                optim
            )

            # 创建预览窗口
            preview_window = tk.Toplevel(self.master)
            preview_window.title(f"Frame: {frame} ({good} Detections)")
            preview_window.geometry("800x600")

            # 创建 matplotlib Figure 和坐标轴
            fig = plt.Figure(figsize=(8, 6))
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
            ax.imshow(np.array(I_cropped), cmap='gray')

            # 绘制 ROI 矩形（加上 0.5 像素偏移以匹配 MATLAB）
            rect = Rectangle((roi[0] + 0.5, roi[1] + 0.5), roi[2], roi[3],
                             edgecolor='r', linewidth=1, fill=False)
            ax.add_patch(rect)

            # 定义圆周角度（10个点）
            rho = np.linspace(0, 2 * np.pi, 10)
            # 遍历检测结果并绘制圆形区域
            if list_arr is not None and list_arr.shape[0] > 0:
                for i in range(list_arr.shape[0]):
                    x_offset = list_arr[i, 1]
                    y_offset = list_arr[i, 0]
                    radius = list_arr[i, 5]
                    center_x = roi[0] + 1 + x_offset
                    center_y = roi[1] + 1 + y_offset
                    xs = center_x + np.cos(rho) * radius
                    ys = center_y + np.sin(rho) * radius
                    polygon = Polygon(np.column_stack((xs, ys)), closed=True,
                                      facecolor=(1, 0, 0), edgecolor='none', alpha=0.5)
                    ax.add_patch(polygon)
                print(f"Detect {list_arr.shape[0]} point")
            else:
                print("No detections")

            canvas = FigureCanvasTkAgg(fig, master=preview_window)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
            preview_window.grid_rowconfigure(0, weight=1)
            preview_window.grid_columnconfigure(0, weight=1)

        except Exception as e:
            messagebox.showerror("Error", f"Error in show_loc_preview: {str(e)}")
            raise

    def render_image(self, data, roi, exf, width, height, int_weight, size_fac, px_size, mode):
        """
        根据提供的参数渲染图像，模仿 MATLAB 的 renderImage 函数。

        参数：
          data: 包含字段 'ctrsX', 'ctrsY', 'photons', 以及（对于动态模式）'precision' 等的字典
          roi: 感兴趣区域 [x, y, width, height]
          exf: 扩展因子
          width, height: 输入图像的宽度和高度
          int_weight: 权重参数
          size_fac: 尺寸缩放因子
          px_size: 像素大小（物理单位）
          mode: 模式标志，1 表示 Fixed, 2 表示 Dynamic
        返回：
          rI: 渲染后的超分辨图像
          new_width, new_height: 渲染图像尺寸
          N: 检测到的粒子数量
        """
        # 计算超分辨图像像素边缘
        edge_y = np.arange(1, height * exf + 1) / exf
        edge_x = np.arange(1, width * exf + 1) / exf

        # 计算 ROI 对应的边缘索引
        y0 = np.searchsorted(edge_y, roi[1], side='left')
        y1 = np.searchsorted(edge_y, roi[1] + roi[3], side='left')
        x0 = np.searchsorted(edge_x, roi[0], side='left')
        x1 = np.searchsorted(edge_x, roi[0] + roi[2], side='left')

        # 计算新图像尺寸
        new_height = y1 - y0 + 1
        new_width = x1 - x0 + 1
        rI = np.zeros((new_height, new_width))

        # 确保数据字段为 numpy 数组
        data = {key: np.asarray(val) for key, val in data.items()}

        if data['ctrsX'].size > 0 and data['ctrsY'].size > 0 and data['photons'].size > 0:
            if mode == 2:  # 动态模式
                # 模拟 MATLAB 中的 histc，用 searchsorted 获得每个数据点的 bin 索引
                lp2pix = (data['precision'] / (px_size / exf)) ** 2 * size_fac ** 2
                yc = np.searchsorted(edge_y[y0:y1], data['ctrsY'], side='left')
                xc = np.searchsorted(edge_x[x0:x1], data['ctrsX'], side='left')
                N = 0
                for i in range(len(xc)):
                    w = int(np.ceil(6 * np.sqrt(lp2pix[i])))
                    # MATLAB 中的索引从1开始，Python 从0开始，这里调整为 >=0 和 <对应值
                    if (xc[i] - w >= 0 and xc[i] + w < np.ceil(roi[2] * exf) and
                            yc[i] - w >= 0 and yc[i] + w < np.ceil(roi[3] * exf)):
                        N += 1
                        refi = np.arange(1, w + 1) - (w // 2)
                        # MATLAB: (refi - (data.ctrsY(i)-roi(2))*exf)' * ones(1,w)
                        ii = (refi - (data['ctrsY'][i] - roi[1]) * exf)[:, None]
                        refj = np.arange(1, w + 1) - (w // 2)
                        jj = refj - (data['ctrsX'][i] - roi[0]) * exf
                        # MATLAB 的索引调整：减1以转换为0-based
                        rows = (np.arange(1, w + 1) - (w // 2) + yc[i] - 1).astype(int)
                        cols = (np.arange(1, w + 1) - (w // 2) + xc[i] - 1).astype(int)
                        # 检查索引是否在范围内（可选）
                        # 累加贡献
                        rI[np.ix_(rows, cols)] += (
                                int_weight * data['photons'][i] / (2 * np.pi * lp2pix[i]) *
                                np.exp(-(1 / (2 * lp2pix[i])) * (ii ** 2 + jj ** 2))
                        )
            elif mode == 1:  # 固定模式
                N = data['photons'].size
                # 使用 searchsorted 来模拟 histc，得到每个检测点的 bin 索引
                idxY = np.searchsorted(edge_y[y0:y1], data['ctrsY'], side='left')
                idxX = np.searchsorted(edge_x[x0:x1], data['ctrsX'], side='left')
                # 计算二维线性索引（注意 Python 数组是0-based）
                linIdx = idxY * new_width + idxX
                # 对 rI 进行扁平化后进行累加
                flat_rI = rI.ravel()
                # 使用 np.add.at 累加 int_weight，注意此处 int_weight 是否需要乘以 data['photons'][i]（动态模式中乘以了）
                # 这里直接将 int_weight 加到对应索引处
                np.add.at(flat_rI, linIdx, int_weight)
                rI = flat_rI.reshape(rI.shape)
                # 构建 PSF，并进行傅里叶变换处理
                psf = self.build_psf(size_fac, new_width, new_height)
                rI = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(rI) * psf))
            else:
                N = 0
        else:
            N = 0
        return rI, new_width, new_height, N

    def imprint_scalebar(self, I, width, height, exf, bar_length, px_size, size_fac, mode):
        """Imprint a scale bar onto the image."""
        bar_unit = self.px_chars(height, width, f"{bar_length}nm", size_fac)

        px_bar_length = round(bar_length / (1000 * px_size / exf))
        px_bar_height = int(np.ceil(max(exf / 3, px_bar_length / 20)))
        bar = np.ones((px_bar_height, px_bar_length))

        tmp = bar.shape[1] - bar_unit.shape[1]
        if tmp > 0:
            bar = np.vstack([
                np.hstack([np.zeros((bar_unit.shape[0], tmp // 2)), bar_unit,
                           np.zeros((bar_unit.shape[0], np.ceil(tmp / 2)))]),
                np.zeros((2, bar.shape[1])),
                bar
            ])
        elif tmp < 0:
            bar = np.vstack([
                bar_unit,
                np.zeros((2, bar_unit.shape[1])),
                np.hstack([np.zeros((px_bar_height, -tmp // 2)), bar, np.zeros((px_bar_height, np.ceil(-tmp / 2)))])
            ])
        else:
            bar = np.vstack([bar_unit, np.zeros((2, bar.shape[1])), bar])

        bar_offset_x = 4 * px_bar_height
        bar_offset_y = 4 * px_bar_height
        offset = height * (width - (bar.shape[1] + bar_offset_x)) + bar_offset_y

        idx = np.where(np.vstack([bar, np.zeros((height - bar.shape[0], bar.shape[1]))]))[0] + offset

        if mode == 1:
            I[idx] = 255
        else:
            I[np.concatenate([idx, idx + height * width, idx + 2 * height * width])] = 255

        return I

    def imprint_timestamp(self, I, width, height, time, size_fac, mode):
        """Imprint a timestamp onto the image."""
        stamp = self.px_chars(height, width, f"{time:.3f}".replace('.', 'p') + '_S', size_fac)

        bar_offset_x = int(np.ceil(max(size_fac * 2, stamp.shape[1] / 5)))
        bar_offset_y = bar_offset_x

        idx = np.where(np.vstack([stamp, np.zeros((height - stamp.shape[0], stamp.shape[1]))]))[0] + \
              bar_offset_x * height + bar_offset_y

        if mode == 1:
            I[idx] = 255
        else:
            I[np.concatenate([idx, idx + height * width, idx + 2 * height * width])] = 255

        return I

    def px_chars(self, height, width, text, size_fac):
        """Placeholder function to create pixel character representation."""
        # This function should create a binary image representation of the text
        # For now, return a dummy array
        text_width = len(text) * size_fac  # Simplified width calculation
        return np.ones((height // 10, text_width), dtype=np.uint8)  # Placeholder for text

    def pixel_intensity_hist(self):
        """显示强度直方图（像素强度）"""
        # 创建新窗口
        fig_new = tk.Toplevel(self.master)
        fig_new.title("Pixel Intensity Distribution")
        fig_new.geometry("650x600")

        # 使用 load_imagestack 加载的图像数据
        image_file = self.image_bin.get('pathname')
        try:
            im = Image.open(image_file)
            I = np.array(im).astype(float)
        except Exception as e:
            print(f"Load image failed: {e}")
            return

        # 创建一个 matplotlib Figure 对象和坐标轴
        fig = plt.Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        # 调用 build_histogram 绘制直方图
        self.build_histogram(I.ravel(), 'counts', ax)
        ax.set_xlabel('Pixel Intensity [counts]')


        # 将 Figure 嵌入到 Tkinter 窗口中，使用 grid 布局
        canvas = FigureCanvasTkAgg(fig, master=fig_new)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky='nsew')

        # —— 添加 Save 按钮 ——#
        def save_plot():
            # 弹出“另存为”对话框，保存为 PNG
            path = filedialog.asksaveasfilename(
                defaultextension='.png',
                filetypes=[('PNG Image', '*.png')],
                title='Save Pixel Intensity Histogram'
            )
            if path:
                try:
                    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
                    print(f">> Saved pixel histogram to: {path}")
                except Exception as err:
                    print(f"Error saving figure: {err}")

        save_btn = tk.Button(fig_new, text="Save", command=save_plot)
        # 按钮放在行 1
        save_btn.grid(row=1, column=0, pady=5)

        # 设置窗口行列权重，使画布和按钮自适应
        fig_new.grid_rowconfigure(0, weight=1)
        fig_new.grid_rowconfigure(1, weight=0)
        fig_new.grid_columnconfigure(0, weight=1)

    def particle_intensity_hist(self):
        """显示粒子强度直方图及2成分高斯混合模型拟合曲线（合成一幅图，双 y 轴）"""
        # 创建新窗口
        fig_new = tk.Toplevel(self.master)
        fig_new.title("Particle Intensity Distribution")
        # 设置窗口尺寸
        fig_new.geometry("650x600")

        cnts_per_photon = self.image_bin['cntsPerPhoton']
        # 计算光子数（单位为 photons），根据你的校准参数
        photons = self.image_bin['signal'] * 2 * np.pi * self.image_bin['radius'] ** 2 / cnts_per_photon

        # 创建一个 Figure 对象，并只添加一个子图
        fig = plt.Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        # 绘制直方图：使用 bar 绘制柱状图（左侧 y 轴）
        freq, bins = np.histogram(photons, bins='fd')
        ax.bar(bins[:-1], freq, width=np.diff(bins), align='edge', color='#4682B4', alpha=0.7,
               label='Photons Intensity Histogram')
        ax.set_xlabel('Particle Intensity (photons)')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right')

        # 使用 ax.twinx() 创建右侧 y 轴，用于绘制高斯混合拟合曲线
        ax2 = ax.twinx()

        # 使用2成分的 Gaussian Mixture 模型进行拟合，并绘制拟合曲线（右侧 y 轴）
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=2)
        gmm.fit(photons.reshape(-1, 1))
        # 生成拟合曲线数据，分布在 data 的最小值到最大值之间
        x_fit = np.linspace(np.min(photons), np.max(photons), 1000).reshape(-1, 1)
        pdf = np.exp(gmm.score_samples(x_fit))
        ax2.plot(x_fit, pdf, 'r--', label="Gaussian Mixture Fit")
        ax2.set_ylabel("Probability Density")
        ax2.legend(loc='upper right',bbox_to_anchor=(1, 0.90))

        fig.tight_layout()

        # 将 Figure 嵌入到 Tkinter 窗口中
        canvas = FigureCanvasTkAgg(fig, master=fig_new)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        # 图像占用行 0
        canvas_widget.grid(row=0, column=0, sticky='nsew')

        # —— 添加 Save 按钮 ——#
        def save_plot():
            # 弹出“另存为”对话框，保存为 PNG
            path = filedialog.asksaveasfilename(
                defaultextension='.png',
                filetypes=[('PNG Image', '*.png')],
                title='Save Particle Intensity Histogram'
            )
            if path:
                try:
                    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)
                    print(f">> Saved particle histogram to: {path}")
                except Exception as err:
                    print(f"Error saving figure: {err}")

        save_btn = tk.Button(fig_new, text="Save", command=save_plot)
        # 按钮放在行 1
        save_btn.grid(row=1, column=0, pady=5)

        # 设置窗口行列权重，使画布和按钮自适应
        fig_new.grid_rowconfigure(0, weight=1)
        fig_new.grid_rowconfigure(1, weight=0)
        fig_new.grid_columnconfigure(0, weight=1)

    def build_movie_preview(self):
        """Build movie preview based on current settings."""
        # Here we would implement the logic for building the movie preview
        # This is just a placeholder implementation
        # 构造fig_params参数
        fig_params = {
            'pathname': self.image_bin.get('pathname', ''),
            'filename': self.image_bin.get('filename', ''),
            'elements': self.image_bin.get('elements', 0),
            'startPnt': self.image_bin.get('startPnt', 0),
            'cntsPerPhoton': self.image_bin.get('cntsPerPhoton', 20.2),
            'pxSize': self.image_bin.get('pxSize', 0.1),
            'isThreshDensity': self.image_bin.get('isThreshDensity', False),
            'isThreshSNR': self.image_bin.get('isThreshSNR', False),
            'isThreshLocPrec': self.image_bin.get('isThreshLocPrec', False),
            'hROI': self.image_bin.get('hROI', None)
        }
        idx = [-1] + list(np.cumsum([1] * 10))  # Placeholder for ctrsN

        # First movie frame setup
        start_point = idx[self.image_bin['r_start']] + 1
        elements = idx[self.image_bin['r_start'] + self.image_bin['r_w']] - max(1, start_point) + 1

        roi = self.image_bin['roi']
        # 设置return_flags
        if self.image_bin['conv_mode'] != 1:
            return_val = [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        else:
            return_val = [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

            # 调用preprocess_data
            data = self.preprocess_data(
                fig_params=fig_params,
                return_flags=return_val,
                roi=roi
            )
        movie_frame, width, height, N = self.render_image_data(data, roi)

        # Create preview window
        preview_fig = tk.Toplevel(self.master)
        preview_fig.title("Movie Preview")
        preview_fig.geometry("800x600")

        # Create a frame for the canvas
        canvas_frame = tk.Frame(preview_fig)
        canvas_frame.grid(row=0, column=0, sticky='nsew')

        # Configure row and column weights for responsiveness
        preview_fig.grid_rowconfigure(0, weight=1)
        preview_fig.grid_columnconfigure(0, weight=1)

        # Create axes for image display
        preview_ax = tk.Canvas(canvas_frame, bg='white')
        preview_ax.grid(row=0, column=0, sticky='nsew')

        # Display the rendered movie frame
        preview_ax.create_image(0, 0, anchor=tk.NW, image=ImageTk.PhotoImage(Image.fromarray(movie_frame)))

        # Add additional UI elements like lines and text
        preview_ax.create_line(0, height, width, height, fill="white", width=3)
        preview_ax.create_text(0.25 * width, 0.75 * height, text='Frame 1', font=('Arial', 15), fill='white')
        preview_ax.create_text(0.75 * width, 0.25 * height, text='Accumulate', font=('Arial', 15), fill='white')

        # Adjust canvas size to fit the image
        preview_ax.config(scrollregion=preview_ax.bbox(tk.ALL))

    def preprocess_data(self, fig_params, return_flags, roi):
        """
        数据预处理主函数 (对应 MATLAB 的 preprocessData)
        :param fig_params: 窗口参数字典
        :param return_flags: 需要返回的数据标志列表
        :param roi: 感兴趣区域 [x, y, width, height]
        :return: 处理后的数据字典
        """
        # 初始化变量
        var_names = [
            'ctrsX', 'ctrsY', 'signal', 'noise', 'offset',
            'radius', 'frame', 'photons', 'precision',
            'snr', 'sbr', 'cluster'
        ]

        # 初始化加载标志
        load_flags = self._init_load_flags(return_flags, fig_params)

        # 加载数据
        data = self._load_data_from_disk(fig_params, load_flags, var_names)

        # ROI过滤
        data = self._apply_roi_filter(data, load_flags, roi, var_names)

        # 计算衍生指标
        data = self._calculate_metrics(data, fig_params, return_flags)

        # 应用阈值过滤
        data = self._apply_thresholds(data, fig_params, load_flags, return_flags, var_names)

        # 密度聚类过滤
        data = self._apply_density_filter(data, fig_params, load_flags, return_flags, var_names)

        # 清理不需要的字段
        data = self._cleanup_fields(data, return_flags, var_names)

        return data

    def _init_load_flags(self, return_flags, fig_params):
        """初始化数据加载标志"""
        load_flags = [False] * 12  # 对应 MATLAB 的 loadValue(1:12)

        # 基础加载逻辑
        load_flags[0:7] = return_flags[0:7]

        # 条件判断加载
        if return_flags[7]: load_flags[5] = True
        if return_flags[8]: load_flags[2:5] = [True] * 3
        if return_flags[9]: load_flags[2:5] = [True] * 3
        if return_flags[10]: load_flags[2:4] = [True] * 2
        if return_flags[11] or fig_params.get('isThreshDensity', False):
            load_flags[0:2] = [True] * 2
            load_flags[5:7] = [True] * 2

        # 其他条件
        if fig_params.get('hROI'):
            load_flags[0:2] = [True] * 2

        return load_flags

    def _load_data_from_disk(self, fig_params, load_flags, var_names):
        """从磁盘加载数据"""
        data = {}
        base_path = os.path.join(fig_params['pathname'], fig_params['filename'])

        for idx, flag in enumerate(load_flags[:7]):
            if flag:
                var_name = var_names[idx]
                file_path = f"{base_path}.{var_name}"

                try:
                    with open(file_path, 'rb') as f:
                        if np.isinf(fig_params.get('elements', 0)):
                            data[var_name] = np.fromfile(f, dtype=np.float64)
                        else:
                            f.seek(fig_params.get('startPnt', 0) * 8)
                            data[var_name] = np.fromfile(
                                f, dtype=np.float64,
                                count=fig_params.get('elements', 0)
                            )
                except FileNotFoundError:
                    print(f"Warning: {file_path} not found")
                    data[var_name] = np.array([])

        return data

    def _apply_roi_filter(self, data, load_flags, roi, var_names):
        """应用ROI过滤"""
        if load_flags[0] and load_flags[1]:
            x, y = data[var_names[0]], data[var_names[1]]
            mask = (
                    (x > roi[0]) &
                    (x < roi[0] + roi[2]) &
                    (y > roi[1]) &
                    (y < roi[1] + roi[3])
            )

            for idx, flag in enumerate(load_flags):
                if flag and idx < len(var_names):
                    data[var_names[idx]] = data[var_names[idx]][mask]

        return data

    def _calculate_metrics(self, data, fig_params, return_flags):
        """计算衍生指标"""
        if 'radius' in data and 'signal' in data:
            # 计算光子数
            cnts_per_photon = fig_params.get('cntsPerPhoton', 1.0)
            data['photons'] = data['signal'] * 2 * np.pi * data['radius'] ** 2 / cnts_per_photon

            # 计算定位精度
            if return_flags[8] or fig_params.get('isThreshLocPrec', False):
                px_size = fig_params.get('pxSize', 0.1)
                data['precision'] = self.calc_loc_precision(
                    data['radius'], px_size,
                    data['photons'], data['noise'] / cnts_per_photon
                )

            # 计算SNR/SBR
            if return_flags[9] or fig_params.get('isThreshSNR', False):
                data['snr'] = data['signal'] / data['noise']

            if return_flags[10]:
                data['sbr'] = data['signal'] / data['offset']

        return data

    def _apply_thresholds(self, data, fig_params, load_flags, return_flags, var_names):
        """应用阈值过滤"""
        thresh_loc = fig_params.get('isThreshLocPrec', False)
        thresh_snr = fig_params.get('isThreshSNR', False)

        if thresh_loc or thresh_snr:
            mask = np.ones(len(data.get(var_names[0], [])), dtype=bool)

            if thresh_loc and thresh_snr:
                min_loc = fig_params.get('minLoc', 0) / 1000
                max_loc = fig_params.get('maxLoc', 1000) / 1000
                min_snr = fig_params.get('minSNR', 0)
                max_snr = fig_params.get('maxSNR', 100)
                mask = (
                        (data['precision'] > min_loc) &
                        (data['precision'] < max_loc) &
                        (data['snr'] > min_snr) &
                        (data['snr'] < max_snr)
                )
            elif thresh_loc:
                min_loc = fig_params.get('minLoc', 0) / 1000
                max_loc = fig_params.get('maxLoc', 1000) / 1000
                mask = (
                        (data['precision'] > min_loc) &
                        (data['precision'] < max_loc)
                )
            elif thresh_snr:
                min_snr = fig_params.get('minSNR', 0)
                max_snr = fig_params.get('maxSNR', 100)
                mask = (
                        (data['snr'] > min_snr) &
                        (data['snr'] < max_snr)
                )

            for idx in range(len(load_flags)):
                if idx < len(var_names) and var_names[idx] in data:
                    data[var_names[idx]] = data[var_names[idx]][mask]

        return data

    def _apply_density_filter(self, data, fig_params, load_flags, return_flags, var_names):
        """应用密度过滤"""
        if fig_params.get('isThreshDensity', False) or return_flags[11]:
            cluster_params = ['clusterScore', 'clusterRadius', 'clusterWeights']

            if not all(k in fig_params for k in cluster_params):
                data, tree = self.density_based_clustering(data)
                del tree  # 手动释放内存

                fig_params.update({
                    'clusterScore': data.get('clusterScore'),
                    'clusterRadius': data.get('clusterRadius'),
                    'clusterWeights': data.get('clusterWeights'),
                    'cluster': data.get('cluster')
                })
            else:
                # 应用已有聚类结果
                start = fig_params.get('startPnt', 0)
                elements = fig_params.get('elements', np.inf)
                cluster = fig_params['cluster']

                if np.isinf(elements):
                    good = slice(start, None)
                else:
                    good = slice(start, start + elements)

                cluster_mask = cluster[good] > 0 if fig_params.get('clusterMode', 1) == 1 else cluster[good] == 0

                for idx in range(len(load_flags)):
                    if idx < len(var_names) and var_names[idx] in data:
                        data[var_names[idx]] = data[var_names[idx]][cluster_mask]

        return data

    def _cleanup_fields(self, data, return_flags, var_names):
        """清理不需要的字段"""
        for idx, flag in enumerate(return_flags):
            if not flag and idx < len(var_names):
                data.pop(var_names[idx], None)
        return data


    def density_based_clustering(self, data):
        """密度聚类实现"""
        # 简化版DBSCAN实现
        points = np.vstack([data['ctrsX'], data['ctrsY']]).T
        tree = cKDTree(points)

        # 这里需要实现具体的聚类算法
        # 示例: 使用固定半径搜索
        clusters = np.zeros(len(points))
        cluster_id = 1

        for i in range(len(points)):
            if clusters[i] == 0:
                neighbors = tree.query_ball_point(points[i], 0.1)  # 示例半径
                if len(neighbors) > 5:  # 示例最小点数
                    clusters[neighbors] = cluster_id
                    cluster_id += 1

        data['cluster'] = clusters
        return data, tree

    def render_image_data(self, data, roi):
        """Placeholder for the rendering logic."""
        # For demonstration, simply return the data as is
        return data  # Replace this with actual rendering logic

    def apply_colormap(self, image, width):
        """Apply colormap to the rendered image."""
        # Implement colormap application logic here
        return image  # Replace this with actual colormap logic

    def build_movie(self):
        """Build a movie based on the current settings."""
        preview_fig = tk.Toplevel(self.master)
        preview_fig.title("Movie Preview")

        # Ask user to accept movie settings
        answer = messagebox.askyesno("Question", "Accept movie settings?")
        if not answer:
            return

        movie_name = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI files", "*.avi")])
        if not movie_name:
            return

        h_progressbar = tk.Toplevel(self.master)
        h_progressbar.title("Progress")

        # Use grid for layout
        tk.Label(h_progressbar, text="Generating Movie...").grid(row=0, column=0, padx=10, pady=10)

        progress = ttk.Progressbar(h_progressbar, length=200, mode='determinate')
        progress.grid(row=1, column=0, padx=10, pady=10)

        # Placeholder for movie creation logic
        for i in range(100):  # Dummy loop for movie frames
            # Simulate frame rendering
            self.master.update_idletasks()
            progress['value'] = i + 1

        h_progressbar.destroy()
        messagebox.showinfo("Info", "Movie generation complete.")

    ####################################max_proj和rve_proj投影
    def max_projection(self):
        """执行最大强度投影"""
        self.perform_projection(projection_type='max')

    def mean_projection(self):
        """执行平均强度投影"""
        self.perform_projection(projection_type='mean')

    def perform_projection(self, projection_type):
        """执行投影的核心逻辑"""
        # 公共初始化部分
        fig = self.master
        roi = self.get_roi()

        # # 直接使用ROI值来定义区域的边界
        roi_y_start, roi_x_start, roi_height, roi_width = roi
        roi_y_end = roi_y_start + roi_height
        roi_x_end = roi_x_start + roi_width


        # 处理 pathname 和 filename 的元组/列表包装
        pathname = self.image_bin['pathname']
        filename = self.image_bin['filename']

        # 增强类型检查和处理
        def safe_unwrap(value):
            if isinstance(value, (tuple, list)):
                if len(value) == 0:
                    messagebox.showerror("Error", "Empty path/filename")
                    return None
                return value[0]
            return str(value)  # 确保是字符串

        pathname = safe_unwrap(pathname)
        filename = safe_unwrap(filename)

        if pathname is None or filename is None:
            return  # 已显示错误信息

        if isinstance(pathname, (tuple, list)):
            pathname = pathname[0]  # 取第一个元素
        if isinstance(filename, (tuple, list)):
            filename = filename[0]  # 取第一个元素

        # 检查 pathname 是否是文件路径
        if os.path.isfile(pathname):
            # 如果 pathname 已经是文件路径，直接使用
            image_path = pathname
        else:
            # 否则，正常拼接路径
            image_path = os.path.join(pathname, filename)

        # 检查文件是否存在
        if not os.path.isfile(image_path):
            messagebox.showerror("Error", f"Image file not found: {image_path}")
            return

        # 创建进度窗口
        progress_window = tk.Toplevel(self.master)
        progress_window.title("Processing")
        progress_bar = ttk.Progressbar(progress_window, mode='determinate')
        progress_bar.grid(row=0, column=0, padx=10, pady=10)

        try:
            # 公共处理逻辑
            if self.image_bin['is_superstack']:
                I = self.process_superstack(image_path, (roi_y_start, roi_y_end, roi_x_start, roi_x_end), progress_bar,
                                            projection_type)
            else:
                I = self.process_single_stack(image_path, (roi_y_start, roi_y_end, roi_x_start, roi_x_end),
                                              progress_bar, projection_type)

            # 检查 I 是否为 None
            if I is None:
                messagebox.showerror("Error", "Projection failed: No image data returned.")
                return

            # 归一化处理
            I = self.normalize_image(I, projection_type)

            # 创建结果窗口
            self.create_projection_window(I, projection_type)

        except Exception as e:
            messagebox.showerror("Error", f"Projection failed: {str(e)}")
        finally:
            progress_window.destroy()

    def get_roi(self):
        """获取ROI区域"""
        if self.image_bin['h_roi'] is None:
            return [0, 0, self.image_bin['height'], self.image_bin['width']]
        roi = np.ceil(self.image_bin['roi']).astype(int)
        return roi.tolist() ##确保返回的列表

    def process_single_stack(self, image_path, roi_bounds, progress_bar, proj_type):
        """处理单个堆栈"""
        try:
            print('image_path_verify:', image_path)
            try:
                img = Image.open(image_path)
                img.verify()  # 验证图像完整性
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {str(e)}")
                return None

            with Image.open(image_path) as img:
                # 检查图像是否有多帧
                if not hasattr(img, 'n_frames'):
                    messagebox.showerror("Error", "Image does not support multiple frames.")
                    return None

                # 检查ROI区域是否合法
                width, height = img.size
                roi_y_start, roi_y_end, roi_x_start, roi_x_end = roi_bounds
                roi_height = roi_y_end - roi_y_start
                roi_width = roi_x_end - roi_x_start

                if (roi_x_end > width) or (roi_y_end > height):
                    messagebox.showerror("Error", "ROI exceeds image dimensions")
                    return None

                frames = img.n_frames
                accumulator = None

                for frame_idx in range(frames):
                    img.seek(frame_idx)
                    # 检查帧是否有效
                    if img.format is None or img.mode is None:
                        messagebox.showerror("Error", f"Invalid frame {frame_idx} in image.")
                        return None

                    # 使用普通索引提取 ROI 区域
                    frame_data = np.array(img.crop((roi_x_start, roi_y_start, roi_x_end, roi_y_end)))

                    if proj_type == 'max':
                        accumulator = frame_data if frame_idx == 0 else np.maximum(accumulator, frame_data)
                    elif proj_type == 'mean':
                        if frame_idx == 0:
                            accumulator = frame_data.astype(np.float64)
                        else:
                            accumulator = (accumulator + frame_data) / 2

                    # 更新进度条
                    progress_bar['value'] = (frame_idx + 1) / frames * 100
                    progress_bar.update_idletasks()

                # if proj_type == 'mean':
                #     accumulator /= frames

            return accumulator

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process single stack: {str(e)}")
            return None

    def process_superstack(self, image_paths, region, progress_bar, proj_type):
        """处理超栈"""
        try:
            total_frames = sum(self.image_bin['stack_size'])
            frame_count = 0
            accumulator = None

            for stack_idx, path in enumerate(image_paths):
                with Image.open(path) as img:
                    # 检查图像是否有多帧
                    if not hasattr(img, 'n_frames'):
                        messagebox.showerror("Error", "Image does not support multiple frames.")
                        return None

                    for frame_idx in range(img.n_frames):
                        img.seek(frame_idx)
                        # 检查帧是否有效
                        if img.format is None or img.mode is None:
                            messagebox.showerror("Error", f"Invalid frame {frame_idx} in image.")
                            return None

                        frame_data = np.array(img.crop(region))

                        # 使用普通索引替代切片
                        if proj_type == 'max':
                            accumulator = frame_data if accumulator is None else np.maximum(accumulator, frame_data)
                        elif proj_type == 'mean':
                            if accumulator is None:
                                accumulator = np.zeros_like(frame_data, dtype=np.float64)
                            else:
                                accumulator = (accumulator + frame_data) / 2


                        # 更新进度条
                        frame_count += 1
                        progress_bar['value'] = frame_count / total_frames * 100
                        progress_bar.update_idletasks()

            return accumulator

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process superstack: {str(e)}")
            return None

    def normalize_image(self, image, proj_type):
        """根据投影类型归一化图像"""
        if proj_type == 'max':
            return self.normalize_8bit(image)
        elif proj_type == 'mean':
            return self.normalize_8bit(image.astype(np.float32))

    def normalize_8bit(self, data):
        # """标准化到0-255范围"""
        data_min = np.min(data)
        data_max = np.max(data)
        return ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)


    def create_projection_window(self, image, proj_type):
        """创建投影结果窗口"""
        title_map = {
            'max': 'Max Projection',
            'mean': 'Mean Projection'
        }

        window = tk.Toplevel(self.master)
        window.title(f"{title_map[proj_type]} - {self.image_bin['filename']}")
        window.geometry('320x320')

        # 显示图像
        img_tk = ImageTk.PhotoImage(Image.fromarray(image))
        label = tk.Label(window, image=img_tk)
        label.image = img_tk
        label.grid(row=0, column=0, sticky='nsew')

        # 工具栏
        toolbar = tk.Frame(window)
        toolbar.grid(row=1, column=0, sticky='ew')

        tk.Button(toolbar, text="Save",
                  command=lambda: self.save_projection(image)).grid(row=0, column=0)

        # 布局配置
        window.grid_rowconfigure(0, weight=1)
        window.grid_columnconfigure(0, weight=1)
        toolbar.grid_columnconfigure(0, weight=1)

    def save_projection(self, image):
        """保存投影结果为 PNG 图片，分辨率为 300 DPI"""
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")]
        )
        if path:
            # 将 numpy 数组转换为 PIL Image 对象
            pil_image = Image.fromarray(image)

            # 保存图像，指定格式为 PNG 并设置分辨率为 300 DPI
            pil_image.save(path, format='PNG', dpi=(300, 300))

    def save_image(self, I):
        """Save the processed image."""
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            Image.fromarray(I).save(file_path)

    def scatter_plot(self):
        """Open scatter plot configuration window."""
        self.encode_type = ['x-coordinate [px]', 'y-coordinate [px]', 'signal amplitude [counts]',
                       'noise amplitude [counts]', 'background level [counts]',
                       'psf radius [px]', 'detection time [frames]', '# collected photons',
                       'localization precision [micron]', 'signal to noise ratio',
                       'signal to background ratio', 'cluster', 'z-coordinate [px]', 'channel']

        menu_fig = tk.Toplevel(self.master)
        menu_fig.title("Encoding Style")
        menu_fig.geometry("300x300")

        # 我们需要设置 5 组控件：1st dim, 2nd dim, 3rd dim, expansion, color
        dims = ['1st dim:', '2nd dim:', '3rd dim:', 'expansion:', 'color:']
        # MATLAB 中默认的 1-based 值：1,2,7,9,7，对应 Python 0-based：0,1,6,8,6
        default_indices = [0, 1, 6, 8, 6]
        controls = {}

        for i, label_text in enumerate(dims):
            # 标签
            lbl = tk.Label(menu_fig, text=label_text, font=("Helvetica", 12))
            lbl.grid(row=i, column=0, padx=5, pady=5, sticky='w')
            # 下拉菜单：选项为 0 到 len(encodeType)-1（显示数字，实际意义由 encodeType 决定）
            var = tk.IntVar(value=default_indices[i])
            opt = tk.OptionMenu(menu_fig, var, *range(len(self.encode_type)))
            opt.config(font=("Helvetica", 12))
            opt.grid(row=i, column=1, padx=5, pady=5, sticky='ew')
            # 权重编辑框
            entry = tk.Entry(menu_fig, font=("Helvetica", 12))
            entry.insert(0, "1")
            entry.grid(row=i, column=2, padx=5, pady=5)
            controls[f'dim{i + 1}'] = (var, entry)

        # 接受设置按钮，点击后调用 build_scatter_plot 回调
        btn = tk.Button(menu_fig, text="Accept Settings", font=("Helvetica", 12),
                        command=lambda: self.build_scatter_plot(controls, menu_fig))
        btn.grid(row=6, column=0, columnspan=3, pady=10)

    def build_scatter_plot(self, controls, settings_window):
        """
        从设置窗口中读取用户选择，然后生成散点图窗口，并添加保存按钮
        """
        # 读取维度和权重
        selected, weights = [], []
        for i in range(1, 6):
            var, entry = controls[f'dim{i}']
            selected.append(var.get())
            try:
                w = float(entry.get())
            except ValueError:
                w = 1.0
            weights.append(w)

        # 关掉设置窗口
        settings_window.destroy()

        # 对应的数据字段
        varNames = ['ctrsX', 'ctrsY', 'signal', 'noise', 'offset', 'radius',
                    'photons', 'precision', 'snr', 'sbr', 'class', 'ctrsZ', 'channel']

        # 准备画布
        fig_new = tk.Toplevel(self.master)
        fig_new.title("2D Scatter")
        fig_new.geometry("550x550")

        fig = plt.Figure(figsize=(5.5, 5.5))  # 正方形的 Figure
        ax = fig.add_subplot(111)

        # 取出 x,y,size,color
        x = np.array(self.image_bin[varNames[selected[0]]])
        y = np.array(self.image_bin[varNames[selected[1]]])
        # size = (self.image_bin[varNames[selected[2]]] * weights[2]
        #         if selected[2] < len(varNames) else 20)
        if selected[2] < len(varNames):
            raw_size = np.array(self.image_bin[varNames[selected[2]]], dtype=np.float64) * float(weights[2])

            # —— 可选对数压缩：对 signal / noise 压缩动态范围；offset/其他保持线性 —— #
            name_for_size = varNames[selected[2]]
            use_log = name_for_size in ('signal', 'noise')
            if use_log:
                # 仅对正值做 log1p，非正值保持原样，避免 nan/inf
                raw_size = np.where(raw_size > 0, np.log1p(raw_size), raw_size)

            # —— 鲁棒百分位裁剪（避免极端值把点变得巨大） —— #
            finite = np.isfinite(raw_size)
            if np.any(finite):
                v = raw_size[finite]
                # 2–98 分位，可按需改，比如 5–95
                lo = np.percentile(v, 2)
                hi = np.percentile(v, 98)
                # 兜底：避免 hi==lo
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo = np.nanmin(v)
                    hi = np.nanmax(v)
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    # 全是常数或无效值：给固定大小
                    size = np.full_like(raw_size, 12.0, dtype=np.float64)
                else:
                    v_clipped = np.clip(raw_size, lo, hi)
                    norm = (v_clipped - lo) / (hi - lo)  # 归一化到 [0,1]

                    # —— 映射到合理的 marker 面积区间（points^2） —— #
                    S_MIN, S_MAX = 6.0, 48.0  # 点还大就把 48 改小，比如 36
                    size = S_MIN + norm * (S_MAX - S_MIN)
            else:
                size = np.full_like(raw_size, 12.0, dtype=np.float64)
        else:
            size = 20.0
        color = (self.image_bin[varNames[selected[3]]] * weights[3]
                 if selected[3] < len(varNames) else None)

        scatter = ax.scatter(x, y, s=size, c=color, cmap='viridis', alpha=0.7)
        ax.set_xlabel(self.encode_type[selected[0]])
        ax.set_ylabel(self.encode_type[selected[1]])
        # ax.set_aspect('equal', 'box')
        ax.set_aspect('auto')
        if color is not None:
            fig.colorbar(scatter, ax=ax, label=self.encode_type[selected[3]])

        # 将 Figure 嵌入 Tkinter
        canvas = FigureCanvasTkAgg(fig, master=fig_new)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()

        # —— 修改开始：用 grid() 而不是 pack() 来布局 ——#
        # 1) 先在 fig_new 中，把 Canvas 放到 row=0, column=0
        canvas_widget.grid(row=0, column=0, sticky='nsew')

        # 2) 让 fig_new 的第 0 行和第 0 列拉伸
        fig_new.grid_rowconfigure(0, weight=1)
        fig_new.grid_columnconfigure(0, weight=1)

        # —— 新增：保存按钮 —— #
        def on_save():
            # —— 基于已加载的 TXT 生成默认文件名 —— #
            # 假设 load_slimfast 已经把 txt 路径存在 self.image_bin['pathname']
            try:
                src_txt = self.image_bin.get('pathname', '')
            except Exception:
                src_txt = ''
            # 默认目录与基名
            if src_txt and os.path.isfile(src_txt):
                init_dir = os.path.dirname(src_txt)
                base = os.path.splitext(os.path.basename(src_txt))[0]  # 去掉 .txt
            else:
                init_dir = os.getcwd()
                base = "scatter"

            default_name = f"{base}_scatter_plot.png"

            # —— 弹出保存对话框（带默认目录与默认文件名）—— #
            filepath = filedialog.asksaveasfilename(
                defaultextension='.png',
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                title="Save scatter figure",
                initialdir=init_dir,
                initialfile=default_name
            )
            if not filepath:
                return

            # —— 统一后缀规则（若用户删了后缀，这里补回去）—— #
            root, ext = os.path.splitext(filepath)
            # 确保扩展名
            if ext.lower() not in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'):
                ext = '.png'
            # 确保带有 _scatter_plot 后缀
            if not root.endswith('_scatter_plot'):
                root = f"{root}_scatter_plot"
            save_path = f"{root}{ext}"

            # —— 覆盖确认 —— #
            if os.path.exists(save_path):
                overwrite = messagebox.askyesno(
                    "文件已存在",
                    f"文件 {os.path.basename(save_path)} 已存在。\n是否覆盖？"
                )
                if not overwrite:
                    return

            # —— 保存（保持正方形输出）—— #
            fig.set_size_inches(6, 6)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            tk.messagebox.showinfo("Saved", f"Figure saved to:\n{save_path}")

        btn_save = tk.Button(fig_new, text="Save", command=on_save)
        btn_save.grid(row=1, column=0,  pady=8)

        # 4) 保证第 1 行（按钮行）不拉伸高度
        fig_new.grid_rowconfigure(1, weight=0)

        fig_new.deiconify()

    def loc_prec(self):
        # 1) 创建新的顶层窗口
        fig_new = tk.Toplevel(self.master)
        fig_new.title("Localization Precision")
        fig_new.geometry("600x500")

        # 2) 计算定位精度数据
        psf_std = self.image_bin['psfStd'] * self.image_bin['px_size']
        px_size = self.image_bin['px_size']
        photons = self.image_bin['photons']* (np.sqrt(np.pi) * 1.03)
        noise = self.image_bin['noise']/self.image_bin['cntsPerPhoton']
        precision_data = np.sqrt(
            (psf_std ** 2 + (px_size ** 2 / 12)) / photons +
            (8 * np.pi * psf_std ** 4 * noise ** 2) / (px_size ** 2 * photons ** 2)
        )

        # 1) 创建 Figure & Axes
        fig = plt.Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        self.build_histogram(precision_data * 1000, 'nm', ax)
        ax.set_xlabel('Localization [nm]')
        ax.set_title(f"Localization precision: {self.image_bin['total_particles']} rendered")
        # ax.axis('off')

        # 2) 嵌入到 Tkinter 窗口
        canvas = FigureCanvasTkAgg(fig, master=fig_new)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky='nsew')

        # 3) 让第 0 行/第 0 列可拉伸
        fig_new.grid_rowconfigure(0, weight=1)
        fig_new.grid_columnconfigure(0, weight=1)

        # 4) 保存按钮，放在第 1 行
        def save_as_png():
            filetypes = [("PNG files", "*.png")]
            save_path = filedialog.asksaveasfilename(
                parent=fig_new,
                title="Save",
                defaultextension=".png",
                filetypes=filetypes,
                initialfile="precision_hist.png"
            )
            if not save_path:
                return
            try:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Saved", f"Histogram saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file:\n{e}")

        save_btn = tk.Button(fig_new, text="Save", command=save_as_png)
        save_btn.grid(row=1, column=0, pady=5)

        # 5) 不让第 1 行拉伸
        fig_new.grid_rowconfigure(1, weight=0)
        fig_new.grid_columnconfigure(0, weight=1)

    def snr_hist(self):
        """Open a new window for SNR histogram."""
        # 1) 创建新的顶层窗口
        fig_new = tk.Toplevel(self.master)
        fig_new.title("Signal to Noise Ratio")
        fig_new.geometry("600x500")

        # 2) 获取 SNR 数据
        snr_data = self.image_bin['signal'] / self.image_bin['noise']

        # 3) 创建 Matplotlib Figure 并绘制直方图
        fig = plt.Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        self.build_histogram(snr_data, '', ax)
        ax.set_xlabel('Signal to Noise Ratio')
        ax.set_title(f"SNR Histogram: {self.image_bin.get('total_particles', 0)} rendered")

        # 4) 嵌入到 Tkinter 弹窗中
        canvas = FigureCanvasTkAgg(fig, master=fig_new)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        # 5) 添加“Save as PNG”按钮
        def save_as_png():
            filetypes = [("PNG files", "*.png")]
            save_path = filedialog.asksaveasfilename(
                parent=fig_new,
                title="Save SNR histogram as...",
                defaultextension=".png",
                filetypes=filetypes,
                initialfile="snr_hist.png"
            )
            if not save_path:
                return
            try:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Saved", f"SNR histogram saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file:\n{e}")

        save_btn = tk.Button(fig_new, text="Save", command=save_as_png)
        save_btn.pack(pady=5)

        # 6) 让窗口自适应调整大小
        fig_new.grid_rowconfigure(0, weight=1)
        fig_new.grid_columnconfigure(0, weight=1)


    def detection_trace(self):
        """Open a new window for detection trace visualization."""
        fig_new = tk.Toplevel(self.master)
        fig_new.title("Detection Trace")
        fig_new.geometry("600x500")

        ctrsN = np.array(self.image_bin['ctrsN'])

        # 1) 创建 Matplotlib Figure 和两个子图
        fig = plt.Figure(figsize=(8, 6))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        # 上半部分：检测数 + LOWESS 平滑
        ax1.plot(ctrsN, label='Particle Detections')
        x_vals = np.arange(len(ctrsN))
        lowess_result = sm.nonparametric.lowess(ctrsN, x_vals, frac=0.25)
        ax1.plot(lowess_result[:, 0], lowess_result[:, 1],
                 color='red', linewidth=2, label='smooth curve')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Particle Detections')
        ax1.legend()
        ax1.autoscale()

        # 下半部分：检测数直方图
        self.build_histogram(ctrsN, '', ax2)
        ax2.set_xlabel('Particle Detections number')
        ax2.set_ylabel('Frequency')

        fig.subplots_adjust(hspace=0.5)

        # 2) 在 fig_new 中创建一个 Frame 专门放 Canvas
        image_frame = tk.Frame(fig_new)
        image_frame.grid(row=0, column=0, sticky='nsew')

        # 3) 嵌入 FigureCanvasTkAgg 到 image_frame
        canvas = FigureCanvasTkAgg(fig, master=image_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky='nsew')

        # 4) 用 grid 布局把“图像区”占第 0 行、“按钮”占第 1 行
        #    并且让第 0 行可拉伸，第 1 行保持固定高度
        fig_new.grid_rowconfigure(0, weight=1)
        fig_new.grid_columnconfigure(0, weight=1)

        image_frame.grid_rowconfigure(0, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)

        # 5) 在第 1 行添加“Save as PNG”按钮
        def save_as_png():
            filetypes = [("PNG files", "*.png")]
            save_path = filedialog.asksaveasfilename(
                parent=fig_new,
                title="Save detection trace as...",
                defaultextension=".png",
                filetypes=filetypes,
                initialfile="detection_trace.png"
            )
            if not save_path:
                return
            try:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Saved", f"Detection trace saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file:\n{e}")

        save_btn = tk.Button(fig_new, text="Save", command=save_as_png)
        save_btn.grid(row=1, column=0, pady=5)

        # 确保第 1 行（按钮行）不被拉伸
        fig_new.grid_rowconfigure(1, weight=0)
        fig_new.grid_columnconfigure(0, weight=1)


    def interaction_study(self):
        """Placeholder for interaction study functionality."""
        messagebox.showinfo("Information", "Interaction Study function is not implemented.")

    def coloc_filter(self):
        """Open co-localization filter process."""
        # 获取通道参数
        imCh = self.image_bin['imCh']
        nImCh = self.image_bin['nImCh']
        roi=self.image_bin['roi']

        # Load data for each channel
        data = []
        for ch in range(nImCh):
            # 构造每个通道的fig_params
            fig_params = {
                'pathname': self.image_bin[f'ch{ch}_pathname'],
                'filename': self.image_bin[f'ch{ch}_filename'],
                'elements': self.image_bin[f'ch{ch}_elements'],
                'startPnt': self.image_bin[f'ch{ch}_startPnt'],
                'cntsPerPhoton': self.image_bin.get('cntsPerPhoton', 1.0),
                'pxSize': self.image_bin.get('pxSize', 0.1),
                'isThreshDensity': self.image_bin.get('isThreshDensity', False),
                'isThreshSNR': self.image_bin.get('isThreshSNR', False),
                'isThreshLocPrec': self.image_bin.get('isThreshLocPrec', False),
                'hROI': self.image_bin.get('hROI', None)
            }

            # 设置通道特定的return_flags
            return_val = [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]  # 示例值

            # 调用preprocess_data
            channel_data = self.preprocess_data(
                fig_params=fig_params,
                return_flags=return_val,
                roi=roi
            )
            data.append(channel_data)

        # Simulate processing for co-localization
        hProgressbar = tk.Toplevel(self.master)
        hProgressbar.title("Processing")

        progress = tk.StringVar(value="Generating Images...")
        tk.Label(hProgressbar, textvariable=progress).grid(row=0, column=0, padx=10, pady=10)

        nColoc = []
        for frame in range(1, 11):  # Example frame range
            idx1, idx2 = self.filter_data(data, frame)
            nColoc.append(len(idx1))
            self.generate_images(data, idx1, idx2, frame)

            # Update progress
            progress.set(f"Generating Images... Frame {frame}/10")
            hProgressbar.update_idletasks()

        hProgressbar.destroy()


    def filter_data(self, data, frame):
        """Filter data based on the current frame."""
        good1 = (data[0]['frame'] == frame)
        good2 = (data[1]['frame'] == frame)

        y_diff = np.subtract.outer(data[0]['ctrsY'][good1], data[1]['ctrsY'][good2]) ** 2
        x_diff = np.subtract.outer(data[0]['ctrsX'][good1], data[1]['ctrsX'][good2]) ** 2
        d = np.sqrt(y_diff + x_diff)

        thresh = 2  # Max allowed distance [px]
        idx1, idx2 = np.where(d <= thresh)
        return idx1, idx2

    def generate_images(self, data, idx1, idx2, frame):
        """Generate images based on co-localization indices."""
        # Placeholder for image generation logic
        pass


    def particle_image_correlation(self):
        fig = self.master  # Current figure reference

        # Set correlation sampling rate
        sampling_rate = 5  # [px^-1]
        d_max = 15  # [px]

        roi = self.image_bin['roi']
        roi = [
            max(roi[0], d_max),
            max(roi[1], d_max),
            min(roi[2], self.image_bin['width'] - d_max),
            min(roi[3], self.image_bin['height'] - d_max)
        ]

        n_im = len(self.image_bin['ctrsN'])
        n_pnts = len(self.image_bin['ctrsX'])

        # Generate 2D-projected tree
        pnt_list = np.column_stack((self.image_bin['ctrsX'], self.image_bin['ctrsY']))
        tree = cKDTree(pnt_list)

        # Average particle count for increasing dt
        pnts_per_frame = np.bincount(self.image_bin['frame'])
        mu_pnts = np.zeros(n_im - 1)

        for dt in range(1, n_im):
            mu_pnts[dt - 1] = np.mean(pnts_per_frame[dt:n_im])

        C_cum = np.zeros((d_max, n_im - 1))

        for d in range(1, int(np.ceil(d_max * sampling_rate)) + 1):
            # Reset count matrix
            cnts = np.zeros((n_pnts, n_im - 1))

            for pnt in range(n_pnts):
                idx = tree.query_ball_point(pnt_list[pnt], d / sampling_rate)
                dt = self.image_bin['frame'][idx] - self.image_bin['frame'][pnt]
                good = dt > 0

                if np.any(good):
                    # Overlap integral for image pairs separated by dt
                    cnts[pnt, :] = np.bincount(dt[good], minlength=n_im - 1)

            # Average over n*dt and normalize to average particle count for n*dt
            C_cum[d - 1, :] = np.sum(cnts, axis=0) / mu_pnts

        # Calculate spatial correction term
        z = []
        x = []
        y = []
        for dt in range(1, n_im):
            lin_start = np.argmax(C_cum[:, dt - 1] > 1)
            z.extend(C_cum[lin_start:, dt - 1])
            y.extend(((lin_start + np.arange(d_max)) / sampling_rate).tolist())
            x.extend([dt] * (d_max - lin_start))

        # Perform regression
        X = np.column_stack((np.ones(len(x)), x, np.array(y) ** 2))
        b, _, _, _ = np.linalg.lstsq(X, z, rcond=None)

        # Generate fitted surface
        X_fit, Y_fit = np.meshgrid(np.arange(1, n_im),
                                   (np.arange(1, int(np.ceil(d_max * sampling_rate))) / sampling_rate) ** 2)
        Z_hat = b[1] * X_fit + b[2] * Y_fit

        # Adjust C_cum
        C_cum -= Z_hat

        # Plot results
        self.plot_results(X_fit, Y_fit, C_cum, z, x, y)

    def plot_results(self, X_fit, Y_fit, C_cum, z, x, y):
        """Plot the results of the correlation analysis."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(X_fit, Y_fit, C_cum, edgecolor='flat', alpha=0.9)
        ax.scatter(x, np.array(y) ** 2, z, color='m', marker='.')
        ax.set_xlabel('dt')
        ax.set_ylabel('Distance (px)')
        ax.set_zlabel('Cumulative Correlation')

        plt.show()

    def localization(self):
        """Run the localization process on the selected image."""
        # Construct the full image path
        self.image_bin['image_name'] = f"{self.image_bin['pathname']}/{self.image_bin['filename']}"

        # Select output filename for localization results
        if isinstance(self.image_bin['filename'], list):
            cnt = 1
            while all(self.image_bin['filename'][0].startswith(fn) for fn in self.image_bin['filename']):
                cnt += 1
            self.image_bin['filename'] = self.image_bin['filename'][0][:cnt - 1]
        else:
            self.image_bin['filename'] = self.image_bin['filename'][:-4]

        # Check ROI
        if not self.image_bin['roi']:
            self.image_bin['roi'] = [0, 0, self.image_bin['width'], self.image_bin['height']]

        # Run localization logic (this needs to be implemented)
        self.run_localization()

    def run_localization(self):
        """执行定位算法，并保存结果到指定的 output_filename."""
        start_time = time.time()
        loc_parallel = self.image_bin.get('locParallel', False)

        # 设置数据容器，用于并行或线性处理
        data = []
        ctrsN = []
        file=self.image_bin['pathname']
        self.image_bin['imageName']=file
        info=imread(file)
        end_pnt=info.shape[0]

        # 获取去掉扩展名的文件名
        filename_without_ext = os.path.splitext(self.image_bin['filename'])[0]


        # 拼接完整路径
        output_path = os.path.join(self.image_bin['pathname'], filename_without_ext)

        if loc_parallel:
            # 并行处理模式
            num_frames = end_pnt - self.image_bin['locStart'] + 1
            with ProcessPoolExecutor(max_workers=self.image_bin['nCores']) as executor:
                futures = [
                    executor.submit(self.localize_particles_Loc_all, self.image_bin, frame, end_pnt, None)
                    for frame in range(self.image_bin['locStart'], end_pnt + 1)
                ]
                for future in futures:
                    result = future.result()
                    data.append(result[0])  # 假设结果的第一个元素是数据
                    ctrsN.append(result[3])  # 假设结果的第4个元素是粒子检测信息

        else:
            # 线性处理模式
            for frame in range(self.image_bin['locStart'] - 1, end_pnt):
                result = self.localize_particles_Loc_all(self.image_bin, frame, end_pnt, info,output_path)
                data.append(result[0])
                ctrsN.append(result[3])

        # elapsed_time = time.time() - start_time
        # print(f"Localization took {elapsed_time:.2f} seconds.")
        messagebox.showinfo("Success", "Localization completed and results saved.")

    def localize_particles_Loc_all(self, image_bin, start_pnt, end_pnt, info,output_path):
        """Run localization on the specified image with the given parameters."""
        imagename = image_bin['imageName']
        region = [
            [image_bin['roi'][1], image_bin['roi'][1] + image_bin['roi'][3] - 1],
            [image_bin['roi'][0], image_bin['roi'][0] + image_bin['roi'][2] - 1]
        ]

        # Initialize optim based on image_bin parameters
        optim = [
            image_bin['maxOptimIter'],
            image_bin['termTol'],
            image_bin['isRadiusTol'],
            image_bin['radiusTol'],
            image_bin['posTol']
        ]

        if image_bin['locParallel']:
            data = [None] * (end_pnt - start_pnt + 1)
            ctrsN = [None] * (end_pnt - start_pnt + 1)

            with ProcessPoolExecutor(max_workers=image_bin['nCores']) as executor:
                # 并行/线性处理前打印
                futures = []
                for frame in range(start_pnt, end_pnt + 1):
                    futures.append(executor.submit(self.detect_et_estime_part_1vue_deflt, imagename, frame, region))

                for i, future in enumerate(futures):
                    data[i], deflatedIm, binaryIm, ctrsN[i] = future.result()


            # 获取去掉扩展名的文件名
            filename_without_ext = os.path.splitext(image_bin['filename'])[0]
            # 拼接完整路径
            output_path = os.path.join(image_bin['pathname'], filename_without_ext)
            self.stream_to_disk(data, output_path)

        else:  # Linear computation
            ctrsN = []
            data = []
            for frame in range(start_pnt - 1, end_pnt):
                print(f"frame {frame}")
                f = tifffile.imread(imagename)
                I = f[frame, :, :]
                # I = imread(imagename)  # Load the image for the current frame
                data_frame, deflatedIm, binaryIm, n_ctrs = self.detect_et_estime_part_1vue_deflt(
                    I, image_bin['w2d'],
                    self.image_bin['psfStd'],
                    chi2.ppf(1 - 1 / 10 ** (image_bin['errorRate'] * -1), 1),
                    image_bin['dfltnLoops'],
                    image_bin['roi'][2],
                    image_bin['roi'][3],
                    image_bin['minInt'],
                    optim  # Use the initialized optim
                )
                # print(f"现在frame是：{frame}.现在的n_ctrs:{n_ctrs}")
                data.append(data_frame)
                ctrsN.append(n_ctrs)

            filename_without_ext = os.path.splitext(image_bin['filename'])[0]

            # 拼接完整路径
            output_path = os.path.join(image_bin['pathname'], filename_without_ext + '_locs.txt')
            self.stream_to_disk(data, output_path)
        return ctrsN


    def check_name_existence(filename, pathname):
        """
        Check if the file with the given filename already exists in the folder.
        If it exists, modify the filename by appending an index.
        Returns (filename, pathname, abort).
        """
        abort = False
        full_path = os.path.join(pathname, filename + '.mat')
        counter = 1
        new_filename = filename
        while os.path.exists(full_path):
            new_filename = f"{filename}_{counter}"
            full_path = os.path.join(pathname, new_filename + '.mat')
            counter += 1
        return new_filename, pathname, abort


    def show_image(self, parent, image_data):
        """在指定父容器显示图像"""
        img = Image.fromarray(image_data.astype('uint8'))
        photo = ImageTk.PhotoImage(img)

        canvas = Canvas(parent)
        canvas.create_image(0, 0, anchor='nw', image=photo)
        canvas.image = photo
        canvas.grid(row=0, column=0, sticky="nsew")

    def draw_tracks(self, parent, tracks, scale_factor):
        """在画布上绘制轨迹"""
        canvas = parent.winfo_children()[0]
        for track in tracks:
            scaled_x = [x * scale_factor - 1 for x in track['x']]
            scaled_y = [y * scale_factor - 1 for y in track['y']]
            canvas.create_line(scaled_x, scaled_y, fill=track['color'], width=track['width'])

    # def load_data(self):
    #     """Load the data from a MAT file."""

#####################################################################################

    def batch_localization(self):
        """Display a window for batch localization settings."""
        menu_fig = tk.Toplevel(self.master)
        menu_fig.title("Batch Container")
        menu_fig.geometry("800x600")

        # Listbox for displaying selected stacks
        h_list = tk.Listbox(menu_fig, font=('Arial', 15), bg='white')
        h_list.grid(row=0, column=0, rowspan=10, sticky='nsew', padx=10, pady=10)

        # Configure grid weights
        menu_fig.grid_rowconfigure(0, weight=1)
        menu_fig.grid_columnconfigure(0, weight=1)

        # Buttons for adding, removing, and managing stacks
        button_frame = tk.Frame(menu_fig)
        button_frame.grid(row=0, column=1, sticky='n', padx=10, pady=10)

        # 预先定义一个统一的尺寸（以字符数为单位）
        BTN_WIDTH = 3  # 例如 6 个字符宽
        BTN_HEIGHT = 1  # 例如 2 行文本高

        tk.Button(
            button_frame,
            text="+",
            font=('Arial', 10),
            width=BTN_WIDTH,
            height=BTN_HEIGHT,
            command=lambda: self.add_stack_optics(h_list)
        ).grid(row=0, column=0, pady=5)

        tk.Button(
            button_frame,
            text="-",
            font=('Arial', 10),
            width=BTN_WIDTH,
            height=BTN_HEIGHT,
            command=lambda: self.del_stack(h_list)
        ).grid(row=1, column=0, pady=5)

        tk.Button(
            button_frame,
            text="UP",
            font=('Arial', 10),
            width=BTN_WIDTH,
            height=BTN_HEIGHT,
            command=lambda: self.move_up_stack(h_list)
        ).grid(row=2, column=0, pady=5)

        tk.Button(
            button_frame,
            text="DOWN",
            font=('Arial', 10),
            width=BTN_WIDTH,
            height=BTN_HEIGHT,
            command=lambda: self.move_down_stack(h_list)
        ).grid(row=3, column=0, pady=5)

        tk.Button(
            button_frame,
            text="START",
            font=('Arial', 10),
            width=BTN_WIDTH,
            height=BTN_HEIGHT,
            command=lambda: self.process_batch(h_list)
        ).grid(row=4, column=0, pady=5)

        # Panel for localization options
        h_loc_opt_panel = tk.Frame(menu_fig, relief=tk.RAISED, borderwidth=2)
        h_loc_opt_panel.grid(row=0, column=2, rowspan=10, sticky='nsew', padx=10, pady=10)

        # **保存引用，方便后面 add_stack_optics 拿到同一个 panel**
        self.h_loc_opt_panel = h_loc_opt_panel

        # 初始化存放子面板的字典
        self.optics_panels = {}  # MOD: 每次打开 batch_localization 都重新初始化

        tk.Label(h_loc_opt_panel, text="Localization", font=('Arial', 10, 'bold')).grid(row=0, column=0, pady=5)

        # Error Rate input
        tk.Label(h_loc_opt_panel, text="Error Rate [10^]:").grid(row=1, column=0, sticky='w')
        self.err_rate_entry = tk.Entry(h_loc_opt_panel)
        self.err_rate_entry.insert(0, "-6")
        self.err_rate_entry.grid(row=1, column=1, pady=5)

        # Detection Box Size input
        tk.Label(h_loc_opt_panel, text="Detection Box [px]:").grid(row=2, column=0, sticky='w')
        self.box_size_entry = tk.Entry(h_loc_opt_panel)
        self.box_size_entry.insert(0, "7")
        self.box_size_entry.grid(row=2, column=1, pady=5)

        # Deflation Loops input
        tk.Label(h_loc_opt_panel, text="Deflation Loops:").grid(row=3, column=0, sticky='w')
        self.deflation_loops_entry = tk.Entry(h_loc_opt_panel)
        self.deflation_loops_entry.insert(0, "0")
        self.deflation_loops_entry.grid(row=3, column=1, pady=5)

        # Intensity Threshold input
        tk.Label(h_loc_opt_panel, text="Intensity Thresh [cnts]:").grid(row=4, column=0, sticky='w')
        self.intensity_thres_entry = tk.Entry(h_loc_opt_panel)
        self.intensity_thres_entry.insert(0, "0")
        self.intensity_thres_entry.grid(row=4, column=1, pady=5)

        # Checkbox for using multiple cores
        self.use_parallel_var = tk.BooleanVar()
        tk.Checkbutton(h_loc_opt_panel, text="Use multiple cores", variable=self.use_parallel_var).grid(row=5, column=0,
                                                                                                   sticky='w')

        # Number of Cores selection
        tk.Label(h_loc_opt_panel, text="Number of Cores:").grid(row=6, column=0, sticky='w')
        self.n_cores_var = tk.StringVar(value='max')
        n_cores_menu = ttk.Combobox(h_loc_opt_panel, textvariable=self.n_cores_var, values=['max', '2', '3','4','5'])
        n_cores_menu.grid(row=6, column=1, pady=5)

        # Optimization settings
        tk.Label(h_loc_opt_panel, text="Optimization Settings", font=('Arial', 10, 'bold')).grid(row=7, column=0,
                                                                                                 pady=5)

        # Max # iterations input
        tk.Label(h_loc_opt_panel, text="Max. # iterations:").grid(row=8, column=0, sticky='w')
        self.max_iter_entry = tk.Entry(h_loc_opt_panel)
        self.max_iter_entry.insert(0, "50")
        self.max_iter_entry.grid(row=8, column=1, pady=5)

        # Termination tolerance input
        tk.Label(h_loc_opt_panel, text="Term. tol. [10^]:").grid(row=9, column=0, sticky='w')
        self.term_tol_entry = tk.Entry(h_loc_opt_panel)
        self.term_tol_entry.insert(0, "-2")
        self.term_tol_entry.grid(row=9, column=1, pady=5)

        # Radius tolerance checkbox
        self.radius_tol_var = tk.BooleanVar()
        tk.Checkbutton(h_loc_opt_panel, text="R0 tolerance [%]:", variable=self.radius_tol_var).grid(row=10, column=0,
                                                                                                sticky='w')

        # Radius tolerance input
        self.radius_tol_entry = tk.Entry(h_loc_opt_panel)
        self.radius_tol_entry.insert(0, "10")
        self.radius_tol_entry.grid(row=10, column=1, pady=5)

        # Max position refinement input
        tk.Label(h_loc_opt_panel, text="Max. pos. refinement [px]:").grid(row=11, column=0, sticky='w')
        self.pos_tol_entry = tk.Entry(h_loc_opt_panel)
        self.pos_tol_entry.insert(0, "1.5")
        self.pos_tol_entry.grid(row=11, column=1, pady=5)

        # 绑定 Listbox 选中事件，用于切换显示哪一个子面板
        h_list.bind("<<ListboxSelect>>", self.on_listbox_select)

    def add_stack_optics(self, h_list):
        """Add a new image stack to the list and populate optics controls
           into the existing h_loc_opt_panel (no new sub‐frames)."""
        # 1) 选择文件并更新 listbox（保持原逻辑不变）
        stack_names = filedialog.askopenfilenames(
            title='Select Image stacks',
            filetypes=[("TIF files", "*.tif")],
            initialdir=self.image_bin.get('search_path', '.')
        )
        if not stack_names:
            return

        # 更新搜索路径 & 列表内容
        self.image_bin['search_path'] = os.path.dirname(stack_names[0])
        current = list(h_list.get(0, tk.END))
        combined = current + list(stack_names)
        h_list.delete(0, tk.END)
        for fn in combined:
            h_list.insert(tk.END, fn)

        # —— 2) 为每个新加入的文件，创建一个子 Frame 并填充控件 ——#
        for fullpath in stack_names:
            # 如果已经创建过，就跳过
            if fullpath in self.optics_panels:
                continue

            # 新建一个子 Frame（对应 MATLAB 的 uipanel），放到 h_loc_opt_panel 内
            panel = tk.Frame(self.h_loc_opt_panel, borderwidth=1, relief='groove')
            # 默认先隐藏：先 grid 再 grid_forget
            panel.grid(row=12, column=0, columnspan=2,sticky='nsew')
            panel.grid_forget()

            # 保存到字典里，key=文件完整路径，value=panel
            self.optics_panels[fullpath] = panel

            # 在 panel 顶部插入小标题
            lbl_title = tk.Label(panel, text="Optics Settings", font=('Arial', 10, 'bold'))
            lbl_title.grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 2))

            # 从第 1 行开始依次创建控件
            row = 1

            # (1) Pixel Size (μm)
            tk.Label(panel, text='Pixel Size (μm):').grid(
                row=row, column=0, sticky='w', padx=5, pady=2
            )
            px_entry = tk.Entry(panel, width=10)
            px_entry.insert(0, '0.16')
            px_entry.grid(row=row, column=1, sticky='w', pady=2)
            # 当用户在此 Entry 中完成输入后自动 recalc PSF
            px_entry.bind("<FocusOut>", lambda e, p=panel: self.calc_psf(p))
            px_entry.bind("<Return>", lambda e, p=panel: self.calc_psf(p))
            panel.px_entry = px_entry
            row += 1

            # (2) Emission WL (nm)
            tk.Label(panel, text='Emission WL (nm):').grid(
                row=row, column=0, sticky='w', padx=5, pady=2
            )
            em_entry = tk.Entry(panel, width=10)
            em_entry.insert(0, '590')
            em_entry.grid(row=row, column=1, sticky='w', pady=2)
            em_entry.bind("<FocusOut>", lambda e, p=panel: self.calc_psf(p))
            em_entry.bind("<Return>", lambda e, p=panel: self.calc_psf(p))
            panel.em_entry = em_entry
            row += 1

            # (3) N.A.
            tk.Label(panel, text='N.A.:').grid(
                row=row, column=0, sticky='w', padx=5, pady=2
            )
            na_entry = tk.Entry(panel, width=10)
            na_entry.insert(0, '1.49')
            na_entry.grid(row=row, column=1, sticky='w', pady=2)
            na_entry.bind("<FocusOut>", lambda e, p=panel: self.calc_psf(p))
            na_entry.bind("<Return>", lambda e, p=panel: self.calc_psf(p))
            panel.na_entry = na_entry
            row += 1

            # (4) PSF Scaling
            tk.Label(panel, text='PSF Scaling:').grid(
                row=row, column=0, sticky='w', padx=5, pady=2
            )
            psf_scale_entry = tk.Entry(panel, width=10)
            psf_scale_entry.insert(0, '1.35')
            psf_scale_entry.grid(row=row, column=1, sticky='w', pady=2)
            psf_scale_entry.bind("<FocusOut>", lambda e, p=panel: self.calc_psf(p))
            psf_scale_entry.bind("<Return>", lambda e, p=panel: self.calc_psf(p))
            panel.psf_scale_entry = psf_scale_entry
            row += 1

            # (5) PSF Std [px]（只读）
            tk.Label(panel, text='PSF Std [px]:').grid(
                row=row, column=0, sticky='w', padx=5, pady=2
            )
            psf_std_entry = tk.Entry(panel, width=10, state='readonly')
            psf_std_entry.insert(0, '1.03')
            psf_std_entry.grid(row=row, column=1, sticky='w', pady=2)
            panel.psf_std_entry = psf_std_entry
            row += 1

            # (6) Counts/Photon
            tk.Label(panel, text='Counts/Photon:').grid(
                row=row, column=0, sticky='w', padx=5, pady=2
            )
            cnts_entry = tk.Entry(panel, width=10)
            cnts_entry.insert(0, '20.2')
            cnts_entry.grid(row=row, column=1, sticky='w', pady=2)
            panel.cnts_entry = cnts_entry
            row += 1

            # (7) Apply spatial correction（复选框）
            spatial_var = tk.BooleanVar(value=False)
            chk = tk.Checkbutton(
                panel,
                text='Apply spatial correction',
                variable=spatial_var
            )
            chk.grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
            panel.spatial_var = spatial_var
            panel.spatial_chk = chk
            row += 1

            # —— 3) 初次创建时就执行一次 calc_psf(panel) 计算初始 psfStd ——#
            self.calc_psf(panel)

        # —— 4) 最后更新面板可见性：只显示 Listbox 里当前选中的文件对应 panel ——#
        self.update_panel_visibility(h_list)

    def del_stack(self, h_list):
        """Delete the selected stack from the list."""
        selected_indices = h_list.curselection()
        if not selected_indices:
            return  # No selection

        # Get the selected index
        bad = selected_indices[0]
        content = list(h_list.get(0, tk.END))  # Get current content as a list
        content.pop(bad)  # Remove the selected item
        h_list.delete(0, tk.END)  # Clear the listbox
        h_list.insert(tk.END, *content)  # Insert updated content

    def move_up_stack(self, h_list):
        """Move the selected stack up in the list."""
        pos = h_list.curselection()
        if pos and pos[0] > 0:
            content = list(h_list.get(0, tk.END))
            content[pos[0] - 1], content[pos[0]] = content[pos[0]], content[pos[0] - 1]  # Swap
            h_list.delete(0, tk.END)  # Clear the listbox
            h_list.insert(tk.END, *content)  # Insert updated content
            h_list.selection_set(pos[0] - 1)  # Update selection

    def move_down_stack(self, h_list):
        """Move the selected stack down in the list."""
        pos = h_list.curselection()
        content = list(h_list.get(0, tk.END))
        if pos and pos[0] < len(content) - 1:
            content[pos[0]], content[pos[0] + 1] = content[pos[0] + 1], content[pos[0]]  # Swap
            h_list.delete(0, tk.END)  # Clear the listbox
            h_list.insert(tk.END, *content)  # Insert updated content
            h_list.selection_set(pos[0] + 1)  # Update selection

    def update_panel_visibility(self, h_list):
        """
        根据 Listbox 选中的文件（Listbox 里存放的是完整路径），
        隐藏所有子 panel，然后只让对应的 panel.grid() 显示在 row=0, col=0。
        """
        # 如果字典为空，直接 return
        if not self.optics_panels:
            return

        # 取 Listbox 里所有项目
        content = list(h_list.get(0, tk.END))
        if not content:
            # 没有任何文件时，把所有 panel 隐藏
            for panel in self.optics_panels.values():
                panel.grid_forget()
            return

        # 取当前选中索引
        sel = h_list.curselection()
        if not sel:
            # 没选中时，也全部隐藏
            for panel in self.optics_panels.values():
                panel.grid_forget()
            return

        selected_index = sel[0]
        selected_file = content[selected_index]

        # 隐藏所有子 panel
        for panel in self.optics_panels.values():
            panel.grid_forget()

            # 只显示对应的那个 panel，且放在 row=12 下
            if selected_file in self.optics_panels:
                self.optics_panels[selected_file].grid(
                    row=12, column=0, columnspan=2, sticky='nsew'
                )

    def on_listbox_select(self, event):
        """
        当 Listbox 选中行发生变化时，调用 update_panel_visibility 切换显示面板。
        """
        # event.widget 就是触发事件的 Listbox
        self.update_panel_visibility(event.widget)

    def calc_psf(self, panel):
        """
        根据 panel 里的 Entry 值计算 PSF Std，并写回到只读 psf_std_entry 中。

        MATLAB 里公式：
          PSF Std [px] = psfScale * 0.55 * (emission_in_um) / ( NA * 1.17 * (pixelSize/2) )

        这里 em_wl(单位 nm) *0.001 → μm；pixelSize 已经是 μm；NA, psfScale 不变。
        """
        try:
            px_size = float(panel.px_entry.get())
            em_wl = float(panel.em_entry.get())
            na = float(panel.na_entry.get())
            psf_sc = float(panel.psf_scale_entry.get())
        except Exception:
            # 任何一项转换失败，则直接返回，不更新
            return

        # 把 em_wl (nm) → μm
        em_um = em_wl * 0.001
        # 计算公式：psf_sc * 0.55 * em_um / (na * 1.17 * (px_size/2))
        denominator = na * 1.17 * (px_size / 2)
        if denominator == 0:
            return
        psf_std = psf_sc * 0.55 * em_um / denominator

        # 把结果写回到 panel.psf_std_entry（只读）
        panel.psf_std_entry.config(state='normal')
        panel.psf_std_entry.delete(0, tk.END)
        panel.psf_std_entry.insert(0, f"{psf_std:.3f}")
        panel.psf_std_entry.config(state='readonly')


    def process_batch(self, h_list):
        """Process each selected image stack."""
        files = list(h_list.get(0, tk.END))
        # print("[DEBUG] 批处理文件数：", len(files))
        for file in files:
            start_time = time.perf_counter()
            self.image_bin = {
                'isOwn': True,
                'isLoaded': True,
                'isSuperstack': False,
                'isTrack': False,
                'frame': 1,

                # Loc Options (Localization)
                'locStart': 1,
                'locEnd': float('inf'),
                'errorRate': float(self.err_rate_entry.get()),
                'w2d': float(self.box_size_entry.get()),  # 假设 w2d 是一个浮点数
                'dfltnLoops': int(self.deflation_loops_entry.get()),
                'minInt': float(self.intensity_thres_entry.get()),
                'locParallel': self.use_parallel_var.get(),  # 直接引用
                'nCores': self.n_cores_var.get(),
                'isRadiusTol': self.image_bin.get('isRadiusTol', True),
                'radiusTol': float(self.radius_tol_entry.get()),
                'posTol': float(self.pos_tol_entry.get()),
                'maxOptimIter': int(self.image_bin.get('maxOptimIter', 100)),
                'termTol': float(self.term_tol_entry.get()),

                # Bar Options
                'isScalebar': False,
                'microBarLength': 1000,
                'isColormap': False,
                'colormapWidth': 20,
                'isTimestamp': False,
                'timestampSize': 2,
                'timestampInkrement': 0.032,

                # Render Options
                'exfOld': 5,
                'exfNew': 5,
                'rW': 50,
                'rStep': 10,
                'rStart': 1,
                'rEnd': float('inf'),
                'rLive': False,
                'fps': 25,
                'movCompression': 1,
                'convMode': 1,
                'intWeight': 0.01,
                'sizeFac': 1,
                'isCumsum': False,

                # Thresh Options
                'isThreshLocPrec': False,
                'minLoc': 0,
                'maxLoc': float('inf'),
                'isThreshSNR': False,
                'minSNR': 0,
                'maxSNR': float('inf'),
                'isThreshDensity': False,

                # Optics Options
                'pxSize': float('100'),
                'cntsPerPhoton': float('0.5'),
                'emWvlnth': float('500'),
                'NA': float('1.4'),
                'psfScale': float('0.7'),
                'psfStd': float('1.03'),
                'spatialCorrection': bool('true'),
                'imageName': file,
                'pathname': '',  # 待更新
                'filename': '',
            }

            # 读取图像文件
            # info = imread(file)
            #读取为内存映射的多帧数组
            # stack=tifffile.memmap(file)
            def safe_load_tiff(path):
                import tifffile, numpy as np
                with tifffile.TiffFile(path) as tif:
                    comp = tif.pages[0].compression
                    comp_name = comp.name if hasattr(comp, 'name') else str(comp)
                    n_pages = len(tif.pages)
                    print(f"Compression: {comp_name}  | pages={n_pages}")
                    try:
                        arr = tif.asarray()  # 保证页数正确
                    except Exception as e:
                        print(f"[warn] tif.asarray failed: {e}; fallback to imread")
                        arr = tifffile.imread(path)

                if arr.ndim == 2:
                    stack = arr[np.newaxis, ...]
                elif arr.ndim >= 3:
                    if arr.shape[0] == n_pages:
                        stack = arr
                    elif arr.shape[-1] == n_pages:
                        stack = np.moveaxis(arr, -1, 0)
                    else:
                        k = [i for i, s in enumerate(arr.shape) if s == n_pages]
                        stack = np.moveaxis(arr, k[0], 0) if k else arr
                    while stack.ndim > 3:
                        stack = stack[:, :, :, 0]  # 丢多余通道
                else:
                    raise RuntimeError(f"Unsupported TIFF shape: {arr.shape}")

                T, H, W = stack.shape[:3]
                return stack

            stack = safe_load_tiff(file)

            # Split the image path and update pathname and filename
            self.image_bin['pathname'], self.image_bin['filename'] = os.path.split(self.image_bin['imageName'])

            # 如果 spatialCorrection 为 True，则设置 spatialCorrPath
            if self.image_bin['spatialCorrection']:
                self.image_bin['spatialCorrPath'] = self.image_bin['spatialCorrection']

            # 根据图像维度判断格式
            if stack.ndim == 2:  # Grayscale image
                h,w = stack.shape
                self.image_bin['width'], self.image_bin['height'] = w,h
                self.image_bin['roi'] = [1, 1,w,h]
                start_pnt,end_pnt = 1,1

            else:  # RGB or multi-channel image
                frames,h,w = stack.shape[:3]
                # 假设 info 的格式为 (frames, height, width) 或 (frames, height, width, channels)
                self.image_bin['width'], self.image_bin['height'] = w,h
                self.image_bin['roi'] = [1, 1, w,h]
                start_pnt,end_pnt = 1,frames

            #从image_bin['roi']解包ROI坐标
            x0,y0,roiWidth,roiHeight = self.image_bin['roi']
            #转成python的0——based切片区间
            x0_py,y0_py = x0-1,y0-1
            x1_py=x0_py+roiWidth
            y1_py=y0_py+roiHeight
            # Call localization function
            self.image_bin['ctrsN'] = self.localize_particles(self.image_bin, start_pnt, end_pnt, stack,x0_py,y0_py,x1_py,y1_py)
            # 获取去掉扩展名的文件名
            filename_without_ext = os.path.splitext(self.image_bin['filename'])[0]

            # 拼接完整路径
            output_path = os.path.join(self.image_bin['pathname'], filename_without_ext)
            output_mat = f"{output_path}.mat"


            # 保存 localization matrix 到 .mat 文件
            try:
                sio.savemat(output_mat, {'imageBin': self.image_bin})
            except Exception as e:
                print("[ERROR] save .mat file failed:", str(e))
                # 记录结束时间，计算耗时
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            # print(f"[DEBUG] Finished processing {file!r} in {elapsed:.2f} seconds")

        messagebox.showinfo("Info", "Batch processing done!")

    def localize_particles(self, image_bin, start_pnt, end_pnt, stack,x0_py,y0_py,x1_py,y1_py):
        """Run localization on the specified image with the given parameters."""
        imagename = image_bin['imageName']
        region = [
            [image_bin['roi'][1], image_bin['roi'][1] + image_bin['roi'][3] - 1],
            [image_bin['roi'][0], image_bin['roi'][0] + image_bin['roi'][2] - 1]
        ]

        # Initialize optim based on image_bin parameters
        optim = [
            image_bin['maxOptimIter'],
            image_bin['termTol'],
            image_bin['isRadiusTol'],
            image_bin['radiusTol'],
            image_bin['posTol']
        ]

        # —— 1. 准备所有不变的常量 ——
        psfStd = image_bin['psfStd']
        wn = int(image_bin['w2d'])  # 窗口大小
        alpha_pfa = 10.0 ** float(image_bin['errorRate'])  # 例如 -3 → 1e-3
        pfa_thresh = chi2.ppf(1 - alpha_pfa, 1)

        T = wn * wn

        # 全图尺寸
        _, H, W = stack.shape

        #  —— 1a. 本地绑定，全局查找变局部 ——
        fft2_local = fft2
        expand_w_local = expand_w
        gausswin2_local = gausswin2
        detect_fn = detect_et_estime_part_1vue_deflt  # 如使用 numba 版本就改这里
        stream_to_disk = self.stream_to_disk

        roi_h = y1_py - y0_py
        roi_w = x1_py - x0_py

        hm = expand_w_local(np.ones((wn, wn)), roi_h, roi_w)  # ★ 用 ROI 尺寸
        TFHM = fft2_local(hm)
        g = gausswin2_local(psfStd, wn, wn)
        gc = g - np.sum(g) / (wn * wn)
        hgc = expand_w_local(gc, roi_h, roi_w)  # ★ 用 ROI 尺寸
        TFHGC = fft2_local(hgc)
        SGC2 = np.sum(gc ** 2)
        TFHM.flags.writeable = False
        TFHGC.flags.writeable = False
        ctx = (TFHM, TFHGC, SGC2)

        if image_bin['locParallel']:
            # 解包所有参数
            imagename = image_bin['imageName']
            y0 = image_bin['roi'][1]
            y1 = image_bin['roi'][1] + image_bin['roi'][3]
            x0 = image_bin['roi'][0]
            x1 = image_bin['roi'][0] + image_bin['roi'][2]
            roiWidth = image_bin['roi'][2]
            roiHeight = image_bin['roi'][3]
            w2d = image_bin['w2d']
            psfStd = image_bin['psfStd']

            dfltnLoops = image_bin['dfltnLoops']
            minInt = image_bin['minInt']
            optim = [
                image_bin['maxOptimIter'],
                image_bin['termTol'],
                image_bin['isRadiusTol'],
                image_bin['radiusTol'],
                image_bin['posTol']
            ]
            o0, o1, o2, o3, o4 = optim

            # ROI 索引也提到外面
            x0_py_l = x0_py
            y0_py_l = y0_py
            x1_py_l = x1_py
            y1_py_l = y1_py
            roi_h = y1_py - y0_py
            roi_w = x1_py - x0_py

            # —— 2. 划分成 N_cores×块 ——
            n_cores = image_bin['nCores']
            if isinstance(n_cores, str) and n_cores.lower() == 'max':
                n_cores = os.cpu_count()
            else:
                n_cores = int(n_cores)

            frames = list(range(start_pnt, end_pnt + 1))
            # block_size = max(1, math.ceil(len(frames) / n_cores))  # 每块大约 len(frames)/n_cores 帧
            block_size = max(1, math.ceil(len(frames) / 12))  # 每块大约 len(frames)/n_cores 帧
            blocks = [frames[i:i + block_size]
                      for i in range(0, len(frames), block_size)]

            data = []
            ctrsN = []

            # —— 3. 定义“块处理”函数 ——
            def _process_block(block):
                block_data = []
                block_ctrs = []
                bd_append = block_data.append
                bc_append = block_ctrs.append
                for t in block:
                    I = stack[t - 1, y0_py_l:y1_py_l, x0_py_l:x1_py_l].astype(np.float64)
                    df, deflt, bim, nct =detect_fn(
                        I, w2d, psfStd, pfa_thresh,
                        dfltnLoops, roi_w, roi_h,
                        minInt, optim, ctx
                    )
                    bd_append(df)
                    bc_append(nct)
                return block_data, block_ctrs

            # —— 4. 提交每个块 ——
            with ThreadPoolExecutor(max_workers=n_cores) as exe:
                futures = [exe.submit(_process_block, blk) for blk in blocks]
                for fut in futures:
                    bd, bc = fut.result()
                    data.extend(bd)
                    ctrsN.extend(bc)

                # 写磁盘
            filename_without_ext = os.path.splitext(image_bin['filename'])[0]
            output_path = os.path.join(
                image_bin['pathname'],
                filename_without_ext + '_locs.txt'
            )
            stream_to_disk(data, output_path)

            return ctrsN

        else:  # Linear computation
            # —— A. 一次性解包，减少循环内部属性查找 ——
            detect_fn = detect_et_estime_part_1vue_deflt
            stream_disk = self.stream_to_disk
            # 这些都是不变的，提前算好
            w2d = image_bin['w2d']
            psfStd = image_bin['psfStd']
            # pfa_thresh = chi2.ppf(1 - 1 / 10 ** (-image_bin['errorRate']), 1)
            dfltnLoops = image_bin['dfltnLoops']
            minInt = image_bin['minInt']
            optim_local = optim  # 上面已经算好的 list

            # ROI 索引也提到外面
            x0_py_l = x0_py
            y0_py_l = y0_py
            x1_py_l = x1_py
            y1_py_l = y1_py
            roi_h = y1_py - y0_py
            roi_w = x1_py - x0_py

            # —— B. 预先绑定 append ——
            data = []
            ctrsN = []
            data_ap = data.append
            ctr_ap = ctrsN.append

            # —— C. 主循环 ——
            # 注意 frame 从 start_pnt-1 到 end_pnt-1
            for frame_idx in range(start_pnt - 1, end_pnt):
                # Load and cast once
                I = stack[frame_idx, y0_py_l:y1_py_l, x0_py_l:x1_py_l].astype(np.float64)
                # 调用本地绑定的检测函数
                df, deflt, bim, nct = detect_fn(
                    I,
                    w2d, psfStd,
                    pfa_thresh,
                    dfltnLoops,
                    roi_w, roi_h,
                    minInt,
                    optim_local, ctx
                )
                # 本地绑定的 append
                data_ap(df)
                ctr_ap(nct)

            # —— D. 写磁盘 ——
            filename_without_ext = os.path.splitext(image_bin['filename'])[0]
            output_path = os.path.join(
                image_bin['pathname'],
                filename_without_ext + '_locs.txt'
            )
            stream_disk(data, output_path)
        return ctrsN

    def stream_to_disk(self, data, output_path):
        """
        将局部化的粒子数据写入磁盘的文本文件。

        参数:
        - data: list, 每个元素为一个包含粒子信息的numpy数组，形状为 (N, 7)
          每个数组中的7列分别为 [y, x, alpha, sig2, offset, r, result_ok]
        - output_path: str, 输出文件路径
        """
        import numpy as np
        import time

        time_0 = time.perf_counter()

        with open(output_path, 'w') as o:
            # —— 若下游脚本依赖旧表头，保持不变 ——（写入的数值已按 MATLAB 规范转换）
            o.write('particle_idx\tframe_idx\ty_coord\tx_coord\talpha\tSig2\toffset\tr\tresult_ok\n')

            particle_idx = 0
            for frame_idx, frame_data in enumerate(data):
                # frame_data: shape (N, 7) → [y, x, alpha, sig2, offset, r, ok]
                for loc in frame_data:
                    y, x, alpha, sig2, offset, r, ok = loc

                    # —— 与 MATLAB 一致的导出量 ——
                    # alpha_peak = alpha / (sqrt(pi) * r)
                    # noise_std  = sqrt(sig2)
                    r_safe = max(float(r), 1e-12)  # 避免除零
                    signal_peak = float(alpha) / (np.sqrt(np.pi) * r_safe)
                    noise_std = np.sqrt(max(float(sig2), 0.0))

                    # —— 写入（顺序与表头一致）——
                    # 这里保持列名 alpha/Sig2，但写入的是 peak 与 std（与 MATLAB 落盘一致）
                    o.write('%d\t' % particle_idx)
                    o.write('%d\t' % frame_idx)
                    o.write('%f\t%f\t' % (y, x))
                    o.write('%f\t%f\t' % (signal_peak, noise_std))  # 转换后的两列
                    o.write('%f\t%f\t%g\n' % (offset, r, ok))

                    particle_idx += 1

    # def detect_et_estime_part_1vue_deflt(self, input_data, wn, r0, pfa, n_deflt, w, h, minInt, optim):
    #     """主要处理逻辑，模仿 MATLAB 函数的功能"""
    #     # 调用检测函数
    #     global TFHM, TFHGC, SGC2
    #     lest, ldec, dfin, Nestime = self.detect_et_estime_part_1vue(input_data, wn, r0, pfa, optim)
    #     # Deflation处理
    #     input_deflt = self.deflat_part_est(input_data, lest, wn)
    #     lestime = lest.copy()  # 保留初始估计
    #     if n_deflt == 0:
    #         border = int(np.ceil(wn / 2))
    #         # 条件1：检测有效标志 (ok)
    #         condition1 = (lestime[:, 6] > 0)
    #         # 条件2：强度与半径比值大于最小阈值
    #         condition2 = (lestime[:, 2] / np.sqrt(np.pi) / lestime[:, 5] > minInt)
    #         # 条件3：行坐标在 [border, h-border)
    #         condition3 = (lestime[:, 0] > border) & (lestime[:, 0] < h - border)
    #         # 条件4：列坐标在 [border, w-border)
    #         condition4 = (lestime[:, 1] > border) & (lestime[:, 1] < w - border)
    #         good = condition1 & condition2 & condition3 & condition4
    #         lestime = lestime[good]
    #         ctrsN = np.sum(good)
    #         return lestime, input_deflt, dfin, ctrsN
    #
    #     for n in range(n_deflt):
    #         l, ld, d, N = self.detect_et_estime_part_1vue(input_deflt, wn, r0, pfa, optim)
    #         lestime = np.vstack((lestime, l))  # 合并结果
    #         dfin = np.logical_or(dfin, d)
    #         input_deflt = self.deflat_part_est(input_deflt, l, wn)
    #
    #         if N == 0 or n == n_deflt - 1:
    #             border = int(np.ceil(wn / 2))
    #
    #             good = (lestime[:, 6] > 0) & \
    #                    (lestime[:, 2] / np.sqrt(np.pi) / lestime[:, 5] > minInt) & \
    #                    (lestime[:, 0] > border) & (lestime[:, 0] < h - border) & \
    #                    (lestime[:, 1] > border) & (lestime[:, 1] < w - border)
    #
    #             good_count = np.sum(good)
    #             lestime = lestime[good]
    #             ctrsN = good_count
    #             return lestime, input_deflt, dfin, ctrsN

    # def detect_et_estime_part_1vue(self, input_data, wn, r0, pfa, optim):
    #     """
    #     与 MATLAB detect_et_estime_part_1vue 功能对应的 Python 版本
    #
    #     Returns:
    #         lestime: shape (N, 7) 的估计结果, 每行 [idx, i, j, alpha, sig², r, ok]
    #         ldetect: 初步检测列表
    #         d:       检测图(布尔)
    #         Nestime: 实际做了高斯牛顿拟合的数量
    #     """
    #     # ========== (1) 计算检测图 ==========
    #     carte_MV, ldetect, d = self.carte_H0H1_1vue(input_data, r0, wn, wn, pfa)
    #     Ndetect = ldetect.shape[0]
    #
    #     # 若没有检测到任何点, 直接返回空
    #     if Ndetect == 0:
    #         lestime = np.zeros((1, 7), dtype=np.float64)
    #         return lestime, ldetect, d, 0
    #
    #     # 预分配
    #     lestime = np.zeros((Ndetect, 7), dtype=np.float64)
    #     Nestime = 0
    #     bord = int(np.ceil(wn / 2))
    #
    #     # ========== (2) 逐点过滤 + GN 拟合 ==========
    #     for n in range(Ndetect):
    #         i_ = ldetect[n, 1]
    #         j_ = ldetect[n, 2]
    #         alpha_ = ldetect[n, 3]
    #
    #         test_bord = (
    #                 (i_ < bord) or (i_ > (input_data.shape[0] - bord)) or
    #                 (j_ < bord) or (j_ > (input_data.shape[1] - bord))
    #         )
    #         if (alpha_ > 0.0) and (not test_bord):
    #             result = self.estim_param_part_GN(input_data, wn, ldetect[n, :], r0, optim)
    #             # 再加调试：看看 GN 返回了多少元素
    #             if len(result) == 7:
    #                 lestime[Nestime, :] = result
    #                 Nestime += 1
    #
    #     if Nestime == 0:
    #         lestime = np.zeros((0, 7), dtype=np.float64)
    #     else:
    #         lestime = lestime[:Nestime, :]
    #     return lestime, ldetect, d, Nestime
    #
    # def carte_H0H1_1vue(self, im, rayon, wn_x, wn_y, s_pfa):
    #     """Detection map generation with debug prints."""
    #     N, M = im.shape
    #     wn_x = int(wn_x)
    #     wn_y = int(wn_y)
    #     T = wn_x * wn_y
    #
    #     # Hypothesis H0
    #     m = np.ones((int(wn_x), int(wn_y)))
    #     hm = self.expand_w(m, N, M)
    #     tfhm = np.fft.fft2(hm)
    #     tfim = np.fft.fft2(im)
    #
    #     m0 = np.real(np.fft.fftshift(np.fft.ifft2(tfhm * tfim))) / T
    #
    #     im2 = im * im
    #     tfim2 = np.fft.fft2(im2)
    #     Sim2 = np.real(np.fft.fftshift(np.fft.ifft2(tfhm * tfim2)))
    #
    #     T_sig0_2 = Sim2 - T * m0 ** 2
    #
    #
    #     # Hypothesis H1
    #     g = self.gausswin2(rayon, wn_x, wn_y, 0, 0)
    #     gc = g - np.sum(g) / T
    #     Sgc2 = np.sum(gc ** 2)
    #     # with open(log_path, 'a') as f:
    #     #     f.write(f"DBG Sgc2 Py = {Sgc2:.6f}\n")
    #     hgc = self.expand_w(gc, N, M)
    #     tfhgc = np.fft.fft2(hgc)
    #
    #     alpha = np.real(np.fft.fftshift(np.fft.ifft2(tfhgc * tfim))) / Sgc2
    #
    #     # Compute carte_MV
    #     test = 1 - (Sgc2 * alpha ** 2) / T_sig0_2
    #     test = np.where(test > 0, test, 1)
    #     carte_MV = -T * np.log(test)
    #     carte_MV[np.isnan(carte_MV)] = 0
    #
    #     detect_masque = carte_MV > s_pfa
    #
    #     n_detect_pixels = np.sum(detect_masque)
    #     # 先算出 local_max 二值图
    #     lm = self.all_max_2d(carte_MV) != 0
    #     if n_detect_pixels == 0:
    #         liste_detect = np.zeros((1, 7))
    #         detect_pfa = np.zeros_like(detect_masque)
    #     else:
    #         detect_pfa = np.logical_and(self.all_max_2d(carte_MV) != 0, detect_masque)
    #         di, dj = np.where(detect_pfa)
    #         n_detect = di.size
    #
    #         # Flatten alpha, sig1_2 in column-major
    #         alpha_colmajor = alpha.flatten(order='F')
    #         vind = dj * N + di  # row=di, col=dj => col-major index
    #         alpha_detect = alpha_colmajor[vind]
    #
    #         sig1_2 = (T_sig0_2 - alpha ** 2 * Sgc2) / T
    #         sig1_2_colmajor = sig1_2.flatten(order='F')
    #         sig2_detect = sig1_2_colmajor[vind]
    #
    #         liste_detect = np.column_stack((
    #             np.arange(1, n_detect + 1),
    #             di,  # row
    #             dj,  # col
    #             alpha_detect,
    #             sig2_detect,
    #             rayon * np.ones(n_detect),
    #             np.ones(n_detect)
    #         ))
    #     return carte_MV, liste_detect, detect_masque

    def expand_w(self, in_array, N, M):
        """
        Expand the input array 'in_array' to size N x M by centering it
        (mimicking the MATLAB function expand_w).
        """
        N_in, M_in = in_array.shape
        out = np.zeros((N, M), dtype=in_array.dtype)

        # 使用浮点运算 + floor，以与 MATLAB 的 floor(N/2 - N_in/2) 一致
        nc = int(np.floor(N / 2.0 - N_in / 2.0))
        mc = int(np.floor(M / 2.0 - M_in / 2.0))

        # MATLAB 中 out((nc+1):(nc+N_in), (mc+1):(mc+M_in)) = in
        # Python 中索引从 0 开始，因此可直接用 out[nc : nc+N_in, mc : mc+M_in]
        # 即与 MATLAB 的 (nc+1) 对应 Python 的 nc
        out[nc: nc + N_in, mc: mc + M_in] = in_array

        return out

    def all_max_2d(self, input_array):
        """Find all local maxima in a 2D array."""
        N, M = input_array.shape
        ref = input_array[1:N - 1, 1:M - 1]

        # 计算各个方向的局部最大值
        pos_max_h = (input_array[0:N - 2, 1:M - 1] < ref) & (input_array[2:N, 1:M - 1] < ref)
        pos_max_v = (input_array[1:N - 1, 0:M - 2] < ref) & (input_array[1:N - 1, 2:M] < ref)
        pos_max_135 = (input_array[0:N - 2, 0:M - 2] < ref) & (input_array[2:N, 2:M] < ref)
        pos_max_45 = (input_array[2:N, 0:M - 2] < ref) & (input_array[0:N - 2, 2:M] < ref)


        carte_max = np.zeros((N, M))
        carte_max[1:N - 1, 1:M - 1] = pos_max_h & pos_max_v & pos_max_135 & pos_max_45
        carte_max = carte_max * input_array

        # # # 确保 carte_max 是布尔类型
        # carte_max = carte_max.astype(bool)

        return carte_max


    def estim_param_part_GN(self, im, wn, liste_info_param, r0, optim):
        """
        Python version of estim_param_part_GN with debug prints.
        """
        pi = liste_info_param[1]+1
        pj = liste_info_param[2]+1

        di_vals = np.arange(1, wn + 1) + (pi - np.floor(wn / 2))
        dj_vals = np.arange(1, wn + 1) + (pj - np.floor(wn / 2))
        di_vals = di_vals.astype(int)
        dj_vals = dj_vals.astype(int)
        # im_part = im[np.ix_(di_vals - 1, dj_vals - 1)]
        # 将 di_vals - 1 和 dj_vals - 1 限制在 [0, im.shape[0]-1] 和 [0, im.shape[1]-1] 的范围内
        di_clipped = np.clip(di_vals - 1, 0, im.shape[0] - 1)
        dj_clipped = np.clip(dj_vals - 1, 0, im.shape[1] - 1)

        im_part = im[np.ix_(di_clipped, dj_clipped)]

        # If optim[0] == 0 => skip... (not used).
        if optim[0] == 0:
            ...
        else:
            bornes_ijr = [
                -optim[4], optim[4],
                -optim[4], optim[4],
                r0 - optim[3] * r0 / 100,
                r0 + optim[3] * r0 / 100
            ]
            r = r0
            i = 0.0
            j = 0.0
            dr = 1
            di_val = 1
            dj_val = 1
            fin = 10 ** optim[1]
            sig2 = np.inf
            # sig2 = np.sum((im_part - np.mean(im_part)) ** 2) / im_part.size
            cpt = 0
            test = True
            iter_max = optim[0]

            while test:
                (r, i, j,
                 dr, di_val, dj_val,
                 alpha, sig2, offset) = self.deplt_GN_estimation(
                    r, i, j, im_part, sig2, dr, di_val, dj_val, optim
                )
                cpt += 1

                if optim[2]:  # use radius
                    cond = max(abs(di_val), abs(dj_val), abs(dr)) > fin
                else:
                    cond = max(abs(di_val), abs(dj_val)) > fin

                if cpt > iter_max:
                    test = False
                else:
                    test = cond

                result_ok = not (
                        i < bornes_ijr[0] or i > bornes_ijr[1] or
                        j < bornes_ijr[2] or j > bornes_ijr[3] or
                        r < bornes_ijr[4] or r > bornes_ijr[5]
                )
                test = test and result_ok
            liste_param = [pi + i, pj + j, alpha, sig2, offset, r, result_ok]

            return liste_param

    def deplt_GN_estimation(self, p_r, p_i, p_j, x, sig2init, p_dr, p_di, p_dj, optim):
        """Gaussian-Newton estimation with detailed debug prints."""
        import numpy as np

        # 初始设置
        r0 = p_r
        i0 = p_i
        j0 = p_j
        prec_rel = 10 ** optim[1]
        verif_crit = 1
        pp_r = r0 - p_dr
        pp_i = i0 - p_di
        pp_j = j0 - p_dj

        wn_i, wn_j = x.shape
        N = wn_i * wn_j
        refi = 0.5 + np.arange(wn_i) - wn_i / 2
        refj = 0.5 + np.arange(wn_j) - wn_j / 2
        again = True
        loops = 0

        while again:
            loops += 1

            # 计算参考坐标偏移
            i = refi - i0
            j = refj - j0
            ii = np.tile(i[:, np.newaxis], (1, wn_j))
            jj = np.tile(j[np.newaxis, :], (wn_i, 1))

            # 计算高斯窗 g
            iiii = ii ** 2
            jjjj = jj ** 2
            iiii_jjjj = iiii + jjjj
            g = (1 / (np.sqrt(np.pi) * r0)) * np.exp(-(1 / (2 * r0 ** 2)) * iiii_jjjj)


            # 去均值，获得 gc 及计算 Sgc2
            gc = g - np.sum(g) / N
            Sgc2 = np.sum(gc ** 2)

            g_div_sq_r0 = (1 / (r0 ** 2)) * g

            # 计算 alpha
            if Sgc2 != 0:
                num_alpha = np.sum(x * gc)
                alpha = num_alpha / Sgc2
            else:
                num_alpha = 0
                alpha = 0

            # 计算残差和统计量
            x_alphag = x - alpha * g
            m = np.sum(x_alphag) / N
            err = x_alphag - m
            sig2 = np.sum(err ** 2) / N

            # 判断是否需要调整（如果误差 sig2 变大）
            if verif_crit and sig2 > sig2init:
                p_di /= 10.0
                p_dj /= 10.0
                i0 = pp_i + p_di
                j0 = pp_j + p_dj
                if optim[2]:
                    p_dr /= 10.0
                    r0 = pp_r + p_dr
                else:
                    p_dr = 0
                    r0 = pp_r

                if np.max([abs(p_dr), abs(p_di), abs(p_dj)]) > prec_rel:
                    n_r = p_r
                    n_i = p_i
                    n_j = p_j
                    dr = di = dj = 0
                    return n_r, n_i, n_j, dr, di, dj, alpha, sig2, m
            else:
                again = False

            if loops > 50:
                again = False

        # --- 计算导数 ---
        der_g_i0 = ii * g_div_sq_r0
        der_g_j0 = jj * g_div_sq_r0
        derder_g_i0 = (-1 + (1 / r0 ** 2) * iiii) * g_div_sq_r0
        derder_g_j0 = (-1 + (1 / r0 ** 2) * jjjj) * g_div_sq_r0

        der_J_i0 = alpha * np.sum(der_g_i0 * err)
        der_J_j0 = alpha * np.sum(der_g_j0 * err)
        derder_J_i0 = alpha * np.sum(derder_g_i0 * err) - alpha ** 2 * np.sum(der_g_i0 ** 2)
        derder_J_j0 = alpha * np.sum(derder_g_j0 * err) - alpha ** 2 * np.sum(der_g_j0 ** 2)

        if optim[2]:
            der_g_r0 = (-1 / r0 + (1 / r0 ** 3) * iiii_jjjj) * g
            derder_g_r0 = (1 - 3 / r0 ** 2 * iiii_jjjj) * g_div_sq_r0 + \
                          (-1 / r0 + (1 / r0 ** 3) * iiii_jjjj) * der_g_r0
            der_J_r0 = alpha * np.sum(der_g_r0 * err)
            derder_J_r0 = alpha * np.sum(derder_g_r0 * err) - alpha ** 2 * np.sum(der_g_r0 ** 2)
            dr = -der_J_r0 / derder_J_r0
            n_r = abs(r0 + dr)
        else:
            dr = 0
            n_r = r0

        di = - der_J_i0 / derder_J_i0
        dj = - der_J_j0 / derder_J_j0

        n_i = i0 + di
        n_j = j0 + dj

        return n_r, n_i, n_j, dr, di, dj, alpha, sig2, m

    def deflat_part_est(self, input_data, liste_est, wn):
        """
        Python 版 deflat_part_est，功能与 MATLAB 版本相同。
        参数：
          input_data: 输入图像，NumPy 数组 (2D)
          liste_est: 检测结果矩阵，格式为 [num, i, j, alpha, sig^2, rayon, ok]
                     注意：i, j 坐标为 MATLAB 1-based 数值
          wn: 初始窗口大小（此参数会在循环内被重新计算）
        返回：
          output: deflation 后的图像，转换为 uint16 类型
        """
        idim, jdim = input_data.shape
        nb_part = liste_est.shape[0]

        # 将输出转换为 float64
        output = input_data.astype(np.float64).copy()

        for part in range(nb_part):
            # 检查有效性：MATLAB判断 liste_est(part,7)==1, 对应 Python 索引6
            if liste_est[part, 6] == 1:
                # MATLAB: i0 = liste_est(part,1); j0 = liste_est(part,2);
                i0 = liste_est[part, 0]
                j0 = liste_est[part, 1]
                alpha = liste_est[part, 2]
                r0 = liste_est[part, 5]
                wn = int(np.ceil(6 * r0))

                # 计算位置及偏差
                pos_i = round(i0)  # MATLAB round(i0)
                pos_j = round(j0)
                dep_i = i0 - pos_i
                dep_j = j0 - pos_j

                # 计算高斯窗
                alpha_g = alpha * self.gausswin2(r0, wn, wn, dep_i, dep_j)

                # 计算 dd, di, dj 按 MATLAB 逻辑（1-based）
                dd = np.arange(1, wn + 1) - int(np.floor(wn / 2))
                di = dd + pos_i  # 此时 di 按 MATLAB 坐标，取值应在 1 到 idim
                dj = dd + pos_j

                # MATLAB条件： iin = di > 0 & di < idim+1
                iin = (di > 0) & (di <= idim)
                jin = (dj > 0) & (dj <= jdim)

                # 转换为 Python 0-based 索引：减 1
                di_valid = (di[iin] - 1).astype(int)
                dj_valid = (dj[jin] - 1).astype(int)

                # 使用 np.ix_ 从 alpha_g 中提取对应子矩阵
                valid_rows = np.flatnonzero(iin)
                valid_cols = np.flatnonzero(jin)
                alpha_g_sub = alpha_g[np.ix_(valid_rows, valid_cols)]

                # 更新输出，确保使用 0-based 索引
                output[np.ix_(di_valid, dj_valid)] -= alpha_g_sub

        return output

    def gausswin2(self, sig, wn_i, wn_j=None, offset_i=0.0, offset_j=0.0):
        """Generate a 2D Gaussian window."""
        if wn_j is None:
            wn_j = wn_i

        refi = 0.5 + np.arange(wn_i) - wn_i / 2
        i = refi - offset_i
        refj = 0.5 + np.arange(wn_j) - wn_j / 2
        j = refj - offset_j
        ii = np.tile(i[:, np.newaxis], (1, wn_j))
        jj = np.tile(j[np.newaxis, :], (wn_i, 1))

        # Unit power
        g = (1 / (np.sqrt(np.pi) * sig)) * np.exp(-(1 / (2 * sig ** 2)) * (ii ** 2 + jj ** 2))
        return g

    def modelfun(self, x, raw, refi, refj, N):
        """
        模型函数，对应 MATLAB 中 modelfun，
        同时将计算出的 alpha 和 m 保存到 self.last_alpha 与 self.last_m 中。
        参数：
          x: 参数向量，其中 x[0] = offset in i, x[1] = offset in j, x[2] = sigma
          raw: 原始图像区域（二维数组）
          refi, refj: 参考向量
          N: 总像素数
        返回：
          sig2: 残差均方误差（noise power）
        """
        i = refi - x[0]
        j = refj - x[1]
        # 构造二维网格
        wn_j = len(refj)
        wn_i = len(refi)
        ii = np.tile(i[:, np.newaxis], (1, wn_j))
        jj = np.tile(j[np.newaxis, :], (wn_i, 1))

        iiii = ii ** 2
        jjjj = jj ** 2
        iiii_jjjj = iiii + jjjj

        g = (1 / (np.sqrt(np.pi) * x[2])) * np.exp(-(1 / (2 * x[2] ** 2)) * iiii_jjjj)
        gc = g - np.sum(g) / N
        Sgc2 = np.sum(gc ** 2)

        # 计算 mean amplitude
        alpha = np.sum(raw * gc) / Sgc2

        # Offset m
        x_alphag = raw - alpha * g
        m = np.sum(x_alphag) / N

        # 计算 residuals 和 noise power
        err = x_alphag - m
        sig2 = np.sum(err ** 2) / N

        # 保存 alpha 和 m 供外部调用
        self.last_alpha = alpha
        self.last_m = m
        return sig2

    def gaussMLE(self, raw, guess, bounds, options):
        """Gaussian Maximum Likelihood Estimation.
        MATLAB版本返回 [x(1), x(2), alpha, fval, m, x(3), 1]
        """
        wn_i, wn_j = raw.shape
        N = wn_i * wn_j
        refi = 0.5 + np.arange(wn_i) - wn_i / 2
        refj = 0.5 + np.arange(wn_j) - wn_j / 2

        # 进行优化
        result = minimize(self.modelfun, guess, args=(raw, refi, refj, N), bounds=bounds, options=options)
        x = result.x
        fval = result.fun
        # 重新计算以获得 alpha 和 m
        sig2 = self.modelfun(x, raw, refi, refj, N)
        alpha = self.last_alpha
        m = self.last_m
        # 返回与 MATLAB 一致的输出向量：[x(1), x(2), alpha, fval, m, x(3), 1]
        output = [x[0], x[1], alpha, fval, m, x[2], 1]
        return output

    def gauss_elliptic_mle(self, raw, guess, bounds, options):
        """Elliptic Gaussian Maximum Likelihood Estimation.
        MATLAB版本返回 [x(1), x(2), alpha, fval, m, x(3), 1]
        """
        wn_i, wn_j = raw.shape
        N = wn_i * wn_j
        refi = 0.5 + np.arange(wn_i) - wn_i / 2
        refj = 0.5 + np.arange(wn_j) - wn_j / 2

        # 生成二维网格，注意设置 indexing='ij' 保持与 MATLAB 相同的行列顺序
        ii, jj = np.meshgrid(refi, refj, indexing='ij')

        iiii = ii ** 2
        jjjj = jj ** 2
        iiii_jjjj = iiii + jjjj

        def modelfun_elliptic(x):
            # 使用多元正态分布密度函数计算 g
            # MATLAB中：g = 2 * sqrt(pi) * multivariate_normal.pdf( sqrt(iiii_jjjj), mean=[x(1), x(2)], cov=[[x(3), x(4)], [x(4), x(5)]] );
            g = 2 * np.sqrt(np.pi) * multivariate_normal.pdf(np.sqrt(iiii_jjjj), mean=[x[0], x[1]],
                                                             cov=[[x[2], x[3]], [x[3], x[4]]])
            g = g - np.sum(g) / N
            Sgc2 = np.sum(g ** 2)
            alpha = np.sum(raw * g) / Sgc2
            x_alphag = raw - alpha * g
            m = np.sum(x_alphag) / N
            err = x_alphag - m
            sig2 = np.sum(err ** 2) / N
            self.last_alpha = alpha
            self.last_m = m
            return sig2

        result = minimize(modelfun_elliptic, guess, bounds=bounds, options=options)
        x = result.x
        fval = result.fun
        # 重新计算以获取 alpha 和 m
        sig2 = modelfun_elliptic(x)
        alpha = self.last_alpha
        m = self.last_m

        # 计算椭圆参数用于显示（与 MATLAB 相同的处理）
        V, D = np.linalg.eig([[x[2], x[3]], [x[3], x[4]]])
        t = np.linspace(0, 2 * np.pi, 20)
        u = np.array([np.cos(t), np.sin(t)])
        # D 需要先转换为对角矩阵再取平方根
        w = V @ np.sqrt(np.diag(D)) @ u
        z = np.repeat(np.array([[x[0]], [x[1]]]), 20, axis=1) + w

        plt.imshow(raw, cmap='gray')
        plt.fill(z[0, :] + 5, z[1, :] + 5, color='red', alpha=0.6)
        plt.show()

        output = [x[0], x[1], alpha, fval, m, x[2], 1]
        return output

    def build_tracks(self):
        # ================== 基本参数 ==================
        start_time = time.perf_counter()
        self.settings = {
            'Width': self.image_bin['width'],
            'Height': self.image_bin['height'],
            'px2micron': self.image_bin['px_size'],
            'Delay': self.image_bin['frame_size'] / 1000
        }

        # Tracking options
        seuil_detec_1vue = chi2.ppf(1 - 10 ** self.image_bin['error_rate'], 1)
        opts = {
            'FinalDetectionTresh': seuil_detec_1vue,
            'SizeDetectionBox': self.image_bin['w2d'],
            'GaussianRadius': self.image_bin['psf_std'],
            'ValidationTresh': self.image_bin['min_int'],
            'NumberDeflationLoops': self.image_bin['dfltn_loops'],
            'AveragingTimeWindow': self.image_bin['stat_win'],
            'BlinkingProbability': self.image_bin['max_off_time'] * -1,
            'MaxDiffusionCoefficient': self.image_bin['Dmax'],
        }
        self.settings['TrackingOptions'] = opts

        # helper 默认值
        self.Boule_free = float(self.image_bin.get('searchExpFac', 1.2))
        self.Nb_combi = int(self.image_bin.get('maxComp', 1))
        self.T = int(self.image_bin.get('stat_win', 10))
        self.T_off = int(-self.image_bin.get('max_off_time', 2))
        self.sig_free = float(getattr(self, 'sig_free', 1.0))

        self.t_red = opts['AveragingTimeWindow'] - opts['BlinkingProbability']  # 注意：max_off_time 为负号传入

        # 0-based 的全局ID
        self.trackStore = defaultdict(list)
        self.current_ids = []
        self.next_id = 0

        ctrsN = np.array(self.image_bin['ctrsN'], dtype=int)
        Nb_STK = len(ctrsN)
        min_frame = int(np.min(self.image_bin['frame']))
        max_frame = int(np.max(self.image_bin['frame']))
        frames_count_from_frames = max_frame + 1
        self.nb_stk = max(Nb_STK, frames_count_from_frames)

        if len(ctrsN) < self.nb_stk:
            ctrsN = np.pad(ctrsN, (0, self.nb_stk - len(ctrsN)), constant_values=0)
            self.image_bin['ctrsN'] = ctrsN

        # MATLAB 是 1-based 的帧，这里沿用外层 1..nb_stk 的循环，但内部使用 0-based
        if self.image_bin.get('loc_start', 1) <= 1:
            start_point = 1
        else:
            start_point = int(np.searchsorted(np.cumsum(ctrsN), self.image_bin['loc_start'], side='right') + 1)
        end_point = int(self.nb_stk)

        self.settings['Frames'] = end_point - start_point + 1

        r0 = self.image_bin['psf_std']
        seuil_alpha = self.image_bin['min_int']

        filename = os.path.splitext(self.image_bin['pathname'])[0].replace('_locs', '') + '.tif'
        stack = tifffile.imread(filename)
        self.im_t = stack[0]

        # 进度条
        progress_win = tk.Toplevel(self.master)
        progress_win.title("Processing")
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_win, variable=progress_var, maximum=self.nb_stk)
        progress_bar.grid(row=0, column=0, padx=10, pady=10)
        progress_win.update()

        # -------------------- 内部小工具 --------------------
        BLOCK = 7

        def blk_col(col_offset, time_block_index):
            """把 (块内偏移 1..7, 块号 0..t_red/..) -> 全表列号(0-based)。"""
            return 1 + time_block_index * BLOCK + (col_offset - 1)

        def ensure_cols(col_idx_max):
            """确保三张表的列数 >= col_idx_max+1（一次性扩展）"""
            for name in ('tab_param', 'tab_var', 'tab_moy'):
                arr = getattr(self, name, None)
                if arr is None or arr.size == 0:
                    continue
                if arr.shape[1] <= col_idx_max:
                    add = col_idx_max + 1 - arr.shape[1]
                    pad = np.zeros((arr.shape[0], add))
                    setattr(self, name, np.hstack([arr, pad]))

        def init_tab(new_traj=None):
            """
            初始化 tab_param/tab_var/tab_moy 的统计值
            new_traj: None -> 初始化当前帧所有轨迹；否则传入 0-based 索引数组/单个索引
            """
            if new_traj is None:
                tab_traj = np.arange(0, int(par_per_frame[t - 1]))
                new_flag = False
            else:
                tab_traj = np.atleast_1d(new_traj).astype(int)
                new_flag = True

            for iTraj in tab_traj:
                # alpha
                local_param = self.tab_param[iTraj, 7 * (self.t_red - 1) + 4]
                self.tab_moy[iTraj, 7 * (self.t_red - 1) + 4] = local_param
                self.tab_var[iTraj, 7 * (self.t_red - 1) + 4] = 0.2 * local_param
                # r
                local_param = self.tab_param[iTraj, 7 * (self.t_red - 1) + 5]
                self.tab_moy[iTraj, 7 * (self.t_red - 1) + 5] = local_param
                self.tab_var[iTraj, 7 * (self.t_red - 1) + 5] = 0.2 * local_param
                # i,j 方差初值
                self.tab_var[iTraj, 7 * (self.t_red - 1) + 2] = self.sig_free
                self.tab_var[iTraj, 7 * (self.t_red - 1) + 3] = self.sig_free
                # blink
                self.tab_moy[iTraj, 7 * (self.t_red - 1) + 7] = self.tab_param[iTraj, 7 * (self.t_red - 1) + 7]
                self.tab_var[iTraj, 7 * (self.t_red - 1) + 7] = self.tab_param[iTraj, 7 * (self.t_red - 1) + 7]

                if new_flag:
                    # 兼容：再填充前一时刻一份
                    self.tab_moy[iTraj, 7 * ((self.t_red - 1) - 1) + 4] = self.tab_param[
                        iTraj, 7 * (self.t_red - 1) + 4]
                    self.tab_var[iTraj, 7 * ((self.t_red - 1) - 1) + 4] = 0.2 * self.tab_param[
                        iTraj, 7 * (self.t_red - 1) + 4]
                    self.tab_moy[iTraj, 7 * ((self.t_red - 1) - 1) + 5] = self.tab_param[
                        iTraj, 7 * (self.t_red - 1) + 5]
                    self.tab_var[iTraj, 7 * ((self.t_red - 1) - 1) + 5] = 0.2 * self.tab_param[
                        iTraj, 7 * (self.t_red - 1) + 5]
                    self.tab_var[iTraj, 7 * ((self.t_red - 1) - 1) + 2] = self.sig_free
                    self.tab_var[iTraj, 7 * ((self.t_red - 1) - 1) + 3] = self.sig_free
                    self.tab_moy[iTraj, 7 * ((self.t_red - 1) - 1) + 7] = self.tab_param[
                        iTraj, 7 * (self.t_red - 1) + 7]
                    self.tab_var[iTraj, 7 * ((self.t_red - 1) - 1) + 7] = self.tab_param[
                        iTraj, 7 * (self.t_red - 1) + 7]

        def calcul_reference(param, iTraj):
            """
            参考统计计算（3->i,4->j,5->alpha,6->r）
            """
            if self.tab_param[iTraj, 7 * (self.t_red - 1) + 7] < 0:
                offset = int(self.tab_param[iTraj, 7 * (self.t_red - 1) + 7])
            else:
                offset = 0

            raw_val = self.tab_param[iTraj, int(7 * ((self.t_red - 1) + offset) + 7)]
            nb_on = int(np.floor(raw_val / max(1, self.nb_stk)))
            T_local = int(self.settings['TrackingOptions']['AveragingTimeWindow'])
            if nb_on > T_local:
                nb_on = T_local

            seuil = 4  # 3+1
            if nb_on >= seuil:
                n = np.arange(0, nb_on, dtype=int)
                cols = 7 * ((self.t_red - 1) + offset - n) + (param - 1)
                local_param = self.tab_param[iTraj, cols]
                sum_param = np.sum(local_param)
                sum_param2 = np.sum(local_param ** 2)
                param_max = np.max(local_param)
                param_ref = sum_param / nb_on
                sig_param = np.sqrt(max(0.0, sum_param2 / nb_on - param_ref ** 2))
                if param == 5:
                    param_ref = param_ref + 1j * param_max
            else:
                pos_info = 1 if offset == 0 else -offset
                param_ref = self.tab_moy[iTraj, int(7 * ((self.t_red - 1) - pos_info) + (param - 1))]
                sig_param = self.tab_var[iTraj, int(7 * ((self.t_red - 1) - pos_info) + (param - 1))]

            return param_ref, sig_param

        def mise_a_jour_tab():
            new_nb_traj_local = self.tab_param.shape[0]
            T_local = int(self.settings['TrackingOptions']['AveragingTimeWindow'])
            for iTraj in range(new_nb_traj_local):
                if self.tab_param[iTraj, 7 * (self.t_red - 1) + 7] > self.T_off:
                    moy, sig_alpha = calcul_reference(5, iTraj)
                    alpha_moy = np.real(moy)
                    alpha_max = np.imag(moy)
                    if alpha_max == 0:
                        alpha_max = 1e-10
                    if sig_alpha == 0:
                        sig_alpha = 1e-10

                    LV_uni = -T_local * np.log(alpha_max)
                    LV_gauss = -T_local / 2.0 * (1 + np.log(2 * np.pi * sig_alpha ** 2))
                    if LV_gauss > LV_uni:
                        self.tab_moy[iTraj, 7 * (self.t_red - 1) + 4] = alpha_moy + 1j * alpha_max
                        self.tab_var[iTraj, 7 * (self.t_red - 1) + 4] = sig_alpha
                    else:
                        self.tab_moy[iTraj, 7 * (self.t_red - 1) + 4] = self.tab_moy[iTraj, 7 * (self.t_red - 2) + 4]
                        self.tab_var[iTraj, 7 * (self.t_red - 1) + 4] = self.tab_var[iTraj, 7 * (self.t_red - 2) + 4]

                    # r
                    moy_r, sig_r = calcul_reference(6, iTraj)
                    self.tab_moy[iTraj, 7 * (self.t_red - 1) + 5] = moy_r
                    self.tab_var[iTraj, 7 * (self.t_red - 1) + 5] = sig_r

                    # i/j
                    moy_i, sig_i = calcul_reference(3, iTraj)
                    moy_j, sig_j = calcul_reference(4, iTraj)
                    sig_ij = np.sqrt(0.5 * (sig_i ** 2 + sig_j ** 2))
                    if self.sig_free < sig_ij:
                        sig_ij = self.sig_free
                    self.tab_var[iTraj, 7 * (self.t_red - 1) + 2] = sig_ij
                    self.tab_var[iTraj, 7 * (self.t_red - 1) + 3] = sig_ij

                    # blink
                    self.tab_moy[iTraj, 7 * (self.t_red - 1) + 7] = self.tab_param[iTraj, 7 * (self.t_red - 1) + 7]
                    self.tab_var[iTraj, 7 * (self.t_red - 1) + 7] = self.tab_param[iTraj, 7 * (self.t_red - 1) + 7]

        # ---------------------- 帧循环 ----------------------
        par_per_frame = []
        for t in range(start_point, end_point + 1):
            # 当前帧的检测
            ind_valid = (self.image_bin['frame'] == (t - 1))
            par_per_frame = np.array(self.image_bin['ctrsN'], copy=True)
            par_per_frame[t - 1] = np.sum(ind_valid)

            n_dets = int(par_per_frame[t - 1])
            ids = np.arange(0, n_dets, dtype=int)
            y = self.image_bin['ctrsY'][ind_valid]
            x = self.image_bin['ctrsX'][ind_valid]
            s = self.image_bin['signal'][ind_valid] * np.sqrt(np.pi) * self.image_bin['radius'][ind_valid]
            n2 = self.image_bin['noise'][ind_valid] ** 2
            r = self.image_bin['radius'][ind_valid]
            ones = np.ones(n_dets)
            if n_dets == 0:
                lest = np.zeros((0, 7))
            else:
                lest = np.column_stack((ids, y, x, s, n2, r, ones))

            # 重新计算 t_red
            T = int(self.settings['TrackingOptions']['AveragingTimeWindow'])
            self.T_off = int(self.settings['TrackingOptions']['BlinkingProbability'])
            self.t_red = T - self.T_off

            Dmax = self.image_bin['Dmax']
            pxSize = self.image_bin['px_size']
            wn = int(np.ceil(self.image_bin['w2d']))
            frameSize = self.image_bin['frame_size'] / 1000.0
            self.sig_free = np.sqrt(Dmax / (pxSize ** 2) * 4 * frameSize)
            self.settings['TrackingOptions']['sig_free'] = self.sig_free

            # ---------- 第一帧初始化 ----------
            if t == start_point:
                n_init = par_per_frame[t - 1]
                if n_init == 0:
                    progress_var.set(t - start_point + 1)
                    progress_win.update_idletasks()
                    continue

                ncols = 1 + BLOCK * (self.t_red + 1)
                self.tab_param = np.zeros((n_init, ncols), dtype=float)
                self.tab_var = np.zeros_like(self.tab_param)
                self.tab_moy = np.zeros_like(self.tab_param)

                # id
                self.tab_param[:, 0] = np.arange(0, n_init, dtype=float)

                # 当前 reduced 块（cur_block = self.t_red - 1）
                cur_block = self.t_red - 1
                col_start = blk_col(1, cur_block)
                ones_col = np.ones((n_init, 1))
                y_col = lest[:, 1].reshape(-1, 1)
                x_col = lest[:, 2].reshape(-1, 1)
                s_col = lest[:, 3].reshape(-1, 1)
                r_col = lest[:, 5].reshape(-1, 1)
                zeros_col = np.zeros((n_init, 1))
                nbstk_col = (self.nb_stk * np.ones((n_init, 1)))
                block_mat = np.hstack([ones_col, y_col, x_col, s_col, r_col, zeros_col, nbstk_col])
                self.tab_param[:, col_start:col_start + BLOCK] = block_mat

                # status 列（这里沿用旧逻辑）
                status_col = blk_col(1, self.t_red)
                if status_col < self.tab_param.shape[1]:
                    self.tab_param[:, status_col] = 2.0

                # var: [ones, zeros(4), n2, Nb_STK]
                self.tab_var[:, 0] = np.arange(0, n_init, dtype=float)
                n2_col = lest[:, 4].reshape(-1, 1)
                block_var = np.hstack([ones_col, np.zeros((n_init, 4)), n2_col, nbstk_col])
                self.tab_var[:, col_start:col_start + BLOCK] = block_var
                if status_col < self.tab_var.shape[1]:
                    self.tab_var[:, status_col] = 2.0

                self.tab_moy = self.tab_var.copy()

                # init 统计
                init_tab(np.arange(0, n_init, dtype=int))

                # 稳定 gid
                M = self.tab_param.shape[0]
                self.current_ids = list(range(self.next_id, self.next_id + M))
                self.next_id += M

                # 存第一帧
                for track in range(M):
                    status_idx = blk_col(1, self.t_red)
                    if self.tab_param[track, status_idx] > 0:
                        gid = self.current_ids[track]
                        block_idx = self.t_red - 1
                        if t > start_point:
                            current_frame = t - 2  # 写出上一帧
                        else:
                            current_frame = t - 1  # 第一帧特例

                        # ---- build row for tracked_table ----
                        i_val = self.tab_param[track, blk_col(2, block_idx)]  # i(y) 位置
                        j_val = self.tab_param[track, blk_col(3, block_idx)]  # j(x) 位置
                        t_val = current_frame  # 帧号 (Python: 0-based)

                        traj_id = gid  # 轨迹编号 (trackID)

                        alpha = self.tab_param[track, blk_col(4, block_idx)]  # alpha = signal*sqrt(pi)*r

                        nb_code = self.tab_var[track, blk_col(6, block_idx)]  # blink/nb 编码
                        sig_ij = self.tab_var[track, blk_col(2, block_idx)]  # 位置方差 σ_ij
                        n2 = self.tab_var[track, blk_col(4, block_idx)]  # 噪声方差 n²

                        row = np.array([
                            j_val,  # 列5 (j, x 坐标)
                            i_val,  # 列4 (i, y 坐标)
                            t_val,  # 列6 (帧号)
                            traj_id,  # 列7 (轨迹 ID)
                            alpha,  # 列8 (alpha)
                            nb_code,  # 列9 (编码)
                            sig_ij,  # 列10 (位置方差)
                            n2  # 列11 (噪声方差)
                        ], dtype=float)

                        self.trackStore[gid].append(row)

            # ---------- 后续帧 ----------
            if t > start_point and getattr(self, 'tab_param', None) is not None and self.tab_param.size > 0:
                im_t = stack[t - 1]
                self.im_t = im_t

                blink_col = 7 * (self.t_red - 1) + 7
                isClosed = (self.tab_param[:, blink_col] == self.T_off)
                if np.any(isClosed):
                    keep = ~isClosed
                    self.tab_param = self.tab_param[keep]
                    self.tab_var = self.tab_var[keep]
                    self.tab_moy = self.tab_moy[keep]
                    self.current_ids = [gid for gid, k in zip(self.current_ids, keep) if k]

                # 重编号（首列）
                if self.tab_param.size > 0:
                    self.tab_param[:, 0] = np.arange(0, len(self.tab_param))
                    self.tab_var[:, 0] = self.tab_param[:, 0]
                    self.tab_moy[:, 0] = self.tab_param[:, 0]

                # 排序（blink 降序）
                part_ordre_blk = np.argsort(-self.tab_param[:, blink_col])

                # —— 本帧写入块统一为 write_block = self.t_red；在写入任何列之前，先扩列 —— #
                cur_block = self.t_red - 1
                write_block = self.t_red
                ensure_cols(blk_col(7, write_block))

                # ====== 遍历所有轨迹，尝试重连 ====== #
                for traj in part_ordre_blk:
                    blink_val = int(self.tab_param[traj, 7 * (self.t_red - 1) + 7])
                    if blink_val > self.T_off:
                        part = self.reconnect_part(traj, self.t_red - 1, lest, wn)
                    else:
                        part = -1

                    if part >= 0:
                        # 写入 write_block 的 [i,j,s,r]
                        cols_param = [blk_col(2, write_block), blk_col(3, write_block),
                                      blk_col(4, write_block), blk_col(5, write_block)]
                        self.tab_param[traj, cols_param] = lest[int(part), [1, 2, 3, 5]]

                        # var 列
                        self.tab_var[traj, blk_col(6, write_block)] = lest[int(part), 4]

                        # nb/blink
                        prev_nb_col = blk_col(7, cur_block)
                        cur_nb_col = blk_col(7, write_block)
                        if self.tab_param[traj, prev_nb_col] > 0:
                            LV_traj_part, flag_full_alpha = self.rapport_detection(traj, cur_block, lest, part, wn)
                            self.tab_param[traj, cur_nb_col] += self.nb_stk
                            if flag_full_alpha == 1:
                                self.tab_param[traj, cur_nb_col] += 1
                            else:
                                self.tab_param[traj, cur_nb_col] -= np.mod(self.tab_param[traj, cur_nb_col],
                                                                           self.nb_stk)
                        else:
                            self.tab_param[traj, cur_nb_col] = self.nb_stk

                    else:
                        # 没连上：复制上一块 i/j；s,r=0；var=0；nb/blink 更新
                        prev_cols = [blk_col(2, cur_block), blk_col(3, cur_block)]
                        prev_vals = self.tab_param[traj, prev_cols].astype(float)
                        cols_new = [blk_col(2, write_block), blk_col(3, write_block),
                                    blk_col(4, write_block), blk_col(5, write_block)]
                        self.tab_param[traj, cols_new] = np.concatenate([prev_vals, [0.0, 0.0]])
                        self.tab_var[traj, blk_col(6, write_block)] = 0.0

                        prev_nb_col = blk_col(7, cur_block)
                        cur_nb_col = blk_col(7, write_block)
                        self.tab_param[traj, cur_nb_col] = (self.tab_param[traj, prev_nb_col] - 1
                                                            if self.tab_param[traj, prev_nb_col] < 0 else -1)
                        if cur_block - 1 >= 0:
                            two_back_nb_col = blk_col(7, cur_block - 1)
                            if (two_back_nb_col < self.tab_param.shape[1]) and (
                                    self.tab_param[traj, two_back_nb_col] == 0):
                                self.tab_param[traj, cur_nb_col] = self.T_off

                    if part >= 0:
                        # 标记已使用
                        lest[int(part), 1] = -lest[int(part), 1]
                        lest[int(part), 2] = -lest[int(part), 2]

                # ====== 检测未分配粒子，创建新轨迹 ====== #
                nb_traj_avant_new = self.tab_param.shape[0]
                nb_non_aff_detect = np.zeros(int(par_per_frame[t - 1]), dtype=bool)

                for p in range(int(par_per_frame[t - 1])):
                    if n_dets == 0:
                        break
                    if lest[p, 1] > 0:
                        glrt_1vue, _ = self.rapport_detection(-1, 0, lest, p, wn)
                        if (glrt_1vue > seuil_detec_1vue) and (lest[p, 3] / (np.sqrt(np.pi) * r0) > seuil_alpha):
                            nb_non_aff_detect[p] = True

                new_traj = int(np.sum(nb_non_aff_detect))

                if new_traj > 0:
                    # 有多少轨迹已经存在
                    nb_traj_avant_new = self.tab_param.shape[0]

                    # 新轨迹 ID，从 nb_traj_avant_new 开始
                    ids_new = np.arange(new_traj, dtype=float) + nb_traj_avant_new

                    # 历史空白块 (7 * t_red 列)
                    zeros7t = np.zeros((new_traj, 7 * self.t_red))

                    # 当前帧号 (Python 0-based)
                    time_col = np.full((new_traj, 1), t)

                    # 提取 [y, x, alpha, r] 四列
                    extracted_columns = lest[nb_non_aff_detect][:, [1, 2, 3, 5]]

                    # 填充：0 和 Nb_STK
                    zeros1 = np.zeros((new_traj, 1))
                    nbcol = (self.nb_stk * np.ones((new_traj, 1)))

                    # 拼接
                    tab_param_new = np.hstack([
                        ids_new.reshape(-1, 1),  # 轨迹ID
                        zeros7t,  # 历史空块
                        time_col,  # 当前帧
                        extracted_columns,  # y, x, alpha, r
                        zeros1,  # m0
                        nbcol  # blink 编码 (初始化 Nb_STK)
                    ])

                    # 追加
                    self.tab_param = np.vstack([self.tab_param, tab_param_new])
                    self.tab_var = np.vstack([self.tab_var, np.zeros((new_traj, 7 * (self.t_red + 1) + 1))])
                    self.tab_moy = np.vstack([self.tab_moy, np.zeros((new_traj, 7 * (self.t_red + 1) + 1))])

                    # 初始化方差 (sig² = lest[:,4] = noise²)
                    self.tab_var[nb_traj_avant_new:nb_traj_avant_new + new_traj,
                    7 * self.t_red + 6] = lest[nb_non_aff_detect, 4]

                # ====== shift block，准备下一帧 ====== #
                cur_block = self.t_red - 1
                write_block = self.t_red
                if cur_block >= 1:
                    start_idx = blk_col(1, 1)
                    end_idx = blk_col(7, write_block) + 1
                    new_nb_traj = self.tab_param.shape[0]

                    self.tab_param = np.hstack([
                        self.tab_param[:, 0:1],
                        self.tab_param[:, start_idx:end_idx],
                        (t - 1) * np.ones((new_nb_traj, 1)),
                        np.zeros((new_nb_traj, 6))
                    ])
                    self.tab_var = np.hstack([
                        self.tab_var[:, 0:1],
                        self.tab_var[:, start_idx:end_idx],
                        (t - 1) * np.ones((new_nb_traj, 1)),
                        np.zeros((new_nb_traj, 6))
                    ])
                    self.tab_moy = np.hstack([
                        self.tab_moy[:, 0:1],
                        self.tab_moy[:, start_idx:end_idx],
                        (t - 1) * np.ones((new_nb_traj, 1)),
                        np.zeros((new_nb_traj, 6))
                    ])

                # 仅初始化新加入的行
                new_nb_traj = self.tab_param.shape[0]
                if new_traj > 0:
                    init_tab(np.arange(nb_traj_avant_new, new_nb_traj))

                # 更新统计
                mise_a_jour_tab()

                # 扩展稳定 gid
                if new_traj > 0:
                    self.current_ids.extend(range(self.next_id, self.next_id + new_traj))
                    self.next_id += new_traj

                # ====== 写入 trackStore ====== #
                for track in range(self.tab_param.shape[0]):
                    status_col_here = blk_col(1, self.t_red)
                    if status_col_here < self.tab_param.shape[1] and self.tab_param[track, status_col_here] > 0:
                        gid = self.current_ids[track]
                        block_idx = self.t_red - 1
                        if t > start_point:
                            current_frame = t - 2
                        else:
                            current_frame = t - 1

                        # 关键：检查这一帧是否真的有 localization
                        i_val = self.tab_param[track, blk_col(2, block_idx)]
                        j_val = self.tab_param[track, blk_col(3, block_idx)]
                        alpha = self.tab_param[track, blk_col(4, block_idx)]

                        # 如果 alpha == 0 (说明这一帧没有粒子，只是复制/空补)，就跳过
                        if alpha == 0:
                            continue

                        nb_code = self.tab_param[track, blk_col(7, block_idx)]
                        sig_ij = self.tab_var[track, blk_col(3, block_idx)]
                        n2 = self.tab_var[track, blk_col(6, block_idx)]

                        new_row = np.array([j_val,i_val, current_frame, gid, alpha, nb_code, sig_ij, n2], dtype=float)
                        self.trackStore[gid].append(new_row)

            # 进度
            progress_var.set(t - start_point + 1)
            progress_win.update_idletasks()

        progress_win.destroy()

        # ============ 保存 ============
        total_written = sum(len(v) for v in self.trackStore.values())
        total_locs = len(self.image_bin['ctrsX'])
        print(f"[CHECK] written={total_written}, locs={total_locs}")

        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text files", "*.txt")],
                                                 initialfile=os.path.splitext(os.path.basename(filename))[0],
                                                 title='Save Tracking Data to')
        if file_path:
            pathname = os.path.dirname(file_path)
            rows = []
            for gid in sorted(self.trackStore.keys()):
                for row in self.trackStore[gid]:
                    rows.append(row.tolist())
            df = pd.DataFrame(rows)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            new_file_name = os.path.join(pathname, f'{base_name}_table.txt')
            df.to_csv(new_file_name, sep='\t', index=False, header=False)
            end_time = time.perf_counter()
            elapsed = end_time - start_time

    def reconnect_part(self, traj, t, lest, wn):
        """
        Find the best matching particle for trajectory `traj` at time-block `t`.
        All indices 0-based.
        Returns part (0-based index into lest) or -1 to indicate blink (no match).
        """
        self.Nb_combi = int(self.image_bin.get('maxComp', getattr(self, 'Nb_combi', 1)))

        ind_boule = self.liste_part_boule([], traj, lest, t)
        nb_part_boule = ind_boule.shape[0]

        if nb_part_boule == 0:
            return -1

        vec_traj_inter = self.liste_traj_inter(ind_boule, traj, lest, t)
        vec_traj = np.concatenate(([traj], vec_traj_inter)).astype(int)
        nb_traj = vec_traj.size

        if nb_traj > self.Nb_combi:
            vec_traj = self.limite_combi_traj_blk(vec_traj, t)
            nb_traj = vec_traj.size

        # consider particles in competition
        ind_boule_O1 = ind_boule.copy()
        for ntraj in range(1, nb_traj):
            traj_inter = int(vec_traj[ntraj])
            ind_boule_O1 = self.liste_part_boule(ind_boule_O1, traj_inter, lest, t)

        nb_part_boule_O1 = ind_boule_O1.shape[0]
        ind_boule = ind_boule_O1
        nb_part_boule = nb_part_boule_O1

        if nb_part_boule == 0:
            return -1

        if nb_part_boule > self.Nb_combi:
            ind_boule = self.limite_combi_part_dst(traj, ind_boule, lest, t)
            nb_part_boule = ind_boule.size

        # prepare vec_part: length = nb_traj, pad with -1
        if nb_traj <= nb_part_boule:
            vec_part = np.asarray(ind_boule[:nb_traj], dtype=int)
        else:
            padding = -1 * np.ones(nb_traj - nb_part_boule, dtype=int)
            vec_part = np.concatenate((ind_boule, padding)).astype(int)

        # choose best configuration
        part = int(self.best_config_reconnex(vec_traj, vec_part, t, lest, wn))

        # best_config_reconnex returns element from vec_part (0-based index or -1)
        return part

    def n_plus_proche(self, ic, jc, liste_i, liste_j, N):
        # 计算每个点到 (ic, jc) 的平方距离
        sq_dist = (liste_i - ic) ** 2 + (liste_j - jc) ** 2
        sq_dist_classe, ind_classe = np.sort(sq_dist), np.argsort(sq_dist)
        if len(liste_i) > N:
            indice = ind_classe[:N]
            dist2 = sq_dist_classe[:N]
        else:
            indice = ind_classe
            dist2 = sq_dist_classe
        return indice, dist2

    def limite_combi_part_dst(self, traj, vec_part_in, lest, t):
        vec_part_in = np.asarray(vec_part_in, dtype=int)
        if vec_part_in.size == 0:
            return vec_part_in

        Nb_combi = int(getattr(self, 'Nb_combi', int(self.image_bin.get('maxComp', 1))))

        # 正确的列：i(y)=offset 2, j(x)=offset 3
        col_i = 7 * t + 2
        col_j = 7 * t + 3
        if col_i >= self.tab_param.shape[1] or col_j >= self.tab_param.shape[1]:
            return vec_part_in

        ic = float(self.tab_param[traj, col_i])  # y
        jc = float(self.tab_param[traj, col_j])  # x

        # lest 列：1->i(y), 2->j(x)
        tabi = lest[vec_part_in, 1]
        tabj = lest[vec_part_in, 2]

        inds, _ = self.n_plus_proche(ic, jc, tabi, tabj, Nb_combi)
        return vec_part_in[inds]

    def limite_combi_traj_blk(self, vec_traj_in, t):
        vec_traj_in = np.asarray(vec_traj_in, dtype=int)
        if vec_traj_in.size <= 1:
            return vec_traj_in

        blink_col = 7 * t + 7
        if blink_col >= self.tab_param.shape[1]:
            return vec_traj_in[:getattr(self, 'Nb_combi', 1)]

        tab_blk = self.tab_param[vec_traj_in[1:], blink_col]
        sorted_idx = np.argsort(tab_blk)[::-1]
        keep_k = max(1, int(getattr(self, 'Nb_combi', 1)) - 1)
        selected = sorted_idx[:keep_k]
        selected_trajs = vec_traj_in[1:][selected]
        vec_traj_out = np.concatenate(([vec_traj_in[0]], selected_trajs))
        return vec_traj_out

    def liste_traj_inter(self, liste_part, traj_ref, lest, t):
        Nb_traj = self.tab_param.shape[0]
        Boule_free = getattr(self, 'Boule_free', float(self.image_bin.get('searchExpFac', 1.2)))
        sigij_resuilt = self.sigij_free_blink(np.arange(Nb_traj), t)  # 每条轨迹的自由扩散半径
        boules = Boule_free * sigij_resuilt
        traj_boule = np.zeros(Nb_traj, dtype=bool)

        for p in liste_part:
            p = int(p)
            i0 = float(lest[p, 1])  # 粒子 i(y)
            j0 = float(lest[p, 2])  # 粒子 j(x)

            i0m = np.maximum(i0 - boules, 0)
            i0M = i0 + boules
            j0m = np.maximum(j0 - boules, 0)
            j0M = j0 + boules

            # 轨迹当前块坐标：offset 2 -> i(y), offset 3 -> j(x)
            traj_boule |= (
                    (self.tab_param[:, 7 * t + 2] < i0M) & (self.tab_param[:, 7 * t + 2] > i0m) &
                    (self.tab_param[:, 7 * t + 3] < j0M) & (self.tab_param[:, 7 * t + 3] > j0m)
            )

        # 下一块还未被占用（存在才判断）
        nb_next_col = 7 * (t + 1) + 7
        if nb_next_col < self.tab_param.shape[1]:
            traj_boule &= (self.tab_param[:, nb_next_col] == 0)

        # 当前块是激活态
        traj_boule &= (self.tab_param[:, 7 * t + 7] > 0)

        # 去掉参考轨迹
        if 0 <= traj_ref < Nb_traj:
            traj_boule[traj_ref] = False
        return np.where(traj_boule)[0]

    def liste_part_boule(self, liste_part_ref, traj, lest, t):
        Nb_part = int(lest.shape[0])

        if liste_part_ref is None:
            liste_part_ref = np.array([], dtype=int)
        else:
            liste_part_ref = np.asarray(liste_part_ref, dtype=int)

        if liste_part_ref.size != 0:
            masque_part_ref = np.ones(Nb_part, dtype=bool)
            masque_part_ref[liste_part_ref] = False
        else:
            masque_part_ref = None

        if traj < 0 or traj >= self.tab_param.shape[0]:
            return np.array([], dtype=int)

        # 正确的列：i(y) 用 offset 2，j(x) 用 offset 3
        col_i = 7 * t + 2
        col_j = 7 * t + 3
        if col_i >= self.tab_param.shape[1] or col_j >= self.tab_param.shape[1]:
            return np.array([], dtype=int)

        ic = float(self.tab_param[traj, col_i])  # 轨迹的 i(y)
        jc = float(self.tab_param[traj, col_j])  # 轨迹的 j(x)

        Boule_free = float(self.image_bin.get('searchExpFac', getattr(self, 'Boule_free', 1.2)))
        boule = Boule_free * float(self.sigij_free_blink(traj, t))

        icm = max(ic - boule, 0.0);
        icM = ic + boule
        jcm = max(jc - boule, 0.0);
        jcM = jc + boule

        # lest[:,1] 是 i(y)，lest[:,2] 是 j(x) —— 和上面的边界匹配
        part_boule = ((lest[:, 1] < icM) & (lest[:, 1] > icm) &
                      (lest[:, 2] < jcM) & (lest[:, 2] > jcm))

        if masque_part_ref is not None:
            part_boule &= masque_part_ref

        liste_new_part = np.where(part_boule)[0]
        return np.concatenate((liste_part_ref, liste_new_part)) if liste_part_ref.size else liste_new_part

    def rapport_detection(self, traj, t, lest, part, wn):
        """
        Returns (score, flag_full_alpha).
        Convention: traj is 0-based trajectory index; traj < 0 => compute GLRT for isolated particle (H1 vs H0).
        part must be 0-based particle index into lest.
        """
        Poids_melange_aplha = self.image_bin['intLawWeight']
        Poids_melange_diff = self.image_bin['diffLawWeight']
        T_off = self.T_off
        im_t = self.im_t

        N = wn * wn

        if traj < 0:
            part = int(part)
            sig2_H1 = lest[int(part), 4]
            Pi = int(round(lest[part, 1])) - 1
            Pj = int(round(lest[part, 2])) - 1
            offset = wn // 2
            di = (np.arange(1, wn + 1) + Pi - offset).clip(0, im_t.shape[0] - 1).astype(int)
            dj = (np.arange(1, wn + 1) + Pj - offset).clip(0, im_t.shape[1] - 1).astype(int)
            im_part = im_t[np.ix_(di, dj)]
            sig2_H0 = np.var(im_part, ddof=1)
            out = N * np.log(sig2_H0 / sig2_H1)
            return out, None

        # reconnect case
        if self.tab_param[traj, 7 * t + 7] < 0:
            nb_blink = -self.tab_param[traj, 7 * t + 7]
            sig_blink = -T_off / 3
            Pblink = 2 / (np.sqrt(2 * np.pi) * sig_blink) * np.exp(-1 / (2 * sig_blink ** 2) * nb_blink ** 2)
            Lblink = np.log(Pblink)
        else:
            Lblink = 0

        alpha = lest[int(part), 3]
        alpha_moy = np.real(self.tab_moy[traj, 7 * t + 4])
        sig_alpha = self.tab_var[traj, 7 * t + 4]
        alpha_max = alpha_moy

        Palpha_gaus = (1 / (np.sqrt(2 * np.pi) * sig_alpha)) * np.exp(
            -1 / (2 * sig_alpha ** 2) * (alpha - alpha_moy) ** 2)
        if alpha < alpha_max:
            Palpha_univ = 1 / alpha_max
        else:
            Palpha_univ = 0.0

        poids = Poids_melange_aplha
        Lalpha = np.log(poids * Palpha_gaus + (1 - poids) * Palpha_univ)

        flag_full_alpha = 1 if Palpha_gaus > Palpha_univ else 0

        i0 = lest[int(part), 1]
        j0 = lest[int(part), 2]
        ic = self.tab_param[traj, 7 * t + 2]
        jc = self.tab_param[traj, 7 * t + 3]

        sig_ij_ref = self.sigij_blink(traj, t)
        sig_free_blk = self.sigij_free_blink(traj, t)

        poids = Poids_melange_diff
        Pn_ref = (1 / (2 * np.pi * sig_ij_ref ** 2)) * np.exp(
            -1 / (2 * sig_ij_ref ** 2) * ((i0 - ic) ** 2 + (j0 - jc) ** 2))
        Pn_free = (1 / (2 * np.pi * sig_free_blk ** 2)) * np.exp(
            -1 / (2 * sig_free_blk ** 2) * ((i0 - ic) ** 2 + (j0 - jc) ** 2))
        Ln0 = np.log(poids * Pn_ref + (1 - poids) * Pn_free)

        r = lest[int(part), 5]
        r_ref = self.tab_moy[traj, 7 * t + 5]
        sig_r_ref = self.tab_var[traj, 7 * t + 5]
        if sig_r_ref != 0:
            Lr = -0.5 * np.log(sig_r_ref) - (1 / (2 * sig_r_ref ** 2)) * (r - r_ref) ** 2
        else:
            Lr = 0

        out = Lalpha + Ln0 + Lblink + 0 * Lr
        return out, flag_full_alpha

    def sigij_blink(self, traj, t):
        traj_arr = np.atleast_1d(traj).astype(int)
        out = np.zeros(traj_arr.shape, dtype=float)
        for k, tr in enumerate(traj_arr):
            if tr < 0 or tr >= self.tab_param.shape[0]:
                out[k] = getattr(self, 'sig_free', 1.0)
                continue
            # 正确：blink/nb 列是 offset 7 => 7*t + 7
            blink_col = 7 * t + 7
            offset = self.tab_param[tr, blink_col] if blink_col < self.tab_param.shape[1] else 0

            nb_blink = -offset if offset < 0 else 0

            # i(y) 的位置不确定性存在 tab_var 的 offset 2
            sig_col = 7 * t + 2
            if sig_col < self.tab_var.shape[1]:
                sig_blk = self.tab_var[tr, sig_col]
            else:
                sig_blk = getattr(self, 'sig_free', 1.0)

            if nb_blink > 0:
                sig_blk = sig_blk * np.sqrt(1 + nb_blink)
            out[k] = sig_blk
        return out[0] if out.size == 1 else out

    def sigij_free_blink(self, traj, t):
        traj_arr = np.atleast_1d(traj).astype(int)
        nb_traj = traj_arr.size
        sig_free = getattr(self, 'sig_free', 1.0)
        out = np.ones(nb_traj, dtype=float) * sig_free
        for k, tr in enumerate(traj_arr):
            if tr < 0 or tr >= self.tab_param.shape[0]:
                nb_blink = 0
            else:
                blink_col = 7 * t + 7  # 修正
                offset = self.tab_param[tr, blink_col] if blink_col < self.tab_param.shape[1] else 0
                nb_blink = -offset if offset < 0 else 0
            out[k] = sig_free * np.sqrt(1 + nb_blink)
        return out[0] if out.size == 1 else out
    def best_config_reconnex(self, vec_traj, vec_part, t, lest, wn):
        nb_traj = int(np.asarray(vec_traj).size)
        if nb_traj == 0:
            return -1

        vec_traj = np.asarray(vec_traj, dtype=int).ravel()
        vec_part = np.asarray(vec_part).ravel()

        # helper: map part -> valid python index or None
        n_lest = int(lest.shape[0]) if hasattr(lest, 'shape') else 0

        def is_valid_part(p):
            try:
                pv = int(p)
            except Exception:
                return False
            return (pv >= 0 and pv <= n_lest - 1)

        # compute baseline using first nb_traj elements
        vec_part_ref = vec_part[:nb_traj]
        best_part = int(vec_part_ref[0]) if vec_part_ref.size > 0 else -1
        vrais = self.vrais_config_reconnex(vec_traj, vec_part_ref, t, lest, wn)

        # try permutations up to a safe limit
        max_exhaustive = 8
        if vec_part.size <= max_exhaustive:
            for perm in itertools.permutations(vec_part, r=nb_traj):
                perm = np.asarray(perm)
                vrais_tmp = self.vrais_config_reconnex(vec_traj, perm[:nb_traj], t, lest, wn)
                if vrais_tmp > vrais:
                    vrais = vrais_tmp
                    best_part = int(perm[0])
            return best_part
        else:
            # greedy fallback
            assigned = set()
            assignment = [-1] * nb_traj
            # precompute per-(traj,part) scores
            scores = {}
            for pidx, pval in enumerate(vec_part):
                for k, traj in enumerate(vec_traj):
                    if is_valid_part(pval):
                        s, _ = self.rapport_detection(int(traj), t, lest, int(pval), wn)
                    else:
                        s = -np.inf
                    scores[(k, pidx)] = float(s)
            for k in range(nb_traj):
                best_pidx = None
                best_s = -np.inf
                for pidx, pval in enumerate(vec_part):
                    if pidx in assigned:
                        continue
                    s = scores.get((k, pidx), -np.inf)
                    if s > best_s:
                        best_s = s
                        best_pidx = pidx
                if best_pidx is None:
                    assignment[k] = -1
                else:
                    assignment[k] = int(vec_part[best_pidx])
                    assigned.add(best_pidx)
            return assignment[0]

    def vrais_config_reconnex(self, vec_traj, vec_part, t, lest, wn):
        vec_traj = np.asarray(vec_traj, dtype=int).ravel()
        vec_part = np.asarray(vec_part).ravel()
        if vec_traj.size != vec_part.size:
            raise ValueError("vec_traj and vec_part must have same length")
        vrais = 0.0
        n_lest = int(lest.shape[0]) if hasattr(lest, 'shape') else 0
        for k in range(vec_part.size):
            part = int(vec_part[k])
            traj = int(vec_traj[k])
            if part < 0:
                continue  # -1 means no particle
            if part >= n_lest:
                continue
            val, _ = self.rapport_detection(traj, t, lest, part, wn)
            vrais += float(val)
        return vrais
    #####tracking结束后，extra开始
    def render_movie(self):
        """
        实现 monoView 模式下 Movie 分支：
          - 计算第一帧加载参数（基于累积检测数）
          - 调用数据预处理与渲染函数生成电影帧
          - 对每一帧进行后处理（添加色图、比例尺、时间戳）
          - 使用 imageio 保存 AVI 电影文件
          - 同时在 SLIMfast Data Viewer 窗口中创建预览，显示电影投影，
            并在图上绘制对角线和提示文本
        返回：
          preview_fig: 预览窗口的句柄
        """
        # 提示用户确认生成电影
        if not messagebox.askyesno("Movie Preview", "是否确认电影设置并生成电影？"):
            return

        # 弹出保存对话框选择 AVI 文件保存路径
        movie_path = filedialog.asksaveasfilename(defaultextension=".avi", title="保存电影为")
        if not movie_path:
            return

        # 从 self.image_bin 中读取电影参数
        fps = self.image_bin.get('fps', 10)
        r_start = self.image_bin.get('r_start', 1)
        r_end = int(self.image_bin['r_end'])
        rW = self.image_bin.get('rW', 20)  # 数据窗口大小
        rStep = self.image_bin.get('rStep', 1)

        # 计算电影帧数，至少 1 帧
        movie_frames = max(int(np.ceil((r_end - r_start + 1 - rW) / rStep)), 1)

        # 创建视频写入器
        writer = imageio.get_writer(movie_path, fps=fps, format='ffmpeg')

        # 计算累积检测数数组 idx, 模拟 MATLAB 中的 [-1; cumsum(ctrsN)]
        ctrsN = np.array(self.image_bin['ctrsN'])
        idx = np.concatenate(([-1], np.cumsum(ctrsN)))

        # 根据转换模式设置返回标志向量（此处未做后续处理，仅保留原逻辑）
        conv_mode = self.image_bin['conv_mode']
        if conv_mode == 1:
            return_val = [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        else:
            return_val = [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]

        # 循环生成每一帧
        first_frame = None  # 用来存储第一帧图像
        #这里改成100,原来是movie_frames+1
        for movie_frame in tqdm(range(1, movie_frames+1), desc="生成电影帧"):
            # 计算当前帧加载参数
            if movie_frame == 1:
                self.image_bin['startPnt'] = idx[r_start] + 1
                self.image_bin['elements'] = idx[r_start + rW] - max(1, self.image_bin['startPnt']) + 1
            elif movie_frame == movie_frames:
                if self.image_bin.get('is_cumsum', False):
                    self.image_bin['startPnt'] = idx[r_start] + 1
                else:
                    self.image_bin['startPnt'] = idx[r_start + (movie_frames - 1) * rStep] + 1
                self.image_bin['elements'] = idx[r_end] - self.image_bin['startPnt'] + 1
            else:
                if self.image_bin.get('is_cumsum', False):
                    self.image_bin['startPnt'] = idx[r_start] + 1
                else:
                    self.image_bin['startPnt'] = idx[r_start + (movie_frame - 1) * rStep] + 1
                self.image_bin['elements'] = idx[r_start + (movie_frame - 1) * rStep + rW] - self.image_bin[
                    'startPnt'] + 1

            # 计算当前帧在全局数据中的起始和结束索引（转换为0-based）
            start_idx = self.image_bin['startPnt'] - 1
            end_idx = start_idx + self.image_bin['elements']

            # 构造当前帧的数据字典，只包括这一帧的粒子数据
            data = {
                'ctrsX': self.image_bin['ctrsX'][start_idx:end_idx],
                'ctrsY': self.image_bin['ctrsY'][start_idx:end_idx],
                'photons': self.image_bin['photons'][start_idx:end_idx],
                'precision': self.image_bin['precision'][start_idx:end_idx]
            }
            # 固定 ROI 为 [0, 0, 320, 320]
            roi_fixed = [0, 0, 320, 320]

            # 调用渲染函数生成当前帧图像
            frame_img, new_width, new_height, N = self.render_image(
                data, roi_fixed,
                self.image_bin['exf_new'],
                self.image_bin['width'],
                self.image_bin['height'],
                self.image_bin['int_weight'],
                self.image_bin['size_fac'],
                self.image_bin['pxSize'],
                conv_mode
            )

            # 如果是第一帧，保存下来用于合成预览图中下三角部分
            if movie_frame == 1:
                first_frame = np.copy(frame_img)

            # 后处理：添加色图、比例尺、时间戳
            if self.image_bin.get('is_colormap', False):
                frame_img = self.imprint_colormap(frame_img, new_width, self.image_bin.get('colormapWidth'), 1)
            if self.image_bin.get('is_scalebar', False):
                frame_img = self.imprint_scalebar(
                    frame_img, new_width, new_height,
                    self.image_bin['exf_new'],
                    self.image_bin.get('micron_bar_length'),
                    self.image_bin['pxSize'],
                    self.image_bin.get('timestampSize'),
                    1
                )
            if self.image_bin.get('is_timestamp', False):
                # 当前 movie_frame 乘以 timestampInkrement 作为时间值
                timestamp = movie_frame * self.image_bin.get('timestampInkrement', 1)
                frame_img = self.imprint_timestamp(
                    frame_img, new_width, new_height,
                    timestamp,
                    self.image_bin.get('timestampSize'),
                    1
                )

            # 写入当前帧到视频文件
            writer.append_data(np.real(frame_img).astype(np.uint8))

        writer.close()
        messagebox.showinfo("Success", f"电影已成功保存：{movie_path}")

        # 生成第一帧（Frame 1）的图像预览
        # 假设在循环中我们保存了第一帧至 first_frame（需要在 movie_frame == 1 时保存）
        first_frame_disp = np.real(first_frame).copy()  # Frame 1

        # 生成累积图（Accumulate Frame），使用所有粒子的坐标进行渲染
        # 构造数据字典，包含全局所有粒子的 x, y, photons, precision
        data_all = {
            'ctrsX': self.image_bin['ctrsX'],
            'ctrsY': self.image_bin['ctrsY'],
            'photons': self.image_bin['photons'],
            'precision': self.image_bin['precision']
        }
        # 固定 ROI 为 [0, 0, 320, 320]（请根据实际情况修改）
        roi_fixed = [0, 0, 320, 320]
        # 这里传入其它参数与之前一致，conv_mode 应该已定义
        accumulate_frame, new_width, new_height, N_all = self.render_image(
            data_all, roi_fixed,
            self.image_bin['exf_new'],
            self.image_bin['width'],
            self.image_bin['height'],
            self.image_bin['int_weight'],
            self.image_bin['size_fac'],
            self.image_bin['pxSize'],
            conv_mode
        )
        accumulate_frame = np.real(accumulate_frame).copy()

        # 假设 first_frame 已在电影帧循环中保存（第一帧图像），
        # 并且 accumulate_frame 已使用全部粒子数据进行渲染获得（最后一帧）
        first_frame_uint8 = np.real(first_frame).astype(np.uint8)
        accumulate_frame_uint8 = np.real(accumulate_frame).astype(np.uint8)

        # 合成预览图像：上三角（包括主对角线）使用第一帧数据， 下三角使用累积数据（不包含主对角线）
        composite = np.triu(accumulate_frame_uint8) + np.tril(first_frame_uint8, k=-1)

        # 创建预览窗口
        preview_fig = tk.Toplevel(self.master)
        preview_fig.title("Movie Preview")
        preview_fig.geometry("800x600")

        fig_movie = plt.Figure(figsize=(8, 6))
        ax_movie = fig_movie.add_subplot(111)
        ax_movie.imshow(composite)

        # 绘制主对角线，从左上 (0,0) 到右下 (new_width-1, new_height-1)
        ax_movie.plot([0, new_width - 1], [0, new_height - 1], color='white', linewidth=3)

        # 计算上三角区域（Frame 1）的质心
        # 上三角包括所有满足 i <= j 的像素，质心可以近似取三个顶点的平均值
        # 取顶点为：左上 (0,0), 右上 (new_width-1, 0), 右下 (new_width-1, new_height-1)
        x_frame1 = (0 + (new_width - 1) + (new_width - 1)) / 3.0  # = 2*(new_width - 1)/3
        y_frame1 = (0 + 0 + (new_height - 1)) / 3.0  # = (new_height - 1)/3

        # 计算下三角区域（Accumulate）的质心
        # 下三角包括所有满足 i > j 的像素，取顶点为：左上 (0,0), 左下 (0, new_height-1), 右下 (new_width-1, new_height-1)
        x_accum = (0 + 0 + (new_width - 1)) / 3.0  # = (new_width - 1)/3
        y_accum = (0 + (new_height - 1) + (new_height - 1)) / 3.0  # = 2*(new_height - 1)/3

        # 在对应区域添加文本标签，调整位置使其远离主对角线
        ax_movie.text(x_frame1, y_frame1, 'Accumulate', fontsize=15, color='white',
                      ha='center', va='center')
        ax_movie.text(x_accum, y_accum, 'Frame 1', fontsize=15, color='white',
                      ha='center', va='center')

        ax_movie.axis('off')

        canvas_movie = FigureCanvasTkAgg(fig_movie, master=preview_fig)
        canvas_movie.draw()
        canvas_movie.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        preview_fig.deiconify()

        return preview_fig

    def export_three_snapshots(self):
        """
        生成三张静态图（300 dpi）：
          1) first_frame.png  —— 第一帧窗口渲染
          2) accumulate.png   —— 累积渲染（全体粒子）
          3) composite_tri.png—— 上三角=累积，下三角=第一帧（无标注文字）
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path

        # -------- 1) 固定输出目录 --------
        out_dir = Path(r"C:\Users\asus\OneDrive\图片\Saved Pictures")
        out_dir.mkdir(parents=True, exist_ok=True)

        # -------- 2) 读取必要参数 --------
        r_start = int(self.image_bin.get('r_start', 1))
        r_end = int(self.image_bin['r_end'])
        rW = int(self.image_bin.get('rW', 20))
        rStep = int(self.image_bin.get('rStep', 1))
        conv_mode = int(self.image_bin.get('conv_mode', 1))

        exf_new = self.image_bin['exf_new']
        width = self.image_bin['width']
        height = self.image_bin['height']
        int_weight = self.image_bin['int_weight']
        size_fac = self.image_bin['size_fac']
        pxSize = self.image_bin['pxSize']

        roi_fixed = [0, 0, width, height]

        # -------- 3) 索引准备 --------
        ctrsN = np.asarray(self.image_bin['ctrsN'], dtype=int)
        idx = np.concatenate(([-1], np.cumsum(ctrsN)))

        # -------- 4) 第一帧窗口数据范围 --------
        startPnt = idx[r_start] + 1
        elements = idx[r_start + rW] - max(1, startPnt) + 1
        s0 = int(startPnt - 1)
        e0 = int(s0 + elements)

        data_first = {
            'ctrsX': self.image_bin['ctrsX'][s0:e0],
            'ctrsY': self.image_bin['ctrsY'][s0:e0],
            'photons': self.image_bin['photons'][s0:e0],
            'precision': self.image_bin['precision'][s0:e0],
        }

        # -------- 5) 渲染第一帧 --------
        frame_first, new_w, new_h, _ = self.render_image(
            data_first, roi_fixed, exf_new, width, height,
            int_weight, size_fac, pxSize, conv_mode
        )
        first_u8 = np.real(frame_first).astype(np.uint8)

        # -------- 6) 渲染累计图 --------
        data_all = {
            'ctrsX': self.image_bin['ctrsX'],
            'ctrsY': self.image_bin['ctrsY'],
            'photons': self.image_bin['photons'],
            'precision': self.image_bin['precision'],
        }
        frame_acc, _, _, _ = self.render_image(
            data_all, roi_fixed, exf_new, width, height,
            int_weight, size_fac, pxSize, conv_mode
        )
        acc_u8 = np.real(frame_acc).astype(np.uint8)

        # -------- 7) 对齐尺寸并合成三角图 --------
        H = min(first_u8.shape[0], acc_u8.shape[0])
        W = min(first_u8.shape[1], acc_u8.shape[1])
        first_u8 = first_u8[:H, :W]
        acc_u8 = acc_u8[:H, :W]
        composite = np.triu(acc_u8) + np.tril(first_u8, k=-1)

        # -------- 8) 保存三张 PNG（300 dpi） --------
        def save_fig(img, path):
            fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=300)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            plt.subplots_adjust(0, 0, 1, 1)
            fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        p1 = out_dir / "first_frame.png"
        p2 = out_dir / "accumulate.png"
        p3 = out_dir / "composite_tri.png"

        save_fig(first_u8, p1)
        save_fig(acc_u8, p2)
        save_fig(composite, p3)

        print(f" Saved 300 dpi images to: {out_dir}")
        print(f"  ├─ {p1.name}")
        print(f"  ├─ {p2.name}")
        print(f"  └─ {p3.name}")

    def imprint_colormap(self, I, width, mapHeight, mode):
        """
        在图像 I 的最后 mapHeight 行叠加颜色条，用于显示色图。

        参数：
          I: 输入图像（如果 mode==1，则为二维灰度图像；否则为三通道彩色图像，
             其形状为 (height, width) 或 (height, width, 3)）
          width: 图像宽度（用于生成色条）
          mapHeight: 色条的高度（像素）
          mode: 模式标志
                - mode == 1：生成二维色条，并将其替换 I 的最后 mapHeight 行；
                - mode == 3：生成三通道色条，其中第三个通道是一个垂直渐变；
                - 其他模式：生成三通道色条，第三通道全为 0。

        返回：
          修改后的图像 I
        """
        if mode == 1:
            # 生成从 0 到 255 的等间隔数列（长度为 width），并复制 mapHeight 行
            ramp = np.round(np.linspace(0, 255, width)).astype(np.uint8)
            colorRamp = np.tile(ramp, (mapHeight, 1))  # shape: (mapHeight, width)
            # 替换图像 I 最后 mapHeight 行（假定 I 为二维数组）
            I[-mapHeight:, :] = colorRamp
        else:
            # 计算 cmapWidth = floor(width/4)
            cmapWidth = int(np.floor(width / 4))
            # 生成色条一部分：从 0 到 255 的等间隔数列，长度为 cmapWidth
            ramp = np.round(np.linspace(0, 255, cmapWidth)).astype(np.uint8)
            # 生成一个全255的向量，长度为 cmapWidth
            ones_255 = np.full((cmapWidth,), 255, dtype=np.uint8)
            # 反转 ramp
            ramp_flipped = np.flip(ramp)
            # 生成长度为 (width - 3*cmapWidth) 的零向量
            zeros_vec = np.zeros((width - 3 * cmapWidth,), dtype=np.uint8)
            # 拼接得到一行色条
            row_vec = np.concatenate([ramp, ones_255, ramp_flipped, zeros_vec])
            # 复制 mapHeight 行，形成二维色条
            colorRamp = np.tile(row_vec, (mapHeight, 1))

            if mode == 3:
                # 生成垂直渐变：从 0 到 255 的等间隔数列，长度为 mapHeight，转置为列向量
                ramp_vertical = np.round(np.linspace(0, 255, mapHeight)).astype(np.uint8).reshape((mapHeight, 1))
                # 水平复制，得到与色条相同尺寸的矩阵
                colorRamp3D = np.tile(ramp_vertical, (1, width))
                # 生成 3 通道色条：第一通道为 colorRamp，第二通道为 np.fliplr(colorRamp)，第三通道为 colorRamp3D
                new_patch = np.dstack((colorRamp, np.fliplr(colorRamp), colorRamp3D))
            else:
                # 其他模式：第三通道全为 0
                new_patch = np.dstack((colorRamp, np.fliplr(colorRamp), np.zeros_like(colorRamp, dtype=np.uint8)))

            # 替换图像 I 最后 mapHeight 行（假定 I 为三通道数组）
            I[-mapHeight:, :, :] = new_patch

        return I


    def show_help(self):
        """Show help documentation."""
        messagebox.showinfo("Help", "This is the help documentation.")

    def get_selected_figures(self):
        """
        获取选中的图像和跟踪窗口句柄 (对应 MATLAB 的 getSelectedFigHandles)
        返回: (im_figures, track_figures)
        """
        # 通道菜单标签映射
        channel_tags = {
            'image': ['menuRedChannel', 'menuGreenChannel',
                      'menuBlueChannel', 'menuGrayChannel'],
            'track': ['menuTrackChannel']
        }

        # 初始化结果容器
        im_figures = [None] * 4
        track_figures = []

        # 查找根窗口
        root_window = self.master

        # 处理图像通道
        for idx, tag in enumerate(channel_tags['image']):
            menu = self.image_bin.get(tag)
            if menu:
                # 获取菜单项选中状态
                for item in self.get_menu_items(menu):
                    if menu.entrycget(item, 'state') == 'checked':
                        im_figures[idx] = self.image_bin[f'{tag}_data']

        # 处理跟踪通道
        track_menu = self.image_bin.get(channel_tags['track'][0])
        if track_menu:
            for item in self.get_menu_items(track_menu):
                if track_menu.entrycget(item, 'state') == 'checked':
                    fig_data = self.image_bin.get(f'{channel_tags["track"][0]}_data')
                    if isinstance(fig_data, list):
                        track_figures.extend(fig_data)
                    else:
                        track_figures.append(fig_data)

        return im_figures, track_figures

    def transfer_var_list(self, target_window, src_windows, mode):
        """
        传输变量列表到目标窗口 (对应 MATLAB 的 transferVarList)
        :param target_window: 目标窗口对象
        :param src_windows: 源窗口对象列表
        :param mode: 传输模式 ('render-mono' 或 'merge')
        """
        # 定义变量映射表
        var_mapping = {
            'render-mono': [
                'is_own', 'is_loaded', 'is_superstack', 'is_track',
                'frame', 'view_mode', 'loc_start', 'loc_end', 'error_rate',
                'w2d', 'dfltn_loops', 'min_int', 'loc_parallel', 'n_cores',
                'spatial_correction', 'is_radius_tol', 'radius_tol', 'pos_tol',
                'max_optim_iter', 'term_tol', 'is_scalebar', 'micron_bar_length',
                'is_colormap', 'colormap_width', 'is_timestamp', 'timestamp_size',
                'timestamp_inkrement', 'cluster_mode', 'exf_old', 'exf_new', 'r_w',
                'r_step', 'r_start', 'r_end', 'r_live', 'fps', 'mov_compression',
                'conv_mode', 'int_weight', 'size_fac', 'is_cumsum', 'is_thresh_loc_prec',
                'min_loc', 'max_loc', 'is_thresh_snr', 'min_snr', 'max_snr',
                'is_thresh_density', 'px_size', 'cnts_per_photon', 'em_wvlnth', 'na',
                'psf_scale', 'psf_std', 'image_name', 'pathname', 'filename',
                'width', 'height', 'roi_x0', 'roi_y0', 'roi', 'h_roi',
                'intensity_range', 'ctrs_n'
            ],
            'merge': [
                'px_size', 'is_scalebar', 'micron_bar_length',
                'is_colormap', 'colormap_width', 'is_timestamp',
                'timestamp_size', 'timestamp_inkrement', 'exf_new',
                'r_step', 'fps', 'mov_compression'
            ]
        }

        var_list = var_mapping.get(mode, [])
        if not var_list:
            raise ValueError(f"无效的传输模式: {mode}")

        # 根据模式处理传输逻辑
        if mode == 'render-mono':
            for var in var_list:
                target_window.image_data[var] = src_windows[0].image_data.get(var)
        elif mode == 'merge':
            for var in var_list:
                values = []
                for src in src_windows:
                    val = src.image_data.get(var)
                    if val is not None:
                        values.append(val)

                # 检查参数一致性
                if self.check_parameter_consistency(values):
                    target_window.image_data[var] = values[0]
                else:
                    self.handle_inconsistent_parameter(target_window, var, values)

    # Helper methods
    def get_menu_items(self, menu):
        """获取菜单项索引列表"""
        return range(menu.index('end') + 1)

    def check_parameter_consistency(self, values):
        """检查参数值是否一致"""
        if not values:
            return False
        return all(v == values[0] for v in values)

    def handle_inconsistent_parameter(self, target_window, param_name, values):
        """处理不一致参数"""
        answer = simpledialog.askstring(
            "参数不一致",
            f"通道间参数 {param_name} 不一致，请输入新值:",
            parent=self.master
        )
        try:
            converted_value = self.convert_value_type(answer, values)
            target_window.image_data[param_name] = converted_value
        except (ValueError, TypeError):
            self.show_error("无效的输入值")

    def convert_value_type(self, value, examples):
        """根据示例值类型转换输入值"""
        if not examples:
            return value

        example_type = type(examples[0])
        if example_type == bool:
            return value.lower() in ('true', '1', 't')
        if example_type == int:
            return int(value)
        if example_type == float:
            return float(value)
        return value

    def export_settings(self, vis_mode):
        """
        导出设置到文本文件 (对应 MATLAB 的 exportsettings)
        :param vis_mode: 可视化模式 ('Image' 或 'Movie')
        """
        # 获取保存路径
        init_dir = self.image_bin.get('search_path', '.')
        file_path = filedialog.asksaveasfilename(
            title='Save Settings',
            defaultextension='.txt',
            initialdir=init_dir,
            filetypes=[('Text files', '*.txt')]
        )

        if not file_path:
            return

        # 更新搜索路径
        self.image_bin['search_path'] = file_path.rsplit('/', 1)[0]

        try:
            with open(file_path, 'w') as fid:
                # 写入基础信息
                fid.write(f"Filename: {self.image_bin.get('pathname', '')}"
                          f"{self.image_bin.get('filename', '')}.mat\n")

                # 处理图像名称
                if self.image_bin.get('isSuperstack', False):
                    for name in self.image_bin.get('imageName', []):
                        fid.write(f"Imagename: {name}\n")
                else:
                    fid.write(f"Imagename: {self.image_bin.get('imageName', '')}\n")

                # 写入时间戳
                fid.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                # 光学选项
                fid.write("\nOptics Options:\n")
                self._write_param(fid, 'Pixelsize [?m] =', 'pxSize', 0.1)
                self._write_param(fid, 'Numerical Aperture =', 'NA', 1.4)
                self._write_param(fid, 'Emission Wavelength [nm] =', 'emWvlnth', 520)
                self._write_param(fid, 'PSF Scaling Factor =', 'psfScale', 1.0)
                self._write_param(fid, 'PSF [px] =', 'psfStd', 1.5)
                self._write_param(fid, 'Counts/Photon =', 'cntsPerPhoton', 100)

                # 定位选项
                fid.write("\nLocalization Options:\n")
                self._write_range(fid, 'Framerange =', 'locStart', 'locEnd', (1, 100))
                self._write_param(fid, 'ErrorRate [10^] =', 'errorRate', 3)
                self._write_param(fid, 'Detectionbox [px] =', 'w2d', 9)
                self._write_param(fid, '# of Deflationloops =', 'dfltnLoops', 5)
                self._write_param(fid, 'Intensity Threshold [Photons] =', 'minInt', 100)

                # 渲染选项
                fid.write("\nRendering Options:\n")
                self._write_range(fid, 'Framerange =', 'rStart', 'rEnd', (1, 100))
                self._write_param(fid, 'Expansion Factor =', 'exfNew', 1.5)

                conv_mode = self.image_bin.get('convMode', 1)
                fid.write("Rendering Method = {}\n".format(
                    "dynamic convolution" if conv_mode == 2 else "fixed convolution"
                ))
                self._write_param(fid, 'Intensity Weighting Factor =', 'intWeight', 1.0)
                self._write_param(fid, 'Size Factor (times Loc. Prec.) =', 'sizeFac', 1.0)

                # 可视化模式特定参数
                fid.write(f"\nVisualization Method = {vis_mode}\n")
                if vis_mode == 'Movie':
                    self._write_param(fid, 'Windowsize =', 'rW', 10)
                    self._write_param(fid, 'Stepsize =', 'rStep', 1)
                    self._write_param(fid, 'Frames/Second =', 'fps', 30)

                    comp_map = {1: 'RLE', 2: 'MSVL', 3: 'no Compression'}
                    comp = comp_map.get(self.image_bin.get('movCompression', 3), 'no Compression')
                    fid.write(f"Compression Algorithm = {comp}\n")

                # 过滤选项
                fid.write("\nFilter Options:\n")
                self._write_switch(fid, 'Localization Precision =', 'isThreshLocPrec')
                self._write_range(fid, 'Filter Range =', 'minLoc', 'maxLoc', (0, 100))
                self._write_switch(fid, 'Signal to Noise Ratio =', 'isThreshSNR')
                self._write_range(fid, 'Filter Range =', 'minSNR', 'maxSNR', (0, 100))

        except Exception as e:
            tk.messagebox.showerror("保存错误", f"文件保存失败: {str(e)}")

    def _write_param(self, fid, label, key, default):
        """写入单个参数"""
        value = self.image_bin.get(key, default)
        fid.write(f"{label} {value:.4f}\n")

    def _write_range(self, fid, label, start_key, end_key, default):
        """写入范围参数"""
        start = self.image_bin.get(start_key, default[0])
        end = self.image_bin.get(end_key, default[1])
        fid.write(f"{label} {start:.1f} to {end:.1f}\n")

    def _write_switch(self, fid, label, key):
        """写入开关状态"""
        state = "Enabled" if self.image_bin.get(key, False) else "Disabled"
        fid.write(f"{label} {state}\n")

    def densitybasedClustering(self, data):
        """密度聚类分析主函数"""
        self.cluster_window = tk.Toplevel(self.master)
        self.cluster_window.title("Densitybased Clustering")
        self.cluster_window.geometry("800x600")

        # 初始化参数
        self.cluster_data = data
        self.scale_factors = {'x': 1.0, 'y': 1.0, 'z': 0.01}
        self.sigma = {'x': 1.0, 'y': 1.0, 'z': 1.0}
        self.kdtree = None
        self.points_num = len(data['ctrsX'])

        # 创建GUI组件
        self._create_cluster_widgets()
        self._create_visualization()
        self._update_status("Ready")

    def _create_cluster_widgets(self):
        """创建控件"""
        # 左侧可视化区域
        self.viz_frame = ttk.Frame(self.cluster_window)
        self.viz_frame.place(relwidth=0.7, relheight=1.0)

        # 右侧控制面板
        control_frame = ttk.Frame(self.cluster_window)
        control_frame.place(relx=0.7, rely=0, relwidth=0.3, relheight=1.0)

        # 缩放控制
        ttk.Label(control_frame, text="Scale(x,y,z):").grid(row=0, column=0, sticky='w')
        self.scale_x = ttk.Entry(control_frame, width=5)
        self.scale_x.insert(0, "1")
        self.scale_x.grid(row=0, column=1)

        self.scale_y = ttk.Entry(control_frame, width=5)
        self.scale_y.insert(0, "1")
        self.scale_y.grid(row=0, column=2)

        self.scale_z = ttk.Entry(control_frame, width=5)
        self.scale_z.insert(0, "0.01")
        self.scale_z.grid(row=0, column=3)

        self.lock_btn = ttk.Checkbutton(
            control_frame, text="lock",
            command=self.toggle_scale_lock
        )
        self.lock_btn.grid(row=0, column=4)

        # 权重设置
        self.weight_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            control_frame, text="Set 3D-Gaussian Weights:",
            variable=self.weight_var,
            command=self.toggle_weights
        ).grid(row=1, column=0, columnspan=4, sticky='w')

        ttk.Label(control_frame, text="Sigma(x,y,z):").grid(row=2, column=0, sticky='w')
        self.sigma_x = ttk.Entry(control_frame, width=5)
        self.sigma_x.insert(0, "1")
        self.sigma_x.grid(row=2, column=1)

        self.sigma_y = ttk.Entry(control_frame, width=5)
        self.sigma_y.insert(0, "1")
        self.sigma_y.grid(row=2, column=2)

        self.sigma_z = ttk.Entry(control_frame, width=5)
        self.sigma_z.insert(0, "1")
        self.sigma_z.grid(row=2, column=3)

        # 分数设置
        ttk.Label(control_frame, text="min Score:").grid(row=3, column=0, sticky='w')
        self.score_entry = ttk.Entry(control_frame, width=8)
        self.score_entry.insert(0, "1")
        self.score_entry.grid(row=3, column=1, columnspan=2)

        ttk.Button(
            control_frame, text="critical",
            command=self.eval_critical_score
        ).grid(row=3, column=3)

        # 搜索半径
        ttk.Label(control_frame, text="Search Radius:").grid(row=4, column=0, sticky='w')
        self.radius_entry = ttk.Entry(control_frame, width=8, state='disabled')
        self.radius_entry.grid(row=4, column=1, columnspan=2)

        self.unit_combo = ttk.Combobox(
            control_frame, values=['px', 'nm'],
            state='disabled', width=5
        )
        self.unit_combo.current(0)
        self.unit_combo.grid(row=4, column=3)

        # 操作按钮
        ttk.Button(control_frame, text="Evaluate", command=self._eval_density).grid(row=5, column=0, columnspan=2,
                                                                                    pady=10)
        ttk.Button(control_frame, text="Accept", command=self._accept_clustering).grid(row=5, column=2, columnspan=2,
                                                                                       pady=10)

    def _create_visualization(self):
        """创建可视化图表"""
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.ax_main = self.fig.add_subplot(111, projection='3d')

    def _build_kdtree(self):
        """构建KD树"""
        try:
            self.scale_factors = {
                'x': float(self.scale_x.get()),
                'y': float(self.scale_y.get()),
                'z': float(self.scale_z.get())
            }

            scaled_points = np.column_stack((
                self.cluster_data['ctrsX'] * self.scale_factors['x'],
                self.cluster_data['ctrsY'] * self.scale_factors['y'],
                self.cluster_data['frame'] * self.scale_factors['z']
            ))

            if self.kdtree is not None:
                del self.kdtree  # 释放之前的内存
            self.kdtree = cKDTree(scaled_points)
            self._update_status("KDTree built successfully")

        except ValueError as e:
            self._show_error(f"Invalid scale value: {str(e)}")

    def _eval_density(self):
        """执行密度聚类"""
        try:
            # 获取参数
            score = float(self.score_entry.get())
            radius = self._get_search_radius()

            # 执行聚类
            labels = self._custom_dbscan(radius, score)

            # 更新可视化
            self._update_visualization(labels)

        except Exception as e:
            self.show_error(f"Clustering failed: {str(e)}")

    def _custom_dbscan(self, eps, min_samples):
        """简化的DBSCAN实现"""
        # 此处应实现完整的DBSCAN算法
        # 返回聚类标签数组
        return np.zeros(len(self.cluster_data['ctrsX']), dtype=int)

    def _eval_critical_score(self):
        """评估临界分数"""
        try:
            # 获取蒙特卡洛参数
            vals = simpledialog.askstring(
                "Critical Score Settings",
                "Enter # simulations and alpha (comma separated):",
                initialvalue="10,0.001"
            )
            iter_num, alpha = map(float, vals.split(','))

            # 计算临界分数
            critical_score = self._estimate_critical_score(iter_num, alpha)
            self.score_entry.delete(0, tk.END)
            self.score_entry.insert(0, f"{critical_score:.2f}")

        except Exception as e:
            self.show_error(f"Critical score evaluation failed: {str(e)}")

    def _estimate_critical_score(self, iterations, alpha):
        """蒙特卡洛模拟估计临界分数"""
        # 实现完整的蒙特卡洛模拟逻辑
        return 1.0  # 示例返回值

    def _accept_clustering(self):
        """接受聚类结果"""
        if self.kdtree is not None:
            del self.kdtree
        self.cluster_window.destroy()

    def _update_visualization(self, labels):
        """更新可视化图表"""
        self.ax_main.clear()

        # 绘制3D散点图
        unique_labels = np.unique(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            self.ax_main.scatter(
                self.cluster_data['ctrsX'][mask],
                self.cluster_data['ctrsY'][mask],
                self.cluster_data['frame'][mask],
                c=[color],
                marker='o',
                s=10
            )

        self.canvas.draw()

    def _update_status(self, message):
        """更新状态栏"""
        if hasattr(self, 'status_var'):
            self.status_var.set(message)

    def dbscan(self, tree, x, y, t, critical_score, radius, sigma=None):
        """
        DBSCAN 密度聚类算法 (对应 MATLAB 的 dbscan 函数)
        参数:
            tree: cKDTree 对象
            x, y, t: 坐标和时间数据 (numpy数组)
            critical_score: 核心点阈值
            radius: 搜索半径
            sigma: 高斯权重参数 [sigma_x, sigma_y, sigma_t]
        返回:
            cluster_id: 聚类编号数组
            pnt_class: 点类别数组 (1:核心点, 0:边界点, -1:噪声)
            pnt_score: 点密度分数数组
        """
        # 初始化参数
        apply_weight = sigma is not None
        n_points = len(x)

        # 初始化结果数组
        cluster_id = np.full(n_points, np.nan)  # 未分类点标记为nan
        pnt_class = np.zeros(n_points, dtype=int)
        pnt_score = np.full(n_points, np.nan)

        current_cluster = 1

        for idx in range(n_points):
            if np.isnan(cluster_id[idx]):
                point = np.array([x[idx], y[idx], t[idx]])
                is_core = self._expand_cluster(
                    tree, x, y, t, point, idx, current_cluster,
                    cluster_id, pnt_class, pnt_score,
                    critical_score, radius, sigma, apply_weight
                )
                if is_core:
                    current_cluster += 1

        # 清理噪声点标记
        pnt_class[(pnt_class < 0) & (cluster_id > 0)] = 0
        return cluster_id, pnt_class, pnt_score

    def _expand_cluster(self, tree, x, y, t, point, idx, cluster_num,
                        cluster_id, pnt_class, pnt_score,
                        critical_score, radius, sigma, apply_weight):
        """扩展聚类核心函数"""
        # 查找邻居
        neighbors, score = self._find_neighbors(tree, point, radius,
                                                x, y, t, sigma, apply_weight)
        pnt_score[idx] = score

        if score < critical_score:
            cluster_id[idx] = 0  # 噪声点
            pnt_class[idx] = -1
            return False

        # 标记为核心点
        cluster_id[neighbors] = cluster_num
        pnt_class[idx] = 1

        # 移除非核心点处理
        seeds = self._prepare_seeds(neighbors, idx, x, y, t)

        while seeds.shape[0] > 0:
            current_point = seeds[0, :3]
            current_idx = int(seeds[0, 3])

            # 查找当前点的邻居
            new_neighbors, new_score = self._find_neighbors(
                tree, current_point, radius, x, y, t, sigma, apply_weight)
            pnt_score[current_idx] = new_score

            if new_score >= critical_score:
                pnt_class[current_idx] = 1

                # 添加新邻居到种子列表
                new_seeds = self._prepare_seeds(new_neighbors, current_idx, x, y, t)
                seeds = self._update_seeds(seeds, new_seeds, cluster_id)

            seeds = np.delete(seeds, 0, axis=0)

        return True

    def _find_neighbors(self, tree, point, radius, x, y, t, sigma, apply_weight):
        """查找邻居并计算分数"""
        # 使用cKDTree进行范围查询
        neighbors = tree.query_ball_point(point, radius)
        neighbors = np.array(neighbors)

        if apply_weight:
            dx = x[neighbors] - point[0]
            dy = y[neighbors] - point[1]
            dt = t[neighbors] - point[2]

            exponent = -(dx ** 2 / (2 * sigma[0]) +
                         dy ** 2 / (2 * sigma[1]) +
                         dt ** 2 / (2 * sigma[2]))
            score = np.nansum(np.exp(exponent)) - 1
        else:
            score = len(neighbors)

        return neighbors, score

    def _prepare_seeds(self, neighbors, idx, x, y, t):
        """准备种子点集合"""
        mask = (neighbors != idx)
        valid_neighbors = neighbors[mask]
        return np.column_stack((
            x[valid_neighbors],
            y[valid_neighbors],
            t[valid_neighbors],
            valid_neighbors
        ))

    def _update_seeds(self, existing_seeds, new_seeds, cluster_id):
        """更新种子列表"""
        # 过滤已分类点
        mask = np.isnan(cluster_id[new_seeds[:, 3].astype(int)])
        new_seeds = new_seeds[mask]

        # 合并并去重
        if existing_seeds.size == 0:
            return new_seeds
        return np.unique(
            np.vstack([existing_seeds, new_seeds]),
            axis=0
        )

    def change_check_mode(self, menu_item):
        """
        切换菜单项选中状态 (对应 MATLAB 的 changeCheckMode)
        :param menu_item: 被点击的菜单项对象
        """
        current_state = menu_item.entrycget("label", "state")
        if current_state == "checked":
            menu_item.entryconfig("label", state="normal")
        else:
            # 获取同级菜单项并取消所有选中
            parent_menu = menu_item.parent
            for index in range(parent_menu.index("end") + 1):
                parent_menu.entryconfig(index, state="normal")
            menu_item.entryconfig("label", state="checked")

    def build_psf(self, sigma, width, height):
        """
        构建点扩散函数 (对应 MATLAB 的 buildPSF)
        :param sigma: 高斯核标准差
        :param width: 图像宽度
        :param height: 图像高度
        :return: 频域中的PSF
        """
        psf = np.zeros((height, width))

        w = int(math.ceil(6 * sigma))  # 99% of gaussian
        x = int(math.floor(width / 2.0 - w / 2.0))
        y = int(math.floor(height / 2.0 - w / 2.0))
        coords = np.arange(-w / 2, w / 2 + 1)
        X, Y = np.meshgrid(coords, coords)
        # 生成高斯核
        gauss = (1.0 / (2.0 * np.pi * sigma ** 2)) * np.exp(- (X ** 2 + Y ** 2) / (2.0 * sigma ** 2))

        try:
            psf[y: y + (w + 1), x: x + (w + 1)] = gauss
        except ValueError as e:
            # 可能越界
            print(f"PSF construction out of range: {e}")
            return None

            # 返回频域PSF
        return fft2(psf)

    def build_histogram(self, data, unit, ax=None):
        """
        创建直方图与累积分布图 (对应 MATLAB 的 buildHistogram)
        :param data: 输入数据数组
        :param unit: 单位字符串
        :param ax: 可选，指定的坐标轴对象
        :return: (主坐标轴，次坐标轴), 图形对象列表
        """
        if ax is None:
            fig, ax1 = plt.subplots(figsize=(10, 6))
        else:
            ax1 = ax

        # 计算直方图
        n_bins = self.calcnbins(data)
        counts, bins, patches = ax1.hist(data, bins=n_bins, alpha=0.7)

        # 创建次坐标轴
        ax2 = ax1.twinx()

        # 计算核密度估计
        kde = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 500)
        cdf = np.array([kde.integrate_box_1d(-np.inf, xi) for xi in x])
        line, = ax2.plot(x, cdf, 'r-', linewidth=3)

        # 设置标签和样式
        ax1.set_xlabel(f'Value ({unit})')
        ax1.set_ylabel('Frequency', color='b')
        ax2.set_ylabel('Cumulative Probability', color='r')

        # 创建图例
        stats_text = (
            f'min = {data.min():.1f} {unit}\n'
            f'max = {data.max():.1f} {unit}\n'
            f'mean = {data.mean():.1f} {unit}\n'
            f'std = {data.std():.1f} {unit}'
        )
        ax1.legend([patches[0], line],
                   [stats_text, 'CDF'],
                   loc='upper right',
                   bbox_to_anchor=(1, 0.95),
                   prop={'family': 'monospace'})

        return (ax1, ax2), [patches, line]



    def calcnbins(self, x, method='middle', minimum=1, maximum=np.inf):
        """
        Calculate the "ideal" number of bins for a histogram using various methods

        Parameters:
            x : array_like - Input data
            method : str - Calculation method ('fd', 'scott', 'sturges', 'all', 'middle')
            minimum : int - Minimum allowable number of bins
            maximum : int - Maximum allowable number of bins

        Returns:
            int or dict - Number of bins based on selected method
        """

        # Input validation: 将数据转换为浮点型数组，确保 np.isnan 能够正常工作
        x = np.asarray(x, dtype=float)

        # Handle complex numbers
        if np.iscomplexobj(x):
            x = x.real
            warnings.warn("Imaginary parts of X will be ignored.", UserWarning)

        # Flatten to vector
        if x.ndim > 1:
            x = x.ravel()
            warnings.warn("X will be coerced to a vector.", UserWarning)

        # Remove NaNs
        x = x[~np.isnan(x)]

        # Validate method
        valid_methods = ['fd', 'scott', 'sturges', 'all', 'middle']
        method = method.lower()
        if method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Choose from {valid_methods}")

        # Main calculation
        if method == 'fd':
            return self._calcfd(x, minimum, maximum)
        elif method == 'scott':
            return self._calcscott(x, minimum, maximum)
        elif method == 'sturges':
            return self._calcsturges(x, minimum, maximum)
        elif method == 'all':
            return {
                'fd': self._calcfd(x, minimum, maximum),
                'scott': self._calcscott(x, minimum, maximum),
                'sturges': self._calcsturges(x, minimum, maximum)
            }
        elif method == 'middle':
            values = [
                self._calcfd(x, minimum, maximum),
                self._calcscott(x, minimum, maximum),
                self._calcsturges(x, minimum, maximum)
            ]
            return int(np.median(values))

    # Helper functions ------------------------------------------------------------

    def _calcfd(self,x, minimum, maximum):
        """Freedman-Diaconis rule"""
        if len(x) < 2:
            return 1

        iqr = np.subtract(*np.percentile(x, [75, 25]))
        if iqr == 0:
            # Use twice median absolute deviation
            mad = np.median(np.abs(x - np.median(x)))
            h = 2 * mad
        else:
            h = 2 * iqr

        if h > 0:
            bins = (np.max(x) - np.min(x)) / (h * len(x) ** (-1 / 3))
        else:
            bins = 1

        return self._confine_range(np.ceil(bins), minimum, maximum)

    def _calcscott(self,x, minimum, maximum):
        """Scott's normal reference rule"""
        if len(x) < 2:
            return 1

        h = 3.5 * np.std(x, ddof=1) * len(x) ** (-1 / 3)
        if h > 0:
            bins = (np.max(x) - np.min(x)) / h
        else:
            bins = 1

        return self._confine_range(np.ceil(bins), minimum, maximum)

    def _calcsturges(self,x, minimum, maximum):
        """Sturges' formula"""
        bins = np.log2(len(x)) + 1 if len(x) > 0 else 1
        return self._confine_range(np.ceil(bins), minimum, maximum)

    def _confine_range(self,value, minimum, maximum):
        """Constrain value to [minimum, maximum] range"""
        return int(np.clip(np.ceil(value), minimum, maximum))


    def figPos(self,ratio):
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        if ratio > 1:
            width = 0.65 * (screen_height / screen_width)
            height = 0.65 / ratio
        elif ratio < 1:
            width = 0.65 * (screen_height / screen_width) * ratio
            height = 0.65
        else:
            width = 0.65 * (screen_height / screen_width)
            height = 0.65

        return [0.3, 0.09, width, height]

    def check_name_existance(self,filename, pathname):
        """
        检查文件名是否存在并处理用户选择

        Parameters:
            filename : str - 目标文件名
            pathname : str - 目标路径

        Returns:
            tuple: (新文件名, 新路径, 标志位)
            标志位: 0-正常, 1-取消操作
        """
        full_path = os.path.join(pathname, filename)
        filename_new, pathname_new, flag = filename, pathname, 0

        if os.path.exists(full_path):
            # 创建隐藏的根窗口
            root = tk.Tk()
            root.withdraw()

            # 弹出覆盖确认对话框
            answer = messagebox.askyesno(
                title='警告',
                message='是否覆盖已存在的文件?',
                icon='warning',
                default='no'
            )
            root.destroy()

            if answer:  # 用户选择覆盖
                # 删除所有扩展名的文件
                pattern = os.path.join(pathname, f"{filename}.*")
                for fpath in glob.glob(pattern):
                    try:
                        os.remove(fpath)
                    except Exception as e:
                        warnings.warn(f"文件删除失败: {str(e)}")

                # 更新搜索路径
                self.image_bin['search_path'] = pathname
            else:  # 用户选择不覆盖
                while True:
                    # 创建新隐藏窗口
                    dlg_root = tk.Tk()
                    dlg_root.withdraw()

                    # 获取初始路径
                    init_path = os.path.join(
                        self.image_bin.get('search_path', pathname),
                        filename
                    )

                    # 弹出保存对话框
                    new_path = filedialog.asksaveasfilename(
                        initialfile=filename,
                        initialdir=self.image_bin.get('search_path', pathname),
                        title="另存为"
                    )
                    dlg_root.destroy()

                    if not new_path:  # 用户取消
                        return (None, None, 1)

                    # 分解路径和文件名
                    pathname_new, filename_new = os.path.split(new_path)
                    if not os.path.exists(new_path):
                        self.image_bin['search_path'] = pathname_new
                        flag = 0
                        break

        return (filename_new, pathname_new, flag)


    ###########


    def show_credits(self):
        """显示 credits 信息。"""
        credits_window = tk.Toplevel(self.master)
        credits_window.title("Credits")
        credits_window.geometry("400x500")
        credits_window.configure(bg="#f4f4f4")

        # 创建 Text 小部件来显示 credits 信息
        credits_text = (
            "Particle Localization:\n"
            "© A. Serge, N. Bertaux,\n"
            "H. Rigneault & D. Marguet\n"
            "Dynamic multiple-target tracing\n"
            "to probe spatiotemporal\n"
            "cartography of cell membranes\n"
            "Nature Methods, Aug. 08\n\n"
            "Dynamic Convolution:\n"
            "M. Parent, T. Gould,\n"
            "S.T. Hess\n\n"
            "kd-Tree & k-NN Search:\n"
            "A. Tagliasacchi\n\n"
            "written by C.P. Richter\n"
            "Division of Biophysics / Group Piehler\n"
            "University of Osnabrueck"
        )

        text_widget = tk.Text(credits_window, wrap=tk.WORD, font=("Times New Roman", 12), bg="#f4f4f4", padx=10, pady=10,
                              height=15, width=40)
        text_widget.insert(tk.END, credits_text)
        text_widget.config(state=tk.DISABLED)  # 禁用编辑

        # 使用 grid 布局
        text_widget.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # 设置窗口的 grid 配置
        credits_window.grid_rowconfigure(0, weight=1)
        credits_window.grid_columnconfigure(0, weight=1)

        credits_window.resizable(False, False)

    def show_change_log(self):
        """显示变更日志的窗口。"""
        changelog_window = tk.Toplevel(self.master)
        changelog_window.title("Change Log")
        changelog_window.geometry("500x500")
        changelog_window.configure(bg="#f4f4f4")

        # 创建 Text 小部件来显示变更日志
        changelog_text = tk.Text(changelog_window, wrap=tk.WORD, font=("Times New Roman", 12), bg="white", padx=10, pady=10,
                                 height=20, width=60)
        changelog_text.insert(tk.END, "\n".join([
            "v1.0.0 - Initial Release",
            "    Integrated complete workflow: Imaging - then - Localization - then - Tracking - then - Analysis",
            "    GUI-based batch processing for .tif image stacks",
            "    Localization module with drift correction and photon count conversion",
            "    Trajectory reconstruction with support for confined and directed motion types",
            "    Built-in 2D/3D scatterplots, MSD, and jump length visualization",
            "    Exportable results compatible with SLIMfast and ISBI formats",
            "    User-friendly ROI control and frame navigation"
        ]))
        changelog_text.config(state=tk.DISABLED)  # 禁用编辑

        # 使用 grid 布局
        changelog_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # 设置窗口的 grid 配置
        changelog_window.grid_rowconfigure(0, weight=1)
        changelog_window.grid_columnconfigure(0, weight=1)

        changelog_window.resizable(False, False)



def detect_et_estime_part_1vue_deflt(input_data, wn, r0, pfa, n_deflt, w, h, minInt, optim, ctx=None):
    """主要处理逻辑，模仿 MATLAB 函数的功能"""
    # 调用检测函数
    lest, ldec, dfin, Nestime = detect_et_estime_part_1vue(input_data, wn, r0, pfa,optim, ctx=ctx)
    # Deflation处理
    input_deflt = deflat_part_est(input_data, lest, wn, ctx=ctx)
    lestime = lest.copy()  # 保留初始估计
    if n_deflt == 0:
        border = int(np.ceil(wn / 2))
        # 条件1：检测有效标志 (ok)
        condition1 = (lestime[:, 6] > 0)
        # 条件2：强度与半径比值大于最小阈值
        condition2 = (lestime[:, 3] / np.sqrt(np.pi) / lestime[:, 5] > minInt)
        # 条件3：行坐标在 [border, h-border)
        condition3 = (lestime[:, 0] > border) & (lestime[:, 0] < h - border)
        # 条件4：列坐标在 [border, w-border)
        condition4 = (lestime[:, 1] > border) & (lestime[:, 1] < w - border)
        print(f"[gate] ok:{condition1.sum()}  int:{condition2.sum()}  inY:{condition3.sum()}  inX:{condition4.sum()}  total:{lestime.shape[0]}")

        good = condition1 & condition2 & condition3 & condition4
        lestime = lestime[good]
        ctrsN = np.sum(good)
        return lestime, input_deflt, dfin, ctrsN

    for n in range(n_deflt):
        l, ld, d, N = detect_et_estime_part_1vue(input_deflt, wn, r0, pfa, optim, ctx=ctx)
        lestime = np.vstack((lestime, l))  # 合并结果
        dfin = np.logical_or(dfin, d)
        input_deflt = deflat_part_est(input_deflt, l, wn, ctx=ctx)

        if N == 0 or n == n_deflt - 1:
            border = int(np.ceil(wn / 2))
            # 各条件的单独人数（看看是卡在哪个门上）
            c1 = (lestime[:, 6] > 0)
            c2 = (lestime[:, 3] / (np.sqrt(np.pi) * np.maximum(lestime[:, 5], 1e-9)) > minInt)
            c3 = (lestime[:, 0] > border) & (lestime[:, 0] < h - border)  # y vs h
            c4 = (lestime[:, 1] > border) & (lestime[:, 1] < w - border)  # x vs w
            print(f"[gate] ok:{c1.sum()}  int:{c2.sum()}  inY:{c3.sum()}  inX:{c4.sum()}  total:{lestime.shape[0]}")

            good = (lestime[:, 6] > 0) & \
                   (lestime[:, 3] / np.sqrt(np.pi) / lestime[:, 5] > minInt) & \
                   (lestime[:, 0] > border) & (lestime[:, 0] < h - border) & \
                   (lestime[:, 1] > border) & (lestime[:, 1] < w - border)

            good_count = np.sum(good)
            lestime = lestime[good]
            ctrsN = good_count
            return lestime, input_deflt, dfin, ctrsN

def detect_et_estime_part_1vue(input_data, wn, r0, pfa, optim, ctx=None):
    """
            与 MATLAB detect_et_estime_part_1vue 功能对应的 Python 版本

            Returns:
                lestime: shape (N, 7) 的估计结果, 每行 [idx, i, j, alpha, sig², r, ok]
                ldetect: 初步检测列表
                d:       检测图(布尔)
                Nestime: 实际做了高斯牛顿拟合的数量
            """
    # ========== (1) 计算检测图 ==========
    carte_MV, ldetect, d = carte_H0H1_1vue(input_data, r0, wn, wn, pfa, ctx=ctx)
    Ndetect = ldetect.shape[0]

    # 若没有检测到任何点, 直接返回空
    if Ndetect == 0:
        lestime = np.zeros((1, 7), dtype=np.float64)
        return lestime, ldetect, d, 0

    # 预分配
    lestime = np.zeros((Ndetect, 7), dtype=np.float64)
    Nestime = 0
    bord = int(np.ceil(wn / 2))

    o0, o1, o2, o3, o4=optim

    for n in range(Ndetect):
        i_ = ldetect[n, 1]
        j_ = ldetect[n, 2]
        alpha_ = ldetect[n, 3]

        test_bord = (
                (i_ < bord) or (i_ > (input_data.shape[0] - bord)) or
                (j_ < bord) or (j_ > (input_data.shape[1] - bord))
        )
        if (alpha_ > 0.0) and (not test_bord):
            # 调用 Numba JIT 版本
            result = estim_param_part_GN_numba(
                input_data,
                wn,
                ldetect[n, :],  # length-7 的一行
                r0,
                o0, o1, o2, o3, o4
            )
            # 直接赋值到 lestime 数组
            lestime[Nestime, :] = result
            Nestime += 1

    if Nestime == 0:
        lestime = np.zeros((0, 7), dtype=np.float64)
    else:
        lestime = lestime[:Nestime, :]
    return lestime, ldetect, d, Nestime

def carte_H0H1_1vue(im, rayon, wn_x, wn_y, s_pfa, ctx=None):
    """Detection map generation with debug prints."""
    if ctx is not None:
        TFHM, TFHGC, SGC2 = ctx

    im = np.asarray(im, dtype=np.float64)
    H, W = im.shape
    if ctx is not None:
        TFHM, TFHGC, SGC2 = ctx
        if TFHM.shape != (H, W) or TFHGC.shape != (H, W):
            raise RuntimeError("FFT kernels & image shape mismatch")
    T = int(wn_x) * int(wn_y)

    # 2.1 图像 FFT
    tfim = fft2(im)

    # 2.2 H0 部分：复用全局 tfhm
    m0 = np.real(fftshift(ifft2(TFHM * tfim))) / T

    # 2.3 计算 Sim2, T_sig0_2
    im2 = im * im
    tfim2 = fft2(im2)
    Sim2 = np.real(fftshift(ifft2(TFHM * tfim2)))
    T_sig0_2 = Sim2 - T * m0 * m0

    # 2.4 H1: 先算 alpha（一定要在 test 之前）
    alpha = np.real(fftshift(ifft2(TFHGC * tfim))) / SGC2

    # 2.5 GLRT：严格按 MATLAB 写法（≤0 直接置 1，再取 log）
    with np.errstate(divide='ignore', invalid='ignore'):
        test = 1.0 - (SGC2 * alpha * alpha) / T_sig0_2
    test = np.where(test > 0.0, test, 1.0)
    carte_MV = -T * np.log(test)
    carte_MV[~np.isfinite(carte_MV)] = 0.0
    mx = float(np.max(carte_MV))
    n_local = int(np.sum(all_max_2d(carte_MV) != 0))
    n_pass = int(np.count_nonzero(carte_MV > s_pfa))

    detect_masque = carte_MV > s_pfa

    n_detect_pixels = np.sum(detect_masque)
    q = np.quantile(carte_MV, [0.5, 0.9, 0.99, 0.999])  # 中位/高分位

    # 先算出 local_max 二值图
    lm = all_max_2d(carte_MV) != 0
    if n_detect_pixels == 0:
        liste_detect = np.zeros((1, 7))
        detect_pfa = np.zeros_like(detect_masque)
    else:
        detect_pfa = np.logical_and(all_max_2d(carte_MV) != 0, detect_masque)
        di, dj = np.where(detect_pfa)
        n_detect = di.size
        alpha_detect = alpha[di,dj]
        sig2_detect = ((T_sig0_2 - alpha * alpha * SGC2) / T)[di,dj]
        liste_detect = np.column_stack((
            np.arange(1, n_detect + 1),
            di + 1,  # row → 1-based
            dj + 1,  # col → 1-based
            alpha_detect,
            sig2_detect,
            rayon * np.ones(n_detect),
            np.ones(n_detect)
        ))
    return carte_MV, liste_detect, detect_masque

def expand_w(in_array, N, M):
    """
    Expand the input array 'in_array' to size N x M by centering it
    (mimicking the MATLAB function expand_w).
     """
    N_in, M_in = in_array.shape
    out = np.zeros((N, M), dtype=in_array.dtype)

    # 使用浮点运算 + floor，以与 MATLAB 的 floor(N/2 - N_in/2) 一致
    nc = int(np.floor(N / 2.0 - N_in / 2.0))
    mc = int(np.floor(M / 2.0 - M_in / 2.0))

    # MATLAB 中 out((nc+1):(nc+N_in), (mc+1):(mc+M_in)) = in
    # Python 中索引从 0 开始，因此可直接用 out[nc : nc+N_in, mc : mc+M_in]
    # 即与 MATLAB 的 (nc+1) 对应 Python 的 nc
    out[nc: nc + N_in, mc: mc + M_in] = in_array

    return out
def all_max_2d(input_array):
    """Find all local maxima in a 2D array."""
    N, M = input_array.shape
    ref = input_array[1:N - 1, 1:M - 1]

    # 计算各个方向的局部最大值
    pos_max_h = (input_array[0:N - 2, 1:M - 1] < ref) & (input_array[2:N, 1:M - 1] < ref)
    pos_max_v = (input_array[1:N - 1, 0:M - 2] < ref) & (input_array[1:N - 1, 2:M] < ref)
    pos_max_135 = (input_array[0:N - 2, 0:M - 2] < ref) & (input_array[2:N, 2:M] < ref)
    pos_max_45 = (input_array[2:N, 0:M - 2] < ref) & (input_array[0:N - 2, 2:M] < ref)

    carte_max = np.zeros((N, M))
    carte_max[1:N - 1, 1:M - 1] = pos_max_h & pos_max_v & pos_max_135 & pos_max_45
    carte_max = carte_max * input_array

    # # # 确保 carte_max 是布尔类型
    # carte_max = carte_max.astype(bool)

    return carte_max
@njit(fastmath=False)
def estim_param_part_GN_numba(im, wn, liste_info_param, r0,
                              optim0, optim1, optim2, optim3, optim4):
    """
    Corrected version that matches MATLAB logic exactly
    """
    # 1) Get detection center (MATLAB 1-based)
    Pi = liste_info_param[1]  # row (i), 1-based
    Pj = liste_info_param[2]  # col (j), 1-based

    # 2) Extract patch
    half = math.floor(wn / 2.0)
    start_i = int(Pi - half)  # 0-based top corner
    start_j = int(Pj - half)

    im_part = im[start_i:start_i + wn, start_j:start_j + wn].astype(np.float64)

    # 3) Set up bounds (like MATLAB bornes_ijr)
    posTol = float(optim4)
    radiusTol = float(optim3)
    bounds_i_min = -posTol
    bounds_i_max = posTol
    bounds_j_min = -posTol
    bounds_j_max = posTol
    bounds_r_min = r0 - radiusTol * r0 / 100.0
    bounds_r_max = r0 + radiusTol * r0 / 100.0

    # 4) GN initialization
    r = float(r0)
    i_off = 0.0
    j_off = 0.0
    dr = 1.0
    di = 1.0
    dj = 1.0
    sig2 = 1e300  # inf replacement
    ITER_MAX = int(optim0)
    fin = 10.0 ** float(optim1)

    # 5) Main optimization loop (matching MATLAB logic)
    cpt = 0
    test = True
    result_ok = 1.0  # Initialize as OK
    alpha = 0.0
    m = 0.0

    while test:
        # Call GN estimation step
        r, i_off, j_off, dr, di, dj, alpha, sig2, m = deplt_GN_estimation_numba(
            r, i_off, j_off, im_part, sig2, dr, di, dj, optim0, optim1, optim2, optim3, optim4
        )
        cpt += 1

        # Convergence test (matching MATLAB)
        if optim2:  # isRadiusTol
            test = (abs(di) > fin) or (abs(dj) > fin) or (abs(dr) > fin)
        else:
            test = (abs(di) > fin) or (abs(dj) > fin)

        # Iteration limit check
        if cpt > ITER_MAX:
            test = False

        # Bounds check (matching MATLAB exactly)
        result_ok_current = 1.0
        if ((i_off < bounds_i_min) or (i_off > bounds_i_max) or
                (j_off < bounds_j_min) or (j_off > bounds_j_max)):
            result_ok_current = 0.0
        if optim2:  # only check radius bounds if enabled
            if (r < bounds_r_min) or (r > bounds_r_max):
                result_ok_current = 0.0

        # Update test condition (MATLAB: test = test & result_ok)
        test = test and (result_ok_current > 0.0)
        result_ok = result_ok_current  # Keep the latest result_ok

    # 6) Return results (matching MATLAB output format)
    out = np.empty(7, np.float64)
    out[0] = Pi + i_off  # y-coordinate (MATLAB: Pi+i)
    out[1] = Pj + j_off  # x-coordinate (MATLAB: Pj+j)
    out[2] = alpha  # mean amplitude
    out[3] = sig2  # noise power
    out[4] = m  # background level (offset)
    out[5] = r  # radius
    out[6] = result_ok  # OK flag
    return out


@njit(fastmath=False)
def deplt_GN_estimation_numba(p_r, p_i, p_j, x, sig2init,
                              p_dr, p_di, p_dj,
                              optim0, optim1, optim2, optim3, optim4):
    """
    MATLAB deplt_GN_estimation 的等价移植。
    输入/输出与原版一致。
    """
    # 拷贝当前参数（对应 r0,i0,j0）
    r0 = float(max(p_r, 1e-12))
    i0 = float(p_i)
    j0 = float(p_j)

    # 相对精度
    prec_rel = 10.0 ** float(optim1)

    # “前一位置” = 当前位置 - 上一步步长
    pp_r = r0 - p_dr
    pp_i = i0 - p_di
    pp_j = j0 - p_dj

    wn_i, wn_j = x.shape
    N = float(wn_i * wn_j)

    # 参考网格
    refi = np.empty(wn_i, np.float64)
    refj = np.empty(wn_j, np.float64)
    half_i = wn_i / 2.0
    half_j = wn_j / 2.0
    for ii in range(wn_i):
        refi[ii] = 0.5 + ii - half_i
    for jj in range(wn_j):
        refj[jj] = 0.5 + jj - half_j

    # ========== 先验质 / 回退阶段（严格照 MATLAB） ==========
    again = True
    loops = 0
    alpha = 0.0
    m = 0.0
    sig2 = sig2init

    while again:
        loops += 1

        # 构造 g / gc
        denom = math.sqrt(math.pi) * r0
        two_r0sq = 2.0 * r0 * r0

        gsum = 0.0
        for a in range(wn_i):
            da = refi[a] - i0
            for b in range(wn_j):
                db = refj[b] - j0
                gsum += math.exp(-(da*da + db*db) / two_r0sq) / denom

        Sgc2 = 0.0
        for a in range(wn_i):
            da = refi[a] - i0
            for b in range(wn_j):
                db = refj[b] - j0
                val = math.exp(-(da*da + db*db) / two_r0sq) / denom
                gc = val - gsum / N
                Sgc2 += gc * gc

        # alpha (MV)
        num_alpha = 0.0
        if Sgc2 != 0.0:
            for a in range(wn_i):
                da = refi[a] - i0
                for b in range(wn_j):
                    db = refj[b] - j0
                    val = math.exp(-(da*da + db*db) / two_r0sq) / denom
                    gc = val - gsum / N
                    num_alpha += x[a, b] * gc
            alpha = num_alpha / Sgc2
        else:
            alpha = 0.0

        # m (MV)
        # 先 g，再 x - alpha*g 的均值
        m_sum = 0.0
        for a in range(wn_i):
            da = refi[a] - i0
            for b in range(wn_j):
                db = refj[b] - j0
                val = math.exp(-(da*da + db*db) / two_r0sq) / denom
                m_sum += (x[a, b] - alpha * val)
        m = m_sum / N

        # 误差与 sig2
        err_ss = 0.0
        for a in range(wn_i):
            da = refi[a] - i0
            for b in range(wn_j):
                db = refj[b] - j0
                val = math.exp(-(da*da + db*db) / two_r0sq) / denom
                resid = x[a, b] - alpha * val - m
                err_ss += resid * resid
        sig2_new = err_ss / N

        # —— 与 MATLAB 同步的“回退”策略：变差就回到 pp_* + 缩小后的步长
        if sig2_new > sig2init:
            p_di = p_di / 10.0
            p_dj = p_dj / 10.0
            i0 = pp_i + p_di
            j0 = pp_j + p_dj
            if optim2:  # 仅半径回退受 isRadiusTol 控制
                p_dr = p_dr / 10.0
                r0 = max(pp_r + p_dr, 1e-12)
            else:
                p_dr = 0.0
                r0 = max(pp_r, 1e-12)

            # 与 MATLAB 一致：若缩步后仍“大于”阈值，直接返回上一解，让外层继续
            if max(abs(p_dr), abs(p_di), abs(p_dj)) > 10.0 ** float(optim1):
                return p_r, p_i, p_j, 0.0, 0.0, 0.0, alpha, sig2init, m
        else:
            sig2 = sig2_new
            again = False

        if loops > 50:   # 与 MATLAB 同步的保险退出
            break

    # ========== 通过验质后，计算导数并做 GN 更新 ==========
    # 预先构造 ii/jj
    ii = np.empty((wn_i, wn_j), np.float64)
    jj = np.empty((wn_i, wn_j), np.float64)
    for a in range(wn_i):
        for b in range(wn_j):
            ii[a, b] = refi[a] - i0
            jj[a, b] = refj[b] - j0

    denom = math.sqrt(math.pi) * r0
    two_r0sq = 2.0 * r0 * r0
    g = np.empty((wn_i, wn_j), np.float64)
    for a in range(wn_i):
        for b in range(wn_j):
            g[a, b] = math.exp(-(ii[a, b]*ii[a, b] + jj[a, b]*jj[a, b]) / two_r0sq) / denom

    g_div_sq_r0 = g / (r0 * r0)
    err = x - alpha * g - m

    # 一阶/二阶导（按 MATLAB）
    iiii = ii * ii
    jjjj = jj * jj
    iiii_jjjj = iiii + jjjj

    der_g_i0 = ii * g_div_sq_r0
    der_g_j0 = jj * g_div_sq_r0

    derder_g_i0 = (-1.0 + (1.0 / (r0 * r0)) * iiii) * g_div_sq_r0
    derder_g_j0 = (-1.0 + (1.0 / (r0 * r0)) * jjjj) * g_div_sq_r0

    der_J_i0 = alpha * np.sum(der_g_i0 * err)
    der_J_j0 = alpha * np.sum(der_g_j0 * err)
    derder_J_i0 = alpha * np.sum(derder_g_i0 * err) - alpha * alpha * np.sum(der_g_i0 * der_g_i0)
    derder_J_j0 = alpha * np.sum(derder_g_j0 * err) - alpha * alpha * np.sum(der_g_j0 * der_g_j0)

    if optim2:  # 允许半径更新
        der_g_r0 = (-1.0 / r0 + (1.0 / (r0 * r0 * r0)) * iiii_jjjj) * g
        derder_g_r0 = (1.0 - 3.0 / (r0 * r0) * iiii_jjjj) * g_div_sq_r0 + \
                      (-1.0 / r0 + (1.0 / (r0 * r0 * r0)) * iiii_jjjj) * der_g_r0
        der_J_r0 = alpha * np.sum(der_g_r0 * err)
        derder_J_r0 = alpha * np.sum(derder_g_r0 * err) - alpha * alpha * np.sum(der_g_r0 * der_g_r0)
        # 与 MATLAB 一致：n_r = abs(r0 + dr)
        dr = -der_J_r0 / derder_J_r0 if derder_J_r0 != 0.0 else 0.0
        n_r = abs(r0 + dr)
    else:
        dr = 0.0
        n_r = r0

    di = -der_J_i0 / derder_J_i0 if derder_J_i0 != 0.0 else 0.0
    dj = -der_J_j0 / derder_J_j0 if derder_J_j0 != 0.0 else 0.0

    n_i = i0 + di
    n_j = j0 + dj

    return n_r, n_i, n_j, dr, di, dj, alpha, sig2, m


def deflat_part_est(input_data, liste_est, wn, ctx=None):
    """
            Python 版 deflat_part_est，功能与 MATLAB 版本相同。
            参数：
              input_data: 输入图像，NumPy 数组 (2D)
              liste_est: 检测结果矩阵，格式为 [num, i, j, alpha, sig^2, rayon, ok]
                         注意：i, j 坐标为 MATLAB 1-based 数值
              wn: 初始窗口大小（此参数会在循环内被重新计算）
            返回：
              output: deflation 后的图像，转换为 uint16 类型
            """
    idim, jdim = input_data.shape
    nb_part = liste_est.shape[0]

    # 将输出转换为 float64
    output = input_data.astype(np.float64).copy()

    for part in range(nb_part):
        # 检查有效性：MATLAB判断 liste_est(part,7)==1, 对应 Python 索引6
        if liste_est[part, 6] == 1:
            # MATLAB: i0 = liste_est(part,1); j0 = liste_est(part,2);
            # 正确的列映射（与 MATLAB 一致）：
            i0 = float(liste_est[part, 1])  # row
            j0 = float(liste_est[part, 2])  # col
            alpha = float(liste_est[part, 3])  # 第4列才是 alpha
            r0 = float(liste_est[part, 5])

            # MATLAB round 等价（不要用 Python 的 round/银行家舍入）
            pos_i = int(np.floor(i0 + 0.5))  #
            pos_j = int(np.floor(j0 + 0.5))  #
            dep_i = i0 - pos_i
            dep_j = j0 - pos_j

            # 窗口大小也建议夹一下，避免越界/浮点：
            wn = max(1, int(np.ceil(6.0 * r0)))
            wn = min(wn, idim, jdim)

            # 计算高斯窗
            alpha_g = alpha * gausswin2(r0, wn, wn, dep_i, dep_j)

            # 计算 dd, di, dj 按 MATLAB 逻辑（1-based）
            dd = np.arange(1, wn + 1) - int(np.floor(wn / 2))
            di = dd + pos_i  # 此时 di 按 MATLAB 坐标，取值应在 1 到 idim
            dj = dd + pos_j

            # MATLAB条件： iin = di > 0 & di < idim+1
            iin = (di > 0) & (di <= idim)
            jin = (dj > 0) & (dj <= jdim)

            # 转换为 Python 0-based 索引：减 1
            di_valid = (di[iin] - 1).astype(int)
            dj_valid = (dj[jin] - 1).astype(int)

            # 使用 np.ix_ 从 alpha_g 中提取对应子矩阵
            valid_rows = np.flatnonzero(iin)
            valid_cols = np.flatnonzero(jin)
            alpha_g_sub = alpha_g[np.ix_(valid_rows, valid_cols)]

            # 更新输出，确保使用 0-based 索引
            output[np.ix_(di_valid, dj_valid)] -= alpha_g_sub

    return output

def gausswin2(sig, wn_i, wn_j=None, offset_i=0.0, offset_j=0.0):
    # ★ 把窗口尺寸统一成整数并下限为 1
    wn_i = int(np.ceil(float(wn_i)))
    if wn_j is None:
        wn_j = wn_i
    else:
        wn_j = int(np.ceil(float(wn_j)))
    wn_i = max(wn_i, 1)
    wn_j = max(wn_j, 1)

    refi = 0.5 + np.arange(wn_i) - wn_i / 2
    i = refi - offset_i
    refj = 0.5 + np.arange(wn_j) - wn_j / 2
    j = refj - offset_j
    ii = np.tile(i[:, np.newaxis], (1, wn_j))
    jj = np.tile(j[np.newaxis, :], (wn_i, 1))

    g = (1 / (np.sqrt(np.pi) * sig)) * np.exp(-(1 / (2 * sig ** 2)) * (ii ** 2 + jj ** 2))
    return g

def modelfun(x, raw, refi, refj, N, store):
    """
    模型函数，对应 MATLAB 中 modelfun，
    现在是顶层函数，不再依赖 self。
    通过传入的 store 字典保存 alpha 和 m 供外部读取。

    参数：
      x: 参数向量，其中 x[0]=offset in i, x[1]=offset in j, x[2]=sigma
      raw: 原始图像区域（二维数组）
      refi, refj: 参考向量（1D array）
      N: 总像素数
      store: dict，用于保存计算出的 'alpha' 和 'm'
    返回：
      sig2: 残差均方误差（noise power）
    """
    # 计算偏移后的坐标网格
    i = refi - x[0]
    j = refj - x[1]
    wn_i, wn_j = len(refi), len(refj)
    ii = np.tile(i[:, np.newaxis], (1, wn_j))
    jj = np.tile(j[np.newaxis, :], (wn_i, 1))

    # 计算 Gaussian 窗
    iiii = ii**2
    jjjj = jj**2
    g = (1/(np.sqrt(np.pi)*x[2])) * np.exp(-(1/(2*x[2]**2)) * (iiii + jjjj))
    gc = g - np.sum(g)/N
    Sgc2 = np.sum(gc**2)

    # 计算 mean amplitude
    alpha = np.sum(raw * gc) / Sgc2

    # 计算 offset m
    x_alphag = raw - alpha * g
    m = np.sum(x_alphag) / N

    # 计算 residuals 和 noise power
    err = x_alphag - m
    sig2 = np.sum(err**2) / N

    # 将 alpha 和 m 保存到外部字典
    store['alpha'] = alpha
    store['m'] = m

    return sig2

def gauss_mle(raw, guess, bounds, options):
    """
    Gaussian Maximum Likelihood Estimation, 顶层函数版本。
    对应 MATLAB 里的 gaussMLE，返回 [x1, x2, alpha, fval, m, sigma, 1]
    """
    # (1) 构造参考网格和像素总数
    wn_i, wn_j = raw.shape
    N = wn_i * wn_j
    refi = 0.5 + np.arange(wn_i) - wn_i/2
    refj = 0.5 + np.arange(wn_j) - wn_j/2

    # (2) 用 store 收集 modelfun 中计算的 alpha 和 m
    store = {}

    # (3) 调用 scipy.minimize，目标函数是把 store 传给 modelfun
    res = minimize(
        lambda x: modelfun(x, raw, refi, refj, N, store),
        x0=guess,
        bounds=bounds,
        options=options
    )

    # (4) 从结果中提取最优参数和目标值
    x_opt = res.x
    fval = res.fun

    # (5) 从 store 里拿到 alpha 和 m
    alpha = store['alpha']
    m     = store['m']

    # (6) 按 MATLAB 格式返回
    return [x_opt[0], x_opt[1], alpha, fval, m, x_opt[2], 1]

def gauss_elliptic_mle(raw, guess, bounds, options):
    """
    顶层版的 Elliptic Gaussian MLE。
    输入：
      - raw: 2D numpy 数组，待拟合的图像块
      - guess: 初始猜测向量 [i0, j0, sigma_i, sigma_ij, sigma_j]
      - bounds: optimize.minimize 的 bounds 参数
      - options: optimize.minimize 的 options 参数
    返回：
      [x, y, alpha, fval, m, sigma, 1]
      与 MATLAB 版本保持一致。
    """
    # 准备参考坐标和常量
    wn_i, wn_j = raw.shape
    N = wn_i * wn_j
    refi = 0.5 + np.arange(wn_i) - wn_i/2
    refj = 0.5 + np.arange(wn_j) - wn_j/2
    ii, jj = np.meshgrid(refi, refj, indexing='ij')
    dist2 = ii**2 + jj**2

    # 用于闭包中保存 alpha, m
    store = {}

    def _modelfun(x):
        # 计算高斯窗 g
        g = 2*np.sqrt(np.pi) * multivariate_normal.pdf(
            np.sqrt(dist2),
            mean=[x[0], x[1]],
            cov=[[x[2], x[3]], [x[3], x[4]]]
        )
        g -= np.sum(g)/N
        Sgc2 = np.sum(g**2)
        # 最大似然估计 alpha 和 offset m
        alpha = np.sum(raw * g) / Sgc2
        x_alphag = raw - alpha * g
        m = np.sum(x_alphag) / N
        # 残差平方和
        err = x_alphag - m
        sig2 = np.sum(err**2) / N

        # 存回 store，供外部读取
        store['alpha'] = alpha
        store['m']     = m
        return sig2

    # 调用 scipy.optimize.minimize
    res = minimize(_modelfun, x0=guess, bounds=bounds, options=options)
    x_opt = res.x
    fval  = res.fun

    # 从 store 中取回 alpha 和 m
    alpha = store['alpha']
    m     = store['m']

    # 可选：画出拟合椭圆
    V, D = np.linalg.eig([[x_opt[2], x_opt[3]], [x_opt[3], x_opt[4]]])
    t = np.linspace(0, 2*np.pi, 20)
    u = np.vstack((np.cos(t), np.sin(t)))
    w = V @ np.sqrt(np.diag(D)) @ u
    z = np.repeat(x_opt[:2].reshape(2,1), 20, axis=1) + w

    plt.imshow(raw, cmap='gray')
    plt.fill(z[0,:] + 5, z[1,:] + 5, color='red', alpha=0.6)
    plt.show()

    # 返回结果，与原 MATLAB 接口对齐
    return [x_opt[0], x_opt[1], alpha, fval, m, x_opt[2], 1]


if __name__ == "__main__":
    root = tk.Tk()
    app = SlimFastApp(root)
    root.mainloop()
