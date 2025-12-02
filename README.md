# **SPTpy: An Integrated Python Toolkit for One-Stop Analysis of Live-Cell Single-Particle Tracking Data**

*A unified and cross-platform Python framework for SPT data processing, nuclear segmentation, trajectory tracking, and biophysical modeling.*  


---

## **1. Overview**

SPTpy is an integrated and open-source Python toolkit designed for end-to-end analysis of live-cell single-particle tracking data.  
Motivated by the operational fragmentation of existing SPT software, SPTpy consolidates essential analytical components—including image pre-processing,single-molecule localization, trajectory reconstruction, nuclear segmentation, kinetic modeling, confinement analysis, and motion-state classification—into a unified workflow.

SPTpy enables robust quantification of transcription factor dynamics, supports cross-platform operation, and is suitable for large-scale SPT datasets commonly generated in modern single-molecule imaging experiments.


## **2. Key Features**

### **Complete SPT Workflow Integration**

- TIFF image stack loading and visualization  
- Per-frame normalization, γ-correction, and optional up-sampling  
- GLRT-based localization with iterative deflation and sub-pixel refinement  
- KNN-based trajectory linking with gap closing  
- Parallel processing for high-throughput datasets

### **Deep Learning–Based Nuclear Segmentation**

- Multi-attention U-Net (MaU-Net) designed for SPT images  
- Dual-stream encoder (Gray vs. Render + Scatter)  
- Cross-stream attention fusion for high-accuracy nuclear masks  
- Offline model training and GPU-accelerated inference

### **Biophysical Analysis**

- Spot-On kinetic modeling  
- Radius of Confinement (RoC) estimation via constrained MSD fitting  
- Motion-State Classification using spatial density labeling  
- Supports analysis of TF dynamics, chromatin interactions, condensate transitions

### **High Throughput & Reproducible**

- Memory-mapped TIFF loading for efficient large data handling  
- Vectorized and parallelized computation throughout  
- Unified GUI ensuring reproducibility and consistency  
- Script-level access for automated workflows

### **User-Friendly GUI**

- Integrated graphical interface for end-to-end analysis  
- Interactive tools for inspecting images, localizations, trajectories  
- Batch-processing support for multiple files and parameters

---

## **3. System Requirements**

| Category             | Requirement                                       |
| :------------------- | :------------------------------------------------ |
| **Operating System** | Windows 10 or higher; Linux                       |
| **Python Version**   | Python ≥ 3.10                                     |
| **CPU**              | Multi-core processor recommended                  |
| **Memory**           | ≥ 16 GB (for large TIFF stacks)                   |
| **GPU (optional)**   | NVIDIA CUDA-enabled GPU (recommended for MaU-Net) |


---

## **4. Installation**

### **1. Clone the repository**

```bash
git clone https://github.com/DongLab-SIAT/SPTpy.git
```

### **2. Create a virtual environment (recommended)**

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### **3. Install Python dependencies**

```bash
pip install -r requirements.txt
```

### **4. Install core algorithm package**

```bash
cd SPTpy
pip install -e .
```

---

## **5. Usage**

SPTpy is primarily used through its graphical interface.  
Launch the GUI with:

```bash
python SPTpy/SPTpy.py
```

---



## **6. Project Structure**

```
SPTpy/
├── spoton_core/                     # Spot-On kinetic modeling engine
│   ├── __init__.py
│   ├── fastSPT_plot.py              # Visualization utilities for Spot-On analysis
│   ├── fastSPT_tools.py             # Helper functions for diffusion fitting
│   ├── fastspt.py                   # Core Spot-On implementation
│   ├── format4DN.py                 # 4DN-compliant trajectory formatting
│   ├── readers.py                   # File readers for SPT data formats
│   ├── version.py                   # Version info for spoton_core
│   └── writers.py                   # Output writers for processed data
│
├── SPTpy.py                         # GUI entry point
├── automated_spt_processor.py       # Automated batch SPT processing workflow
├── msd_analyzer.py                  # MSD computation and diffusion coefficient analysis
├── radius_calculator.py             # Radius of Confinement (RoC) analysis
├── rc_rg_combined_auto.py           # Combined RoC/RG analysis module
├── setup.py                         # Package installation configuration
├── track_processor.py               # Single-particle trajectory linking and filtering
├── unet_attention_model.py          # Multi-attention U-Net model for nuclear segmentation
│
├── .gitignore                       # Git ignore rules
├── LICENSE                          # License file
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependency list
```


---

---

## **7. Example Datasets**

### **Zenodo Archive**

The datasets used in the manuscript have been deposited on Zenodo:

**📦 Zenodo DOI:** https://doi.org/10.5281/zenodo.17783061

To comply with size limitations and data-sharing constraints, **raw TIFF image stacks are not included** in this archive.  
Instead, the Zenodo dataset contains the processed results and intermediate files required to fully reproduce our analyses.

The archive includes the following three components:

#### **1. CREB SPT Dataset**

- Single-molecule localization results  
- Trajectory files used for diffusion, RoC, and motion-state analysis  
- Rendered localization density maps  


#### **2. Cohesin-Loss Dataset**

- Localization and trajectory files for histones (H2B, H2A.Z)  
- Transcriptional regulators (OCT4, BRD4, MED1, MED6, TBP)  
- Processed data used in the manuscript to analyze diffusion behavior, confinement, and condensate transitions  

#### **3. MaU-Net Training and Validation Dataset**

- Training set used for MaU-Net (Gray / Render / Scatter inputs with corresponding nuclear masks)  
- Validation set used for model selection and performance evaluation  
- Pre-processed grayscale image frames and their associated Render / Scatter channels  
- Manually annotated nuclear masks (DAPI-based ground truth)  
- Best-performing MaU-Net model checkpoint selected based on validation loss


These datasets contain all information necessary to reproduce the analyses presented in the manuscript.

---

## **8. Citation**

If you use SPTpy for your research, please cite:

```bibtex
@article{Liao2025SPTpy,
  title={SPTpy: An Integrated Python Toolkit for One-Stop Analysis of Live-Cell Single-Particle Tracking Data},
  author={Liao, Shasha and Yang, Xin and Wang, Jinhong and Zhu, Hongni and Song, Yi and Liu, Yajie and Dong, Peng},
  journal={Bioinformatics},
  year={2025},
}
```

---

## **9. License**

Distributed under the **MIT License**.

---

## **10. Contact**

For questions, data requests, or collaboration:

**Peng Dong**  
Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences  
📧 p.dong@siat.ac.cn

