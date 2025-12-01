# SPTpy

SPTpy is a graphical user interface application for Single Particle Tracking (SPT) analysis, particularly suitable for biological imaging and particle analysis.

## Features

- **Image Loading and Visualization**: Support for loading and visualizing TIFF image stacks and super stacks
- **Particle Localization**: Deep learning-based particle detection and localization
- **Trajectory Tracking**: Automatic tracking of particle motion trajectories
- **Data Analysis**: MSD analysis, diffusion coefficient calculation, radius analysis, etc.
- **Batch Processing**: Support for automated batch data processing workflows

## System Requirements

- **Operating System**: Windows 10 or higher / Linux
- **Python Version**: Python 3.10 or higher
- **GPU**: Optional, supports CUDA acceleration (for deep learning models)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DongLab-SIAT/SPTpy.git
cd SPTpy
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Core Algorithm Package

```bash
cd SPTpy
pip install -e .
```

## Usage

### Launching the Application

Run the main program directly:

```bash
python SPTpy/SPTpy.py
```

Or double-click the `SPTpy/SPTpy.py` file (if Python file association is configured).

### Menu Functions

#### Load Menu

- **Imagestack**: Load TIFF image stack files (`.tif` format)
  - Supports multi-frame TIFF files
  - Images can be viewed and browsed in the image window after loading

- **Batch Localization**: Batch localization function
  - Used for batch processing of particle localization for multiple files

- **Particle Data** → **SPTpy**: Load SPTpy particle localization data
  - Supports text file format (`.txt`)
  - File format: Tab-separated, first row is header
  - Localization results can be visualized in the image window after loading

- **Tracking Data**: Load trajectory tracking data
  - Supports text file format (`.txt`)
  - Data format: `x_coordinate y_coordinate frame_number trajectory_id` (tab-separated)
  - Automatically filters trajectories with length less than 5

#### Analysis Menu

- **spot on**: Spot-On analysis tool
  - Select trajectory file (`.txt` format)
  - Set parameters: frame interval, pixel size, etc.
  - Perform diffusion coefficient analysis and visualization

- **RoC**: Radius analysis (Radius of Confinement)
  - Opens radius analysis parameter settings window
  - Calculates particle confinement radius

- **Motion trajectory**: Motion trajectory analysis
  - Select trajectory table file (`.txt` format)
  - Analyzes trajectory motion patterns, including:
    - Free trajectory
    - Condensate trajectory
    - Condensate to free
    - Free to condensate

#### Extras Menu

- **Credits**: Display author and copyright information
- **Changelog**: Display version update log
- **Help**: Help function (currently disabled)

### Basic Workflow

1. **Load Image Data**
   - Click `Load` → `Imagestack` to select TIFF image file
   - Image will be displayed in the main window

2. **Load Particle Localization Data**
   - Click `Load` → `Particle Data` → `SLIMfast`
   - Select localization text file
   - Localization points will be overlaid on the image

3. **Load Trajectory Data**
   - Click `Load` → `Tracking Data`
   - Select trajectory tracking text file
   - Trajectories will be visualized in the image window

4. **Perform Data Analysis**
   - Click `Analysis` → `spot on` for diffusion analysis
   - Click `Analysis` → `RoC` for radius analysis
   - Click `Analysis` → `Motion trajectory` for motion pattern analysis


## Project Structure

```
SPTpy_v3/
├── SPTpy/
│   ├── SPTpy.py                    # Main program entry
│   ├── unet_attention_model.py     # UNet attention model
│   ├── automated_spt_processor.py  # Automated SPT processing module
│   ├── rc_rg_combined_auto.py      # RC/RG analysis module
│   ├── track_processor.py          # Trajectory processing module
│   ├── msd_analyzer.py             # MSD analysis module
│   ├── radius_calculator.py        # Radius calculation module
│   ├── spoton_core/                # Core algorithm package
│   ├── MonteCarloParams_1_gap.mat  # Monte Carlo parameter file
│   └── setup.py                    # Installation configuration
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore file
└── README.md                       # This file
```

## Dependencies

Main dependencies include:
- PyTorch (Deep learning framework)
- NumPy, SciPy (Numerical computing)
- Matplotlib (Visualization)
- Tkinter (GUI interface)
- scikit-image, scikit-learn (Image processing and machine learning)
- tifffile, imageio (Image I/O)
- pandas (Data processing)
- lmfit (Fitting analysis)
- numba (Performance optimization)

See `requirements.txt` for the complete dependency list.


## Author

- **Liao Shasha**
- Institute: SIAT (Shenzhen Institute of Advanced Technology)
- Location: Shenzhen, China

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Issues

If you encounter any problems or have suggestions, please submit them via GitHub Issues.
