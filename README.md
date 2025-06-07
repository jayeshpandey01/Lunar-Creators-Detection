# Lunar Creators Detection

This repository contains code and resources for detecting lunar features using machine learning. The project is structured into several directories, each serving a specific purpose—from model training and evaluation to logging and visualization.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Model Files](#model-files)
- [Logs](#logs)
- [Source Code](#source-code)
- [License](#license)

## Project Structure

```
.
├── README.md
├── logs/
│   └── training_20241116_130941.log
├── models/
│   └── best_model.pth
├── notebooks/
│   ├── dataset_visualization_Modeling.ipynb
│   └── wandb/
├── src/
│   ├── __pycache__/
│   ├── analyze_runs.py
│   ├── config.yaml
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── main.py
│   ├── model.py
│   ├── setup_project.py
│   ├── split_dataset.py
│   ├── split_single_tiff.py
│   ├── trainer.py
│   ├── utils.py
│   └── visualization_utils.py
```
> **Note:** Only the first 10 files in `src/` are shown due to API limitations. [View all files in src/](https://github.com/jayeshpandey01/Lunar-Creators-Detection/tree/main/src)

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/jayeshpandey01/Lunar-Creators-Detection.git
   cd Lunar-Creators-Detection
   ```
2. **Install dependencies:**  
   (List dependencies and installation instructions here, if available.)

3. **Configure the project:**  
   Adjust parameters in `src/config.yaml` as needed.

## Usage

- The main entry point for training or evaluation is likely `src/main.py` or `src/trainer.py`.
- Utility scripts are provided for data loading (`data_loader.py`), evaluation (`evaluate.py`), visualization (`visualization_utils.py`), and more.

## Notebooks

- The `notebooks/` directory contains Jupyter notebooks for data exploration and visualization, such as:
  - [`dataset_visualization_Modeling.ipynb`](https://github.com/jayeshpandey01/Lunar-Creators-Detection/blob/main/notebooks/dataset_visualization_Modeling.ipynb)
- Additional experiment tracking may be located in the `notebooks/wandb/` directory.

## Model Files

- Trained models are stored in the `models/` directory.  
  Example: [`best_model.pth`](https://github.com/jayeshpandey01/Lunar-Creators-Detection/blob/main/models/best_model.pth)

## Logs

- Training and evaluation logs are stored in the `logs/` directory.  
  Example: [`training_20241116_130941.log`](https://github.com/jayeshpandey01/Lunar-Creators-Detection/blob/main/logs/training_20241116_130941.log)

## Source Code

- All core logic, data processing, and model code is found in the `src/` directory:
  - `main.py`: Likely the main script for the project
  - `trainer.py`: Model training loop
  - `model.py`: Model architecture
  - `data_loader.py`: Data loading utilities
  - `evaluate.py`: Evaluation scripts
  - `visualization_utils.py`: Visualization tools
  - `utils.py`: Miscellaneous utilities
  - `config.yaml`: Project configuration

  [Browse all source files](https://github.com/jayeshpandey01/Lunar-Creators-Detection/tree/main/src)

---
