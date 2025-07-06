# SpliceREAD

A deep learning framework for canonical and non-canonical splice site classification with synthetic data augmentation and SHAP-based interpretability.

[See SpliceRead Wiki for Full Documentation on Installation and Usage](https://github.com/OluwadareLab/SpliceRead/wiki)

**Authors:**

* Khushali Samderiya
* Rohit Menon
* Sahil Thapa
* Prof. Oluwatosin Oluwadare

___________________
#### OluwadareLab, University of Colorado, Colorado Springs
___________________


## Overview

**SpliceREAD** is a deep learning model for accurate splice site classification, especially focused on improving the detection of non-canonical splice sites. Traditional models often overlook these rare variants due to extreme class imbalance and sequence variability. SpliceREAD addresses this gap by combining residual convolutional neural networks with synthetic data augmentation using ADASYN, enabling improved learning of subtle genomic signals. The framework is robust across multiple species, sequence lengths, and validation settings, consistently outperforming existing tools in both canonical and non-canonical detection tasks.

---

## Folder Structure

```
SpliceRead/
+-- data/                 # Placeholder folder to be replaced with downloaded dataset
+-- models/               # Placeholder folder to be replaced with pretrained models
+-- output/               # Stores generated synthetic sequences and visualization outputs
+-- scripts/              # All training, generation, evaluation, and visualization scripts
¦   +-- training/         # Classifier training logic
¦   +-- data_augmentation/  # Synthetic data generation logic
¦   +-- evaluation/       # Performance evaluation scripts
¦   +-- visualization/    # Plotting utilities
¦   +-- interpretation/   # SHAP-based interpretation utilities
+-- Dockerfile            # Containerized environment for reproducibility
+-- README.md             # Project documentation
```

---

## Setup Instructions

### Step 1: Clone Repository

```bash
git clone https://github.com/OluwadareLab/SpliceRead.git
cd SpliceRead
```

### Step 2: Download Data and Models

Download the Zenodo archive from the link below:

[https://doi.org/10.5281/zenodo.15538290](https://doi.org/10.5281/zenodo.15538290)

### Step 3: Place Files

* Extract the `SpliceRead_Files.zip` archive.
* Replace the `data/` folder in the repo with the extracted `data/` folder.
* Replace the `models/` folder in the repo with the extracted `models/` folder.

### Step 4: Build Docker Image

```bash
docker build -t spliceread .
```

### Step 5: Start Docker Container

```bash
docker run -it --name spliceread-container \
  -v /path/to/local/SpliceRead:/workspace/SpliceRead \
  spliceread /bin/bash
```

Inside the container:

```bash
cd /workspace/SpliceRead
```

---

## Execution Instructions

### 1.Train & Evaluate

**Purpose**: Trains and evaluate the SpliceREAD model using 5-fold cross-validation.

**Arguments**:

| Flag                              | Description                                                      |
|-----------------------------------|------------------------------------------------------------------|
| `--three_class_no_synthetic`      | Use to train the model **without** synthetic data                |
| `--three_class`                   | Use to train the **with** synthetic data (default)               |
| `--synthetic_ratio <float>`       | % ratio of non-canonical to canonical sequences (default: 100.0) |
| `--use_smote`                     | Generate synthetic data with **SMOTE** (instead of ADASYN)       |
| `--show_progress`                 | Display progress bars during data loading                        |
| `--sequence_length <400\|600>`    | Sequence length in bp (default: 600)                             |
| `--train_dir <path>`              | Directory for training data                                      |
| `--test_dir <path>`               | Directory for test data                                          |
| `--model_dir <path>`              | Where trained models are saved                                   |
| `--output_dir <path>`             | Where evaluation outputs & logs go                               |
| `--model_path <path>`             | Path to a pre-trained model for evaluation                       |
| `--folds <int>`                   | Number of cross-validation folds (default: 5)                    |



**Command**:

```bash
python3 run.py [options]
```

**Example**:

```bash
python3 run.py --three-class \
  --synthetic_ratio 100 \
  --sequence_length 600 \
  --train_dir ./to_zenodo/data_600bp/train \
  --test_dir ./to_zenodo/data_600bp/test \
  --model_dir ./trained_model \
  --output_dir ./output \
  --show_progress
```
To evaluate a pretrained 600 bp model, download the model_files archive from Zenodo.

**Arguments**:
| Flag                          | Description                                                          |
|-------------------------------|----------------------------------------------------------------------|
| `--model_path <path>`         | Path to the trained model file (HDF5, e.g. `.h5`) (required)         |
| `--test_data <path>`          | Directory containing test sequences organized by class (required)    |
| `--out_dir <path>`            | Directory to save evaluation report (default: `./evaluation_results`)|
| `--sequence_length <600>`     | Sequence length in base pairs (default: `600`)                       |
| `--show_progress`             | Show progress bars while loading test data    

**Example**
```bash
python3 scripts/evaluation/run_evaluation.py \
  --model_path ./to_zenodo/model_files/best_model_fold_5.h5 \
  --test_data to_zenodo/data_600bp/test \
  --out_dir ./my_eval_results \
  --show_progress
```

---

### 2.`Visualization`
**Purpose**: Visualizes GC/AT nucleotide content using scatter plots.

To visualize GC/AT nucleotide content, first we have to generate synthetic sequences using ADASYN. Synthetic data is already provided in zenodo archive you can use that as well.

#### 2.1. `run_generator.py`

**Purpose**: Generates synthetic non-canonical sequences using ADASYN in one-hot encoding space.

**Arguments**:

| Flag                          | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `--ratio <float>`             | % ratio of non-canonical to canonical sequences (e.g., `100` for equal)     |
| `--acc_can <path>`            | Path to canonical acceptor sequences                                        |
| `--acc_nc <path>`             | Path to non-canonical acceptor sequences                                    |
| `--don_can <path>`            | Path to canonical donor sequences                                           |
| `--don_nc <path>`             | Path to non-canonical donor sequences                                       |
| `--out_acc <path>`            | Output folder to save synthetic acceptor sequences                          |
| `--out_don <path>`            | Output folder to save synthetic donor sequences                             |
| `--use-smote`                 | Use SMOTE instead of ADASYN for synthetic sequence generation (optional)    |


**Example**:

```bash
python3 scripts/data_augmentation/run_generator.py \
  --ratio 100 \
  --acc_can ./to_zenodo/data_400bp/train/POS/ACC/CAN \
  --acc_nc ./to_zenodo/data_400bp/train/POS/ACC/NC \
  --don_nc ./to_zenodo/data_400bp/train/POS/DON/CAN \
  --don_nc ./to_zenodo/data_400bp/train/POS/DON/NC \
  --out_acc ./to_zenodo/data_400bp/train/POS/ACC/ADASYN/ADASYN_100 \ # make sure that ADASYN, CAN, NC are in same folder or place
  --out_don ./to_zenodo/data_400bp/train/POS/DON/ADASYN/ADASYN_100 
```


#### 2.2. `run_visualization.py`

**Purpose**: Visualizes GC/AT nucleotide content using scatter plots.

**Arguments**:

| Flag                          | Description                                                           |
|-------------------------------|-----------------------------------------------------------------------|
| `--canonical <path>`          | Directory containing canonical sequences                              |
| `--noncanonical <path>`       | Directory containing non-canonical sequences                          |
| `--synthetic <path>`          | Directory containing synthetic sequences                              |
| `--title <str>`               | Title prefix for each plot (used in figure titles and filenames)      |
| `--output <path>`             | Directory to save generated plots                                     |


**Example**:

```bash
python3 scripts/visualization/run_visualization.py \
  --canonical ./to_zenodo/data_600bp/train/POS/ACC/CAN \
  --noncanonical ./to_zenodo/data_600bp/train/POS/ACC/NC \
  --synthetic ./to_zenodo/data_600bp/train/POS/ACC/ADASYN/ADASYN_100 \
  --title "Acceptor Sequences" \
  --output ./plots
```

---

### 3. Interpretation

**Purpose**: Computes SHAP values and generates sequence logo plots to interpret model decisions.

**Input**:

* Trained model: `models/SpliceRead_model.h5`
* Data folder: `data/train/POS/ACC`

**Output**:

* SHAP logo plot: `output/acceptor_shap_logo.png`

**Command**:

```bash
python3 scripts/interpretation/run_shap_logo.py \
  --model models/SpliceRead_model.h5 \
  --data data/train/POS/ACC \
  --samples 100 \
  --class_index 1 \
  --output output/acceptor_shap_logo.png
```

---

## Output Summary

* **Synthetic Sequences**  `data/train/POS/[ACC|DON]/ADASYN/ADASYN_`
* **Trained Models**  `model_files/`
* **Visualizations**  `output/`

  * GC/AT Content Plots
  * SHAP Sequence Logos

---

## Citation

If you use SpliceREAD in your research, please cite our repository:

**Zenodo DOI**: [https://doi.org/10.5281/zenodo.15538290](https://doi.org/10.5281/zenodo.15538290)

---

## Contact

For questions, please contact [Khushali Samderiya](mailto:ksamderi@uccs.edu) or [Prof. Oluwatosin Oluwadare](mailto:ooluwada@uccs.edu)

---

## License

MIT License. See `LICENSE` file for details.
