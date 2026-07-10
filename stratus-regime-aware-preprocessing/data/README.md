# Dataset setup

The repository does not redistribute raw eye-tracking data. Download each dataset from its original source and preserve the source citation and license.

## Expected folders

```text
data/raw/etdd70/
  Subject_1003_combined_raw.csv
  Subject_....csv

data/raw/autism/
  1.csv
  2.csv
  ...
```

The notebook reads at most the first ten CSV exports from each directory, matching the reported experiment.

## ETDD70

**Dataset:** ETDD70: Eye-Tracking Dataset for Classification of Dyslexia Using AI-Based Methods  
**Authors:** Nicol Dostalova, Roman Svaricek, Jan Sedmidubsky, Wolf Culemann, Cenek Sasinka, Pavel Zezula, and Jiri Cenek  
**Repository:** https://zenodo.org/records/13332134

The loader expects raw combined subject files with timestamps and left/right gaze coordinates. The implementation averages available valid eyes into canonical `x` and `y` coordinates.

## Eye-Tracking Autism dataset

**Dataset article:** Federica Cilia, Romuald Carette, Mahmoud Elbattah, Jean-Luc Guérin, and Gilles Dequen. *Eye-Tracking Dataset to Support the Research on Autism Spectrum Disorder* (2022).  
**Repository:** https://www.kaggle.com/datasets/imtkaggleteam/eye-tracking-autism

A single CSV export may contain several participants. The loader therefore separates sequences by participant, trial, and stimulus before extracting reference windows. The train/test split is participant-level.

## Alternative locations

Set environment variables instead of copying data into the repository:

### PowerShell

```powershell
$env:STRATUS_ETDD70_DIR = "D:\datasets\etdd70"
$env:STRATUS_AUTISM_DIR = "D:\datasets\autism"
```

### Bash

```bash
export STRATUS_ETDD70_DIR=/data/etdd70
export STRATUS_AUTISM_DIR=/data/autism
```

## Data protection

Do not commit participant-level raw coordinates, local dataset copies, or files that violate the source license. Aggregate output tables and plots from the final run are already included in `results/`.
