# 0. Required Materials

This repository assumes that the following resources are available locally:

- **MIMIC-III v1.4**  
  Download from PhysioNet: https://physionet.org/content/mimiciii/1.4/  
  The dataset must be placed at a user-defined location referenced as `mimic_dir` in the configuration file.

- **CAML splits (top-50 labels)**  
  If not already available in this repository, the splits can be obtained from:  
  https://github.com/mireiahernandez/icd-continuous-prediction/blob/main/dataset/splits/caml_splits.csv

- **Pretrained Language Model (PLM) checkpoint**  
  Bio-LM models can be found at:  
  https://github.com/facebookresearch/bio-lm  
  The downloaded checkpoint should be referenced via `base_checkpoint` in the configuration file.

---

# 1. Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

# 2) Data Preprocessing
The script for preprocessing the data is located in Patient_Journey_Modelling/Data
/Preprocessing/pj_preprocessor.py.

This module constructs patient-admission–level (HADM_ID) event timelines by aggregating and aligning information from multiple MIMIC-III tables, including:

* clinical notes

* laboratory events

* microbiology results

* drug prescriptions

All events are sorted chronologically and written to a JSON Lines file (one admission per line).

In addition, the script builds token dictionaries for structured event types (laboratory, drug, and microbiology events) in order to standardize their representation during model training and inference.
The generated token_dicts.json file may contain NaN values.
These must be manually converted to the string "None" before running the training pipeline. This correction is straightforward and can be performed quickly using any text editor or a small script.
# 3) Run the code 

Model training is controlled via the configuration file.

The parameter fusion specifies the fusion mechanism used in the model architecture:
* NOMA-G: set "fusion": "gating"
* NOMA-M: set "fusion": "mha"

# Notice 
TThis codebase is built on top of the following repository: https://github.com/mireiahernandez/icd-continuous-prediction 
In particular, it reuses and extends components related to the LAHST architecture.