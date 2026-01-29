# Circadian pipeline (ER4000 / INTELLICAGE)

This repository implements a **3-stage circadian analysis pipeline** for ER4000 and INTELLICAGE data.

## Pipeline overview

- **Stage 1 (`run_stage_1.py`)**  
  Discovers raw experiments, generates and validates per-experiment metadata, then extracts **per-animal, per-parameter** `circadipy` protocols into  
  `data/derived/stage1_protocols/…`

- **Stage 2 (`run_stage_2.py`)**  
  Loads Stage 1 protocols, optionally filters / resamples / normalizes them, concatenates data across experiments (ordered by date), and optionally runs **cosinor** analysis.

- **Stage 3 (`run_stage_3.py`)**  
  Loads Stage 2 concatenated protocols and cosinor results, then generates individual and group plots into  
  `data/exports/figures/…`

---

## MUST-HAVE folder structure

Your repository **must** contain a `data/` directory with at least the following structure:

```text
data/
├─ raw/
│  ├─ ER4000/
│  │  ├─ <experiment_folder_1>/
│  │  ├─ <experiment_folder_2>/
│  │  └─ ...
│  └─ INTELLICAGE/
│     └─ ...
├─ derived/
│
├─ exports/
│  
└─ ...

````

* **Stage 1** discovers experiments under `data/raw`
* **Stage 1 outputs** are written to
  `data/derived/stage1_protocols/ER4000/<experiment>/…`

---

## MUST-HAVE naming conventions

### 1) Experiment folder names (ER4000)

Stage 2 **orders experiments by date parsed from the folder name**.
Folders that do not match the expected date pattern are skipped.

**Recommended format:**

```text
dados_03062025
```

* Date format: **DDMMYYYY**
* Date must appear after an underscore

---

### 2) Raw file names inside each ER4000 experiment

Stage 1 expects **exactly** this naming scheme:

```text
ANIMAL NAME - ALE.asc
ANIMAL NAME - TEMP.asc
```

Example:

```text
FEMEA 1 - ALE.asc
FEMEA 1 - TEMP.asc
FEMEA 2 - ALE.asc
FEMEA 2 - TEMP.asc
```

⚠️ **Important**

* Capitalization matters
* Spaces matter
* Do **not** rename files after metadata is generated

---

### 3) Stage 1 derived protocol filenames

Stage 1 saves extracted protocols as:

```text
data/derived/stage1_protocols/ER4000/<experiment>/<ANIMAL>_<PARAM>.pkl
```

Example:

```text
FEMEA_1_ALE.pkl
FEMEA_1_TEMP.pkl
```

---

### 4) Stage 2 concatenated filenames

Stage 2 writes flat, per-animal outputs:

```text
<ANIMAL>_<PARAM>.pkl
<ANIMAL>_<PARAM>_NORM-<NORM_TYPE>.pkl
```

---

## Config file: naming & location

All stage scripts import configuration from:

```text
config/analysis_config.py
```

**Rules:**

* File must be named **exactly** `analysis_config.py`
* Must live inside the `config/` package
* Do not break imports such as:

```python
from config.analysis_config import ...
```

---

## What everything does in `analysis_config.py`

### A) Core paths

These define the project layout and are used directly by the stages:

* `DATA`, `RAW`, `EXPORTS`
* `ER4000_DATA_FOLDER`
* `STAGE1_DIR`
* `STAGE2_DIR`
* `COSINOR_DIR`
* Figure output dirs:

  * `ACTOGRAM_FIGURES_DIR`
  * `TIME_SERIES_FIGURES_DIR`
  * `COSINOR_FIGURES_DIR`

---

### B) Experiment metadata file

Stage 1 uses a **single metadata file**:

```python
METADATA_FILE
```

Stage 1 will:

* Generate a template if none exists
* Stop if metadata does not match discovered experiment folders
* Infer missing `cycle_days`, write metadata, **then stop so you can review**

---

### C) Per-experiment metadata fields (editable)

For each experiment (e.g. `dados_03062025`):

* `zt0_time` (**must not be `None`**)
* `labels`

  * `cycle_types`
  * `test_labels`
  * `cycle_days`
* `files` (**must not be changed manually**)

  * Pattern + resolved file list
    ⚠️ **Do not manually edit file lists**

---

### D) Stage 2 processing flags

These control Stage 2 behavior:

* `RUN_COSINOR` - whether to run cosinor analysis
* `FORCE_COSINOR`- whether to rerun cosinor even if results exist
* `SAVE_RAW` - whether to save concatenated raw protocols
* `SAVE_NORM` - whether to save concatenated normalized protocols
* `NORM_TYPE`- normalization method to use (**must be one of**): 
  * `zscore`
  * `minmax`
* `NORM_PER_DAY` - whether to normalize per-day instead of per-protocol (default: `False`)
* `COSINOR_CONFIG` - cosinor analysis parameters (see `circadipy` docs)

---

### E) Device selection (Stage 1)

* `ER4000_ANALYSIS`
* `INTELLICAGE_ANALYSIS`

If a flag is `False`, that device is skipped.

---

## Group metadata (required for group plots)

Group plots require a `GROUP_METADATA` mapping:

```python
"FEMEA 1": {
    "group": "CONTROL",
    "exclude": False
}
```

**Rules:**

1. Every animal appearing in Stage 2 outputs **must exist** in group metadata
2. `group` is used to split, as of now, hardocded **CONTROL vs HIPO**
3. Matching is case-insensitive

---

## Running the pipeline with your `circadipy` environment

The only requirement is that **the Python interpreter you use** has `circadipy` installed.

### Option A — activate environment (recommended)

> [!NOTE]
> All commands to run the scripts must be run from the repository root.
> That is, the main folder, `Circadian_analysis/`.
> From there, you should be able to run using referenced paths.
> For example: To run `run_stage_1.py`, you would do:
> `python src.stage_1.run_stage_1.py`
> Thats because `src/` is a subpackage of the main repository package.
> And the scripts use relative imports to know where everything is.

So, assuming you have a `circadipy` environment named `circadipy_env`
And your repository is at `/path/to/Circadian_analysis`
And you are using **bash** (Linux, Mac, Windows WSL):

```bash
conda activate circadipy_env
cd /path/to/Circadian_analysis
python src.stage_1.run_stage_1.py
```

---

### Option B — explicit interpreter path (Windows)

> [!NOTE]
> In this case you can modify the .bat files in the main folder to point to your interpreter.
> That is. edit the path to your `python.exe` inside each `.cmd` file to match the one from your `circadipy` environment.
> You can verify where your interpreter is by running this inside the activated environment:

```bash
where python
```

Then, to run Stage 1, with a full path to your interpreter, double click the .cmd file, without changing it from the main folder:

```bash
C:\path\to\miniconda3\envs\circadipy_env\python.exe src\stage_1\run_stage_1.py
```

---

## Simple “how to run”

### 0) Place raw files correctly

```text
data/raw/ER4000/dados_03062025/FEMEA 1 - ALE.asc
data/raw/ER4000/dados_03062025/FEMEA 1 - TEMP.asc
```

Repeat for all experiments and animals.

---

### 1) Stage 1 — discovery + metadata + extraction

```bash
python src/stage_1/run_stage_1.py
```

What happens:

* Discovers experiment folders
* Generates metadata template if missing
* Stops if metadata needs editing
* Extracts protocols using `circadipy`
* Writes `.pkl` files to `stage1_protocols`

---

### 2) Stage 2 — concatenation + normalization + cosinor

```bash
python src/stage_2/run_stage_2.py
```

What happens:

* Loads Stage 1 outputs
* Sorts experiments by parsed date
* Writes concatenated `.pkl` files
* Optionally runs cosinor analysis

---

### 3) Stage 3 — plotting

```bash
python src/stage_3/run_stage_3.py
```

What happens:

* Loads concatenated protocols
* Loads cosinor results if available
* Produces individual and group plots under:

```text
data/exports/figures/groups/<ALE|TEMP>/
```