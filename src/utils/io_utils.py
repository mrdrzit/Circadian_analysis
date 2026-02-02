import re
import json
import natsort
import pickle
import unicodedata
import sys
import copy
import pandas as pd
import numpy as np
from circadipy import chrono_reader as chr
from scipy.stats import sem, t
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from config.analysis_config import DEFAULT_METADATA_BY_DEVICE, KNOWN_DEVICES, ER4000_FOLDER_REGEX, DATE_REGEX, GROUPS_METADATA_PATH, RESAMPLE_TO_VALUE, FILTER_WINDOW, COSINOR_DIR

_cosinor_cache = {}

FILE_PATTERNS_BY_DEVICE = {
    "ER4000": "*.asc",
    "INTELLICAGE": "*.csv",
}

UNICODE_BOUNDARY = r'(?<![a-z])(?:{})(?![a-z])'

FEMALE_REGEX = re.compile(
    UNICODE_BOUNDARY.format(
        r'f(?:emale|em|emea)?|fem'
    )
)

MALE_REGEX = re.compile(
    UNICODE_BOUNDARY.format(
        r'm(?:ale|asc|acho)?|masc|mach'
    )
)

def resolve_files_for_experiment(exp_dir: Path, device: str):
    pattern = FILE_PATTERNS_BY_DEVICE.get(device)

    if pattern is None:
        raise KeyError(f"No file pattern defined for device '{device}'")

    files = list(exp_dir.glob(pattern))
    return natsort.os_sorted(files)
    
def discover_experiments(raw_root: Path):
    """
    Discovers experiments per device, explicitly handling:
    - missing device folders
    - empty device folders
    - populated device folders

    Adds:
    - folder_name_ok
    - files_resolved
    - n_files
    """
    registry = {}

    for device_name in KNOWN_DEVICES.values():
        device_dir = raw_root / device_name

        # Case 1: device folder does not exist
        if not device_dir.exists():
            registry[device_name] = {
                "present": False,
                "empty": True,
                "experiments": [],
            }
            continue

        exp_dirs = [p for p in device_dir.iterdir() if p.is_dir()]

        # Case 2: device folder exists but has no experiments
        if not exp_dirs:
            registry[device_name] = {
                "present": True,
                "empty": True,
                "experiments": [],
            }
            continue

        experiments = []
        for exp_dir in exp_dirs:

            if device_name == "ER4000":
                folder_name_ok = bool(ER4000_FOLDER_REGEX.match(exp_dir.name))
            else:
                folder_name_ok = None

            files = resolve_files_for_experiment(exp_dir, device_name)

            experiments.append({
                "name": exp_dir.name,
                "device": device_name,
                "raw_dir": exp_dir,
                "folder_name_ok": folder_name_ok,
                "files_resolved": files,          # authoritative
                "n_files": len(files),
            })

        registry[device_name] = {
            "present": True,
            "empty": False,
            "experiments": experiments,
        }

    return registry

def generate_metadata(registry):
    """
    Generate a metadata template with device-specific defaults
    and a snapshot of resolved files.
    """
    experiments = {}

    for info in registry.values():
        if not info["present"] or info["empty"]:
            continue

        for exp in info["experiments"]:
            device = exp["device"]

            if device not in DEFAULT_METADATA_BY_DEVICE:
                raise KeyError(
                    f"No default metadata defined for device '{device}'"
                )

            meta = deepcopy(DEFAULT_METADATA_BY_DEVICE[device])

            # Snapshot resolved files (relative paths only)
            meta["files"]["resolved"] = [
                str(p.relative_to(exp["raw_dir"]))
                for p in exp["files_resolved"]
            ]
            meta["files"]["n_files"] = exp["n_files"]

            experiments[exp["name"]] = meta

    metadata = {
        "_meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_experiments": len(experiments),
        },
        "experiments": experiments,
    }

    return metadata

def load_experiment_metadata(metadata_path: Path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Hard guard
    if "_meta" not in metadata or "experiments" not in metadata:
        raise ValueError(
            "Invalid metadata file structure. "
            "Expected top-level keys: '_meta', 'experiments'."
        )

    return metadata

def save_experiment_metadata(
    metadata: dict,
    metadata_path: Path,
    overwrite: bool = False,
):
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(
            f"Metadata file already exists at {metadata_path}. "
            "Use overwrite=True if you really want to replace it."
        )

    if "_meta" not in metadata or "experiments" not in metadata:
        raise ValueError(
            "Invalid metadata structure. Expected keys: '_meta', 'experiments'."
        )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)

    print(f"[INFO] Metadata saved to {metadata_path}")

def parse_er4000_filename(file_name: str):
    """
    Expected: 'FEMEA 1 - ALE.asc' or 'FEMEA 1 - TEMP.asc'
    Returns: (protocol_name, parameter)
    """
    protocol_name = file_name.split(" -")[0].strip()
    parameter = file_name.split("-")[-1].split(".")[0].strip()  # ALE/TEMP
    return protocol_name, parameter

def build_er4000_file_index(raw_root: Path, registry: dict, metadata: dict):
    """
    Returns:
      experiments_ordered: list[str] experiment folder names (filesystem order)
      exp_index: dict[exp_name][protocol_name][parameter] = Path
      exp_info: dict[exp_name] = metadata for that experiment
    """
    exp_info = metadata["experiments"]

    # ---- ORDER COMES FROM DISCOVERY, NOT METADATA ----
    experiments_ordered = []
    for info in registry.values():
        if not info["present"] or info["empty"]:
            continue

        for exp in info["experiments"]:
            if exp["device"] == "ER4000":
                experiments_ordered.append(exp["name"])

    # Optional: make it deterministic
    experiments_ordered = sorted(experiments_ordered)

    exp_index = {}

    for exp_name in experiments_ordered:
        if exp_name not in exp_info:
            raise RuntimeError(
                f"Experiment '{exp_name}' found on disk but missing from metadata."
            )

        exp_dir = raw_root / "ER4000" / exp_name
        files_meta = exp_info[exp_name]["files"]["resolved"]

        exp_index[exp_name] = {}

        for rel in files_meta:
            p = exp_dir / rel
            file_name = p.name

            animal_name, parameter = parse_er4000_filename(file_name)

            exp_index[exp_name].setdefault(animal_name, {})
            exp_index[exp_name][animal_name][parameter] = p

    return experiments_ordered, exp_index, exp_info

def save_protocol(
    protocol,
    out_dir: Path,
    animal: str,
    parameter: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{animal.replace(' ', '_')}_{parameter}.pkl"
    out_path = out_dir / fname

    with open(out_path, "wb") as f:
        pickle.dump(protocol, f)

    return out_path

def load_protocol(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def experiment_date(exp_name: str) -> datetime:
    """
    Extract date from experiment folder name like 'dados_03062025'
    Returns a datetime.date for sorting.
    """
    m = DATE_REGEX.match(exp_name)
    if not m:
        raise ValueError(
            f"Cannot extract date from experiment name '{exp_name}'"
        )

    day, month, year = map(int, m.groups())
    return datetime(year, month, day)

def load_stage2_protocol(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def validate_group_metadata(protocol_animals, metadata):
    protocol_animals = set(protocol_animals)
    metadata_animals = set(metadata.keys())

    missing = protocol_animals - metadata_animals
    extra = metadata_animals - protocol_animals

    if missing:
        raise ValueError(
            "Animals present in concatenated data but missing in group metadata:\n"
            + "\n".join(sorted(missing))
        )

    if extra:
        raise ValueError(
            "Animals present in group metadata but not found in concatenated data:\n"
            + "\n".join(sorted(extra))
        )

def load_group_metadata():
    with open(GROUPS_METADATA_PATH, "r") as f:
        metadata = json.load(f)

    if not isinstance(metadata, dict):
        raise ValueError("groups.json must be a dict keyed by animal ID")

    for animal, info in metadata.items():
        if "group" not in info:
            raise ValueError(f"Animal {animal} missing 'group' field")

    return metadata

def normalize_filename(s: str) -> str:
    return unicodedata.normalize("NFKD", s).casefold()

def infer_sex_from_name(name: str):
    norm = normalize_filename(name)

    is_female = bool(FEMALE_REGEX.search(norm))
    is_male = bool(MALE_REGEX.search(norm))

    if is_female and is_male:
        return None
    if is_female:
        return "F"
    if is_male:
        return "M"
    return None

def ensure_group_metadata(
    animals,
    group_metadata_path: Path,
):
    if group_metadata_path.exists():
        return

    print("[INFO] Group metadata not found.")
    print(f"[INFO] Creating template at: {group_metadata_path}")

    metadata = {}

    for animal in sorted(animals):
        metadata[animal] = {
            "group": "control",  
            "sex": infer_sex_from_name(animal),
            "exclude": False,
        }

    group_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(group_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n[IMPORTANT]")
    print("A group metadata template has been created.")
    print("Please edit the file and fill in the 'group' field.")
    print("THEN RE-RUN STAGE 3 TO USE IT.\n")

    print("After editing, the entry might look like:")
    print("""
    "FEMEA 1": {
        "group": "Control",
        "sex": "F",
        "exclude": false
    }
    """)

    print("\nCapitalization and spacing matters!\n")
    print("So please, if possible don't use spaces in group names and use single letter only for sex (M/F).\n")
    sys.exit(0)

def compute_ci(series):
    n = series.count()
    if n < 2:
        return (None, None)
    m = series.mean()
    se = sem(series, nan_policy="omit")
    h = se * t.ppf(0.975, df=n - 1)
    return (m - h, m + h)

def plot_with_ci(ax, grouped_df, ci_df, param, color, label, show_ci=True):
    # --- get mean series from grouped_df (handles MultiIndex columns) ---
    try:
        # MultiIndex: (param, "mean")
        mean = grouped_df[(param, "mean")]
    except Exception:
        # Alternative: grouped_df[param]["mean"]
        mean = grouped_df[param]["mean"]

    mean = pd.to_numeric(mean, errors="coerce")

    # --- draw mean line ---
    ax.plot(mean.values, mean.index.values, color=color, linewidth=2, label=label)

    if not show_ci or ci_df is None or param not in ci_df.columns:
        return

    ci_col = ci_df[param]

    # --- unpack CI if compute_ci returns (lo, hi) tuples/lists ---
    first_valid = next((v for v in ci_col.values if v is not None and not (isinstance(v, float) and np.isnan(v))), None)

    if isinstance(first_valid, (tuple, list)) and len(first_valid) == 2:
        lower = ci_col.apply(lambda x: x[0] if isinstance(x, (tuple, list)) and len(x) == 2 else np.nan)
        upper = ci_col.apply(lambda x: x[1] if isinstance(x, (tuple, list)) and len(x) == 2 else np.nan)
    else:
        # If your compute_ci already returns numeric series, treat it as missing CI structure
        lower = pd.Series(np.nan, index=mean.index)
        upper = pd.Series(np.nan, index=mean.index)

    lower = pd.to_numeric(lower, errors="coerce")
    upper = pd.to_numeric(upper, errors="coerce")

    # --- align + drop invalid rows ---
    dfp = pd.concat({"mean": mean, "lower": lower, "upper": upper}, axis=1).dropna()
    if dfp.empty:
        return

    ax.fill_betweenx(
        dfp.index.values,
        dfp["lower"].values,
        dfp["upper"].values,
        color=color,
        alpha=0.25,
        linewidth=0,
    )

def filter_protocol(raw_protocol):
    protocol = copy.deepcopy(raw_protocol)
    protocol.resample(RESAMPLE_TO_VALUE, method="last")
    protocol.apply_filter(window=FILTER_WINDOW, type="moving_average", order=2)
    return protocol

def load_cosinor(animal, parameter, cosinor_root):
    safe_animal = animal.replace(" ", "_")
    cos_dir = cosinor_root / f"{safe_animal}_{parameter}"
    key = (animal, parameter)
    if key in _cosinor_cache:
        return _cosinor_cache[key]

    p1 = cos_dir / "best_models.pkl"
    p2 = cos_dir / "best_models_fixed.pkl"

    if not (p1.exists() and p2.exists()):
        _cosinor_cache[key] = (None, None)
        return (None, None)

    with open(p1, "rb") as f:
        bm = pickle.load(f)
    with open(p2, "rb") as f:
        bmf = pickle.load(f)

    _cosinor_cache[key] = (bm, bmf)
    return (bm, bmf)

def fig_exists(out_dir, suffix):
    return any(out_dir.glob(f"*{suffix}*.png"))

def infer_total_days_er4000(file_path: str, zt0_time: int, consider_first_day: bool = False) -> int:
    dummy_labels = {
        "cycle_types": ["LD"],
        "test_labels": ["all"],
        "cycle_days": [],  # let circadipy infer
    }
    p = chr.read_protocol(
        name="__probe__",
        file=file_path,
        zt_0_time=int(zt0_time),
        labels_dict=dummy_labels,
        type="er4000",
        consider_first_day=consider_first_day,
    )
    return int(len(np.unique(p.days)))

def create_prism_table(dataframe):
    param_name = dataframe.columns[-1]
    pivot_df = dataframe.pivot(index="int_day", columns="animal", values=param_name)
    group_row = dataframe.drop_duplicates("animal")[["animal", "group"]].set_index("animal").T
    group_row.index = ["Group"]
    pivot_with_group = pd.concat([group_row, pivot_df])
    pivot_with_group.reset_index(inplace=True)
    pivot_with_group.rename(columns={"index": "int_day"}, inplace=True)
    return pivot_with_group