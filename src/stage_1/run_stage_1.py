from circadipy import chrono_reader as chr
from tkinter import filedialog
from config.analysis_config import DATA, RAW, ER4000_DATA_FOLDER, METADATA_FILE
from src.utils.io_utils import discover_experiments, generate_metadata, load_experiment_metadata, save_experiment_metadata, build_er4000_file_index, save_protocol, load_protocol, infer_total_days_er4000
import tkinter as tk
import os
import warnings
warnings.filterwarnings('ignore')

# DEFINE IF FOLDERS SELECTION IS INTERACTIVE OR PRE-DEFINED AND GLOBAL VARIABLES -------------------------------------------------------------------------------
INTERACTIVE = False  # If True, the user will be prompted to select the folders containing the data files. If False, the pre-defined paths will be used.
ER4000_ANALYSIS = True
FORCE_METADATA = False  # If True, forces the use of existing metadata even if it does not match the discovered experiments.

# This script was constructed assuming a specific nomenclature for the data files, and may not work correctly otherwise.
# Therefore, we suggest that all the file's names follow this logic:
# ANIMAL 1 - ALE ; the first part is the animal identifier, then a - , and lastly the type of data, with ALE meaning spontaneous locomotor activity and TEMP temperature

# DEFINE DE DATA AND FIGURES FOLDERS -------------------------------------------------------------------------------
# All defined folders are relative to the current script's directory
# This allows the script to be run from different locations without hardcoding paths
# However, the script assumes that the folder structure is maintained as described, the data folders are present, the figures folder exists
# and the scripts are run from the correct directory. That is, the "scripts" folder, at the same level as the "raw_data_er4000" and "raw_data_intellicage" folders.

if INTERACTIVE:
    # Open a Tkinter file dialog to select the folders
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()  # Hide the main window

    print("Select the folder containing ER-4000 data files.")
    ER4000_DATA_FOLDER = filedialog.askdirectory(title="Select ER-4000 Data Folder")
else:
    if not os.path.exists(ER4000_DATA_FOLDER):
        raise FileNotFoundError(f"ER-4000 data folder not found at {ER4000_DATA_FOLDER}. Please check the path or enable INTERACTIVE mode.")

# DISCOVER AVAILABLE EXPERIMENTS -------------------------------------------------------------------------------

registry = discover_experiments(RAW)

for device, info in registry.items():
    if not info["present"]:
        print(f"[INFO] {device} folder not present, skipping.")
        continue

    if info["empty"]:
        print(f"[INFO] No data found for {device}, skipping.")
        continue

    print(f"[INFO] Found {len(info['experiments'])} experiments for {device}")

    if device == "ER4000":
        bad = [e for e in info["experiments"] if not e["folder_name_ok"]]
        if bad:
            warnings.warn(
                f"{len(bad)} ER4000 folders have invalid names: "
                + ", ".join(e["name"] for e in bad)
            )

# CREATE / LOAD METADATA -------------------------------------------------------------------------------
experiment_metadata = {}

metadata_existed_before = METADATA_FILE.exists()

if metadata_existed_before and not FORCE_METADATA:
    print(f"[INFO] Loading existing metadata from {METADATA_FILE}")
    experiment_metadata = load_experiment_metadata(METADATA_FILE)

    # ---- STRUCTURE VALIDATION ----
    if "_meta" not in experiment_metadata or "experiments" not in experiment_metadata:
        raise RuntimeError(
            "Invalid metadata structure. Expected top-level keys: '_meta' and 'experiments'."
        )

    discovered_names = {
        exp["name"]
        for info in registry.values()
        if info["present"] and not info["empty"]
        for exp in info["experiments"]
    }
    metadata_names = set(experiment_metadata["experiments"].keys())

    missing_in_metadata = discovered_names - metadata_names
    extra_in_metadata = metadata_names - discovered_names

    if missing_in_metadata or extra_in_metadata:
        print("\n[WARNING] Metadata mismatch detected — stopping so you can fix it manually.")

        if missing_in_metadata:
            print(f"  Experiments missing metadata: {sorted(missing_in_metadata)}")

        if extra_in_metadata:
            print(f"  Metadata entries with no matching folder: {sorted(extra_in_metadata)}")

        raise SystemExit(
            f"\n[STOP] Please edit {METADATA_FILE} to match the discovered folders, "
            "then re-run Stage 1."
        )

else:
    if metadata_existed_before and FORCE_METADATA:
        print("[INFO] FORCE_METADATA enabled — regenerating metadata (will overwrite).")
    else:
        print(f"[INFO] No metadata file found at {METADATA_FILE} — generating a template.")

    experiment_metadata = generate_metadata(registry)
    save_experiment_metadata(experiment_metadata, METADATA_FILE, overwrite=FORCE_METADATA)

# BUILD ER4000 FILE INDEX -------------------------------------------------------------------------------
experiments_ordered, exp_index, exp_info = build_er4000_file_index(RAW, registry, experiment_metadata)

# INFER / FIX MISSING METADATA FIELDS USING CIRCADIPY -------------------------------------------------------------------------------
changed = False
assert all(isinstance(v, dict) and "labels" in v for v in exp_info.values())
for exp_name in experiments_ordered:
    meta = exp_info[exp_name]

    if meta["zt0_time"] is None:
        raise RuntimeError(f"[STOP] zt0_time is None for experiment '{exp_name}' in {METADATA_FILE}")

    labels = meta.get("labels", {})
    cycle_types = labels.get("cycle_types") or []
    test_labels = labels.get("test_labels") or []
    cycle_days = labels.get("cycle_days") or []

    # Ensure at least 1 block exists (required for circadipy)
    if len(cycle_types) == 0 or len(test_labels) == 0:
        cycle_types = ["LD"]
        test_labels = ["all"]
        cycle_days = []  # let it infer
        changed = True

    # Use ANY one protocol/parameter file in this experiment as a probe
    if not exp_index.get(exp_name):
        raise RuntimeError(f"Empty exp_index for {exp_name}")
    any_protocol = next(iter(exp_index[exp_name]))
    entry = exp_index[exp_name][any_protocol]
    probe_file = str(entry.get("ALE") or entry.get("TEMP")) 
    if probe_file is None:
        raise RuntimeError(f"No ALE/TEMP file found to probe days for {exp_name}")
    N = infer_total_days_er4000(probe_file, meta["zt0_time"], consider_first_day=False)

    # If cycle_days missing -> fill as single inferred block
    if len(cycle_days) == 0:
        cycle_days = [N]
        # also force single-block labels to match lengths
        cycle_types = [cycle_types[0]]
        test_labels = [test_labels[0]]
        changed = True
    else:
        # If multi-block lengths mismatch, safest non-breaking fallback:
        if not (len(cycle_days) == len(cycle_types) == len(test_labels)):
            print(f"[WARNING] labels length mismatch in {exp_name}; forcing single block to avoid crash")
            cycle_days = [N]
            cycle_types = [cycle_types[0]]
            test_labels = [test_labels[0]]
            changed = True
        else:
            # Adjust total days to match circadipy's expected N
            delta = N - sum(cycle_days)
            if delta != 0:
                cycle_days[0] = int(cycle_days[0] + delta)
                changed = True

    # Write back
    meta["labels"] = {"cycle_types": cycle_types, "test_labels": test_labels, "cycle_days": cycle_days}

# Save updated metadata once
if changed:
    experiment_metadata["experiments"].update(exp_info)
    save_experiment_metadata(experiment_metadata, METADATA_FILE, overwrite=True)
    print(
        f"\n[STOP] Metadata is written to {METADATA_FILE}.\n"
        "Please review and configure it manually (e.g., zt0_time, labels, exclusions),\n"
        "The cycle_days field has been filled in for your convenience.\n"
        "Now, please, review and adjust any other fields as needed.\n"
        "Then re-run Stage 1 with FORCE_METADATA=False.\n"
    )
    print("""An example of filled metadata for an ER4000 experiment is shown below:

        dados_03062025:
            files:
                n_files: 14
                pattern: "*.asc"
                resolved:
                    - "FEMEA 1 - ALE.asc"
                    - "FEMEA 1 - TEMP.asc"
                    - "FEMEA 2 - ALE.asc"
                    - "FEMEA 2 - TEMP.asc"
            labels:
                cycle_types: ["LD"]
                test_labels: ["A"]
                cycle_days: [12]
            notes: ""
            zt0_time: 20
        """
    )
    print("\n[OBS] Capitalization and scaping matters.")
    print("[OBS] So, please, make sure to follow the format exactly.")
    print("[OBS] All caps for cycle types (e.g., \"LD\", \"DD\").")
    print("[OBS] Quotes for test labels (e.g., \"A\", \"B\", \"ALL\").")
    print("[OBS] Always use double quotes (\" \"), never single quotes (\' \').")
    print("[OBS] Don\'t change the following fields: n_files, pattern, resolved.")
    raise SystemExit(2)

# ---- CONSISTENCY CHECKS ----
if len(set(experiments_ordered)) != len(experiments_ordered):
    raise RuntimeError("Duplicate experiment names detected in registry.")

if set(experiments_ordered) != set(experiment_metadata["experiments"].keys()):
    raise RuntimeError("Discovery/metadata experiment mismatch.")

# ---- PROTOCOL CONSISTENCY CHECK ----
protocol_sets = [set(exp_index[exp].keys()) for exp in experiments_ordered]
common_protocols = set.intersection(*protocol_sets) if protocol_sets else set()

if not common_protocols:
    raise RuntimeError(
        "No common protocols found across experiments "
        "(e.g., 'FEMEA 1')."
    )

for exp_name, protos in zip(experiments_ordered, protocol_sets):
    missing = common_protocols - protos
    extra = protos - common_protocols
    if missing or extra:
        print(
            f"[WARNING] Protocol mismatch in {exp_name}: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )

# ---- MAIN ER4000 PROCESSING LOOP ----
DERIVED = DATA / "derived" / "stage1_protocols"
step = 0
total_steps = len(common_protocols) * 2 * len(experiments_ordered)

for protocol_name in sorted(common_protocols):
    for parameter in ("ALE", "TEMP"):
        for exp_name in experiments_ordered:
            step += 1

            out_dir = DERIVED / "ER4000" / exp_name
            out_file = out_dir / f"{protocol_name.replace(' ', '_')}_{parameter}.pkl"

            print(
                f"[{step}/{total_steps}] "
                f"{protocol_name} {parameter} | {exp_name}",
                end="\r",
            )

            try:
                if out_file.exists():
                    # ---- LOAD FROM CACHE ----
                    protocol = load_protocol(out_file)

                else:
                    # ---- READ FROM RAW ----
                    file_path = str(exp_index[exp_name][protocol_name][parameter])
                    meta = exp_info[exp_name]

                    try: 
                        protocol = chr.read_protocol(
                            name=protocol_name,
                            file=file_path,
                            zt_0_time=meta["zt0_time"],
                            labels_dict=meta["labels"],
                            type="er4000",
                            consider_first_day=False,
                        )
                    except Exception as e:
                        print(f"circadipy read_protocol failed for {exp_name} {protocol_name} {parameter}: {e}")
                        raise SystemExit(
                            f"\n[STOP] Metadata is written to {METADATA_FILE}.\n"
                            "Please review and configure it manually (e.g., zt0_time, labels, exclusions), "
                            "then re-run Stage 1 with FORCE_METADATA=False."
                        )

                    save_protocol(
                        protocol=protocol,
                        out_dir=out_dir,
                        animal=protocol_name,
                        parameter=parameter,
                    )

            except Exception as e:
                print(
                    f"\n[ERROR] Failed {protocol_name} {parameter} "
                    f"in {exp_name}: {e}"
                )
                continue

print("\n[INFO] ER4000 protocol extraction complete.")