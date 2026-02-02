from config.analysis_config import COSINOR_CONFIG, COSINOR_DIR, STAGE1_DIR, STAGE2_DIR, RUN_COSINOR, FORCE_COSINOR, SAVE_RAW, SAVE_NORM, NORM_TYPE, NORM_PER_DAY, EXPORTS
from src.utils.io_utils import load_protocol, experiment_date, filter_protocol
from circadipy import chrono_rhythm as chrt
from collections import defaultdict
from copy import deepcopy
import pickle
import warnings

warnings.filterwarnings("ignore", category=FutureWarning,)

all_protocols = {} 

print("[INFO] Loading and filtering Stage 1 protocols...")

# ------------------------------------------------------------
# CACHE: if concatenated stage2 output exists, skip building it
# ------------------------------------------------------------
def _stage2_out_exists(animal: str, parameter: str) -> bool:
    stem = f"{animal.replace(' ', '_')}_{parameter}"
    raw_pkl  = STAGE2_DIR / f"{stem}.pkl"
    norm_pkl = STAGE2_DIR / f"{stem}_NORM-{NORM_TYPE}.pkl"

    raw_ok  = (not SAVE_RAW)  or raw_pkl.exists()
    norm_ok = (not SAVE_NORM) or norm_pkl.exists()
    return raw_ok and norm_ok

def _cosinor_exists(variant_dir, animal, parameter):
    safe = animal.replace(" ", "_")
    out_dir = variant_dir / f"{safe}_{parameter}"
    return (out_dir / "best_models.pkl").exists() and (out_dir / "best_models_fixed.pkl").exists()


# Build a set of keys we should skip entirely
skip_keys = set()
for animal_dir in [STAGE2_DIR]:  # keeping simple: stage2 is flat
    for p in animal_dir.glob("*.pkl"):
        s = p.stem
        if "_NORM-" in s:
            base = s.split("_NORM-")[0]
        else:
            base = s
        # base like FEMEA_1_ALE
        try:
            a, par = base.rsplit("_", 1)
            skip_keys.add((a.replace("_", " "), par))
        except ValueError:
            pass

print(f"[INFO] Stage2 cache found for {len(skip_keys)} animal/parameter pairs; will skip recompute.")

# ------------------------------------------------------------
# STREAM: sort experiments, filter per-exp, concat immediately
# ------------------------------------------------------------

# only keep real experiment dirs; ignore non-date folders safely
exp_dirs = [p for p in STAGE1_DIR.iterdir() if p.is_dir()]
exp_dirs_sorted = []
for p in exp_dirs:
    try:
        _ = experiment_date(p.name)
        exp_dirs_sorted.append(p)
    except ValueError:
        # ignore folders that don't match the date pattern
        continue

exp_dirs_sorted = sorted(exp_dirs_sorted, key=lambda p: experiment_date(p.name))
concatenated_protocols = {}
announced_skips = set()
n_segments = defaultdict(int)  # (animal, parameter) -> how many experiments were appended
step = 0
total = sum(len(list(p.glob("*.pkl"))) for p in exp_dirs_sorted)

for exp_dir in exp_dirs_sorted:
    exp_name = exp_dir.name

    for pkl_file in exp_dir.glob("*.pkl"):
        step += 1
        stem = pkl_file.stem
        animal, parameter = stem.rsplit("_", 1)
        animal = animal.replace("_", " ")
        key = (animal, parameter)

        if _stage2_out_exists(animal, parameter):
            if key not in announced_skips:
                print(f"[INFO] Using cached concat {animal} {parameter} (Stage2)")
                announced_skips.add(key)

            # IMPORTANT: load cached raw protocol so cosinor can run
            cached_raw = STAGE2_DIR / f"{animal.replace(' ', '_')}_{parameter}.pkl"
            if cached_raw.exists():
                concatenated_protocols.setdefault(animal, {})[parameter] = load_protocol(cached_raw)
            continue


        try:
            raw_protocol = load_protocol(pkl_file)
            protocol = filter_protocol(raw_protocol)
        except Exception as e:
            print(f"[ERROR] Failed filtering {animal} {parameter} in {exp_name}: {e}")
            continue

        concatenated_protocols.setdefault(animal, {})

        if parameter not in concatenated_protocols[animal]:
            concatenated_protocols[animal][parameter] = protocol
        else:
            concatenated_protocols[animal][parameter].concat_protocols(protocol, method="last")

        n_segments[(animal, parameter)] += 1
        print(f"[{step}/{total}] Filtering and concatenating {animal} {parameter} | {exp_name}", end="\r")
print() 

# Warn about incomplete series
for (animal, parameter), k in sorted(n_segments.items()):
    if k < 2:
        print(f"[WARNING] Not enough protocols to concat for {animal} {parameter} (n={k})")

print("[INFO] Streaming concat complete.")

# ------------------------------------------------------------
# COSINOR
# ------------------------------------------------------------
if RUN_COSINOR:
    print("[INFO] Running cosinor analysis on concatenated protocols...")

    # RAW variant
    RAW_DIR = COSINOR_DIR / "raw"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RAW_TABLES_DIR = EXPORTS / "tables"/ "raw"
    RAW_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # NORM variant (match your Stage 3 expectation)
    NORM_DIR = COSINOR_DIR / f"norm_{NORM_TYPE}"
    NORM_DIR.mkdir(parents=True, exist_ok=True)
    NORM_TABLES_DIR = EXPORTS / "tables"/ f"norm_{NORM_TYPE}"
    NORM_TABLES_DIR.mkdir(parents=True, exist_ok=True)


    items = [
        (animal, parameter, protocol_raw)
        for animal, params in concatenated_protocols.items()
        for parameter, protocol_raw in params.items()
    ]

    for i, (animal, parameter, protocol_raw) in enumerate(items, start=1):

        # --- RAW cosinor ---
        if SAVE_RAW and (FORCE_COSINOR or not _cosinor_exists(RAW_DIR, animal, parameter)):
            safe = animal.replace(" ", "_")
            out_dir = RAW_DIR / f"{safe}_{parameter}"
            out_dir.mkdir(parents=True, exist_ok=True)
            table_dir = RAW_TABLES_DIR/f"{safe}_{parameter}"
            table_dir.mkdir(parents=True, exist_ok=True)
    
            try:
                best_models, best_models_file = chrt.fit_cosinor(protocol_raw, dict=COSINOR_CONFIG, save_folder=str(table_dir))
                best_models_fixed, best_models_fixed_file = chrt.fit_cosinor_fixed_period(protocol_raw, best_models, save_folder=str(table_dir))
                with open(out_dir / "best_models.pkl", "wb") as f:
                    pickle.dump(best_models, f)
                with open(out_dir / "best_models_fixed.pkl", "wb") as f:
                    pickle.dump(best_models_fixed, f)
                print(f"[{i}/{len(items)}] Completed RAW cosinor for {animal} {parameter}")
            except Exception as e:
                print(f"[ERROR] RAW cosinor failed for {animal} {parameter}: {e}")

        # --- NORM cosinor ---
        if SAVE_NORM and (FORCE_COSINOR or not _cosinor_exists(NORM_DIR, animal, parameter)):
            safe = animal.replace(" ", "_")
            out_dir = NORM_DIR / f"{safe}_{parameter}"
            out_dir.mkdir(parents=True, exist_ok=True)
            table_dir = NORM_TABLES_DIR/f"{safe}_{parameter}"
            table_dir.mkdir(parents=True, exist_ok=True)

            try:
                protocol_norm = deepcopy(protocol_raw)
                protocol_norm.normalize_data(type=NORM_TYPE, per_day=NORM_PER_DAY)

                best_models, best_models_file = chrt.fit_cosinor(protocol_norm, dict=COSINOR_CONFIG, save_folder=str(table_dir))
                best_models_fixed, best_models_fixed_file = chrt.fit_cosinor_fixed_period(protocol_norm, best_models, save_folder=str(table_dir))

                with open(out_dir / "best_models.pkl", "wb") as f:
                    pickle.dump(best_models, f)
                with open(out_dir / "best_models_fixed.pkl", "wb") as f:
                    pickle.dump(best_models_fixed, f)
                print(f"[{i}/{len(items)}] Completed NORM cosinor for {animal} {parameter}")
            except Exception as e:
                print(f"[ERROR] NORM cosinor failed for {animal} {parameter}: {e}")

# ------------------------------------------------------------
# SAVE stage 2 outputs
# ------------------------------------------------------------
print("[INFO] Saving Stage 2 concatenated protocols...")
STAGE2_DIR.mkdir(parents=True, exist_ok=True)

for animal, params in concatenated_protocols.items():
    safe_animal = animal.replace(" ", "_")

    for parameter, protocol_raw in params.items():
        # --- RAW save (real units) ---
        if SAVE_RAW:
            out_raw = STAGE2_DIR / f"{safe_animal}_{parameter}.pkl"
            with open(out_raw, "wb") as f:
                pickle.dump(protocol_raw, f)
            print(f"[INFO] Saved RAW protocol for {safe_animal} {parameter}")

        # --- NORM save (phase-friendly units) ---
        if SAVE_NORM:
            protocol_norm = deepcopy(protocol_raw)
            protocol_norm.normalize_data(type=NORM_TYPE, per_day=NORM_PER_DAY)

            out_norm = STAGE2_DIR / f"{safe_animal}_{parameter}_NORM-{NORM_TYPE}.pkl"
            with open(out_norm, "wb") as f:
                pickle.dump(protocol_norm, f)
            print(f"[INFO] Saved NORM protocol for {safe_animal} {parameter}")