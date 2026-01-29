# disable matplotlib interactive mode
import matplotlib
matplotlib.use("Agg")

import ast
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from circadipy import chrono_plotter as chp
from src.utils.io_utils import load_protocol, validate_group_metadata, ensure_group_metadata, compute_ci, plot_with_ci, load_cosinor, fig_exists
from config.analysis_config import ACTOGRAM_FIGURES_DIR, COSINOR_DIR, TIME_SERIES_FIGURES_DIR, STAGE2_DIR, EXPORTS, GROUPS_METADATA_PATH, NORM_TYPE

# ---- suppress circadipy/pandas FutureWarnings (like "closed" -> "inclusive") ----
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Argument `closed` is deprecated.*")

print("[INFO] Stage 3: plotting concatenated protocols and cosinor results...")

PLOT_FINAL_GROUP_FIGURES = True
PLOT_FINAL_INDIVIDUAL_FIGURES = True
PLOT_COSINOR_DATA_AND_ACTOGRAMS = True
FORCE_PLOTS = False
NORM_TAG = f"NORM-{NORM_TYPE}"
USE_NORMALIZED = True   # True -> load *_NORM-zscore.pkl; False -> load raw
COSINOR_VARIANT_DIR = COSINOR_DIR / (f"norm_{NORM_TYPE}" if USE_NORMALIZED else "raw")
COSINOR_FIGURES_DIR = EXPORTS / "figures" / "cosinor" / (f"norm_{NORM_TYPE}" if USE_NORMALIZED else "raw")
ACTOGRAM_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TIME_SERIES_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
COSINOR_VARIANT_DIR.mkdir(parents=True, exist_ok=True)
COSINOR_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

concatenated_protocols = {}

print("[INFO] Loading Stage 2 concatenated protocols...")

pattern = f"*_{NORM_TAG}.pkl" if USE_NORMALIZED else "*.pkl"

for pkl_file in STAGE2_DIR.glob(pattern):
    if not USE_NORMALIZED and pkl_file.stem.endswith(f"_{NORM_TAG}"):
        continue

    stem = pkl_file.stem  # FEMEA_1_ALE
    parts = stem.split("_")

    # raw: FEMEA_1_ALE
    # norm: FEMEA_1_ALE_NORM-zscore
    if USE_NORMALIZED:
        animal = " ".join(parts[:-2])      # FEMEA 1
        parameter = parts[-2]              # ALE
    else:
        animal = " ".join(parts[:-1])
        parameter = parts[-1]

    try:
        protocol = load_protocol(pkl_file)
    except Exception as e:
        print(f"[ERROR] Failed loading {pkl_file}: {e}")
        continue

    concatenated_protocols.setdefault(animal, {})
    concatenated_protocols[animal][parameter] = protocol

print(
    f"[INFO] Loaded {sum(len(v) for v in concatenated_protocols.values())} "
    "concatenated protocols."
)

# Generate group metadata if needed and filter excluded animals ------------------------------------------------------------
animals = list(concatenated_protocols.keys())
ensure_group_metadata(animals=animals, group_metadata_path=GROUPS_METADATA_PATH)

with open(GROUPS_METADATA_PATH, "r", encoding="utf-8") as f:
    GROUP_METADATA = json.load(f)

validate_group_metadata(
    protocol_animals=animals,
    metadata=GROUP_METADATA,
)

# --- apply exclude flag ---
excluded = {a for a, meta in GROUP_METADATA.items() if meta.get("exclude", False)}
if excluded:
    print(f"[INFO] Excluding {len(excluded)} animals via group_metadata: {sorted(excluded)}")

# filter protocols dict
concatenated_protocols = {
    animal: params
    for animal, params in concatenated_protocols.items()
    if animal not in excluded
}

# refresh animals list after filtering
animals = list(concatenated_protocols.keys())

if PLOT_COSINOR_DATA_AND_ACTOGRAMS:
    for animal, params in concatenated_protocols.items():
        for parameter, protocol in params.items():

            safe_animal = animal.replace(" ", "_")

            actogram_dir = ACTOGRAM_FIGURES_DIR / f"{safe_animal}_{parameter}"
            time_series_dir = TIME_SERIES_FIGURES_DIR / f"{safe_animal}_{parameter}"

            cosinor_fig_dir = COSINOR_FIGURES_DIR / f"{safe_animal}_{parameter}"
            cosinor_fig_dir.mkdir(parents=True, exist_ok=True)

            actogram_dir.mkdir(parents=True, exist_ok=True)
            time_series_dir.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] Plotting {animal} {parameter}")

            # ---- LOAD COSINOR RESULTS ----
            cosinor_dir = COSINOR_VARIANT_DIR / f"{safe_animal}_{parameter}"
            best_models_path = cosinor_dir / "best_models.pkl"
            best_models_fixed_path = cosinor_dir / "best_models_fixed.pkl"

            if best_models_path.exists() and best_models_fixed_path.exists():
                best_models, best_models_fixed = load_cosinor(animal, parameter, cosinor_root=COSINOR_VARIANT_DIR)
            else:
                best_models = None
                best_models_fixed = None
                print(
                    f"[WARNING] Missing cosinor results for {animal} {parameter}. "
                    "Skipping model-based plots."
                )

            # ------------------------------------------------------------------
            # TIME SERIES
            # ------------------------------------------------------------------
            ts_suffix = f"concatenated_time_series"

            if fig_exists(time_series_dir, ts_suffix) and not FORCE_PLOTS:
                print(f"[SKIP] Time series already exists for {animal} {parameter}")
            else:
                print(f"[PLOT] Time series for {animal} {parameter}")
                chp.time_serie(
                    protocol,
                    labels=[
                        f"{animal} - concatenated protocol after processing",
                        "Time (Days)",
                        "Amplitude",
                    ],
                    color="midnightblue",
                    save_folder=str(time_series_dir),
                    save_suffix=ts_suffix,
                )

            # ------------------------------------------------------------------
            # ACTOGRAMS
            # ------------------------------------------------------------------
            if parameter.lower() == "ale":
                act_suffix = f"actogram_bar"

                if fig_exists(actogram_dir, act_suffix) and not FORCE_PLOTS:
                    print(f"[SKIP] Actogram bar already exists for {animal}")
                else:
                    print(f"[PLOT] Actogram bar for {animal}")
                    chp.actogram_bar(
                        protocol,
                        first_hour=0,
                        save_folder=str(actogram_dir),
                        save_suffix=act_suffix,
                        x_label="ZT",
                    )

            elif parameter.lower() == "temp":
                act_suffix = f"actogram_colormap"

                if fig_exists(actogram_dir, act_suffix) and not FORCE_PLOTS:
                    print(f"[SKIP] Actogram colormap already exists for {animal}")
                else:
                    print(f"[PLOT] Actogram colormap for {animal}")
                    chp.actogram_colormap(
                        protocol,
                        first_hour=0,
                        save_folder=str(actogram_dir),
                        save_suffix=act_suffix,
                        x_label="ZT",
                    )

            # ------------------------------------------------------------------
            # COSINOR / MODEL-BASED FIGURES
            # ------------------------------------------------------------------
            if best_models is not None and best_models_fixed is not None:

                mod_det_suffix = f"model_overview_detailed"

                if fig_exists(cosinor_fig_dir, mod_det_suffix) and not FORCE_PLOTS:
                    print(f"[SKIP] Model overview detailed exists for {animal} {parameter}")
                else:
                    print(f"[PLOT] Model overview detailed for {animal} {parameter}")
                    chp.model_overview_detailed(
                        protocol,
                        best_models_fixed,
                        save_folder=str(cosinor_fig_dir),
                        save_suffix=mod_det_suffix,
                    )

                mod_suffix = f"model_overview"

                if fig_exists(cosinor_fig_dir, mod_suffix) and not FORCE_PLOTS:
                    print(f"[SKIP] Model overview exists for {animal} {parameter}")
                else:
                    print(f"[PLOT] Model overview for {animal} {parameter}")
                    chp.model_overview(
                        protocol,
                        best_models,
                        save_folder=str(cosinor_fig_dir),
                        save_suffix=mod_suffix,
                    )

                mos_suffix = f"model_over_signal"

                if fig_exists(cosinor_fig_dir, mos_suffix) and not FORCE_PLOTS:
                    print(f"[SKIP] Model over signal exists for {animal} {parameter}")
                else:
                    print(f"[PLOT] Model over signal for {animal} {parameter}")
                    chp.model_over_signal(
                        protocol,
                        best_models,
                        position="head",
                        mv_avg_window=1,
                        save_folder=str(cosinor_fig_dir),
                        save_suffix=mos_suffix,
                    )

    print("[INFO] Stage 3 plotting complete.")

else:
    print("[INFO] Stage 3 actogram and cosinor plotting skipped as per configuration.")

if PLOT_FINAL_INDIVIDUAL_FIGURES:
    PAPER_FIGURES_DIR = EXPORTS / "figures" / "paper" / "individual"
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    PARAMS = ["acrophase_zt", "mesor", "amplitude", "period"]
    XLABELS = ["ZT or CT", "COUNTS", "COUNTS", "HOURS"]
    TITLES = ["ACROPHASE", "MESOR", "AMPLITUDE", "PERIOD"]

    print("[INFO] Paper-style individual rhythmic-parameter plots...")

    # Flatten list for progress
    items = [
        (animal, parameter)
        for animal, params in concatenated_protocols.items()
        for parameter in params.keys()
    ]
    total = len(items)
    i = 0

    for animal, parameter in items:
        i += 1
        protocol = concatenated_protocols[animal][parameter]
        safe_animal = animal.replace(" ", "_")

        out_dir = PAPER_FIGURES_DIR / f"{safe_animal}_{parameter}"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_png = out_dir / f"{safe_animal}_{parameter}_rhythmic_params.png"
        if fig_exists(out_dir, f"{safe_animal}_{parameter}_rhythmic_params") and not FORCE_PLOTS:
            print(f"[{i}/{total}] [SKIP] {animal} {parameter} (exists)")
            continue

        # ---- LOAD COSINOR RESULTS (required) ----
        cosinor_dir = COSINOR_VARIANT_DIR / f"{safe_animal}_{parameter}"
        best_models_fixed_path = cosinor_dir / "best_models_fixed.pkl"

        if not best_models_fixed_path.exists():
            print(f"[{i}/{total}] [WARNING] Missing best_models_fixed for {animal} {parameter} -> {best_models_fixed_path}")
            continue

        print(f"[{i}/{total}] [INFO] Plotting individual rhythmic params: {animal} {parameter}")

        try:
            _, best_models_fixed = load_cosinor(animal, parameter, cosinor_root=COSINOR_VARIANT_DIR)
            if best_models_fixed is None:
                print(f"[{i}/{total}] [WARNING] best_models_fixed is None for {animal} {parameter}")
                continue

            # best_models_fixed is typically a DataFrame
            df = best_models_fixed.copy()

            # Keep only rows that look like real daily fits
            # (your output had some rows with test=nan / missing day)
            if "day" in df.columns:
                df = df[df["day"].notna()].copy()

            # Sort chronologically
            if "day" in df.columns:
                df = df.sort_values("day").reset_index(drop=True)

            if df.empty:
                print(f"[{i}/{total}] [WARNING] Empty best_models_fixed after filtering for {animal} {parameter}")
                continue

            # Define y positions like original (1..N) with inverted y-axis
            days = list(range(1, len(df) + 1))

            fig, axes = plt.subplots(1, 4, figsize=(12, 8))

            for ax, p, xlabel, title in zip(axes, PARAMS, XLABELS, TITLES):
                ax.set_title(title)
                ax.set_xlabel(xlabel)

                # ---- main series ----
                if p not in df.columns:
                    ax.text(0.5, 0.5, f"Missing: {p}", ha="center", va="center", transform=ax.transAxes)
                    ax.set_yticks(np.linspace(days[0], days[-1], min(len(days), 6), dtype=int))
                    ax.invert_yaxis()
                    continue

                x = df[p].values
                ax.plot(x, days, color="black", linewidth=1.5)

                # # ---- CI shading (if available) ----
                # if p == "acrophase_zt":
                #     # You already have lower/upper columns
                #     if "acrophase_zt_lower" in df.columns and "acrophase_zt_upper" in df.columns:
                #         lo = df["acrophase_zt_lower"].values
                #         hi = df["acrophase_zt_upper"].values
                #         ax.fill_betweenx(days, lo, hi, alpha=0.25)
                # elif p == "mesor" and "CI(mesor)" in df.columns:
                #     ci = df["CI(mesor)"].apply(_parse_ci_cell)
                #     lo = [t[0] for t in ci]
                #     hi = [t[1] for t in ci]
                #     if any(v is not None for v in lo) and any(v is not None for v in hi):
                #         ax.fill_betweenx(days, lo, hi, alpha=0.25)
                # elif p == "amplitude" and "CI(amplitude)" in df.columns:
                #     ci = df["CI(amplitude)"].apply(_parse_ci_cell)
                #     lo = [t[0] for t in ci]
                #     hi = [t[1] for t in ci]
                #     if any(v is not None for v in lo) and any(v is not None for v in hi):
                #         ax.fill_betweenx(days, lo, hi, alpha=0.25)

                ax.set_yticks(np.linspace(days[0], days[-1], min(len(days), 6), dtype=int))
                ax.invert_yaxis()  # day 1 at top (same as original)

                # Optional: mimic your original x-lims (tune as needed)
                if p == "acrophase_zt":
                    ax.set_xlim(0, 24)
                elif p == "period":
                    ax.set_xlim(20, 28)

            axes[0].set_ylabel("DAYS")

            fig.suptitle(f"{animal} — {parameter.upper()} (daily cosinor fixed-period)", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            fig.savefig(out_png, dpi=300)
            plt.close(fig)

            print(f"[{i}/{total}] [OK] Saved: {out_png}")

        except Exception as e:
            print(f"[{i}/{total}] [ERROR] Failed plotting {animal} {parameter}: {e}")
            continue

    print("[INFO] Individual paper-style rhythmic-parameter plots complete.")

else:
    print("[INFO] Individual paper-style rhythmic-parameter plots skipped as per configuration.")

if PLOT_FINAL_GROUP_FIGURES:

    # ------------------------------------------------------------------
    # Load cosinor results into long dataframe (PORT OF ORIGINAL)
    # ------------------------------------------------------------------

    all_individual_values = []

    for animal in animals:
        safe_animal = animal.replace(" ", "_")

        for parameter in ("ALE", "TEMP"):
            cos_dir = COSINOR_VARIANT_DIR / f"{safe_animal}_{parameter}"
            fixed_path = cos_dir / "best_models_fixed.pkl"

            if not fixed_path.exists():
                continue

            _, best_models_fixed = load_cosinor(animal, parameter, cosinor_root=COSINOR_VARIANT_DIR)
            if best_models_fixed is None:
                print(f"[WARNING] best_models_fixed is None for {animal} {parameter}")
                continue

            relevant_params_df = best_models_fixed.loc[:, [
                "acrophase_zt",
                "acrophase_zt_lower",
                "acrophase_zt_upper",
                "mesor",
                "amplitude",
                "period",
            ]].copy()

            relevant_params_df["animal"] = animal
            if animal not in GROUP_METADATA:
                print(f"[WARNING] {animal} missing in GROUP_METADATA; skipping.")
                continue
            relevant_params_df["group"] = GROUP_METADATA[animal]["group"]
            relevant_params_df["day"] = np.arange(1, len(relevant_params_df) + 1)
            relevant_params_df["parameter"] = parameter

            all_individual_values.extend(
                relevant_params_df.to_dict(orient="records")
            )

    all_individual_df = pd.DataFrame(all_individual_values)
    included_animals = [a for a in animals if a not in excluded]
    all_individual_df = all_individual_df[all_individual_df["animal"].isin(included_animals)].copy()
    missing = excluded - set(concatenated_protocols.keys())
    if missing:
        print(f"[WARNING] group_metadata excludes animals not found in Stage2: {sorted(missing)}")

    # ------------------------------------------------------------------
    # Group separation and plotting (per parameter)
    # ------------------------------------------------------------------

    for parameter in ("ALE", "TEMP"):

        param_df = all_individual_df[
            all_individual_df["parameter"] == parameter
        ]

        if param_df.empty:
            print(f"[WARNING] No data for {parameter}, skipping group plots.")
            continue

        ctrl_data = param_df[param_df["group"].str.lower() == "control"]
        hipo_data = param_df[param_df["group"].str.lower() == "hipo"]

        if ctrl_data.empty or hipo_data.empty:
            print(
                f"[WARNING] Missing one group for {parameter} "
                f"(CTRL n={len(ctrl_data)}, HIPO n={len(hipo_data)})"
            )
            continue

        # ------------------------------------------------------------------
        # Aggregation
        # ------------------------------------------------------------------

        grouped_CTRL_data = (
            ctrl_data
            .select_dtypes(include="number")
            .groupby(ctrl_data["day"])
            .agg(["mean", "std", "count"])
        )

        grouped_HIPO_data = (
            hipo_data
            .select_dtypes(include="number")
            .groupby(hipo_data["day"])
            .agg(["mean", "std", "count"])
        )

        ci_CTRL_df = ctrl_data.groupby("day").agg({
            "acrophase_zt": compute_ci,
            "mesor": compute_ci,
            "amplitude": compute_ci,
            "period": compute_ci,
        })

        ci_HIPO_df = hipo_data.groupby("day").agg({
            "acrophase_zt": compute_ci,
            "mesor": compute_ci,
            "amplitude": compute_ci,
            "period": compute_ci,
        })

        # ------------------------------------------------------------------
        # Plotting
        # ------------------------------------------------------------------

        FIG_DIR = EXPORTS / "figures" / "groups" / parameter
        FIG_DIR.mkdir(parents=True, exist_ok=True)

        parameters = ["acrophase_zt", "mesor", "amplitude", "period"]
        xlabels = ["ZT or CT", "COUNTS", "COUNTS", "HOURS"]
        titles = ["ACROPHASE", "MESOR", "AMPLITUDE", "PERIOD"]

        # ---- CONTROL ----
        ctrl_fig, ctrl_ax = plt.subplots(1, 4, figsize=(12, 8))
        for i, (param, xlabel, title) in enumerate(zip(parameters, xlabels, titles)):
            ax = ctrl_ax[i]
            plot_with_ci(ax, grouped_CTRL_data, ci_CTRL_df, param, "#9e9e9e", "CONTROL", show_ci=False)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.invert_yaxis()

        ctrl_ax[0].set_ylabel("DAYS")
        ctrl_fig.legend(
            handles=[mpatches.Patch(color="#9e9e9e", label="CONTROL")],
            loc="lower center",
            ncol=1,
        )
        ctrl_fig.suptitle(f"CONTROL DATA — {parameter}", fontsize=16)
        ctrl_fig.savefig(FIG_DIR / f"{parameter}_control_data.png")
        plt.close(ctrl_fig)
        print("[INFO] Saved control figures for", parameter)

        # ---- HIPO ----
        hipo_fig, hipo_ax = plt.subplots(1, 4, figsize=(12, 8))
        for i, (param, xlabel, title) in enumerate(zip(parameters, xlabels, titles)):
            ax = hipo_ax[i]
            plot_with_ci(ax, grouped_HIPO_data, ci_HIPO_df, param, "#f7a3a3", "HIPO", show_ci=False)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.invert_yaxis()

        hipo_ax[0].set_ylabel("DAYS")
        hipo_fig.legend(
            handles=[Line2D([0], [0], color="#f7a3a3", lw=2, label="HIPO")],
            loc="lower center",
            ncol=1,
        )
        hipo_fig.suptitle(f"HIPO DATA — {parameter}", fontsize=16)
        hipo_fig.savefig(FIG_DIR / f"{parameter}_hipo_data.png")
        plt.close(hipo_fig)
        print("[INFO] Saved HIPO figures for", parameter)

        # ---- BOTH ----
        all_fig, all_ax = plt.subplots(1, 4, figsize=(12, 8))
        for i, (param, xlabel, title) in enumerate(zip(parameters, xlabels, titles)):
            ax = all_ax[i]
            plot_with_ci(ax, grouped_CTRL_data, ci_CTRL_df, param, "#9e9e9e", "CONTROL", show_ci=True)
            plot_with_ci(ax, grouped_HIPO_data, ci_HIPO_df, param, "#f7a3a3", "HIPO", show_ci=True)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.invert_yaxis()

        all_ax[0].set_ylabel("DAYS")
        all_fig.legend(
            handles=[
                Line2D([0], [0], color="#9e9e9e", lw=2, label="CONTROL"),
                Line2D([0], [0], color="#f7a3a3", lw=2, label="HIPO"),
            ],
            loc="lower center",
            ncol=2,
        )
        all_fig.suptitle(f"CONTROL AND HIPO DATA — {parameter}", fontsize=16)
        all_fig.savefig(FIG_DIR / f"{parameter}_both_groups_data.png")
        plt.close(all_fig)
        print("[INFO] Saved control vs hipo figure for", parameter)

else:
    print("[INFO] Group figures skipped as per configuration.")
