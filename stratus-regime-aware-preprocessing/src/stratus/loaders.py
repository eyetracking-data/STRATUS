from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "dataset",
    "source_file",
    "participant_id",
    "segment_id",
    "time_s",
    "x",
    "y",
]


def _first_n_csv_files(folder: str | Path, n: int = 5) -> list[Path]:
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return files[:n]


def _finite_xy(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.isfinite(df["x"].to_numpy(dtype=float))
        & np.isfinite(df["y"].to_numpy(dtype=float)),
        index=df.index,
    )


def load_dyslexia_file(path: str | Path) -> pd.DataFrame:
    """Load one ETDD70 combined raw file into the canonical schema."""
    path = Path(path)
    df = pd.read_csv(path, low_memory=False)

    needed = ["time", "gaze_x_left", "gaze_y_left", "gaze_x_right", "gaze_y_right"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["gaze_x_left", "gaze_y_left", "gaze_x_right", "gaze_y_right"]:
        df.loc[df[c] == 0, c] = np.nan

    x = df[["gaze_x_left", "gaze_x_right"]].mean(axis=1, skipna=True)
    y = df[["gaze_y_left", "gaze_y_right"]].mean(axis=1, skipna=True)

    time_s = (df["time"] - df["time"].iloc[0]) / 1_000_000.0

    if "subject_id" in df.columns:
        participant = df["subject_id"].astype("string").fillna(path.stem)
    else:
        participant = pd.Series([path.stem] * len(df), dtype="string")

    if "task" in df.columns and "stimfile" in df.columns:
        recording = df["task"].astype(str) + "__" + df["stimfile"].astype(str)
    elif "task" in df.columns:
        recording = df["task"].astype(str)
    else:
        recording = pd.Series(["recording"] * len(df))

    out = pd.DataFrame({
        "dataset": "ETDD70",
        "source_file": path.name,
        "participant_id": participant.astype(str),
        "segment_id": recording.astype(str),
        "time_s": time_s,
        "x": x,
        "y": y,
    })
    return out[CANONICAL_COLUMNS]


def load_autism_file(path: str | Path) -> pd.DataFrame:
    """Load one autism export without combining different participants.

    The public CSV exports contain multiple participants in a single file.
    Participant identity is therefore retained explicitly and included in the
    recording key used for segment extraction.
    """
    path = Path(path)
    df = pd.read_csv(path, na_values=["-"], low_memory=False)

    time_col = "RecordingTime [ms]"
    rx, ry = "Point of Regard Right X [px]", "Point of Regard Right Y [px]"
    lx, ly = "Point of Regard Left X [px]", "Point of Regard Left Y [px]"
    needed = [time_col, rx, ry, lx, ly, "Participant"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    for c in [time_col, rx, ry, lx, ly]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in [rx, ry, lx, ly]:
        df.loc[df[c] == 0, c] = np.nan

    x = df[[rx, lx]].mean(axis=1, skipna=True)
    y = df[[ry, ly]].mean(axis=1, skipna=True)
    participant = df["Participant"].astype("string").fillna("unknown").astype(str)

    if "Trial" in df.columns:
        trial = df["Trial"].astype("string").fillna("trial").astype(str)
    else:
        trial = pd.Series(["trial"] * len(df), index=df.index)
    if "Stimulus" in df.columns:
        stimulus = df["Stimulus"].astype("string").fillna("stimulus").astype(str)
    else:
        stimulus = pd.Series(["stimulus"] * len(df), index=df.index)

    recording = participant + "__" + trial + "__" + stimulus

    # Normalize time independently within each participant recording. This
    # prevents unrelated participants from being interpreted as one sequence.
    raw_time = df[time_col].astype(float)
    time_s = raw_time.groupby(recording, sort=False).transform(
        lambda s: (s - s.min()) / 1000.0
    )

    out = pd.DataFrame({
        "dataset": "Autism",
        "source_file": path.name,
        "participant_id": participant,
        "segment_id": recording,
        "time_s": time_s,
        "x": x,
        "y": y,
    })
    return out[CANONICAL_COLUMNS]


def load_dataset(folder: str | Path, dataset: str, n_files: int = 5) -> pd.DataFrame:
    files = _first_n_csv_files(folder, n=n_files)
    loader = load_dyslexia_file if dataset.lower() in {"dyslexia", "etdd70"} else load_autism_file
    parts = []
    for f in files:
        try:
            parts.append(loader(f))
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}")
    if not parts:
        raise RuntimeError(f"No files could be loaded for dataset={dataset}")
    return pd.concat(parts, ignore_index=True)


def _group_columns(df: pd.DataFrame) -> list[str]:
    cols = ["source_file"]
    if "participant_id" in df.columns:
        cols.append("participant_id")
    cols.append("segment_id")
    return cols


def estimate_sampling_rate_hz(df: pd.DataFrame) -> float:
    """Robust median sampling-rate estimate within participant recordings."""
    dts = []
    for _, g in df.groupby(_group_columns(df), sort=False, dropna=False):
        t = np.sort(g["time_s"].to_numpy(dtype=float))
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt):
            dts.extend(dt.tolist())
    if not dts:
        return np.nan
    return 1.0 / float(np.median(dts))


def estimate_segment_sampling_rate_hz(df: pd.DataFrame) -> float:
    t = np.sort(df["time_s"].to_numpy(dtype=float))
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(dt):
        return np.nan
    return 1.0 / float(np.median(dt))


def _first_exact_window(
    g: pd.DataFrame,
    segment_seconds: float,
    max_dt_factor: float,
) -> pd.DataFrame | None:
    """Return the first fully finite, temporally continuous window."""
    g = g.sort_values("time_s").drop_duplicates("time_s", keep="first").reset_index(drop=True)
    if len(g) < 2:
        return None

    t = g["time_s"].to_numpy(dtype=float)
    valid = _finite_xy(g).to_numpy(dtype=bool) & np.isfinite(t)
    positive_dt = np.diff(t)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0)]
    if not len(positive_dt):
        return None
    median_dt = float(np.median(positive_dt))

    # A run is broken by invalid coordinates, non-increasing timestamps, or a
    # timestamp gap substantially larger than the recording's normal cadence.
    run_start = None
    for i in range(len(g) + 1):
        if i == len(g):
            ok = False
        elif not valid[i]:
            ok = False
        elif i > 0 and run_start is not None:
            dt = t[i] - t[i - 1]
            ok = np.isfinite(dt) and dt > 0 and dt <= max_dt_factor * median_dt
        else:
            ok = True

        if ok and run_start is None:
            run_start = i
        elif not ok and run_start is not None:
            run_end = i
            run_t = t[run_start:run_end]
            if len(run_t) >= 2 and run_t[-1] - run_t[0] >= segment_seconds:
                # Earliest endpoint that reaches the requested duration.
                end_rel = int(np.searchsorted(run_t, run_t[0] + segment_seconds, side="left"))
                if end_rel < len(run_t):
                    return g.iloc[run_start:run_start + end_rel + 1].copy()
            run_start = i if i < len(g) and valid[i] else None
    return None


def extract_valid_segments(
    df: pd.DataFrame,
    segment_seconds: float = 8.0,
    max_segments_per_participant: int = 1,
    min_valid_fraction: float = 0.98,
    selection_seed: int = 42,
    max_dt_factor: float = 3.0,
) -> pd.DataFrame:
    """Extract exact-duration reference windows and sample them by participant.

    Candidate windows are constructed separately for every participant
    recording. At most `max_segments_per_participant` candidates are then
    selected with a fixed seed, preventing the first file or task from
    deterministically dominating the case study.
    """
    if "participant_id" not in df.columns:
        raise ValueError("participant_id is required for leakage-safe extraction")

    candidates: list[pd.DataFrame] = []
    group_cols = ["participant_id", "source_file", "segment_id"]
    for (participant, source_file, seg_id), g in df.groupby(group_cols, sort=False, dropna=False):
        chunk = _first_exact_window(g, segment_seconds=segment_seconds, max_dt_factor=max_dt_factor)
        if chunk is None:
            continue
        if _finite_xy(chunk).mean() < min_valid_fraction:
            continue
        chunk["candidate_key"] = f"{participant}__{source_file}__{seg_id}"
        candidates.append(chunk)

    if not candidates:
        raise RuntimeError("No valid reference segments found")

    candidate_df = pd.concat(candidates, ignore_index=True)
    rng = np.random.default_rng(selection_seed)
    selected = []
    out_id = 0
    for participant, pg in candidate_df.groupby("participant_id", sort=True, dropna=False):
        keys = np.array(sorted(pg["candidate_key"].unique()), dtype=object)
        take = min(max_segments_per_participant, len(keys))
        chosen = rng.choice(keys, size=take, replace=False)
        for key in sorted(chosen.tolist()):
            chunk = pg[pg["candidate_key"].eq(key)].copy()
            chunk["case_segment"] = f"{chunk['dataset'].iloc[0]}__{participant}__seg{out_id}"
            chunk["case_index"] = out_id
            selected.append(chunk.drop(columns=["candidate_key"]))
            out_id += 1

    return pd.concat(selected, ignore_index=True)


def infer_coordinate_bounds(
    df: pd.DataFrame,
    upper_quantile: float = 0.999,
    round_to: int = 100,
) -> tuple[float, float]:
    """Infer an empirical coordinate plausibility domain before corruption."""
    finite_x = df.loc[np.isfinite(df["x"]), "x"].to_numpy(dtype=float)
    finite_y = df.loc[np.isfinite(df["y"]), "y"].to_numpy(dtype=float)
    if len(finite_x) == 0 or len(finite_y) == 0:
        raise ValueError("Cannot infer coordinate bounds from empty data.")
    width = float(np.ceil(np.quantile(finite_x, upper_quantile) / round_to) * round_to)
    height = float(np.ceil(np.quantile(finite_y, upper_quantile) / round_to) * round_to)
    return max(width, float(round_to)), max(height, float(round_to))


def sampling_rate_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-file and participant-aware sampling-rate diagnostics."""
    rows = []
    for source_file, g in df.groupby("source_file", sort=False):
        rates = []
        recordings = 0
        participants = set()
        for _, seg in g.groupby(_group_columns(g), sort=False, dropna=False):
            rate = estimate_segment_sampling_rate_hz(seg)
            if np.isfinite(rate):
                rates.append(rate)
                recordings += 1
                participants.update(seg["participant_id"].astype(str).unique())
        if rates:
            rows.append({
                "source_file": source_file,
                "median_hz": float(np.median(rates)),
                "min_recording_hz": float(np.min(rates)),
                "max_recording_hz": float(np.max(rates)),
                "n_recordings": recordings,
                "n_participants": len(participants),
            })
    return pd.DataFrame(rows)
