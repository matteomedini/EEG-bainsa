"""
BCI Competition IV 2a loader (GDF) using MNE.

Run:
  python scripts/load_bci_iv_2a.py --data_dir data --subject 1 --session T
"""

import os
import argparse
import numpy as np
import mne

EVENT_DICT = {"769": 769, "770": 770, "771": 771, "772": 772}
LABEL_MAP = {769: 0, 770: 1, 771: 2, 772: 3}

def load_subject(data_dir: str, subject: int, session: str = "T",
                 tmin: float = 0.0, tmax: float = 4.0,
                 l_freq: float = 8.0, h_freq: float = 30.0):

    if session not in {"T", "E"}:
        raise ValueError("session must be 'T' or 'E'")

    fname = f"A0{subject}{session}.gdf"
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Missing file: {fpath}\n"
            f"Put A01T.gdf etc in: {os.path.abspath(data_dir)}"
        )

    raw = mne.io.read_raw_gdf(fpath, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True)

    if l_freq is not None and h_freq is not None:
        raw.filter(l_freq, h_freq, fir_design="firwin", verbose="ERROR")

    events, _ = mne.events_from_annotations(raw, verbose="ERROR")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=EVENT_DICT,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )

    X = epochs.get_data()
    y_raw = epochs.events[:, -1]
    y = np.array([LABEL_MAP[v] for v in y_raw], dtype=np.int64)

    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--session", type=str, default="T")
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=4.0)
    ap.add_argument("--l_freq", type=float, default=8.0)
    ap.add_argument("--h_freq", type=float, default=30.0)
    args = ap.parse_args()

    X, y = load_subject(
        args.data_dir, args.subject, args.session,
        tmin=args.tmin, tmax=args.tmax,
        l_freq=args.l_freq, h_freq=args.h_freq
    )

    print("Loaded:", f"A0{args.subject}{args.session}.gdf")
    print("X shape:", X.shape)
    print("y classes:", sorted(set(y.tolist())))
    print("Trials per class:", {c: int((y == c).sum()) for c in sorted(set(y.tolist()))})


if __name__ == "__main__":
    main()