"""Download Oracle's Elixir match data from its public Google Drive folder.

    python -m pro fetch      the current season, plus any season from FIRST_YEAR that's missing

The folder is listed fresh each time, since a file's Drive id can change when it's re-uploaded.
A download only replaces the local file if it looks right: the columns pro.load needs, at least
as many games as before, and a last game that isn't older than before.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .load import COLS

FOLDER = "https://drive.google.com/drive/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH"  # linked from oracleselixir.com/tools/downloads
FIRST_YEAR = 2024
NAME = "{}_LoL_esports_match_data_from_OraclesElixir.csv"


def summary(path: Path) -> tuple[int, pd.Timestamp]:
    df = pd.read_csv(path, usecols=["gameid", "date"])
    return df.gameid.nunique(), pd.to_datetime(df.date).max()


def check(new: Path, old: Path):
    """Raise ValueError if the downloaded file `new` shouldn't replace `old`."""
    missing = set(COLS) - set(pd.read_csv(new, nrows=0).columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    if old.exists():
        (n_new, last_new), (n_old, last_old) = summary(new), summary(old)
        if n_new < n_old or last_new < last_old:
            raise ValueError(f"older than the local file ({n_new} games to {last_new} vs {n_old} to {last_old})")


def download(file_id: str, out: Path):
    """Plain public download; Google Drive refuses it for a while when a file is popular
    ("too many users have viewed or downloaded this file"). With GOOGLE_API_KEY set (a free
    Google Cloud API key with the Drive API enabled), fall back to the Drive API."""
    import gdown

    try:
        if gdown.download(id=file_id, output=str(out), quiet=True):
            return
        error = "download failed"
    except Exception as e:  # gdown raises on the quota page
        error = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(f"{error} (Google Drive quota; retry later, or set GOOGLE_API_KEY)")
    import shutil
    import urllib.request

    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={key}"
    with urllib.request.urlopen(url, timeout=300) as r, open(out, "wb") as f:
        shutil.copyfileobj(r, f)


def fetch(dest=Path("pro_play_data")) -> list[Path]:
    import gdown  # only needed here

    dest.mkdir(parents=True, exist_ok=True)
    listing = gdown.download_folder(FOLDER, skip_download=True, quiet=True) or []
    ids = {Path(f.path).name: f.id for f in listing}
    current = datetime.now(timezone.utc).year
    updated, failed = [], []
    for year in range(FIRST_YEAR, current + 1):
        name = NAME.format(year)
        path = dest / name
        if year < current and path.exists():
            continue  # finished seasons don't change
        if name not in ids:
            failed.append(f"{name}: not in the Drive folder")
            continue
        part = path.with_name(name + ".part")
        try:
            download(ids[name], part)
            check(part, path)
        except Exception as e:
            part.unlink(missing_ok=True)
            failed.append(f"{name}: {e}")
            continue
        os.replace(part, path)
        n, last = summary(path)
        print(f"{name}: {n:,} games, last on {last:%Y-%m-%d %H:%M}")
        updated.append(path)
    if failed:
        raise SystemExit("could not update:\n  " + "\n  ".join(failed))
    return updated
