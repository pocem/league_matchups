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


DRIVE = "https://www.googleapis.com/drive/v3/files"
SCOPES = ["https://www.googleapis.com/auth/drive"]  # copying a file you don't own needs the full scope


def http(url, *, data=None, token=None, method=None, out: Path | None = None):
    """A Drive API request with urllib; returns parsed JSON, or streams the body into `out`.
    Drive's JSON error message (e.g. downloadQuotaExceeded) is turned into a readable ValueError."""
    import json
    import shutil
    import urllib.error
    import urllib.request

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if isinstance(data, dict):
        data, headers["Content-Type"] = json.dumps(data).encode(), "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            if out is not None:
                with open(out, "wb") as f:
                    shutil.copyfileobj(r, f)
                return None
            body = r.read()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP {e.code}: {google_error(e)}") from None


def google_error(e) -> str:
    """Google's explanation from an error response body."""
    import json

    try:
        body = json.loads(e.read())
    except Exception:
        return str(e.reason)
    err = body.get("error")
    if isinstance(err, str):  # the token endpoint: {"error": "invalid_grant", "error_description": ...}
        return f"{err}: {body.get('error_description', '')}".rstrip(": ")
    reasons = ", ".join(x.get("reason", "") for x in (err or {}).get("errors", []))
    return f"{(err or {}).get('message', e.reason)} ({reasons})"


def oauth_token() -> str | None:
    """An access token for your Google account, from GOOGLE_CLIENT_ID / _SECRET / _REFRESH_TOKEN
    (python -m pro auth creates them). None when they aren't set."""
    import urllib.parse

    env = {k: os.environ.get(f"GOOGLE_{k}") for k in ("CLIENT_ID", "CLIENT_SECRET", "REFRESH_TOKEN")}
    if not all(env.values()):
        return None
    form = urllib.parse.urlencode({"client_id": env["CLIENT_ID"], "client_secret": env["CLIENT_SECRET"],
                                   "refresh_token": env["REFRESH_TOKEN"], "grant_type": "refresh_token"})
    return http("https://oauth2.googleapis.com/token", data=form.encode())["access_token"]


def download(file_id: str, out: Path):
    """Try three routes, as Google Drive often blocks downloads of popular files for hours:
    1. the plain public download,
    2. the Drive API with GOOGLE_API_KEY (blocked by the same quota once it's hit),
    3. signed in as you: copy the file into your own Drive, download the copy, delete it.
       A copy is a new file, so the original's download quota doesn't apply."""
    import gdown

    errors = []
    try:
        if gdown.download(id=file_id, output=str(out), quiet=True):
            return
        errors.append("plain download failed")
    except Exception as e:  # gdown raises on the quota page
        text = str(e)
        errors.append("plain download blocked (too many recent downloads)" if "Too many users" in text
                      else (text.strip().splitlines() or [type(e).__name__])[0])

    if key := os.environ.get("GOOGLE_API_KEY"):
        try:
            return http(f"{DRIVE}/{file_id}?alt=media&key={key}", out=out)
        except ValueError as e:
            errors.append(f"API key: {e}")

    try:
        token = oauth_token()
    except ValueError as e:
        errors.append(f"signing in: {e}")
        token = None
    if token:
        try:
            copy = http(f"{DRIVE}/{file_id}/copy?fields=id", token=token,
                        data={"name": f"{out.name} (temporary copy for league_matchups)"})["id"]
            try:
                return http(f"{DRIVE}/{copy}?alt=media", token=token, out=out)
            finally:
                http(f"{DRIVE}/{copy}", token=token, method="DELETE")
        except ValueError as e:
            errors.append(f"copy in your Drive: {e}")
    elif not any(e.startswith("signing in") for e in errors):
        errors.append("no Google sign-in set up for the copy route (python -m pro auth)")
    raise ValueError("; ".join(errors))


def authorize(client_secrets: Path):
    """One-time: sign in with your Google account and print the three values for GitHub secrets.
    client_secrets is the OAuth client JSON (type "Desktop app") from Google Cloud Console."""
    import json

    from google_auth_oauthlib.flow import InstalledAppFlow  # pip install google-auth-oauthlib

    creds = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES).run_local_server(
        port=0, access_type="offline", prompt="consent")
    client = json.loads(client_secrets.read_text())["installed"]
    print("\nAdd these as GitHub repo secrets (Settings -> Secrets and variables -> Actions):\n")
    print(f"GOOGLE_CLIENT_ID      {client['client_id']}")
    print(f"GOOGLE_CLIENT_SECRET  {client['client_secret']}")
    print(f"GOOGLE_REFRESH_TOKEN  {creds.refresh_token}")
    print("\nThen delete the client JSON file; never commit it or these values.")


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
