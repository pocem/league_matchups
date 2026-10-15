"""python -m draft train   -> fit the solo queue draft model, models/draft_net.pt
python -m draft score   -> draft score for every pro game, data/pro/draft_scores.parquet
python -m draft patch-eval -> same-patch vs all-patch draft models on pro games (see patch_eval.py)
python -m draft freeze  -> champion strength table from recent patches, models/draft_strength.json"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np  # before torch: both ship libiomp5md.dll, and on Windows numpy's must load first
import pandas as pd
import torch

from . import data, model


def load_solo(path: Path):
    matches = pd.read_parquet(path / "matches.parquet").sort_values("game_creation", ignore_index=True)
    champions = pd.read_csv(path / "champions.csv")
    voc = data.vocab(champions)
    ch = torch.tensor(data.solo_drafts(matches, champions, voc), dtype=torch.long)
    y = torch.tensor(matches.blue_win.to_numpy(), dtype=torch.float32)
    return matches, voc, ch, y


def train(data_dir: Path, out_dir: Path):
    torch.set_num_threads(6)
    matches, voc, ch, y = load_solo(data_dir)
    n = len(y)
    a, b = int(n * 0.8), int(n * 0.9)  # time split: oldest 80% train, next 10% valid, newest 10% test
    tr, va, te = (ch[:a], y[:a]), (ch[a:b], y[a:b]), (ch[b:], y[b:])
    print(f"{n:,} solo queue matches; train {a:,}  valid {b - a:,}  test {n - b:,} "
          f"(test from {matches.game_creation.iloc[b]:%Y-%m-%d})")

    p = float(tr[1].mean())
    const = lambda yy: float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())
    results = []
    for cfg in model.grid(len(voc) + 1):
        m, epoch, val = model.fit(*tr, cfg, val=va)
        results.append({"cfg": cfg, "epoch": epoch, "valid": val, "test": model.evaluate(m, *te)})
        print(f"  valid {val:.5f}  epoch {epoch:2d}  {cfg['model']}  wd {cfg['weight_decay']}", flush=True)

    best = min((r for r in results if r["cfg"]["model"]["kind"] == "net"), key=lambda r: r["valid"])
    strength = results[0]
    summary = {
        "matches": n, "test_from": str(matches.game_creation.iloc[b]),
        "test_log_loss": {"always blue rate": const(te[1]), "champion strength only": strength["test"],
                          "draft net": best["test"]},
        "valid_log_loss": {"always blue rate": const(va[1]), "champion strength only": strength["valid"],
                           "draft net": best["valid"]},
        "best_config": best["cfg"], "best_epoch": best["epoch"],
    }
    print(json.dumps(summary["test_log_loss"], indent=2))

    # final model: best config refit on every match for the chosen number of epochs
    final, _, _ = model.fit(ch, y, best["cfg"], epochs=best["epoch"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": best["cfg"], "state_dict": final.state_dict(), "vocab": voc}, out_dir / "draft_net.pt")
    (out_dir / "draft_metadata.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"saved {out_dir / 'draft_net.pt'}")


def freeze(data_dir: Path, out_dir: Path, patches: list[str]):
    """Champion strength per role from the given patches only, saved as a plain JSON table.

    The strength model matched the draft network everywhere we tested, and as a table it can
    be used by the website's JavaScript directly: draft logit = sum over roles of
    value[blue champion][role] - value[red champion][role]."""
    torch.set_num_threads(6)
    matches, voc, ch, y = load_solo(data_dir)
    keep = torch.tensor(matches.patch.isin(patches).to_numpy())
    ch, y = ch[keep], y[keep]
    cut = int(len(y) * 0.9)
    cfg = {"model": {"kind": "strength", "n_champ": len(voc) + 1}, "lr": 3e-3, "weight_decay": 0.0,
           "batch_size": 512}
    _, epoch, holdout = model.fit(ch[:cut], y[:cut], cfg, val=(ch[cut:], y[cut:]))
    final, _, _ = model.fit(ch, y, cfg, epochs=max(epoch, 1))
    w = final.scorer.w.weight.detach().numpy()
    p = float(y[:cut].mean())
    out = {"patches": patches, "matches": int(len(y)), "epochs": epoch,
           "holdout_log_loss": holdout,
           "holdout_log_loss_always_blue_rate": float(-(y[cut:] * np.log(p) + (1 - y[cut:]) * np.log(1 - p)).mean()),
           "roles": data.PRO_ROLES,
           "values": {name: [round(float(v), 5) for v in w[i]] for name, i in voc.items()}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "draft_strength.json").write_text(json.dumps(out, indent=1))
    print(f"champion strength from {len(y):,} games on {patches}: holdout log loss {holdout:.4f} "
          f"(always blue rate {out['holdout_log_loss_always_blue_rate']:.4f}) -> {out_dir / 'draft_strength.json'}")


def score(model_dir: Path, pro_dir: Path):
    saved = torch.load(model_dir / "draft_net.pt", weights_only=False)
    m = model.DraftModel(**saved["cfg"]["model"])
    m.load_state_dict(saved["state_dict"])
    m.eval()
    games = pd.read_parquet(pro_dir / "games.parquet")
    ch = data.pro_drafts(games, saved["vocab"])
    with torch.no_grad():
        logit = m.draft_logit(torch.tensor(ch, dtype=torch.long)).numpy()
    out = pd.DataFrame({"gameid": games.gameid, "draft_logit": logit})
    out.to_parquet(pro_dir / "draft_scores.parquet", index=False)
    print(f"scored {len(out):,} pro drafts; unknown champion slots: {(ch == 0).sum()}")


def main():
    ap = argparse.ArgumentParser(prog="draft")
    ap.add_argument("cmd", choices=["train", "score", "patch-eval", "freeze"])
    ap.add_argument("--patches", nargs="+", default=["16.19", "16.20"], help="freeze: solo queue patches to use")
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    ap.add_argument("--min-games", type=int, default=20_000, help="patch-eval: solo games a patch needs")
    args = ap.parse_args()
    if args.cmd == "train":
        train(Path(args.data), Path(args.models))
    elif args.cmd == "score":
        score(Path(args.models), Path(args.data) / "pro")
    elif args.cmd == "freeze":
        freeze(Path(args.data), Path(args.models), args.patches)
    else:
        from . import patch_eval
        torch.set_num_threads(6)
        patch_eval.run(Path(args.data), Path(args.models), args.min_games)


if __name__ == "__main__":
    main()
