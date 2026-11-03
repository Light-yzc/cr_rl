"""Reject incompatible deck forms using real simulator resets and legal actions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from cr_rl.envs.crforge import CrForgeVector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    league = yaml.safe_load(args.candidates.read_text())
    accepted, rejected = {}, {}
    rng = np.random.default_rng(20261103)
    with CrForgeVector(1, 1) as env:
        for name, deck in league["decks"].items():
            try:
                forms = league["slots"][name]
                env.reset([20261103], deck, deck, forms, forms)
                for _ in range(120):
                    if env.dones[0]:
                        break
                    for seat in range(2):
                        legal = np.flatnonzero(env.masks[0, seat])
                        plays = legal[legal != 0]
                        env.actions[0, seat] = rng.choice(plays) if len(plays) else 0
                    env.step()
                info = env.vector.info(0)
                if info["fault"]:
                    raise ValueError(f"simulator fault {info}")
                accepted[name] = deck
            except Exception as error:
                rejected[name] = str(error)
    league["decks"] = accepted
    for key in ("slots", "provenance"):
        league[key] = {name: league[key][name] for name in accepted}
    league["compatibility"] = (
        "reset and 120 legal-action decisions passed; not exhaustive mechanics QA"
    )
    args.out.write_text(yaml.safe_dump(league, sort_keys=False))
    # Split by actual cards AND forms, so duplicate source names cannot leak a
    # held-out combination back into the PPO league under another identifier.
    heldout = set()
    for name, deck in accepted.items():
        key = json.dumps(sorted(zip(deck, league["slots"][name], strict=True)))
        if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 5 == 0:
            heldout.add(name)
    heldout_cards = {tuple(sorted(accepted[name])) for name in heldout}
    heldout = {name for name, deck in accepted.items() if tuple(sorted(deck)) in heldout_cards}
    for split, selected in (("eval", heldout), ("train", set(accepted) - heldout)):
        subset = {**league, "split": split}
        for key in ("decks", "slots", "provenance"):
            subset[key] = {name: value for name, value in league[key].items() if name in selected}
        args.out.with_name(args.out.stem + f"_{split}.yaml").write_text(
            yaml.safe_dump(subset, sort_keys=False)
        )
    args.out.with_suffix(".audit.json").write_text(
        json.dumps({"accepted": list(accepted), "rejected": rejected}, indent=2)
    )
    if len(accepted) < 8:
        raise RuntimeError(f"Too few valid league decks: {len(accepted)}")
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected)}), flush=True)


if __name__ == "__main__":
    main()
