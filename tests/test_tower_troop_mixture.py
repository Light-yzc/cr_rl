from __future__ import annotations

import json
from pathlib import Path
import yaml

from cr_rl.training.colab_pipeline import ppo_command
from cr_rl.training.state_selfplay import _parse_tower_troop_mixture


def test_parse_tower_troop_mixture_and_colab_pipeline_forwarding(tmp_path):
    assert _parse_tower_troop_mixture(None) is None
    assert _parse_tower_troop_mixture("   ") is None

    names, probs = _parse_tower_troop_mixture(
        "King_PrincessTowers:0.55,King_KnifeTowers:0.15,King_CannonTowers:0.15,King_ChefTowers:0.15"
    )
    assert names == [
        "King_PrincessTowers",
        "King_KnifeTowers",
        "King_CannonTowers",
        "King_ChefTowers",
    ]
    assert abs(float(probs.sum()) - 1.0) < 1e-9
    assert abs(float(probs[0]) - 0.55) < 1e-9

    prep = tmp_path / "prepared"
    prep.mkdir(parents=True)
    (prep / "source_metadata.json").write_text(json.dumps({"training_args": {}}))
    cfg = {
        "run_name": "test_run",
        "ppo_only": True,
        "backup": False,
        "wandb": False,
        "tower_troop_mixture": "King_PrincessTowers:0.55,King_KnifeTowers:0.15,King_CannonTowers:0.15,King_ChefTowers:0.15",
    }
    cmd = ppo_command(tmp_path, cfg)
    assert "--tower-troop-mixture" in cmd
    idx = cmd.index("--tower-troop-mixture")
    assert "King_KnifeTowers" in cmd[idx + 1]


def test_full_meta_expansion_achieves_100pct_166353_base_and_evo_hero_coverage():
    tdir = Path.home() / ".crforge/tables/16.635.3"
    if not tdir.is_dir():
        return
    from cr_rl.client.schema_v4 import SPELL_TABLES, load_table_rows
    from cr_rl.training.deck_augmentation import augment_forms

    tables_rows = {}
    for tbl in SPELL_TABLES:
        if (tdir / f"{tbl}.json").is_file():
            tables_rows.update(load_table_rows(tdir, tbl))

    base = yaml.safe_load(Path("configs/decks_all51.yaml").read_text(encoding="utf-8"))
    extra = yaml.safe_load(Path("configs/decks_expansion_full_meta.yaml").read_text(encoding="utf-8"))
    aug_base = augment_forms(base, tables_rows)
    aug_extra = augment_forms(extra, tables_rows)

    used_base = set()
    used_ev = set()
    used_hero = set()
    for pool in (aug_base, aug_extra):
        decks = pool.get("decks", {})
        slots = pool.get("slots", {})
        for name, cards in decks.items():
            used_base.update(cards)
            for c, f in zip(cards, slots.get(name, [0] * 8)):
                if f & 1:
                    used_ev.add(c)
                if f & 2:
                    used_hero.add(c)

    all_base = set()
    all_ev = set()
    all_hero = set()
    for tn in ("spells_characters", "spells_buildings", "spells_other"):
        for nm, r in load_table_rows(tdir, tn).items():
            if r.get("NotInUse") or r.get("NotVisible"):
                continue
            cost = r.get("ManaCost")
            if isinstance(cost, (int, float)) and cost > 0:
                all_base.add(nm)
                for lnk in (r.get("EvolvedSpells") or []):
                    row = tables_rows.get(lnk, {})
                    if not row or row.get("NotInUse") or row.get("NotVisible"):
                        continue
                    if row.get("CardForm") == "HeroForm":
                        all_hero.add(nm)
                    elif lnk.endswith("_EV1"):
                        all_ev.add(nm)

    assert all_base - used_base == set(), f"Missing base cards: {sorted(all_base - used_base)}"
    assert all_ev - used_ev == set(), f"Missing EV1 cards: {sorted(all_ev - used_ev)}"
    assert all_hero - used_hero == set(), f"Missing Hero cards: {sorted(all_hero - used_hero)}"
