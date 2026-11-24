"""Unit tests for SchemaV4, Vocabulary, TileMap, DeployMask, and PlacementSearch."""

from __future__ import annotations

from pathlib import Path

import pytest

from cr_rl.client.deploy_mask import (
    DeployCatalog,
    MaskEntity,
    TileMap,
    build_legal_mask,
    landing_tiles,
)
from cr_rl.client.schema_v4 import SchemaV4, Vocabulary

TABLES_16_635_3 = Path(r"C:\Users\Administrator\.crforge\tables\16.635.3")


def _towers(catalog: DeployCatalog, *, enemy_left_alive: bool = True) -> list[MaskEntity]:
    king = catalog.units["KingTower"]
    princess = catalog.units["PrincessTower"]
    return [
        MaskEntity.from_unit(king, 0, 9000, 3000, True),
        MaskEntity.from_unit(princess, 0, 3500, 6500, True),
        MaskEntity.from_unit(princess, 0, 14500, 6500, True),
        MaskEntity.from_unit(king, 1, 9000, 29000, True),
        MaskEntity.from_unit(princess, 1, 3500, 25500, enemy_left_alive),
        MaskEntity.from_unit(princess, 1, 14500, 25500, True),
    ]


def test_schema_v4_coordinates_and_actions() -> None:
    assert SchemaV4.SEAT_INTS == 7024
    assert SchemaV4.ACTIONS == 2307
    assert SchemaV4.HEADER_INTS == 224
    assert SchemaV4.ENTITY_INTS == 34

    assert SchemaV4.to_seat_x(0, 3500) == 14500
    assert SchemaV4.to_seat_y(0, 6500) == 6500
    assert SchemaV4.to_seat_x(1, 3500) == 3500
    assert SchemaV4.to_seat_y(1, 6500) == 25500

    assert SchemaV4.game_x(0, 0) == 17500
    assert SchemaV4.game_y(0, 0) == 500
    assert SchemaV4.game_x(1, 0) == 500
    assert SchemaV4.game_y(1, 0) == 31500

    a = SchemaV4.play(2, 7, 14)
    assert SchemaV4.decode_action(a) == {
        "kind": "play",
        "slot": 2,
        "col": 7,
        "row": 14,
        "tile": 14 * 18 + 7,
    }
    assert SchemaV4.decode_action(SchemaV4.WAIT) == {"kind": "wait"}
    assert SchemaV4.decode_action(SchemaV4.ability(1)) == {"kind": "ability", "slot": 1}


@pytest.mark.skipif(not TABLES_16_635_3.exists(), reason="16.635.3 tables not present locally")
def test_vocabulary_hashes_and_rarities() -> None:
    vocab = Vocabulary.from_tables(TABLES_16_635_3)
    assert len(vocab.names) == 1466
    assert vocab.sha256_prefix == "a1edba1fe5a0216d"
    assert len(vocab.buff_names) == 265
    assert vocab.buff_sha256_prefix == "290034fb8f6fbd11"
    assert vocab.relative_levels["Common"] == 0
    assert vocab.relative_levels["Rare"] == 2
    assert vocab.relative_levels["Epic"] == 5
    assert vocab.relative_levels["Legendary"] == 8
    assert vocab.relative_levels["Champion"] == 10
    assert vocab.row_id("Knight") >= 0


@pytest.mark.skipif(not TABLES_16_635_3.exists(), reason="16.635.3 tables not present locally")
@pytest.mark.parametrize(
    ("card_name", "expected_towers_up", "expected_pocket_open"),
    [
        ("Knight", 222, 259),
        ("Cannon", 116, 129),
        ("Xbow", 116, 129),
        ("Miner", 448, 457),
        ("GoblinBarrel", 544, 544),
        ("Log", 270, 324),
        ("BarbLog", 270, 324),
        ("Fireball", 576, 576),
        ("RoyalDelivery", 270, 308),
        ("IceWizard", 222, 259),
        ("RoyalRecruits", 52, 52),
        ("RoyalHogs", 168, 198),
        ("GoblinDrill", 334, 350),
    ],
)
def test_landing_tiles_matches_crforge_oracle(
    card_name: str,
    expected_towers_up: int,
    expected_pocket_open: int,
) -> None:
    catalog = DeployCatalog.from_tables(TABLES_16_635_3)
    tmap = TileMap.standard_1v1()
    card = catalog.cards[card_name]

    lands_up = landing_tiles(tmap, card, 0, _towers(catalog, enemy_left_alive=True))
    lands_pocket = landing_tiles(tmap, card, 0, _towers(catalog, enemy_left_alive=False))
    lands_seat1 = landing_tiles(tmap, card, 1, _towers(catalog, enemy_left_alive=True))

    assert sum(lands_up) == expected_towers_up
    assert sum(lands_pocket) == expected_pocket_open
    assert sum(lands_seat1) == expected_towers_up


@pytest.mark.skipif(not TABLES_16_635_3.exists(), reason="16.635.3 tables not present locally")
def test_legal_mask_gating_mirror_and_seat_symmetry() -> None:
    catalog = DeployCatalog.from_tables(TABLES_16_635_3)
    tmap = TileMap.standard_1v1()
    towers = _towers(catalog, enemy_left_alive=True)

    # Mirror with no last played card -> 0 legal tiles even with full elixir
    mask = build_legal_mask(
        catalog,
        tmap,
        own_seat=0,
        entities=towers,
        hand_cards=["Knight", "Cannon", "Mirror", "Fireball"],
        hand_costs=[3, 3, 1, 4],
        hand_pending=[False, False, True, False],
        available_elixir_whole=4,
        last_played_card=None,
        ability_ready=(True, False),
    )
    assert mask[SchemaV4.WAIT] is True
    assert sum(mask[1 : 1 + 576]) == 222  # Knight
    assert sum(mask[1 + 576 : 1 + 2 * 576]) == 116  # Cannon
    assert sum(mask[1 + 2 * 576 : 1 + 3 * 576]) == 0  # Pending Mirror w/o last card
    assert sum(mask[1 + 3 * 576 : 1 + 4 * 576]) == 576  # Fireball
    assert mask[SchemaV4.ability(0)] is True
    assert mask[SchemaV4.ability(1)] is False

    # Mirror repeating Cannon (3+1 = 4 elixir): needs 4 elixir, blocked at 3
    mask_mirror_3 = build_legal_mask(
        catalog,
        tmap,
        own_seat=0,
        entities=towers,
        hand_cards=["Mirror", None, None, None],
        hand_costs=[0, 0, 0, 0],
        hand_pending=[False, False, False, False],
        available_elixir_whole=3,
        last_played_card="Cannon",
    )
    assert sum(mask_mirror_3[1 : 1 + 576]) == 0

    mask_mirror_4 = build_legal_mask(
        catalog,
        tmap,
        own_seat=0,
        entities=towers,
        hand_cards=["Mirror", None, None, None],
        hand_costs=[0, 0, 0, 0],
        hand_pending=[False, False, False, False],
        available_elixir_whole=4,
        last_played_card="Cannon",
    )
    assert sum(mask_mirror_4[1 : 1 + 576]) == 116

    # Seat 0 and Seat 1 place PrincessTower/own back line identically in seat frame
    m0 = build_legal_mask(
        catalog, tmap, own_seat=0, entities=towers,
        hand_cards=["Knight", None, None, None], hand_costs=[3, 0, 0, 0],
        hand_pending=[False] * 4, available_elixir_whole=5,
    )
    m1 = build_legal_mask(
        catalog, tmap, own_seat=1, entities=towers,
        hand_cards=["Knight", None, None, None], hand_costs=[3, 0, 0, 0],
        hand_pending=[False] * 4, available_elixir_whole=5,
    )
    # Row 0 (own back line behind King) has cols 6..11 open in both seat 0 and seat 1
    assert sum(m0[1 : 1 + 18]) == 6
    assert sum(m1[1 : 1 + 18]) == 6
    assert m0[SchemaV4.play(0, 6, 0)] is True
    assert m1[SchemaV4.play(0, 6, 0)] is True
    assert m0[SchemaV4.play(0, 3, 20)] is False
    assert m1[SchemaV4.play(0, 3, 20)] is False
