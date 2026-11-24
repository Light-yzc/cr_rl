"""Offline native UnitData calibration at one reference level, not live level."""
import json
from pathlib import Path

import jpype
from crforge_env import VectorBattle,LadderConfig

ROOT=Path(__file__).resolve().parents[1]
names=['Skeleton','Bat','Goblin_Stab','SpearGoblin','Minion','Archer','Barbarian',
       'Knight','Musketeer','MiniPekka','Giant','Golem','Golemite','BabyDragon',
       'SkeletonWarrior','Balloon','Witch','Wizard','Prince','Pekka']
out=ROOT/'artifacts/army_elixir/fallback_calibration.json'
rows=[]
with VectorBattle(1,1,install=ROOT/'.cache/vendor/crforge/env/build/install/env',
        tables=Path.home()/'.crforge/tables/16.635.3',config=LadderConfig(decision_ticks=8),
        jvm_args=['-Xmx512m']) as env:
    records=jpype.JClass('org.crforge.core.battle.data.BattleRecords')(env._tables)
    scaling=jpype.JClass('org.crforge.core.pathfinding.combat.LevelScaling')
    packed=jpype.JClass('org.crforge.core.pathfinding.combat.PackedLevel')
    globals_=jpype.JClass('org.crforge.core.pathfinding.combat.ScalingGlobals').standard()
    amounts=jpype.JClass('org.crforge.core.battle.projectile.ProjectileAmounts')
    mode=jpype.JClass('org.crforge.core.pathfinding.combat.ScalingMode').CARD_DAMAGE
    for name in names:
        try: unit=records.unit(name)
        except Exception: continue
        rarity=unit.rarity();level=packed.fromLevel(11,rarity)
        hp=int(scaling.hitpoints(globals_,unit.hitpoints(),level,rarity,False,False))
        damage=int(scaling.scale(globals_,unit.damage(),level,mode,rarity))
        projectile=unit.projectile()
        if damage==0 and projectile is not None:
            damage=int(amounts.damage(globals_,projectile,packed.fromLevel(11,projectile.rarity())))
        rows.append(dict(name=name,base_hp=int(unit.hitpoints()),base_damage=int(unit.damage()),
            reference_hp=hp,reference_damage=damage,rarity=str(rarity.name()),
            projectile=str(projectile.name()) if projectile is not None else None,
            bucket='low' if hp<350 and damage<150 else 'high'))
report=dict(reference_level=11,H=350,D=150,rows=rows,
    rationale='Below 350HP AND 150 single-hit damage separates Skeleton/Bat/Goblins/Minion/Archer from medium bodies; not a DPS or optimal value claim',
    backend_commit='7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6',data_version='16.635.3')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
print(json.dumps(report))
