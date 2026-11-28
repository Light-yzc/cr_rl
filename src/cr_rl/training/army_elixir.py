"""Fixed cast budgets. Native provenance is reward-only; no policy features change."""
from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
import yaml


@dataclass
class CastBudget:
    side: int
    paid: float
    factor: float
    # A slot is reserved once, including future members; never recycled after death.
    slots: dict
    members: dict = field(default_factory=dict)


class ArmyElixirLedger:
    def __init__(self):
        self.reset(0)

    def reset(self, episode):
        self.episode = episode
        self.groups = {}
        self.entity_to_slot = {}
        self.retired_ids = set()
        self.budget_violation_count = 0

    def add_cast(self, key, side, paid, factor, slots, *, confirmed):
        if not confirmed: return False
        if key in self.groups: raise ValueError('Duplicate cast identity')
        shares = dict(slots)
        if (side not in (0,1) or not math.isfinite(paid) or paid<0
                or not math.isfinite(factor) or factor<0
                or not all(math.isfinite(v) and v>=0 for v in shares.values())
                or sum(shares.values())>1+1e-12):
            self.budget_violation_count += 1
            raise ValueError('Invalid or excessive fixed cast budget')
        self.groups[key] = CastBudget(side,paid,factor,shares)
        return True

    def attach(self, key, slot, entity_id):
        group = self.groups[key]
        if (slot not in group.slots or slot in group.members
                or entity_id in self.entity_to_slot or entity_id in self.retired_ids):
            self.budget_violation_count += 1
            raise ValueError('Reused entity ID or budget slot')
        group.members[slot] = int(entity_id)
        self.entity_to_slot[int(entity_id)] = (key,slot)

    def values(self, entities):
        """entities: world-id -> (side,hp,max_hp); missing/dead gives zero."""
        totals=np.zeros(2,dtype=np.float64)
        mapped=0
        for identity,(side,hp,maximum) in entities.items():
            link=self.entity_to_slot.get(identity)
            if (link is None or not math.isfinite(hp) or not math.isfinite(maximum)
                    or hp<=0 or maximum<=0): continue
            key,slot=link;group=self.groups[key]
            if side != group.side: continue  # e.g. conversion is unsupported
            health=.3+.7*min(1.,max(0.,hp/maximum))
            totals[side] += group.paid*group.factor*group.slots[slot]*health
            mapped+=1
        alive=sum(hp>0 for _,hp,maximum in entities.values())
        balance=float(totals[0]-totals[1])
        return np.array([balance,-balance]),dict(army_value_own=float(totals[0]),
            army_value_enemy=float(totals[1]),mapped_member_count=mapped,
            unknown_lineage_count=alive-mapped,mapped_member_rate=mapped/alive if alive else 1.,
            budget_violation_count=self.budget_violation_count)


def load_values(path):
    config=yaml.safe_load(Path(path).read_text(encoding='utf8'))
    if config.get('version') not in ('army-elixir-direct-v1','army-elixir-stat-v1','army-elixir-conserved-v2'):
        raise ValueError('Unsupported cost table version')
    if config.get('version')=='army-elixir-stat-v1':
        base=load_values(Path(path).parent/config['base_exact_config'])
        config={**base,**config}
    if config.get('version')=='army-elixir-conserved-v2':
        config.setdefault('cards', {})
    if config.get('unknown') not in ('zero_and_report','stat_two_bucket'):
        raise ValueError('Unknown lineage policy')
    if config.get('unknown')=='stat_two_bucket':
        fallback=config['fallback']
        if (fallback['reference_level']<1 or not all(np.isfinite(fallback[k]) and fallback[k]>0
                for k in ('hp_threshold','damage_threshold','low','high','side_cap'))
                or fallback['low']>fallback['high']):
            raise ValueError('Invalid fallback calibration')
    if not all(np.isfinite(v) and v>=0 for v in config['category_factors'].values()):
        raise ValueError('Invalid category factors')
    return config


def fallback_values(entities, exact_ids, stats_by_id, settings):
    """No reallocation: fixed two-bucket estimate, separate cap for each side."""
    totals=np.zeros(2,dtype=np.float64)
    counts=dict(fallback_low_count=0,fallback_high_count=0,still_unknown_count=0)
    for identity,(side,hp,maximum) in entities.items():
        if identity in exact_ids or hp<=0: continue
        stats=stats_by_id.get(identity)
        if (side not in (0,1) or stats is None or maximum<=0
                or not math.isfinite(hp) or not math.isfinite(maximum)):
            counts['still_unknown_count']+=1;continue
        base_hp,damage=stats
        if not math.isfinite(base_hp) or not math.isfinite(damage) or base_hp<=0 or damage<0:
            counts['still_unknown_count']+=1;continue
        low=base_hp<settings['hp_threshold'] and damage<settings['damage_threshold']
        amount=settings['low'] if low else settings['high']
        totals[side]+=amount*(.3+.7*min(1.,max(0.,hp/maximum)))
        counts['fallback_low_count' if low else 'fallback_high_count']+=1
    counts['fallback_cap_hit_count']=int((totals>=settings['side_cap']).sum())
    totals=np.minimum(totals,settings['side_cap'])
    counts.update(fallback_value_own=float(totals[0]),fallback_value_enemy=float(totals[1]))
    return totals,counts


class NativeArmyElixir:
    """Track new successful Play records; cache unit references outside policy tensors.

    Special summoning has no exact provenance. Optional stat estimates never
    infer parentage. No per-step file reads or hashes.
    """
    def __init__(self, env, config):
        import jpype
        self.env=env
        self.config=config
        self.character_class=jpype.JClass('org.crforge.core.battle.unit.CharacterEntity')
        self._stat_cache={}
        self._exact_card_cache={}
        self._conserved_card_cache={}
        if config.get('unknown')=='stat_two_bucket':
            self._scaling=jpype.JClass('org.crforge.core.pathfinding.combat.LevelScaling')
            self._packed=jpype.JClass('org.crforge.core.pathfinding.combat.PackedLevel')
            self._globals=jpype.JClass('org.crforge.core.pathfinding.combat.ScalingGlobals').standard()
            self._amounts=jpype.JClass('org.crforge.core.battle.projectile.ProjectileAmounts')
            self._damage_mode=jpype.JClass('org.crforge.core.pathfinding.combat.ScalingMode').CARD_DAMAGE
        metadata=env.metadata()
        if (metadata.get('commit')!=config['backend_commit']
                or env.vector.data_version!=config['data_version']):
            raise ValueError('Army cost configuration/backend provenance mismatch')
        self.lanes=[None]*env.count
        self.episodes=[0]*env.count

    def _standard(self,index):
        vector=self.env.vector
        if hasattr(vector,'lanes'): return vector.lanes[index]._env.env(0).battle()
        return vector._env.env(index).battle()

    def reset_lane(self,index):
        self.episodes[index]+=1
        self.lanes[index]=dict(ledger=ArmyElixirLedger(),cursor=0,refs={},
                               unknown_cast_count=0)
        self.lanes[index]['ledger'].reset(self.episodes[index])

    def _safe_card(self,std,name):
        if name in self._exact_card_cache: return self._exact_card_cache[name]
        category=None
        try:
            card=std.getWorld().getRecords().card(name)
            units=[]
            if card.namesCharacter() and card.unit() is not None: units.append(card.unit())
            if card.secondary() is not None: units.append(card.secondary())
            units.extend(item.unit() for item in card.listed())
            safe=bool(units) and not card.casts() and not card.onExecuteAction()
            for unit in units:
                # Conservative rejection of every spawn/action/transform route.
                for method in ('deathSpawnCharacter','deathSpawnCharacter2','deathSpawnProjectile',
                        'spawnCharacter','spawnCharacter2','spawnCharacter3','spawnPathfindMorph',
                        'onStartingAction','onDeathAction','onKilledAction','onKilledDoneAction',
                        'onStartingAttackAction','onAttackAction','onStartChargingAction',
                        'shieldLostAction','onDamageTakenAction','onAttackSelfAction','onHitTargetAction','ability'):
                    if getattr(unit,method)(): safe=False
                if unit.champion() or unit.spawnAttach() or not unit.unmodelledDeathColumns().isEmpty(): safe=False
                for projectile in (unit.projectile(), unit.customFirstProjectile()):
                    if projectile is not None and (projectile.spawnCharacter() or projectile.deflectedCharacterSpawn()):
                        safe=False
            if safe:
                buildings=[bool(unit.building()) for unit in units]
                if all(buildings): category='building'
                elif not any(buildings): category='troop'
        except Exception:
            category=None
        self._exact_card_cache[name]=category
        return category

    def _unit_stats(self,data):
        name=str(data.name())
        if name in self._stat_cache: return self._stat_cache[name]
        value=None
        try:
            if not data.building() and not data.king() and not data.summonerTower() and data.rarity() is not None:
                rarity=data.rarity();level=self.config['fallback']['reference_level']
                packed=self._packed.fromLevel(level,rarity)
                hp=int(self._scaling.hitpoints(self._globals,data.hitpoints(),packed,rarity,False,False))
                damage=int(self._scaling.scale(self._globals,data.damage(),packed,self._damage_mode,rarity))
                projectile=data.projectile()
                if damage==0 and projectile is not None:
                    damage=int(self._amounts.damage(self._globals,projectile,
                        self._packed.fromLevel(level,projectile.rarity())))
                # Unknown special/action-only damage is not guessed as zero attack.
                if hp>0 and damage>0: value=(hp,damage)
        except Exception:
            value=None
        self._stat_cache[name]=value
        return value

    def _classify_conserved_card(self, name, field_code, units):
        key = (name, int(field_code))
        if key in self._conserved_card_cache:
            return self._conserved_card_cache[key]
        factors = self.config['category_factors']
        category = self.config.get('cards', {}).get(name)
        if category is None and units:
            try:
                datas = [u.getData() for u in units]
                buildings = [bool(d.building()) for d in datas]
                if all(buildings):
                    is_spawner = any(
                        bool(getattr(d, m)())
                        for d in datas
                        for m in ('spawnCharacter', 'spawnCharacter2', 'spawnCharacter3',
                                  'deathSpawnCharacter', 'deathSpawnCharacter2')
                    )
                    if is_spawner and 'spawner' in factors:
                        category = 'spawner'
                    elif name == 'Elixir Collector' and 'collector' in factors:
                        category = 'collector'
                    else:
                        category = 'building'
                else:
                    is_split = any(
                        bool(getattr(d, m)())
                        for d in datas
                        for m in ('deathSpawnCharacter', 'deathSpawnCharacter2', 'spawnPathfindMorph')
                    )
                    category = 'split_troop' if (is_split and 'split_troop' in factors) else 'troop'
            except Exception:
                category = 'troop'
        self._conserved_card_cache[key] = category
        return category

    def poll(self,indices=None):
        phi=np.zeros((self.env.count,2),dtype=np.float64)
        stats=[]
        allow_variants = bool(self.config.get('allow_variants', False))
        universal_direct = bool(self.config.get('universal_direct_units', False))
        for index in (range(self.env.count) if indices is None else indices):
            if self.lanes[index] is None: self.reset_lane(index)
            lane=self.lanes[index];std=self._standard(index);plays=std.getPlays()
            if len(plays)<lane['cursor']: raise ValueError('Native battle reset without ledger reset')
            for j in range(lane['cursor'],len(plays)):
                play=plays[j];result=play.result()
                if result is None or not result.placed(): continue
                item=play.evolution()
                if play.mirror() is not None:
                    lane['unknown_cast_count']+=1;continue
                if not allow_variants:
                    if play.variant() is not None or item is None or int(item.field())!=0:
                        lane['unknown_cast_count']+=1;continue
                    name=str(item.spell().name())
                    paid_cost=int(item.cost())
                    field_code=0
                else:
                    if item is not None:
                        name=str(item.spell().name())
                        paid_cost=max(0, int(item.cost()))
                        field_code=int(item.field())
                    elif play.variant() is not None:
                        var=play.variant()
                        spell_obj=var.spell()
                        name=str(spell_obj.name()) if hasattr(spell_obj, 'name') else str(spell_obj)
                        paid_cost=max(0, int(var.cost()))
                        field_code=2
                    else:
                        lane['unknown_cast_count']+=1;continue
                units=list(play.units())
                category=self.config['cards'].get(name)
                if category is None and universal_direct and units:
                    category=self._classify_conserved_card(name, field_code, units)
                elif category is None and self.config.get('auto_exact_safe_direct'):
                    category=self._safe_card(std,name)
                if category is None or not units:
                    lane['unknown_cast_count']+=1;continue
                # Exact initial placement roster, including members that already died.
                ids=[int(unit.getId()) for unit in units]
                if len(set(ids))!=len(ids): raise ValueError('Duplicate native spawn ID')
                key=(self.episodes[index],int(play.side()),j)
                ledger=lane['ledger']
                ledger.add_cast(key,int(play.side()),paid_cost,
                    self.config['category_factors'][category],
                    {k:1/len(units) for k in range(len(units))},confirmed=True)
                for slot,unit in enumerate(units):
                    ledger.attach(key,slot,int(unit.getId()))
                    lane['refs'][int(unit.getId())]=unit
            lane['cursor']=len(plays)
            entities={}
            stat_fallback={}
            # Only world characters, excluding towers, projectiles and area effects.
            for entity in std.getBattle().getHolder().entities():
                if not isinstance(entity,self.character_class): continue
                identity=int(entity.getId())
                if identity in lane['refs'] and not entity.equals(lane['refs'][identity]):
                    raise ValueError('Native entity ID reused within a battle')
                hp=entity.getHitPoints()
                if hp is not None:
                    entities[int(entity.getId())]=(int(entity.side()),int(hp.getHitPoints()),int(hp.getMaximum()))
                    if (self.config.get('unknown')=='stat_two_bucket'
                            and identity not in lane['ledger'].entity_to_slot):
                        stat_fallback[identity]=self._unit_stats(entity.getData())
            balance,metrics=lane['ledger'].values(entities)
            if self.config.get('unknown')=='stat_two_bucket':
                values,counts=fallback_values(entities,lane['ledger'].entity_to_slot,
                    stat_fallback,self.config['fallback'])
                extra=float(values[0]-values[1])
                balance+=np.array([extra,-extra])
                metrics.update(counts)
                alive=metrics['mapped_member_count']+metrics['unknown_lineage_count']
                metrics['still_unknown_rate']=counts['still_unknown_count']/alive if alive else 0.
                metrics['exact_mapped_rate']=metrics['mapped_member_rate']
            phi[index]=balance
            metrics.update(lane=index,unknown_cast_count=lane['unknown_cast_count'],
                natural_expiry_count=None,natural_expiry_status='no_cause_event_in_direct_play_adapter')
            stats.append(metrics)
        return phi,stats
