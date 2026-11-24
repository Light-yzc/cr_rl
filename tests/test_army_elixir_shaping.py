"""Fixed-budget algebra and real native deployments; no GPU/model needed."""
import copy
from pathlib import Path

import numpy as np
import pytest

from cr_rl.training.army_elixir import ArmyElixirLedger, NativeArmyElixir, load_values
from cr_rl.training.rewards import RewardConfig, shaped_rewards

ROOT=Path(__file__).resolve().parents[1]


def test_swarm_budget_stays_fixed_after_death_and_repeated_cards_are_disjoint():
    ledger=ArmyElixirLedger()
    ledger.add_cast('first',0,3,1,{i:.2 for i in range(5)},confirmed=True)
    for i in range(5): ledger.attach('first',i,i)
    world={i:(0,100,100) for i in range(5)}
    assert ledger.values(world)[0][0]==pytest.approx(3)
    del world[0]
    assert ledger.values(world)[0][0]==pytest.approx(2.4)
    ledger.add_cast('second',0,4,1,{0:1},confirmed=True)
    ledger.attach('second',0,6);world[6]=(0,100,100)
    assert ledger.values(world)[0][0]==pytest.approx(6.4)
    assert ledger.groups['first'].slots=={i:.2 for i in range(5)}


def test_reserved_split_and_spawner_slots_cannot_create_budget_or_recycle_it():
    ledger=ArmyElixirLedger()
    slots={'parent':.7,'child1':.15,'child2':.15}
    ledger.add_cast('split',0,8,1,slots,confirmed=True)
    ledger.attach('split','parent',1)
    assert ledger.values({1:(0,100,100)})[0][0]==pytest.approx(5.6)
    ledger.attach('split','child1',2);ledger.attach('split','child2',3)
    assert ledger.values({2:(0,100,100),3:(0,100,100)})[0][0]==pytest.approx(2.4)
    with pytest.raises(ValueError,match='Reused'): ledger.attach('split','parent',4)
    ledger=ArmyElixirLedger()
    ledger.add_cast('hut',0,5,.6,{'body':.5,**{i:.05 for i in range(10)}},confirmed=True)
    ledger.attach('hut','body',0)
    for i in range(10): ledger.attach('hut',i,i+1)
    assert ledger.values({i:(0,100,100) for i in range(11)})[0][0]==pytest.approx(3)
    with pytest.raises(ValueError): ledger.attach('hut',11,12)
    with pytest.raises(ValueError): ledger.add_cast('extra',1,4,1,{0:1,1:1},confirmed=True)


def test_rejected_and_unknown_sources_never_create_value_and_reset_isolated():
    ledger=ArmyElixirLedger()
    assert not ledger.add_cast('refused',0,8,1,{0:1},confirmed=False)
    assert ledger.values({3:(0,100,100),4:(1,-1,100)})[0][0]==0
    ledger.add_cast('yes',1,4,1,{0:1},confirmed=True);ledger.attach('yes',0,3)
    phi,stats=ledger.values({3:(1,100,100),5:(0,100,0)})
    np.testing.assert_array_equal(phi,[-4,4])
    assert stats['mapped_member_count']==1
    restored=copy.deepcopy(ledger)  # continuing rollout chunk preserves slots
    np.testing.assert_allclose(restored.values({3:(1,50,100)})[0],[-2.6,2.6])
    restored.reset(2)
    assert restored.values({3:(1,100,100)})[0][0]==0
    assert ledger.values({3:(1,100,100)})[0][0]==-4
    assert not ledger.values({3:(1,100,-1),9:(0,float('nan'),100)})[0].any()


def test_enemy_damage_is_positive_own_damage_negative_with_same_beta():
    ledger=ArmyElixirLedger()
    for side in (0,1):
        ledger.add_cast(side,side,4,1,{0:1},confirmed=True);ledger.attach(side,0,side)
    before=ledger.values({0:(0,100,100),1:(1,100,100)})[0]
    enemy_damaged=ledger.values({0:(0,100,100),1:(1,50,100)})[0]
    own_damaged=ledger.values({0:(0,50,100),1:(1,100,100)})[0]
    assert (.99*enemy_damaged-before)[0]>0
    assert (.99*own_damaged-before)[0]<0


@pytest.mark.parametrize('gamma',[0.,.9,.9995,1.])
def test_elixir_terminal_telescope_truncation_and_faults(gamma):
    from test_reward_shaping import arena
    schema,obs=arena()
    cfg=RewardConfig(enabled=True,princess=0,king=0,crown=0,
        army_reward_mode='elixir_potential',army_elixir_values={'version':'test'},army_elixir_beta=.004)
    values=[np.array([[4.,-4.]]),np.array([[2.,-2.]]),np.array([[6.,-6.]])]
    total=np.zeros((1,2))
    for t in range(3):
        before=values[t];after=values[(t+1)%3]
        reward,_=shaped_rewards(obs,obs,np.zeros((1,2)),np.zeros((1,2)),[t==2],schema,gamma,cfg,
                               army_values=(before,after))
        assert reward.sum()==pytest.approx(0)
        total+=gamma**t*reward
    np.testing.assert_allclose(total,-values[0]*.004,atol=1e-8)
    reward,_=shaped_rewards(obs,obs,np.zeros((1,2)),np.zeros((1,2)),[False],schema,gamma,cfg,
                           army_values=(values[0],values[1]))
    np.testing.assert_allclose(reward,(gamma*values[1]-values[0])*.004,atol=1e-8)
    reward,_=shaped_rewards(obs,obs,np.zeros((1,2)),np.zeros((1,2)),[True],schema,gamma,cfg,
                           army_values=(values[0],values[1]),faulted=[True])
    assert not reward.any()


def test_old_reward_metadata_exactly_unchanged_and_new_metadata_detects_migration():
    cfg=RewardConfig(enabled=True,army_potential_weight=.015)
    assert cfg.metadata()==dict(version=1,formula='gamma_phi_next_minus_phi_current',
        princess=.15,king=.2,crown=.15,overflow=0.,enabled=True,army_potential_weight=.015)
    new=RewardConfig(enabled=True,army_reward_mode='elixir_potential',
        army_elixir_values=load_values(ROOT/'configs/army_elixir_values.yaml'))
    assert new.metadata()['version']==2 and cfg.metadata()!=new.metadata()
    assert new.metadata()['ledger_lifecycle']=='native_world_reset_on_resume'
    assert 'army_elixir_bar_factor' not in new.metadata()
    from dataclasses import replace
    resource = replace(new, army_elixir_bar_factor=.75)
    assert resource.metadata() == {**new.metadata(), 'version': 3, 'army_elixir_bar_factor': .75}


def test_resource_potential_uses_available_bar_and_paid_board_transfer():
    from test_reward_shaping import arena

    schema, initial = arena()
    schema.scales = {'elixir_scale': 10_000}
    schema.header['elixir_reserved'] = [initial.shape[-1]]
    schema.header['enemy_elixir'] = [initial.shape[-1] + 1]
    initial = np.pad(initial, ((0, 0), (0, 0), (0, 2)))
    own = schema.header['elixir'][0]
    reserved = schema.header['elixir_reserved'][0]
    initial[..., own] = [[6 * 10_000, 2 * 10_000]]
    initial[..., reserved] = [[10_000, 2 * 10_000]]
    deployed = initial.copy()
    deployed[..., own] -= [[5 * 10_000, 0]]
    cfg = RewardConfig(enabled=True, princess=0, king=0, crown=0,
        army_reward_mode='elixir_potential', army_elixir_values={'version': 'test'},
        army_elixir_bar_factor=.75)
    zeros = np.zeros((1, 2))
    board = np.array([[5., -5.]])
    reward, comp = shaped_rewards(initial, deployed, zeros, zeros, [False], schema, 1., cfg,
                                  army_values=(zeros, board))
    np.testing.assert_allclose(reward, [[.005, -.005]], atol=1e-8)
    np.testing.assert_array_equal(reward, comp['army'])
    for terminated, faulted, expected in [
        (False, False, .004 * (.99 * 5 - .75 * 5)),
        (True, False, -.004 * .75 * 5),
        (True, True, 0.),
    ]:
        reward, _ = shaped_rewards(initial, deployed, zeros, zeros, [terminated], schema, .99, cfg,
                                   army_values=(zeros, board), faulted=[faulted])
        np.testing.assert_allclose(reward, [[expected, -expected]], atol=1e-8)
    waited = initial.copy()
    waited[0, 0, own] += 10_000
    reward, _ = shaped_rewards(initial, waited, zeros, zeros, [False], schema, 1., cfg,
                               army_values=(zeros, zeros))
    np.testing.assert_allclose(reward, [[.003, -.003]], atol=1e-8)


def test_resource_potential_for_single_seat_batch_of_two():
    from test_reward_shaping import arena

    schema, initial = arena()
    schema.scales = {'elixir_scale': 10_000}
    schema.header['elixir_reserved'] = [initial.shape[-1]]
    schema.header['enemy_elixir'] = [initial.shape[-1] + 1]
    initial = np.pad(initial[0], ((0, 0), (0, 2)))
    initial[:, schema.header['elixir'][0]] = [40_000, 60_000]
    initial[:, schema.header['elixir_reserved'][0]] = [10_000, 0]
    initial[:, schema.header['enemy_elixir'][0]] = [10_000, 30_000]
    cfg = RewardConfig(enabled=True, princess=0, king=0, crown=0,
        army_reward_mode='elixir_potential', army_elixir_values={'version': 'test'},
        army_elixir_bar_factor=.75)
    zeros = np.zeros(2)
    reward, _ = shaped_rewards(initial, initial, zeros, zeros, [True, False], schema, .99, cfg,
                               army_values=(zeros, zeros))
    np.testing.assert_allclose(reward, [-.006, -.00009], atol=1e-8)


def native_env():
    from crforge_env import _jvm
    _jvm.start(ROOT/'.cache/vendor/crforge/env/build/install/env',['-Xmx512m'])
    from cr_rl.envs.crforge import CrForgeVector
    return CrForgeVector(1,1,independent=True,decision_ticks=8)


@pytest.mark.parametrize('target,cost,count,factor',[
    ('Knight',3,1,1),('Archer',3,2,1),('MinionHorde',5,6,1),('Cannon',3,1,.4),
    ('Golem',8,0,0),('Tombstone',3,0,0),('Graveyard',5,0,0)])
def test_real_native_successful_play_fixed_roster_and_unsupported_fail_closed(target,cost,count,factor):
    deck=['Knight','Archer','MinionHorde','Cannon','Golem','Tombstone','Graveyard','Mirror']
    with native_env() as env:
        for seed in range(10,40):
            env.vector.reset([seed],[deck],[deck])
            hand=env.observations[0,0,list(env.vector.schema.header['hand_card'])]
            if env.vector.vocabulary.index(target) in hand: break
        native=NativeArmyElixir(env,load_values(ROOT/'configs/army_elixir_values.yaml'))
        native.reset_lane(0)
        chosen=False
        for step in range(45):
            actions=np.zeros((1,2),dtype=np.int32)
            if not chosen:
                for slot in np.flatnonzero(hand==env.vector.vocabulary.index(target)):
                    legal=np.flatnonzero(env.masks[0,0,1+slot*576:1+(slot+1)*576])
                    if len(legal): actions[0,0]=1+slot*576+legal[len(legal)//2];chosen=True
            env.step(actions)
            if env.vector.fault(0): pytest.fail(str(env.vector.fault(0)))
            phi,stats=native.poll()
            ledger=native.lanes[0]['ledger']
            if len(env.vector.lanes[0]._env.env(0).battle().getPlays()): break
        assert chosen
        assert len(ledger.groups)==(1 if count else 0)
        if count:
            group=next(iter(ledger.groups.values()))
            assert len(group.members)==count and sum(group.slots.values())==pytest.approx(1)
            assert group.paid==cost and group.factor==factor
            assert abs(phi[0,0])<=cost*factor+1e-8
            assert stats[0]['unknown_lineage_count']==0
        else: assert phi[0,0]==0 and native.lanes[0]['unknown_cast_count']==1
        assert phi[0,0]==-phi[0,1]
        # Same native world across a collector chunk: nothing is reallocated.
        assert native.poll()[0][0,0]==phi[0,0]
        env.vector.reset([seed+1],[deck],[deck])
        native.reset_lane(0)
        assert native.poll()[0][0,0]==0


def test_real_native_resource_potential_uses_both_bars():
    deck = ['Knight', 'Archer', 'MinionHorde', 'Cannon', 'Golem', 'Tombstone', 'Graveyard', 'Mirror']
    values = load_values(ROOT/'configs/army_elixir_values.yaml')
    configs = [
        RewardConfig(enabled=True, army_reward_mode='elixir_potential',
                     army_elixir_values=values, army_elixir_bar_factor=factor)
        for factor in (0., .75)
    ]
    with native_env() as env:
        env.vector.reset([19], [deck], [deck])
        schema = env.vector.schema
        native = NativeArmyElixir(env, values)
        native.reset_lane(0)
        full_actions = np.zeros((1, 2), dtype=np.int32)
        saw_bar_transfer = False
        for _ in range(45):
            current = env.observations.copy()
            before = native.poll()[0]
            actions = full_actions.copy()
            if not saw_bar_transfer:
                legal = np.flatnonzero(env.masks[0, 0, 1:2305])
                if len(legal):
                    actions[0, 0] = int(legal[0]) + 1
            env.step(actions)
            following = env.observations.copy()
            after = native.poll()[0]
            assert not env.vector.fault(0)
            components = [
                shaped_rewards(current, following, actions, np.zeros((1, 2)), [False], schema, .9995,
                               cfg, army_values=(before, after))[1]['army']
                for cfg in configs
            ]
            own = schema.header['elixir'][0]
            reserved = schema.header['elixir_reserved'][0]
            cur = np.maximum(current[..., own] - current[..., reserved], 0)
            nxt = np.maximum(following[..., own] - following[..., reserved], 0)
            expected = (.9995*np.diff(nxt, axis=-1) - np.diff(cur, axis=-1)) * .75 * .004
            np.testing.assert_allclose(components[1] - components[0],
                                       np.concatenate((-expected, expected), axis=-1)
                                       / schema.scales['elixir_scale'], atol=1e-8)
            if actions[0, 0] and nxt[0, 0] < cur[0, 0]:
                saw_bar_transfer = True
                break
        assert saw_bar_transfer


def test_conserved_v2_covers_sparky_deck_and_evolutions_without_empty_drop_freebie():
    sparky_deck = ['ZapMachine', 'Elixir Collector', 'Wizard', 'Rage', 'Giant', 'DarkPrince', 'MiniPekka', 'MinionHorde']
    evo_deck = ['Cannon', 'Hunter', 'Knight', 'IceSpirits', 'Log', 'Arrows', 'Snowball', 'Zap']
    evo_forms = [1, 1, 2, 0, 0, 0, 0, 0]  # Evo Cannon, Evo Hunter, Hero Knight!
    values = load_values(ROOT / 'configs/army_elixir_conserved.yaml')
    cfg = RewardConfig(
        enabled=True,
        princess=0.0,
        king=0.0,
        crown=0.0,
        army_reward_mode='elixir_potential',
        army_elixir_beta=0.004,
        army_elixir_values=values,
        army_elixir_bar_factor=0.75,
    )
    with native_env() as env:
        schema = env.vector.schema
        # 1. Verify Elixir Collector (drawn after 1 cycle play per CR rules!), Wizard, DarkPrince, MinionHorde all register conserved ledger entries (< 0.75 * cost)
        for target, expected_cat in [('Elixir Collector', 'collector'), ('Wizard', 'troop'), ('DarkPrince', 'troop'), ('MinionHorde', 'troop')]:
            target_id = env.vector.vocabulary.index(target)
            for seed in range(10, 80):
                env.vector.reset([seed], [sparky_deck], [evo_deck], [[0]*8], [evo_forms])
                hand = env.observations[0, 0, list(schema.header['hand_card'])]
                nxt_card = int(env.observations[0, 0, schema.header['next_card'][0]])
                if target_id in hand or nxt_card == target_id:
                    break
            native = NativeArmyElixir(env, values)
            native.reset_lane(0)
            cycled = target_id in hand
            played = False
            groups_before_play = 0
            target_group_seen = False
            for _ in range(95):
                hand = env.observations[0, 0, list(schema.header['hand_card'])]
                actions = np.zeros((1, 2), dtype=np.int32)
                if not cycled and target_id not in hand:
                    # Elixir Collector never starts in initial 4-card hand; cycle 1 card first
                    for s_cyc in range(4):
                        legal = np.flatnonzero(env.masks[0, 0, 1 + s_cyc * 576 : 1 + (s_cyc + 1) * 576])
                        if len(legal):
                            actions[0, 0] = 1 + s_cyc * 576 + int(legal[0])
                            cycled = True
                            break
                elif not played and target_id in hand:
                    for slot in np.flatnonzero(hand == target_id):
                        legal = np.flatnonzero(env.masks[0, 0, 1 + slot * 576 : 1 + (slot + 1) * 576])
                        if len(legal):
                            groups_before_play = len(native.lanes[0]['ledger'].groups)
                            actions[0, 0] = 1 + slot * 576 + int(legal[len(legal) // 2])
                            played = True
                            break
                env.step(actions)
                assert not env.vector.fault(0)
                phi, _ = native.poll()
                if played and len(native.lanes[0]['ledger'].groups) > groups_before_play:
                    target_group_seen = True
                    break
            assert played and target_group_seen, f"{target} was not tracked!"
            grp = list(native.lanes[0]['ledger'].groups.values())[-1]
            assert grp.factor == pytest.approx(values['category_factors'][expected_cat])
            assert grp.factor < cfg.army_elixir_bar_factor
            assert phi[0, 0] > 0

        # 2. Verify Hero Knight & Evolved Cannon/Hunter on seat 1 are ALSO tracked (zero Evo/Hero blind spot!)
        for evo_target in ('Cannon', 'Hunter', 'Knight'):
            for seed in range(10, 80):
                env.vector.reset([seed], [sparky_deck], [evo_deck], [[0]*8], [evo_forms])
                hand1 = env.observations[0, 1, list(schema.header['hand_card'])]
                if env.vector.vocabulary.index(evo_target) in hand1:
                    break
            native = NativeArmyElixir(env, values)
            native.reset_lane(0)
            played = False
            for _ in range(55):
                actions = np.zeros((1, 2), dtype=np.int32)
                if not played:
                    for slot in np.flatnonzero(hand1 == env.vector.vocabulary.index(evo_target)):
                        legal = np.flatnonzero(env.masks[0, 1, 1 + slot * 576 : 1 + (slot + 1) * 576])
                        if len(legal):
                            actions[0, 1] = 1 + slot * 576 + int(legal[len(legal) // 2])
                            played = True
                env.step(actions)
                assert not env.vector.fault(0)
                phi, _ = native.poll()
                if native.lanes[0]['ledger'].groups:
                    break
            assert played and len(native.lanes[0]['ledger'].groups) == 1, f"Evo/Hero {evo_target} was not tracked!"
            assert phi[0, 1] > 0 and phi[0, 0] == -phi[0, 1]
