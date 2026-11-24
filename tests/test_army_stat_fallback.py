import copy
from pathlib import Path

import numpy as np
import pytest

from cr_rl.training.army_elixir import fallback_values,load_values,NativeArmyElixir

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/army_elixir_stat_fallback.yaml'


def test_two_buckets_both_side_caps_and_exact_never_double_counted():
    settings=load_values(CONFIG)['fallback']
    world={1:(0,100,100),2:(0,100,100),3:(1,100,100),4:(1,50,100)}
    stats={1:(81,81),2:(400,10),3:(100,150),4:(350,100)}
    values,count=fallback_values(world,{1},stats,settings)
    np.testing.assert_allclose(values,[.3,.3+.3*.65])
    assert count['fallback_low_count']==0 and count['fallback_high_count']==3
    world={i:(i%2,100,100) for i in range(200)}
    values,count=fallback_values(world,{},dict.fromkeys(world,(81,81)),settings)
    np.testing.assert_array_equal(values,[1,1])
    assert count['fallback_cap_hit_count']==2


def test_bad_stats_and_hp_are_zero_and_no_fixed_reward_for_survival():
    settings=load_values(CONFIG)['fallback']
    world={1:(0,100,100),2:(0,-1,100),3:(1,100,0),4:(1,100,100)}
    values,count=fallback_values(world,{}, {1:None,4:(-1,32)},settings)
    assert not values.any() and count['still_unknown_count']==3
    values,_=fallback_values({1:(0,100,100)}, {}, {1:(81,81)},settings)
    assert values[0]==pytest.approx(.1)
    assert (1.*values-values)[0]==0
    assert (.99*values-values)[0]==pytest.approx(-.001)  # discount term only


def test_native_summons_have_fallback_without_claiming_exact_lineage():
    from test_army_elixir_shaping import native_env
    deck=['Knight','Archer','MinionHorde','Cannon','Golem','Tombstone','Graveyard','Mirror']
    observations=[]
    with native_env() as env:
        for target in ('Tombstone','Graveyard','Golem'):
            for seed in range(10,40):
                env.vector.reset([seed],[deck],[deck])
                hand=env.observations[0,0,list(env.vector.schema.header['hand_card'])]
                if env.vector.vocabulary.index(target) in hand: break
            tracker=NativeArmyElixir(env,load_values(CONFIG));tracker.reset_lane(0)
            chosen=False;seen=0
            child_seen=0
            for step in range(650 if target=='Golem' else 180):
                actions=np.zeros((1,2),dtype=np.int32)
                if not chosen:
                    for slot in np.flatnonzero(hand==env.vector.vocabulary.index(target)):
                        legal=np.flatnonzero(env.masks[0,0,1+slot*576:1+(slot+1)*576])
                        if len(legal): actions[0,0]=1+slot*576+legal[len(legal)//2];chosen=True
                env.step(actions)
                assert not env.vector.fault(0)
                phi,stats=tracker.poll();row=stats[0]
                assert row['fallback_value_own']<=1 and row['fallback_value_enemy']<=1
                assert row['mapped_member_count']==0
                assert phi[0,0]==-phi[0,1]
                seen+=row['fallback_low_count']+row['fallback_high_count']
                if target=='Golem':
                    native=env.vector.lanes[0]._env.env(0).battle()
                    child_seen+=sum(str(u.getData().name())=='Golemite'
                        for u in native.getBattle().getHolder().entities() if hasattr(u,'getData'))
                if env.dones[0]: break
            assert seen>0,target
            if target=='Golem': assert child_seen>0,'Native Golem did not split in this probe'
            observations.append(seen)
    assert all(observations)


def test_native_static_group_retains_exact_priority_with_fallback_enabled():
    from test_army_elixir_shaping import native_env
    deck=['Knight','Archer','MinionHorde','Cannon','Golem','Tombstone','Graveyard','Mirror']
    with native_env() as env:
        for seed in range(10,40):
            env.vector.reset([seed],[deck],[deck])
            hand=env.observations[0,0,list(env.vector.schema.header['hand_card'])]
            if env.vector.vocabulary.index('MinionHorde') in hand: break
        tracker=NativeArmyElixir(env,load_values(CONFIG))
        chosen=False
        for step in range(45):
            actions=np.zeros((1,2),dtype=np.int32)
            if not chosen:
                for slot in np.flatnonzero(hand==env.vector.vocabulary.index('MinionHorde')):
                    legal=np.flatnonzero(env.masks[0,0,1+slot*576:1+(slot+1)*576])
                    if len(legal): actions[0,0]=1+slot*576+legal[len(legal)//2];chosen=True
            env.step(actions);phi,stats=tracker.poll()
            if tracker.lanes[0]['ledger'].groups: break
        assert tracker.lanes[0]['ledger'].groups
        assert stats[0]['fallback_low_count']==stats[0]['fallback_high_count']==0
        assert tracker._safe_card(env.vector.lanes[0]._env.env(0).battle(),'GoblinGang')=='troop'
        assert tracker._safe_card(env.vector.lanes[0]._env.env(0).battle(),'Golem') is None
        assert tracker._safe_card(env.vector.lanes[0]._env.env(0).battle(),'Tombstone') is None
