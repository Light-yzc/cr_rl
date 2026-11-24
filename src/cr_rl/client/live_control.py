"""Pending action reconciliation using the existing reader/adapter/action APIs."""
from dataclasses import dataclass


class BattleClockGate:
    """Use observed tick AND time advancement when the raw flag is unreliable.

    Live trace 20261115: the raw running byte becomes false mid-match even
    while both counters continue advancing. It cannot independently gate play.
    End-of-match evidence always takes precedence; first unflagged frame waits
    for a second observation rather than assuming the battle has started.
    """
    def __init__(self):
        self.previous = None

    def allow(self, *, tick, time_ms, running_flag, ended):
        progressed = (self.previous is not None and tick > self.previous[0]
                      and time_ms > self.previous[1])
        self.previous = (tick, time_ms)
        return not ended and tick >= 8 and (running_flag or progressed)


@dataclass
class PendingPlay:
    match_id: str
    played_count: int
    hand: tuple[int, ...]
    slot: int
    sent_at: float
    decision_id: int | None = None
    battle_target: tuple[int, int] | None = None


@dataclass
class PendingAbility:
    match_id: str
    slot: int
    deck_idx: int
    unit_eid: int
    charges: int
    cooldown_ms: int
    sent_at: float
    decision_id: int | None = None


def reconcile_ability(pending, *, match_id, abilities, now, timeout=2.):
    if match_id!=pending.match_id:return 'timed_out'
    current=next((a for a in abilities or () if a.slot==pending.slot),None)
    if current is not None and current.deck_idx==pending.deck_idx and current.unit_eid==pending.unit_eid:
        consumed=pending.charges>=0 and 0<=current.charges<pending.charges
        cooldown=current.cooldown_ms>pending.cooldown_ms
        if current.state in (7,10) or consumed or cooldown:return 'confirmed'
    if now-pending.sent_at>=timeout:return 'timed_out'
    return None


def pending_targets_for(pending, current_hand):
    """Recover the requested battle point only for the same deck card.

    This is command context from the local sender, not a newly read server
    position. The adapter only applies it to a matching server pending command.
    """
    targets = [None] * 4
    if not isinstance(pending,PendingPlay) or pending.battle_target is None:
        return tuple(targets)
    deck_index = pending.hand[pending.slot]
    if deck_index in current_hand:
        targets[list(current_hand).index(deck_index)] = pending.battle_target
    return tuple(targets)


def reconcile_play(pending, *, match_id, played_count, hand, now, timeout=2.):
    if pending is None:
        return None
    if match_id != pending.match_id:
        return 'timed_out'
    # Count alone does not identify this command. Require the expected slot
    # to rotate too; external manual play must be absent during model control.
    if played_count == pending.played_count+1 and hand[pending.slot] != pending.hand[pending.slot]:
        return 'confirmed'
    if now-pending.sent_at >= timeout:
        return 'timed_out'
    return None
