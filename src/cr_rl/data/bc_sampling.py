"""Complete action-label coverage with shard-local history access."""

import numpy as np


def epoch_rows(
    plays,
    waits,
    rng,
    play_fraction,
    sample_count=None,
    offsets=None,
    ability_rows=None,
    ability_fraction=0.0,
    shard_mix_window=1,
):
    if shard_mix_window < 1:
        raise ValueError("shard_mix_window must be positive")
    if not 0 <= ability_fraction < 1:
        raise ValueError("ability fraction is conditional on non-WAIT draws and must be in [0,1)")
    abilities = np.asarray(ability_rows if ability_rows is not None else [], dtype=np.int64)
    cards = plays[~np.isin(plays, abilities)] if len(abilities) else plays
    if sample_count is None:
        selected_plays = rng.permutation(plays)
        if ability_fraction and len(abilities):
            required = int(np.ceil(len(cards) * ability_fraction / (1 - ability_fraction)))
            extra = max(0, required - len(abilities))
            if extra:
                selected_plays = np.concatenate((selected_plays, rng.choice(abilities, extra)))
        wait_count = max(1, round(len(selected_plays) * (1 - play_fraction) / play_fraction))
    else:
        n_play = max(1, int(sample_count * play_fraction))
        if ability_fraction and len(abilities) and len(cards) and n_play >= 2:
            n_ability = min(n_play - 1, max(1, round(n_play * ability_fraction)))
            selected_plays = np.concatenate(
                (
                    rng.choice(cards, n_play - n_ability, replace=n_play - n_ability > len(cards)),
                    rng.choice(abilities, n_ability, replace=n_ability > len(abilities)),
                )
            )
        else:
            selected_plays = rng.choice(plays, n_play, replace=n_play > len(plays))
        wait_count = max(1, sample_count - n_play)
    selected = np.concatenate(
        [
            selected_plays,
            rng.choice(waits, wait_count, replace=wait_count > len(waits)),
        ]
    )
    rng.shuffle(selected)
    if offsets is not None:
        shards = np.searchsorted(offsets[1:], selected, side="right")
        order = rng.permutation(len(offsets) - 1)
        ranks = np.empty(len(order), dtype=np.int64)
        ranks[order] = np.arange(len(order))
        # Randomize within a bounded working set, rather than fitting one shard
        # at a time. Window=1 preserves the historical ordering for old resumes.
        windows = ranks[shards] // shard_mix_window
        selected = selected[np.argsort(windows, kind="stable")]
    return selected


def class_prior_weights(natural_actions, sampled_actions):
    def frequency(actions):
        groups = np.where(actions == 0, 0, np.where(actions < 2305, 1, 2))
        return np.bincount(groups, minlength=3) / len(groups)

    natural, sampled = frequency(natural_actions), frequency(sampled_actions)
    if ((natural > 0) & (sampled == 0)).any():
        raise ValueError("A natural action class is absent from the sampled epoch")
    return np.divide(natural, sampled, out=np.zeros(3), where=sampled > 0)
