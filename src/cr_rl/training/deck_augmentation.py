"""Bounded deck-form augmentation and family-balanced PPO sampling."""
from collections import defaultdict
from copy import deepcopy


def augment_forms(pool, tables):
    """Append ordinary, single-evolution, single-hero and mixed variants.

    Only explicit, enabled alternate rows linked by the base card are eligible.
    Original deck names, order and flags remain unchanged for checkpoint resumes.
    """
    result = deepcopy(pool)
    decks = result['decks']
    slots = result.setdefault('slots', {})
    existing = {(tuple(cards), tuple(slots.get(name, [0]*8))) for name, cards in decks.items()}
    families = {}
    for name, cards in list(decks.items()):
        families.setdefault(tuple(sorted(cards)), (name, cards))

    def append(name, cards, flags):
        spec = (tuple(cards), tuple(flags))
        if spec in existing:
            return
        if name in decks:
            raise ValueError(f'Augmentation name collision: {name}')
        decks[name] = list(cards)
        slots[name] = list(flags)
        existing.add(spec)

    for name, cards in families.values():
        append(name+'__ordinary', cards, [0]*8)
        eligible = {1: [], 2: []}
        for i, card in enumerate(cards):
            base = tables.get(card, {})
            for linked in base.get('EvolvedSpells', []) or []:
                row = tables.get(linked, {})
                if not row or row.get('NotInUse') or row.get('NotVisible'):
                    continue
                kind = row.get('CardForm')
                flag = 2 if kind == 'HeroForm' else 1 if linked.endswith('_EV1') else 0
                if flag and i not in eligible[flag]:
                    eligible[flag].append(i)
        # At most three upgraded candidates per base family. Avoid a combinatorial
        # explosion; ordinary sampling receives its own independent probability.
        for flag, label in ((1, 'evo'), (2, 'hero')):
            if eligible[flag]:
                flags = [0]*8
                flags[eligible[flag][0]] = flag
                append(name+'__'+label, cards, flags)
        pair = next(((e, h) for e in eligible[1] for h in eligible[2] if e != h), None)
        if pair:
            flags = [0]*8
            flags[pair[0]], flags[pair[1]] = 1, 2
            append(name+'__evo_hero', cards, flags)
    result['deck_sampling'] = {'mode': 'family_balanced', 'ordinary_fraction': .5}
    result['pool_purpose'] = 'ppo_training_with_form_augmentation'
    return result


def deck_probabilities(pool, names):
    """Uniform families, then 50% ordinary / 50% upgraded within each family."""
    config = pool.get('deck_sampling', {})
    if not config:
        return None  # Preserve legacy uniform sampling and RNG consumption.
    if config.get('mode') != 'family_balanced':
        raise ValueError('Unknown deck sampling mode')
    ordinary = float(config.get('ordinary_fraction', .5))
    if not 0 <= ordinary <= 1:
        raise ValueError('ordinary_fraction must lie in [0, 1]')
    families = defaultdict(lambda: ([], []))
    for i, name in enumerate(names):
        cards = pool['decks'][name]
        flags = pool.get('slots', {}).get(name, [0]*8)
        families[tuple(sorted(cards))][int(any(flags))].append(i)
    probabilities = [0.] * len(names)
    for plain, upgraded in families.values():
        if not plain:
            raise ValueError('Every augmented family requires an ordinary deck')
        for indices, weight in ((plain, ordinary if upgraded else 1.),
                                (upgraded, 1.-ordinary)):
            if indices:
                for i in indices:
                    probabilities[i] = weight / len(indices) / len(families)
    return probabilities
