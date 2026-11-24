"""Generate reproducible enabled form variants from the pinned simulator tables."""
import argparse
import json
from pathlib import Path

import yaml

from cr_rl.training.deck_augmentation import augment_forms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('configs/decks_all51.yaml'))
    p.add_argument('--out', type=Path, default=Path('configs/decks_augmented.yaml'))
    p.add_argument('--tables', type=Path, default=Path.home()/'.crforge/tables/16.635.3')
    args = p.parse_args()
    tables = {}
    for filename in sorted(args.tables.glob('spells_*.json')):
        for name, row in json.loads(filename.read_text(encoding='utf-8'))['rows'].items():
            tables[name] = row['columns']
    pool = yaml.safe_load(args.source.read_text(encoding='utf-8'))
    result = augment_forms(pool, tables)
    result['augmentation_source'] = str(args.source)
    result['augmentation_table_version'] = args.tables.name
    args.out.write_text(yaml.safe_dump(result, sort_keys=False), encoding='utf-8')
    print(f"Decks: {len(pool['decks'])} -> {len(result['decks'])}; output: {args.out}")


if __name__ == '__main__':
    main()
