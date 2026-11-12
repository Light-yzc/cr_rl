# League candidate confirmation

`loc_transfer/confirm_league_candidates.py` is a bounded, opt-in confirmation job for the league diversity trial. It waits for all eight 8-game screening reports, ranks only non-anchor historical candidates by screening opponent score (descending, generation ascending on ties), and fixes the top two.

Each selected candidate gets two fresh evaluator chunks at pair starts 8 and 16. Each chunk has eight held-out deck pairs and both learner seats, for 32 complete games per candidate. The pairing seed stays `20261201`; policy sampling uses the independent base seed `20261202`. Workers are pinned to the two registered V100 UUIDs and run their two chunks sequentially; each subprocess advertises the four GiB service cap. Existing complete reports are reused only when all protocol, model SHA, paired-seat, validity, code, deck-pool, and aggregate checks pass. Partial directories are renamed with an `.interrupted_<timestamp>` suffix before retry.

On success, `confirmation.json` records all results. Candidates reaching opponent score 0.55 or higher produce `expert_manifest.json` with labels `independent_counter_<generation>` and a sibling `diverse_pool/opponent_pool.json` derived from the prepared baseline. An existing generated branch is renamed with a `.previous_<timestamp>` suffix before replacement; the baseline and screening evidence are never modified. If neither candidate clears the gate, the script writes `no_experts.json` and does not create a usable manifest.

Use `--help` or `--fixture` for a no-GPU check. Running without those options starts evaluators and should be done only by the training operator after screening is complete.
