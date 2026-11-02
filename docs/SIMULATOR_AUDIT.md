# Simulator acceptance audit

Status: **candidate selected; local acceptance pending**.

## Candidate evidence

- `crforge` commit `7ee59b39bf1d41d5c01645f01c47ed8c8d3355d6`.
- Compatibility notes: reference client 16.635.1; client 16.635.7 observed 2026-10-31; data 16.635.3.
- Core uses 50 ms integer ticks; Python `VectorBattle` runs vectorised 1v1 battles and exports flat observations/action masks.
- Reference battle/conformance fixtures exist. Their coverage does not prove all Sparky matchups or the entire current ladder distribution.
- Current Mac initially had Java 17; crforge requires Java 21. Bootstrap installs/selects Java 21.

## Must pass before meaningful training

- [ ] Resolve all eight configured card names in the 16.635.3 catalogue and verify costs/rarities.
- [ ] Verify deck construction and first-hand/cycle semantics.
- [ ] Sparky charge, wind-up, projectile, retarget, stun/reset and death interactions.
- [ ] Giant pathing/body blocking and Sparky follow positioning.
- [ ] Zap damage/stun, Minions air targeting, Arrows area damage.
- [ ] Legal action masks, elixir reservation, delayed command execution, failed deploy reporting.
- [ ] Side symmetry, same-prestate action fairness, deterministic seeded reset.
- [ ] Replay round-trip and terminal result parity on selected fixtures.
- [ ] 1/4/8/32 environment throughput and memory measurement on target hardware.

## Important interface discrepancy

The simulator's Python environment uses a flat 18x32 tile action codec (with WAIT), seat-relative frames, and an observation schema of its own. Do not pretend it already matches the proposed 192-entity observation / conditional card-slot-position policy. Implement a versioned adapter and preserve raw simulator observations. A future client adapter must map to the same canonical schema and explicitly mask unavailable fields.

## Version policy

The backend commit and data version are pinned in `configs/backends.yaml`. Never silently update either during a training run. Every run stores simulator commit, data content hash, schema version, deck, card levels, tower level, seed range, opponent checkpoint IDs, and policy version.
