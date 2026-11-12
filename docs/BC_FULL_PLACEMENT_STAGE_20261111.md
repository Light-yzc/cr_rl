# Full BC learned placement stage (2026-11-11)

Entry notebook: `notebooks/colab_bc040_full.ipynb`. Rebased on the actual cloud
`source_parallel8_noearlystop_20261111.tar.gz`, not the older sampled-data notebook.

## Training contract

- Existing 10,654 games / 333 source shards, 0.4-second frames, all original action labels.
- Preserve the current fixed train / validation / heldout identities. Approximately
  10,223 train, 426 validation, 5 heldout; random 16 validation games per epoch.
- Initialize from the verified latest full BC v3 parent at old optimizer step 9,887
  (`epoch=0`, `next_episode=5120`). Parent weights are retained; its Adam, epoch and
  scaler are not restored into the new stage.
- Freeze the entity encoder, six KDA / two GQA temporal layers and card / WAIT heads.
  Train only the tested `card_prior_cnn_001` placement adapter, with 64 CNN channels.
  The per-card location table is learned, initialized from TRAIN labels in this
  actual full dataset, with sigma=1 and alpha=.01. No old 293-game fitted decoder
  or its temperature calibration is transplanted.
- Eight epochs, constant learning rate 3e-4, no warmup, temperature 1. Parallel eight
  episode seats, frame microbatch 256, existing FP16 autocast / GradScaler safeguards.
- New directory `experiment_placement_priorcnn64_20261111`, new optimizer and W&B
  identity. This is a separate placement stage, not joint training of the trunk.

## Resume and storage

The controller uses `--upgrade-placement-from` only for the initial parent
handoff, with the parent's bound dataset manifest. Same-stage restarts use
`--resume` and restore the versioned decoder, frozen mode, Adam, scaler, RNG and
cursor. It rejects changed replay splits, representation, vocabulary or dataset
growth in this release. A repack with unchanged game identities / splits is
allowed for initial parent handoff. Prior statistics bind to the new dataset
fingerprint and train split.

Only `latest.pt` (full resume) and `best.pt` (inference) are retained, with atomic
Drive publication. Packed data loads once to `/content`; expanded per-game mmap
files are reclaimed after use. No local full checkpoint download was performed
while preparing this release.

All-WAIT chunks advance temporal state but do not call backward when the frozen
model has no trainable loss. An all-WAIT episode group does not advance Adam.
Padded finished streams skip the spatial decoder; active inference frames still
evaluate all four hand cards. Nonfinite aggregate gradient norms reject an
update before weights or Adam change. StateEpisode diagnostics omit mmap arrays
from repr, avoiding reads of already-closed mappings when reporting exceptions.

## User launch

Reopen the updated `colab_bc040_full.ipynb`, select L4, then run code cells 1–6
in order. Code cell 5 starts training; rerunning it reconnects without launching
a duplicate trainer. W&B reads the existing Colab secret / environment variable.
The preparation task itself uses only an existing CPU runtime; it does not start
GPU training. First preparation performs one TRAIN-only location-statistics scan.

The prior experiments improved heldout placement metrics, but full-data learning
and real match win rate must be assessed after this new stage has run. Freezing
card/WAIT heads deliberately does not change their learned behavior.

## Release implementation

The source bundle keeps the latest cloud `policy.py`, persistence and packed-data
implementation. Seven runtime overlays add the adapter and training integration;
this avoids pulling unrelated local PPO / holding-prior changes into BC.
The full notebook has a reproducible JSON template and verifies embedded helper
hashes. Release manifests and CPU preflight receipts accompany the archive.
