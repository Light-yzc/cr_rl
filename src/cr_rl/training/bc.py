from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BCBatchNormalization:
    """CPU label counts for one effective batch, including all its microbatches.

    ``act`` includes ordinary card plays and ability actions; ``play`` includes
    only ordinary card plays. Counts are fixed before any microbatch forward so
    each loss call can contribute to the same objective without a GPU sync.
    """

    rows: int
    wait: int
    act: int
    play: int

    def __post_init__(self):
        if any(not isinstance(value, int) or isinstance(value, bool)
               for value in (self.rows, self.wait, self.act, self.play)):
            raise TypeError("BC normalization counts must be Python integers")
        if self.rows <= 0 or min(self.wait, self.act, self.play) < 0:
            raise ValueError("BC normalization requires positive rows and nonnegative counts")
        if self.wait + self.act != self.rows or self.play > self.act:
            raise ValueError("BC normalization requires wait + act == rows and play <= act")

    @classmethod
    def from_actions(cls, actions: torch.Tensor) -> BCBatchNormalization:
        """Count CPU labels; do not transfer a GPU label tensor for accounting."""
        if actions.device.type != "cpu":
            raise ValueError("Build BC normalization from CPU labels before device transfer")
        wait = int((actions == 0).sum())
        play = int(((actions > 0) & (actions < 2305)).sum())
        return cls(rows=actions.numel(), wait=wait, act=actions.numel() - wait, play=play)


def hierarchical_action_predictions(joint_log_probs: torch.Tensor):
    """Compare WAIT with total play mass; choose a card conditional on playing.

    Comparing WAIT with each individual slot biases the diagnostic toward WAIT
    simply because play probability is distributed over four cards and 576 tiles.
    """
    logp = joint_log_probs.float()
    placements = logp[..., 1:2305].reshape(*logp.shape[:-1], 4, 576)
    slots = torch.logsumexp(placements, dim=-1)
    card_logp = torch.cat((logp[..., :1], slots, logp[..., 2305:2307]), dim=-1)
    predicts_play = torch.logsumexp(slots, dim=-1) > logp[..., 0]
    return card_logp, predicts_play, slots.argmax(dim=-1)


def behaviour_cloning_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    legal_mask: torch.Tensor,
    weights: torch.Tensor | None = None,
):
    """Masked action cross-entropy. Position-only/conditional losses can be layered on later."""
    if not torch.all(legal_mask.any(dim=-1)):
        raise ValueError("BC batch contains a state without a legal action")
    masked = logits.masked_fill(~legal_mask.bool(), -1e9)
    per_item = F.cross_entropy(
        masked.reshape(-1, masked.shape[-1]), actions.long().reshape(-1), reduction="none"
    ).reshape(actions.shape)
    if weights is not None:
        per_item = per_item * weights
    return per_item.mean()


def hierarchical_bc_loss(
    joint_log_probs: torch.Tensor,
    actions: torch.Tensor,
    legal_mask: torch.Tensor,
    position_coef: float = 0.5,
    card_class_weights: torch.Tensor | None = None,
    tile_label_sigma: float = 0.0,
    conditional_card_loss: bool = False,
    timing_prior: float | None = None,
    sampled_action_prior: float | None = None,
    normalization: BCBatchNormalization | None = None,
):
    """Separate card/WAIT supervision from the conditional tile decision.

    The policy still represents the exact joint distribution. This objective gives the
    strategically important decision to play or wait a clean gradient before asking the
    location head to explain a particular tile among 576 possibilities.

    By default, return the original objective averaged within this call. If
    ``normalization`` contains the whole effective batch's counts, both the loss
    and every returned metric are this microbatch's additive contribution to that
    whole-batch objective. Sum these contributions across microbatches; do not
    divide the loss or the gradients by the accumulation count again. The
    fractions also contribute local counts divided by effective-batch rows.
    """
    if joint_log_probs.shape[-1] != 2307 or legal_mask.shape != joint_log_probs.shape:
        raise ValueError("Expected crforge v4's 2307-action codec")
    if not torch.all(legal_mask.any(dim=-1)):
        raise ValueError("BC batch contains a state without a legal action")
    if normalization is not None and actions.numel() > normalization.rows:
        raise ValueError("BC microbatch is larger than its effective-batch normalization")

    slot_log_probs = [joint_log_probs[..., :1]]
    for slot in range(4):
        start = 1 + slot * 576
        slot_log_probs.append(
            torch.logsumexp(joint_log_probs[..., start : start + 576], dim=-1, keepdim=True)
        )
    slot_log_probs.extend((joint_log_probs[..., 2305:2307],))
    card_log_probs = torch.cat(slot_log_probs, dim=-1)

    is_wait = actions == 0
    is_play = (actions > 0) & (actions < 2305)
    is_ability = actions >= 2305
    card_target = torch.zeros_like(actions, dtype=torch.long)
    card_target = torch.where(is_play, 1 + (actions - 1) // 576, card_target)
    card_target = torch.where(is_ability, 5 + actions - 2305, card_target)
    card_loss = -card_log_probs.gather(-1, card_target.unsqueeze(-1)).squeeze(-1)
    unweighted_card_loss = card_loss
    is_act = ~is_wait
    wait_denominator = (
        is_wait.sum().clamp_min(1) if normalization is None else max(normalization.wait, 1)
    )
    act_denominator = (
        is_act.sum().clamp_min(1) if normalization is None else max(normalization.act, 1)
    )
    play_denominator = (
        is_play.sum().clamp_min(1) if normalization is None else max(normalization.play, 1)
    )

    def row_mean(values: torch.Tensor) -> torch.Tensor:
        if normalization is None:
            return values.mean()
        return values.sum() / normalization.rows

    act_logp = torch.logsumexp(card_log_probs[..., 1:], dim=-1)
    timing_loss = -torch.where(is_act, act_logp, card_log_probs[..., 0])
    conditional_loss = torch.where(is_act, unweighted_card_loss + act_logp, 0.0)
    if conditional_card_loss:
        # Rebalance the decision to act, but never suppress the conditional
        # card/ability gradient with the sparse natural action prior.
        timing_wait = torch.where(is_wait, timing_loss, 0.0).sum() / wait_denominator
        timing_act = torch.where(is_act, timing_loss, 0.0).sum() / act_denominator
        present = (
            (is_wait.any().float() + is_act.any().float()).clamp_min(1)
            if normalization is None
            else max(int(normalization.wait > 0) + int(normalization.act > 0), 1)
        )
        timing = (timing_wait + timing_act) / present
        if timing_prior is not None:
            if not (
                0 < timing_prior < 1
                and sampled_action_prior is not None
                and 0 < sampled_action_prior < 1
            ):
                raise ValueError("Natural timing requires target and sampled priors in (0, 1)")
            # Importance weighting changes ONLY timing. Conditional card/tile
            # gradients retain their action-rich sampling density. The sampled
            # prior is measured over the epoch, not estimated per microbatch.
            weights = torch.where(
                is_act,
                timing_prior / sampled_action_prior,
                (1 - timing_prior) / (1 - sampled_action_prior),
            )
            timing = row_mean(timing_loss * weights)
        slot_loss = conditional_loss.sum() / act_denominator
        card_objective = timing + slot_loss
    elif card_class_weights is not None:
        groups = torch.where(is_wait, 0, torch.where(is_play, 1, 2))
        card_loss = card_loss * card_class_weights.to(card_loss)[groups]
        card_objective = row_mean(card_loss)
    else:
        card_objective = row_mean(card_loss)

    safe_slot = ((actions - 1) // 576).clamp(0, 3)
    safe_tile = ((actions - 1) % 576).clamp(0, 575)
    tile_rows = joint_log_probs[..., 1:2305].reshape(*actions.shape, 4, 576)
    chosen_slot = tile_rows.gather(
        -2, safe_slot.unsqueeze(-1).unsqueeze(-1).expand(*actions.shape, 1, 576)
    ).squeeze(-2)
    conditional_tile_log_probs = chosen_slot - card_log_probs.gather(
        -1, (safe_slot + 1).unsqueeze(-1)
    )
    tile_loss = -conditional_tile_log_probs.gather(-1, safe_tile.unsqueeze(-1)).squeeze(-1)
    hard_tile_loss = torch.where(is_play, tile_loss, torch.zeros_like(tile_loss))
    if tile_label_sigma < 0:
        raise ValueError("tile label sigma cannot be negative")
    if tile_label_sigma > 0:
        grid = torch.arange(576, device=actions.device)
        distance_squared = (grid % 18 - safe_tile[..., None] % 18).float().square()
        distance_squared += (grid // 18 - safe_tile[..., None] // 18).float().square()
        tile_legal = legal_mask[..., 1:2305].reshape(*actions.shape, 4, 576)
        chosen_legal = tile_legal.gather(
            -2, safe_slot[..., None, None].expand(*actions.shape, 1, 576)
        ).squeeze(-2)
        targets = torch.exp(-distance_squared / (2 * tile_label_sigma**2)) * chosen_legal
        targets = targets / targets.sum(-1, keepdim=True).clamp_min(1e-12)
        tile_loss = -(targets * conditional_tile_log_probs.float()).sum(-1)
    tile_loss = torch.where(is_play, tile_loss, torch.zeros_like(tile_loss))

    # WAIT and ability labels supervise only the card/action-type decision.
    total = card_objective + position_coef * tile_loss.sum() / play_denominator
    return total, {
        "card_loss": card_objective.detach(),
        "timing_loss": (timing if conditional_card_loss else row_mean(timing_loss)).detach(),
        "conditional_slot_loss": (conditional_loss.sum() / act_denominator).detach(),
        "unweighted_card_loss": row_mean(unweighted_card_loss).detach(),
        "position_loss": (tile_loss.sum() / play_denominator).detach(),
        "position_hard_loss": (hard_tile_loss.sum() / play_denominator).detach(),
        "wait_fraction": row_mean(is_wait.float()).detach(),
        "play_fraction": row_mean(is_play.float()).detach(),
    }
