from __future__ import annotations

import torch
import torch.nn.functional as F


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
):
    """Separate card/WAIT supervision from the conditional tile decision.

    The policy still represents the exact joint distribution. This objective gives the
    strategically important decision to play or wait a clean gradient before asking the
    location head to explain a particular tile among 576 possibilities.
    """
    if joint_log_probs.shape[-1] != 2307 or legal_mask.shape != joint_log_probs.shape:
        raise ValueError("Expected crforge v4's 2307-action codec")
    if not torch.all(legal_mask.any(dim=-1)):
        raise ValueError("BC batch contains a state without a legal action")

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
    act_logp = torch.logsumexp(card_log_probs[..., 1:], dim=-1)
    timing_loss = -torch.where(is_act, act_logp, card_log_probs[..., 0])
    conditional_loss = torch.where(is_act, unweighted_card_loss + act_logp, 0.0)
    if conditional_card_loss:
        # Rebalance the decision to act, but never suppress the conditional
        # card/ability gradient with the sparse natural action prior.
        timing_wait = torch.where(is_wait, timing_loss, 0.0).sum() / is_wait.sum().clamp_min(1)
        timing_act = torch.where(is_act, timing_loss, 0.0).sum() / is_act.sum().clamp_min(1)
        present = (is_wait.any().float() + is_act.any().float()).clamp_min(1)
        timing = (timing_wait + timing_act) / present
        slot_loss = conditional_loss.sum() / is_act.sum().clamp_min(1)
        card_objective = timing + slot_loss
    elif card_class_weights is not None:
        groups = torch.where(is_wait, 0, torch.where(is_play, 1, 2))
        card_loss = card_loss * card_class_weights.to(card_loss)[groups]
        card_objective = card_loss.mean()
    else:
        card_objective = card_loss.mean()

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
    total = card_objective + position_coef * tile_loss.sum() / is_play.sum().clamp_min(1)
    return total, {
        "card_loss": card_objective.detach(),
        "timing_loss": (timing if conditional_card_loss else timing_loss.mean()).detach(),
        "conditional_slot_loss": (conditional_loss.sum() / is_act.sum().clamp_min(1)).detach(),
        "unweighted_card_loss": unweighted_card_loss.mean().detach(),
        "position_loss": (tile_loss.sum() / is_play.sum().clamp_min(1)).detach(),
        "position_hard_loss": (hard_tile_loss.sum() / is_play.sum().clamp_min(1)).detach(),
        "wait_fraction": is_wait.float().mean().detach(),
        "play_fraction": is_play.float().mean().detach(),
    }
