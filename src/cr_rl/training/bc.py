from __future__ import annotations

import torch
import torch.nn.functional as F


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
    tile_loss = torch.where(is_play, tile_loss, torch.zeros_like(tile_loss))

    # WAIT and ability labels supervise only the card/action-type decision.
    total = card_loss.mean() + position_coef * tile_loss.sum() / is_play.sum().clamp_min(1)
    return total, {
        "card_loss": card_loss.mean().detach(),
        "position_loss": (tile_loss.sum() / is_play.sum().clamp_min(1)).detach(),
        "wait_fraction": is_wait.float().mean().detach(),
        "play_fraction": is_play.float().mean().detach(),
    }
