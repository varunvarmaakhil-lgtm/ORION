#!/usr/bin/env python3
"""
ORION auxiliary loss: episode-success value head.

Predicts P(the current episode will eventually succeed | state) from the
PPO policy's recurrent features. Habitat's success flag (1 only on the final
step of a successful episode) is treated as a sparse terminal reward, and
targets are TD(lambda) success returns with gamma = 1:

    if transition t ends the episode:  G_t = success_t
    otherwise:                          G_t = (1 - lam) * p_{t+1} + lam * G_{t+1}

p_{t+1} is the head's own detached prediction. The last rollout step is used
only as the bootstrap value. Loss = BCE(logits, G) * loss_scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_auxiliary_loss(name="orion")
class ORIONSuccessLoss(nn.Module):
    def __init__(
        self,
        action_space,
        net,
        loss_scale: float = 0.05,
        hidden_dim: int = 256,
        target_temperature: float = 1.0,
        td_lambda: float = 0.95,
        hidden_size=None,
    ):
        super().__init__()
        if hidden_size is not None:
            hidden_dim = hidden_size
        if loss_scale < 0:
            raise ValueError("loss_scale must be non-negative.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if target_temperature <= 0:
            raise ValueError("target_temperature must be positive.")
        if not 0.0 <= td_lambda <= 1.0:
            raise ValueError("td_lambda must be in [0, 1].")

        self.loss_scale = float(loss_scale)
        self.target_temperature = float(target_temperature)
        self.td_lambda = float(td_lambda)

        self.success_head = nn.Sequential(
            nn.Linear(net.output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for module in self.success_head:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def predict_logits(self, rnn_output):
        return self.success_head(rnn_output).squeeze(-1) / self.target_temperature

    @torch.no_grad()
    def compute_targets(self, probs, success, masks):
        # probs, success, masks: (T, N). masks[t] = 0 -> obs t starts a new
        # episode, so transition t ended its episode iff masks[t + 1] == 0.
        T = probs.shape[0]
        lam = self.td_lambda
        targets = torch.empty_like(probs[: T - 1])
        g = probs[T - 1]
        for t in reversed(range(T - 1)):
            done_t = 1.0 - masks[t + 1]
            bootstrap = (1.0 - lam) * probs[t + 1] + lam * g
            g = done_t * success[t] + (1.0 - done_t) * bootstrap
            targets[t] = g
        return targets

    def _zero(self, reference):
        zero = reference.sum() * 0.0
        return {"loss": zero, "pred_mean": zero.detach(), "target_mean": zero.detach()}

    def forward(self, aux_loss_state, batch):
        rnn_output = aux_loss_state.get("rnn_output")
        if rnn_output is None:
            return self._zero(next(self.success_head.parameters()))

        logits_flat = self.predict_logits(rnn_output)
        success = batch.get("success_targets")
        masks = batch.get("masks")
        num_envs = batch.get("orion_num_envs")
        if success is None or masks is None or num_envs is None:
            return self._zero(logits_flat)

        n = int(num_envs)
        total = logits_flat.shape[0]
        if n <= 0 or total % n != 0:
            raise ValueError(
                f"ORION: cannot reshape {total} predictions into (T, {n})."
            )
        T = total // n
        if T < 2:
            return self._zero(logits_flat)

        logits = logits_flat.view(T, n)
        success = success.to(logits.dtype).reshape(T, n)
        masks = masks.to(logits.dtype).reshape(T, n)
        if not torch.all((success == 0) | (success == 1)):
            raise ValueError("ORION success flags must be 0 or 1.")

        probs = torch.sigmoid(logits).detach()
        targets = self.compute_targets(probs, success, masks)
        loss = F.binary_cross_entropy_with_logits(logits[: T - 1], targets)

        return {
            "loss": self.loss_scale * loss,
            "pred_mean": probs[: T - 1].mean(),
            "target_mean": targets.mean(),
        }


# Backwards-compatible name used in habitat_baselines/rl/ppo/__init__.py
ORIONStopLoss = ORIONSuccessLoss
