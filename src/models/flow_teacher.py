"""Subsystem 2: the continuous flow-matching teacher, at upstream's protocol seam.

What this replaces, and what it keeps
-------------------------------------
Upstream's ``CrowdESSimulatorModel`` is an *encoder stack* followed by a
deterministic MLP decoder that emits the whole future chunk in one reshape
(``simulator_model.py:176-180``). It is not a diffusion model and not a neural
ODE -- a point worth stating because the design docs assume otherwise.

We keep the encoder stack **verbatim**, by holding a real
``CrowdESSimulatorModel`` as ``self.net`` and calling its own submodules. That
buys three things:

1. the head can be warm-started from a parity checkpoint, giving a clean
   "same encoder, different decoder" ablation;
2. ``src/systems/simulator_system.py:78`` dereferences
   ``self.model.net.config.env_dim`` unconditionally, and that keeps working;
3. the B=8 latent stays exactly upstream's, so the behaviour-mode labels that
   Subsystems 3-5 consume are the same objects upstream clusters.

Only the decoder changes: instead of regressing the future, we learn a
conditional velocity field and integrate it.

Why the B=8 latent is kept rather than reinvented
-------------------------------------------------
``endpoint_cluster_centers`` are KMeans centroids of the future endpoint in a
frame *rotated onto the navmesh waypoint and scaled by waypoint distance*
(``simulator_model.py:200-220``). That is already "heading and pace relative to
A* guidance", which is the mode space design doc 05 asks for. Reuse it.

Caveat on ``preds``
-------------------
Upstream's ``preds`` is a deterministic regression of the future. Ours is a
*sample*. ADE/FDE therefore is not the comparable statistic -- minADE_K/minFDE_K
is. See ``src/models/README_flow_teacher.md``.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional, Sequence

import torch
import torch.nn as nn

from CrowdES.layers import ConcatSquashLinear
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from src._upstream import (
    CrowdESSimulatorConfig,
    CrowdESSimulatorModel,
    CrowdESSimulatorModelOutput,
)
from src.models.endpoint_clusters import EndpointClusterMixin

#: Dropped beside an HF export so nothing mistakes it for a runnable CrowdES
#: simulator. The exported ``net`` still carries upstream's *unused* regression
#: decoder, so ``CrowdESSimulatorModel.from_pretrained`` would load it happily
#: and silently produce a wrong baseline. ``src/evaluate_scene.py`` refuses it.
FLOW_HEAD_MARKER = "FLOW2BT_HEAD.json"


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Standard sinusoidal embedding. ``t`` is (B,) in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class VelocityField(nn.Module):
    r"""$v_\theta(x_t, t, C_s)$ over the flattened future chunk.

    ``ConcatSquashLinear`` is upstream's own conditioning primitive
    (``CrowdES/layers.py:75``): ``layer(x) * sigmoid(gate(ctx)) + bias(ctx)``.
    Using it keeps the head idiomatic to this codebase and keeps the parameter
    count within ~20% of the decoder it replaces.
    """

    def __init__(
        self,
        data_dim: int,
        cond_dim: int,
        hidden_dims: Sequence[int] = (256, 512, 512, 256),
        time_embed_dim: int = 128,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        ctx_dim = cond_dim + time_embed_dim
        dims = [data_dim] + list(hidden_dims) + [data_dim]
        self.layers = nn.ModuleList(
            ConcatSquashLinear(i, o, ctx_dim) for i, o in zip(dims[:-1], dims[1:])
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # torchdiffeq hands back a 0-dim t; the training path hands a (B,) t.
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        ctx = torch.cat([cond, self.time_mlp(timestep_embedding(t, self.time_embed_dim))], dim=1)
        h = x
        for index, layer in enumerate(self.layers):
            h = layer(ctx, h)
            if index < len(self.layers) - 1:
                h = self.activation(h)
        return h


class FlowMatchingSimulator(EndpointClusterMixin):
    """Satisfies ``src/models/protocol.py``; swap in via ``model_cfg._target_``."""

    def __init__(
        self,
        history_length: int = 10,
        future_length: int = 10,
        env_size=(64, 64),
        env_dim: int = 8,
        neighbor_max_num: int = 4,
        latent_dim: int = 8,
        hidden_dim: int = 2048,
        velocity_hidden_dims: Sequence[int] = (256, 512, 512, 256),
        time_embed_dim: int = 128,
        ode_steps: int = 5,
        ode_method: str = "midpoint",
        eval_seed: int = 0,
    ):
        super().__init__()
        hf_config = CrowdESSimulatorConfig(
            history_length=history_length,
            future_length=future_length,
            env_size=list(env_size),
            env_dim=env_dim,
            neighbor_max_num=neighbor_max_num,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        self.net = CrowdESSimulatorModel(config=hf_config)
        self._init_cluster_buffers(latent_dim)

        self.data_dim = future_length * 2
        self.velocity = VelocityField(
            data_dim=self.data_dim,
            cond_dim=hidden_dim + latent_dim,
            hidden_dims=velocity_hidden_dims,
            time_embed_dim=time_embed_dim,
        )
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        self.ode_steps = int(ode_steps)
        self.ode_method = str(ode_method)
        self.eval_seed = int(eval_seed)

    # ------------------------------------------------------------- encoding
    def _context(self, traj_hist, goal, attr, control, neighbor, env_data) -> torch.Tensor:
        """Upstream's encoder, line for line (``simulator_model.py:106-137``).

        Upstream computes this inline inside ``forward`` and never exposes it,
        so it has to be reproduced rather than called. Keep it in step with the
        submodule: it reads upstream's own modules, so only the *wiring* is ours.
        """
        net = self.net
        batch_size = traj_hist.shape[0]
        device = traj_hist.device
        traj_hist = traj_hist.flatten(1)  # (B, T_hist*2)

        if goal is None:
            goal = torch.zeros(batch_size, 2, device=device)
        if attr is None:
            attr = torch.zeros(batch_size, 2, device=device)
        if control is None:
            control = torch.zeros(batch_size, 2, device=device)

        if env_data is not None:
            env_embedding = net.env_encoder(env_data)
        else:
            env_embedding = torch.zeros(batch_size, net.config.env_embedding_dim, device=device)

        if neighbor is not None:
            neighbor = neighbor.flatten(2)  # (B, N, T_hist*2)
            neighbor_embedding = torch.cat([traj_hist.unsqueeze(1), neighbor], dim=1)
            for neighbor_encoder in net.neighbor_encoders:
                neighbor_embedding = neighbor_encoder(neighbor_embedding)
            neighbor_embedding = neighbor_embedding[:, 0]
        else:
            neighbor_embedding = torch.zeros(
                batch_size, net.config.neighbor_embedding_dim, device=device
            )

        embedding = torch.cat(
            [traj_hist, goal, attr, control, neighbor_embedding, env_embedding], dim=1
        )
        for traj_encoder in net.traj_encoders:
            embedding = traj_encoder(embedding, embedding)
        return embedding

    # -------------------------------------------------------------- latents
    def _latent_logits(self, embedding: torch.Tensor) -> torch.Tensor:
        logits = self.net.lantent_predictor(embedding)  # upstream's typo, kept
        logits = logits - logits.mean(dim=1, keepdim=True)
        return logits.clamp(min=-5, max=5)

    def _gt_behavior(self, traj_fut: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        """Nearest cluster centre in the rotation/scale-normalised frame.

        NOTE ``normalize_rotation_scale`` aliases and mutates ``control``
        in place. Upstream is only safe here because the encoder has already
        consumed ``control`` by this point -- so this call must stay *after*
        ``_context``, exactly as upstream orders it.
        """
        endpoint = traj_fut[:, -1]
        endpoint_norm, *_ = self.net.normalize_rotation_scale(endpoint, control)
        distance = endpoint_norm.unsqueeze(1).detach() - self.endpoint_cluster_centers.detach()
        return torch.argmin(torch.linalg.norm(distance, ord=2, dim=2), dim=1).detach()

    @staticmethod
    def _sample_latent(logits, latent_dim, sampling, previous_state, alpha, tmp):
        """Upstream's inference-time behaviour selection (``simulator_model.py:159-174``)."""
        if sampling:
            if previous_state is not None:
                probs = (logits / tmp).softmax(dim=1)
                probs = probs + previous_state * alpha
                probs = probs / probs.sum(dim=1, keepdim=True)
                return torch.distributions.OneHotCategorical(probs=probs).sample()
            return torch.distributions.OneHotCategorical(logits=logits).sample()
        index = torch.argmax(logits, dim=1)
        return torch.nn.functional.one_hot(index, num_classes=latent_dim).float()

    # -------------------------------------------------------------- forward
    def forward(
        self,
        traj_hist: torch.Tensor,
        traj_fut: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        attr: Optional[torch.Tensor] = None,
        control: Optional[torch.Tensor] = None,
        neighbor: Optional[torch.Tensor] = None,
        env_data: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        sampling: bool = True,
        previous_state: Optional[torch.Tensor] = None,
        alpha: Optional[float] = 1.0,
        tmp: Optional[float] = 1.0,
    ) -> CrowdESSimulatorModelOutput:
        # As upstream: the presence of traj_fut -- not self.training -- selects
        # the training branch and the presence of .loss.
        is_training = traj_fut is not None
        self._bind_centers()

        embedding = self._context(traj_hist, goal, attr, control, neighbor, env_data)
        logits = self._latent_logits(embedding)

        loss_l = None
        if is_training:
            gt_behavior = self._gt_behavior(traj_fut, control)
            latent = torch.nn.functional.one_hot(
                gt_behavior, num_classes=self.net.config.latent_dim
            ).float()
            loss_l = nn.CrossEntropyLoss()(logits, gt_behavior)
        else:
            latent = self._sample_latent(
                logits, self.net.config.latent_dim, sampling, previous_state, alpha, tmp
            )

        cond = torch.cat([embedding, latent], dim=1)

        if is_training:
            x_1 = traj_fut.flatten(1)  # (B, T_fut*2)
            x_0 = torch.randn_like(x_1)
            t = torch.rand(x_1.shape[0], device=x_1.device)
            path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
            velocity = self.velocity(path_sample.x_t, t, cond)

            loss_fm = nn.functional.mse_loss(velocity, path_sample.dx_t)
            loss = loss_fm + loss_l * self.net.config.loss_l_scaling

            # Under CondOT, x_t = (1-t) x_0 + t x_1 and dx_t = x_1 - x_0, so
            # x_1 = x_t + (1-t) dx_t exactly. With v ~= dx_t this is the implied
            # endpoint -- free, and it makes a train-time ADE proxy loggable.
            # It is NOT a sample from the model; do not report it as one.
            with torch.no_grad():
                implied = path_sample.x_t + (1.0 - t).unsqueeze(-1) * velocity
            preds = implied.view(-1, self.net.config.future_length, 2)
        else:
            loss = None
            preds = self._integrate(cond, sampling=sampling).view(
                -1, self.net.config.future_length, 2
            )

        if not return_dict:
            output = (preds, logits, latent)
            return ((loss,) + output) if loss is not None else output

        return CrowdESSimulatorModelOutput(loss=loss, logits=logits, preds=preds, states=latent)

    # ------------------------------------------------------------ sampling
    def _integrate(self, cond: torch.Tensor, sampling: bool = True) -> torch.Tensor:
        """Solve dx/dt = v_theta(x, t, cond) from t=0 to t=1.

        ``sampling=False`` must be reproducible, because it is what
        ``validation_step`` scores. Upstream gets determinism by taking the
        argmax latent of a regression decoder; a flow model also needs a fixed
        ``x_0``, so we draw it from a seeded generator. The result is a
        *deterministic sample*, not a conditional mean -- a different statistic
        from upstream's ``preds``.
        """
        batch_size = cond.shape[0]
        device = cond.device
        if sampling:
            x_0 = torch.randn(batch_size, self.data_dim, device=device)
        else:
            generator = torch.Generator(device=device).manual_seed(self.eval_seed)
            x_0 = torch.randn(batch_size, self.data_dim, device=device, generator=generator)

        solver = ODESolver(velocity_model=lambda x, t, **_: self.velocity(x, t, cond))
        return solver.sample(
            x_init=x_0,
            step_size=1.0 / self.ode_steps,
            method=self.ode_method,
            time_grid=torch.tensor([0.0, 1.0]),
        )

    # --------------------------------------------------------------- export
    def export_pretrained(self, out_dir: str) -> None:
        """Write the encoder as an HF dir, plus the flow head and a refusal marker.

        ``net.save_pretrained`` emits upstream's exact key set, including the
        regression decoder this head never uses. That directory therefore loads
        cleanly into ``CrowdESSimulatorModel.from_pretrained`` and would run the
        untrained decoder. The marker is what stops that being a silent result.
        """
        self.net.save_pretrained(out_dir, safe_serialization=False)
        torch.save(
            {
                "velocity": self.velocity.state_dict(),
                "ode_steps": self.ode_steps,
                "ode_method": self.ode_method,
                "eval_seed": self.eval_seed,
                "data_dim": self.data_dim,
            },
            os.path.join(out_dir, "flow_head.pt"),
        )
        with open(os.path.join(out_dir, FLOW_HEAD_MARKER), "w") as handle:
            json.dump(
                {
                    "head": "FlowMatchingSimulator",
                    "refuse_crowdes_framework": True,
                    "why": (
                        "net.* carries upstream's unused regression decoder. Loading this "
                        "directory with CrowdESSimulatorModel.from_pretrained would run that "
                        "decoder and produce a plausible but wrong baseline. Use "
                        "src/runtime/flow2bt_framework.py, which loads flow_head.pt."
                    ),
                },
                handle,
                indent=2,
            )

    def load_flow_head(self, out_dir: str) -> None:
        payload = torch.load(os.path.join(out_dir, "flow_head.pt"), map_location="cpu")
        self.velocity.load_state_dict(payload["velocity"])
        self.ode_steps = int(payload["ode_steps"])
        self.ode_method = str(payload["ode_method"])
        self.eval_seed = int(payload["eval_seed"])

    # ------------------------------------------------------------ warm start
    @torch.no_grad()
    def warm_start_from_parity(self, ckpt_path: str, strict_encoder: bool = True) -> dict:
        """Copy encoder weights out of a ``ParityCrowdESSimulator`` Lightning ckpt.

        Returns the load report so a caller can assert what actually transferred
        rather than trusting that it did.
        """
        state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        encoder_state = {
            key[len("model.net.") :]: value
            for key, value in state.items()
            if key.startswith("model.net.")
        }
        missing, unexpected = self.net.load_state_dict(encoder_state, strict=False)
        if strict_encoder:
            leaked = [key for key in missing if not key.startswith("traj_decoders.")]
            if leaked:
                raise RuntimeError(f"warm start missed non-decoder encoder keys: {leaked[:8]}")
        centers = state.get("model.endpoint_cluster_centers")
        if centers is not None:
            self.endpoint_cluster_centers.copy_(centers)
            self.centers_fitted.fill_(bool(state.get("model.centers_fitted", torch.tensor(True))))
        return {"missing": list(missing), "unexpected": list(unexpected)}
