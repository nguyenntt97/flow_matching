"""The contract every simulator model in ``src/`` satisfies.

It is upstream's ``CrowdESSimulatorModel.forward`` signature, verbatim, because
``CrowdES/inference_model.py::process_simulator`` calls it positionally and
reads ``.preds`` and ``.states`` off the result. Anything that wants to run
through upstream's inference path -- including the later flow-matching head --
has to keep this shape.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import torch

from src._upstream import CrowdESSimulatorModelOutput


@runtime_checkable
class SimulatorModel(Protocol):
    #: Whether ``on_fit_start`` should fit KMeans endpoint clusters for this model.
    needs_endpoint_clusters: bool

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
        """Note: ``traj_fut is not None`` -- not ``self.training`` -- selects the
        training branch and the presence of ``.loss``."""
        ...

    def export_pretrained(self, out_dir: str) -> None:
        """Write a directory ``CrowdESSimulatorModel.from_pretrained`` can load."""
        ...
