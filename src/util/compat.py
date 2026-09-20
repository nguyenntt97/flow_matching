"""Dependency checks for constraints upstream's code silently assumes."""

from __future__ import annotations

_checked = False


def check_pandas() -> None:
    """Refuse to start a dataset build on a pandas that upstream cannot run.

    ``utils/trajectory.py::groupby_sliding_window`` reads ``x.agent_id`` inside
    a ``groupby(['agent_id']).apply(...)``. pandas 3.0 stopped passing grouping
    columns into ``apply``, so the group has no such column and the build dies
    with ``AttributeError: 'DataFrame' object has no attribute 'agent_id'`` --
    inside a joblib worker, after several seconds of scene loading, with the
    real cause buried under a BrokenProcessPool or a re-raised remote error.

    Measured: 2.3.3 builds correctly (with a FutureWarning), 3.0.3 fails.
    """
    global _checked
    if _checked:
        return

    import pandas as pd

    major = int(pd.__version__.split(".")[0])
    if major >= 3:
        raise RuntimeError(
            f"pandas {pd.__version__} cannot build the CrowdES dataset.\n\n"
            "utils/trajectory.py::groupby_sliding_window reads `x.agent_id` inside a\n"
            "groupby(['agent_id']).apply(...). pandas 3.0 no longer passes grouping\n"
            "columns into apply, so that attribute does not exist and the build fails\n"
            "in a worker process with a confusing traceback.\n\n"
            "Fix:  pip install 'pandas>=2,<3'\n"
            "See src/environment.yml."
        )
    _checked = True
