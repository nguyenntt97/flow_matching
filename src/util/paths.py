"""Repo-root anchoring for configs.

Upstream's yamls use CWD-relative paths (``./datasets/...``) and assume the CWD
is the submodule directory -- which cannot see the data at all, because the
preprocessed tree lives under the *repo root* ``datasets/`` symlink and the
submodule's own ``datasets/`` holds only two JSON files.

Rather than chdir'ing, every path in our configs is written as
``${repo:some/relative/path}`` and resolved to an absolute path at compose
time. The DotDict handed to upstream is then CWD-independent.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(relative: str) -> str:
    """Resolve ``relative`` against the repo root. Absolute inputs pass through."""
    candidate = Path(relative)
    if candidate.is_absolute():
        return str(candidate)
    # Do not call .resolve() on the joined path: datasets/ is a symlink to
    # another filesystem, and resolving it would bake the mount point into
    # cache filenames and logs. Only normalise '..' segments.
    return str(Path(REPO_ROOT / candidate))


def register_resolvers() -> None:
    """Register the ``${repo:...}`` resolver. Safe to call repeatedly."""
    OmegaConf.register_new_resolver("repo", repo_path, replace=True)
