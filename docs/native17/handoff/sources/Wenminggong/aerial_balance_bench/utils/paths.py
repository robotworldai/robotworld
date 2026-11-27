"""Path helpers for resources bundled with this project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOURCES_DIR = PROJECT_ROOT / "resources"


def resource_path(*parts: str) -> Path:
    """Return an absolute path inside the local resources directory."""
    return RESOURCES_DIR.joinpath(*parts)

